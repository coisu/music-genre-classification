import os
import numpy as np
import librosa
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GRU, Reshape, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import traceback
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp3'}

# fma_medium 데이터셋을 사용할 경로
DATASET_PATH = 'fma_medium'
METADATA_PATH = 'fma_metadata/tracks.csv'

# 미리 처리된 데이터를 저장할 파일명
preprocessed_data_file = 'preprocessed_data_fma_medium.npz'

GENRES = ['Classical', 'Pop', 'Rock', 'Jazz', 'Hip-Hop', 'Country', 'Metal', 'Reggae']

# Upload allowed file check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Feature extraction function using mel-spectrogram
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to decibels
        pad_width = max_pad_len - mel_spec_db.shape[1]
        if pad_width > 0:
            mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :max_pad_len]
        return mel_spec_db
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None


# Function to save prediction results as a bar chart image
def save_genre_prediction_plot(prediction, output_image_path):
    plt.figure(figsize=(10, 6))
    plt.bar(GENRES, prediction[0] * 100)
    plt.xlabel('Genres')
    plt.ylabel('Percentage Match (%)')
    plt.title('Music Genre Prediction')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()

# Load the trained model or train it if it doesn't exist
def load_or_train_model():
    if not os.path.exists('music_genre_classifier_saved_model'):
        print("No pre-trained model found. Training a new model...")
        train_model()
    else:
        print("Pre-trained model found. Skipping training.")
    return load_model()


def save_class_weights(y_train, file_path='class_weights.txt'):
    y_train_numeric = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_numeric), y=y_train_numeric)
    class_weights_dict = dict(enumerate(class_weights))

    with open(file_path, 'w') as f:
        for genre_index, weight in class_weights_dict.items():
            f.write(f"Genre Index: {genre_index}, Weight: {weight}\n")
    
    print("Class weights saved to", file_path)

# Train and save the model
def train_model():
    if os.path.exists(preprocessed_data_file):
        print("Loading preprocessed data from cache...")
        with np.load(preprocessed_data_file) as data:
            X = data['X']
            y = data['y']
    else:
        print("Processing fma_medium data...")

        # Load metadata
        metadata = pd.read_csv(METADATA_PATH, header=[0, 1])
        track_genre = metadata['track', 'genre_top']

        # Find existing files
        existing_files = set()
        for root, dirs, files in os.walk(DATASET_PATH):
            for file in files:
                if file.endswith('.mp3'):
                    file_id = os.path.splitext(file)[0]
                    existing_files.add(file_id)

        X, y = [], []
        for i, (track_id, genre) in enumerate(track_genre.items()):
            try:
                track_id_str = f"{int(track_id):06}"  # Format track ID
                if track_id_str in existing_files and genre in GENRES:
                    file_path = os.path.join(DATASET_PATH, track_id_str[:3], f"{track_id_str}.mp3")
                    mel_spec_db = extract_features(file_path)
                    if mel_spec_db is not None:
                        X.append(mel_spec_db)
                        y.append(GENRES.index(genre))
                    else:
                        print(f"Could not extract features from {file_path}")
                    if (i + 1) % 100 == 0:
                        print(f"Processed {i + 1} files so far...")
            except ValueError:
                print(f"Invalid track ID: {track_id}")

        if len(X) == 0 or len(y) == 0:
            print("No valid data found in the dataset.")
            return

        X = np.array(X)
        y = np.array(y)
        np.savez(preprocessed_data_file, X=X, y=y)

    X = X[..., np.newaxis]  # Add channel dimension
    X = X.astype('float32') / 255.0
    y = to_categorical(y, num_classes=len(GENRES))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train_numeric = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_numeric), y=y_train_numeric)
    class_weights = dict(enumerate(class_weights))

    save_class_weights(y_train)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 174, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Reshape((30 * 42, 64)))  
    model.add(GRU(64, return_sequences=False))
    model.add(Dropout(0.3))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(GENRES), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weights)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    model.save('music_genre_classifier_saved_model', save_format='tf')

# Load the trained model
def load_model():
    try:
        model = tf.keras.models.load_model('music_genre_classifier_saved_model')
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        model = load_or_train_model()
        if model is not None:
            mel_spec_db = extract_features(file_path)
            if mel_spec_db is not None:
                mel_spec_db = mel_spec_db[np.newaxis, ..., np.newaxis]
                mel_spec_db = mel_spec_db.astype('float32') / 255.0
                prediction = model.predict(mel_spec_db)
                predicted_genre = GENRES[np.argmax(prediction)]

                prediction_chart_path = os.path.join('static', 'prediction_chart.png')
                save_genre_prediction_plot(prediction, prediction_chart_path)

                return render_template('result.html', genre=predicted_genre, prediction_chart='prediction_chart.png')
    return redirect(request.url)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    if not os.path.exists('static'):
        os.makedirs('static')

    app.run(host='0.0.0.0', port=5000, debug=True)
