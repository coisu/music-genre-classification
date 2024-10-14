import os
import numpy as np
import librosa
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import traceback
import tensorflow as tf
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp3'}
GENRES = ['Classical', 'Pop', 'Rock', 'Jazz', 'Hip-Hop', 'Country', 'Metal', 'Reggae']
DATASET_PATH = 'fma_small'


# 메타데이터에서 장르 정보 가져오기
metadata = pd.read_csv('fma_metadata/tracks.csv', header=[0, 1])

# 'track_id'를 인덱스로 설정하고 'genre_top'을 가져옵니다.
track_genre = metadata['track', 'genre_top']

preprocessed_data_file = 'preprocessed_data.npz'

# Upload allowed file check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Feature extraction function
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc
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

# Train and save the model
def train_model():
    # Load or preprocess data
    if os.path.exists(preprocessed_data_file):
        print("Loading preprocessed data from cache...")
        with np.load(preprocessed_data_file) as data:
            X = data['X']
            y = data['y']
    else:
        print("Processing fma_small data...")
        metadata = pd.read_csv('fma_metadata/tracks.csv', header=[0, 1])

        track_genre = metadata['track', 'genre_top']

        # Scan for existing files
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
                    mfcc = extract_features(file_path)
                    if mfcc is not None:
                        X.append(mfcc)
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

    # Normalize and prepare data
    X = X.astype('float32') / 255.0
    y = to_categorical(y, num_classes=len(GENRES))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(40, 174, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(GENRES), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Save model in SavedModel format
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

# Flask route for file upload
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

        # Load the model and make a prediction
        model = load_or_train_model()
        if model is not None:
            mfcc = extract_features(file_path)
            if mfcc is not None:
                mfcc = mfcc.astype('float32') / 255.0
                mfcc = mfcc[np.newaxis, ..., np.newaxis]
                prediction = model.predict(mfcc)
                predicted_genre = GENRES[np.argmax(prediction)]

                # Save the genre prediction chart as an image
                prediction_chart_path = os.path.join('static', 'prediction_chart.png')
                save_genre_prediction_plot(prediction, prediction_chart_path)  # Save the chart

                return render_template('result.html', genre=predicted_genre, prediction_chart='prediction_chart.png')
    return redirect(request.url)

if __name__ == "__main__":
    # Ensure upload directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Ensure static directory exists for storing the prediction chart image
    if not os.path.exists('static'):
        os.makedirs('static')

    # Start the Flask app in debug mode
    app.run(host='0.0.0.0', port=5000, debug=True)
