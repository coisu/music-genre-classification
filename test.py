import os
import numpy as np
import librosa
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GRU, Reshape, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
import traceback
import tensorflow as tf
import matplotlib.pyplot as plt
import threading
from flask_socketio import SocketIO, emit


app = Flask(__name__)
socketio = SocketIO(app)

progress = 0  # 글로벌 진행도 변수
is_training = True  # 트레이닝 플래그

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp3'}

tf.config.set_visible_devices([], 'GPU')

# fma_medium 데이터셋을 사용할 경로
DATASET_PATH = 'fma_medium'
METADATA_PATH = 'fma_metadata/tracks.csv'

# 미리 처리된 데이터를 저장할 파일명
preprocessed_data_file = 'preprocessed_data_fma_medium.npz'

GENRES = ['Classical', 'Pop', 'Rock', 'Jazz', 'Hip-Hop', 'Country', 'Metal', 'Reggae']

# Upload allowed file check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Train the model in a background thread so it doesn't block the server startup
def train_if_needed():
    global is_training
    try:
        if not os.path.exists('music_genre_classifier_saved_model'):
            print("\nNo pre-trained model found. Training a new model...")
            train_model()
        else:
            print("Pre-trained model found. Skipping training.")
    finally:
        is_training = False

# 진행 상태 업데이트 함수
# Store the logs globally
logs = []  

# Modify the update_progress function to include logs
def update_progress(current, total, message=None):
    global progress, logs
    progress = int((current / total) * 100)
    
    # If a log message is passed, append it to the log list
    if message:
        logs.append(message)
    
    # Emit progress and the latest log entry to the frontend
    socketio.emit('progress_update', {
        'progress': progress,
        'log': logs[-1] if logs else ''
    })

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

# Train and save the model
def train_model():
    global progress, is_training
    progress = 0

    if os.path.exists(preprocessed_data_file):
        print("Loading preprocessed data from cache...")
        with np.load(preprocessed_data_file) as data:
            X = data['X']
            y = data['y']
    else:
        print("Processing fma_medium data...")

        # Load metadata
        metadata = pd.read_csv(METADATA_PATH, header=[0, 1], low_memory=False)
        track_genre = metadata['track', 'genre_top']

        # Find existing files
        existing_files = set()
        for root, dirs, files in os.walk(DATASET_PATH):
            for file in files:
                if file.endswith('.mp3'):
                    file_id = os.path.splitext(file)[0]
                    existing_files.add(file_id)

        X, y = [], []
        # total_files = len(track_genre)  # Total number of tracks
        for i, (track_id, genre) in enumerate(track_genre.items()):
            update_progress(i, len(track_genre), f"Processing track {track_id}, genre: {genre}")  # Include log message

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

    # **클래스 가중치 계산**
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',  # 'balanced'는 데이터셋의 빈도에 맞게 가중치를 자동으로 계산함
        classes=np.unique(np.argmax(y_train, axis=1)),  # 클래스 레이블을 추출
        y=np.argmax(y_train, axis=1)  # one-hot 인코딩된 레이블을 원래 형태로 변환
    )
    class_weights = dict(enumerate(class_weights))  # 클래스 가중치를 딕셔너리로 변환

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

    # **Add learning rate to Adam optimizer**
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("-----------------------Starting model training...")

    # **Early Stopping Callback to avoid overfitting**
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # **클래스 가중치 적용하여 모델 훈련**
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test),
              callbacks=[early_stopping], verbose=1, class_weight=class_weights)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Ensure the model directory exists before saving
    model_dir = 'music_genre_classifier_saved_model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(model_dir, save_format='tf')
    
    progress = 100  # Training completed
    is_training = False
    print("Training Finished and model saved.")


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

# Index route
@app.route('/')
def index():
    if is_training:
        return render_template('progress.html')
    else:
        return render_template('upload.html')


# Progress API
@app.route('/progress')
def progress_status():
    global progress
    return jsonify({'progress': progress})

# File upload route
@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        model = load_model()
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

    # Start training in a separate thread
    threading.Thread(target=train_if_needed).start()

    # socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
