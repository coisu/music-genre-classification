import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


preprocessed_data_file = 'preprocessed_data.npz'

DATASET_PATH = 'fma_small'
GENRES = ['Classical', 'Pop', 'Rock', 'Jazz', 'Hip-Hop', 'Country', 'Metal', 'Reggae']

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
        print(f"Error encountered while parsing file: {file_path} - {e}")
        return None

# 캐시 파일이 존재하면 파일에서 불러옴
if os.path.exists(preprocessed_data_file):
    print("Loading preprocessed data from cache...")
    with np.load(preprocessed_data_file) as data:
        X = data['X']
        y = data['y']

else:
    print("Processing fma_small data...")

    # 메타데이터 로드
    metadata = pd.read_csv('fma_metadata/tracks.csv', header=[0, 1])

    # 필요한 열만 추출 (트랙 ID와 장르)
    track_genre = metadata['track', 'genre_top']

    # 데이터 디렉토리 설정
    # DATASET_PATH = 'fma_small'
    # GENRES = ['Classical', 'Pop', 'Rock', 'Jazz', 'Hip-Hop', 'Country', 'Metal', 'Reggae']

    existing_files = set()
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith('.mp3'):
                file_id = os.path.splitext(file)[0]
                existing_files.add(file_id)

    # 특징 추출 함수
    # def extract_features(file_path, max_pad_len=174):
    #     try:
    #         audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    #         mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    #         pad_width = max_pad_len - mfcc.shape[1]
    #         if pad_width > 0:
    #             mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    #         else:
    #             mfcc = mfcc[:, :max_pad_len]
    #         return mfcc
    #     except Exception as e:
    #         print(f"Error encountered while parsing file: {file_path} - {e}")
    #         return None

    # 데이터와 레이블 저장
    X = []
    y = []

    for i, (track_id, genre) in enumerate(track_genre.items()):
        try:
            track_id_int = int(track_id)
            track_id_str = f"{track_id_int:06}"  # 트랙 ID를 6자리 문자열로 포맷팅

            # 실제 파일이 존재하고, 장르가 유효한 경우만 처리
            if track_id_str in existing_files and genre in GENRES:
                file_path = os.path.join(DATASET_PATH, track_id_str[:3], f"{track_id_str}.mp3")

                # 처리 중인 파일 및 인덱스 출력
                print(f"Processing file {i+1}: {file_path}")

                mfcc = extract_features(file_path)
                if mfcc is not None:
                    X.append(mfcc)
                    y.append(GENRES.index(genre))

                # 매 100번째 파일마다 상태 출력
                if (i+1) % 100 == 0:
                    print(f"Processed {i+1} files so far...")

        except ValueError:
            print(f"Invalid track ID: {track_id}")

    # 데이터를 numpy 배열로 변환
    X = np.array(X)
    y = np.array(y)

    # 캐시에 저장
    np.savez(preprocessed_data_file, X=X, y=y)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


X = X.astype('float32') / 255.0

# 레이블을 원-핫 인코딩으로 변환
y = to_categorical(y, num_classes=len(GENRES))

# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

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

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=32,
                    validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

model.save('music_genre_classifier.keras')

# model.save('music_genre_classifier.h5')

def predict_genre(file_path):
    mfcc = extract_features(file_path)
    if mfcc is not None:
        mfcc = mfcc.astype('float32') / 255.0
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        prediction = model.predict(mfcc)
        predicted_genre = GENRES[np.argmax(prediction)]
        return predicted_genre
    else:
        return "Error in processing the file."


file_path = 'TooSweet_Hozier.mp3'
predicted_genre = predict_genre(file_path)
print(f"Tested File: {file_path}")
print(f"Predicted genre: {predicted_genre}")


