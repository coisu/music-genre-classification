import os
from app import train_model

if not os.path.exists('music_genre_classifier_saved_model'):
    print("Training model as it doesn't exist.")
    train_model()
else:
    print("Model already exists. No need to train.")
