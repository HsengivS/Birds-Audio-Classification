import os
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Audio
import librosa
import librosa.display


model = keras.models.load_model('birds_classify_v2_300_epoch.h5')

classes = ['AmericanCrow', 'BlueJay', 'EasternWoodPewee','NorthernWaterthrush', 'Ovenbird', 'Veery']


def predict_audio(audio_path, classes):
  try:
  
    y, sr = librosa.load(audio_path, duration=4)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    to_predict = np.asarray(mfccs)

    to_predict = to_predict.reshape(1, 40, 173, 1)

    prediction = model.predict(to_predict)

    result = classes[np.argmax(prediction)]

    return result
  except:
    return "file is corrupted"
