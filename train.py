# Pandas
import pandas as pd

# Scikit learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.utils import class_weight

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import to_categorical

# Audio
import librosa
import librosa.display

# Plot
import matplotlib.pyplot as plt

# Utility
import os
import glob
import numpy as np
from tqdm import tqdm
import itertools

data_folders = ["/content/drive/My Drive/BIRDS_AUDIO_CLASSIFIER/"+i for i in os.listdir("/content/drive/My Drive/BIRDS_AUDIO_CLASSIFIER")]
data_folders

dataset = pd.DataFrame(dataset)
dataset = shuffle(dataset, random_state=42)

dataset.info()

dataset.label.value_counts()

plt.figure(figsize=(12,6))
dataset.label.value_counts().plot(kind='bar', title="Dataset distribution")
plt.show()

train, test = train_test_split(dataset, test_size=0.05, random_state=42)

print("Train: %i" % len(train))
print("Test: %i" % len(test))

"""Extract Audio Features"""

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, duration=4)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfccs

extract_features(dataset.iloc[0,0]).shape

# Commented out IPython magic to ensure Python compatibility.
%%time
x_train, x_test = [], []
print("Extract features from TRAIN  and TEST dataset")
for idx in tqdm(range(len(train))):
    x_train.append(extract_features(train.filename.iloc[idx]))

for idx in tqdm(range(len(test))):
    x_test.append(extract_features(test.filename.iloc[idx]))
    

# print(x_train[0:2])

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
print("X train:", x_train.shape)
print("X test:", x_test.shape)

"""Encode Labels"""

# Commented out IPython magic to ensure Python compatibility.
%%time
encoder = LabelEncoder()
encoder.fit(train.label)

y_train = encoder.transform(train.label)
y_test = encoder.transform(test.label)

"""Compute class weights"""

class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

class_weights

"""Shape the input"""

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



print("X train:", x_train.shape)
print("Y train:", y_train.shape)
print()
print("X test:", x_test.shape)
print("Y test:", y_test.shape)

"""Build Model"""

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))
model.add(GlobalAveragePooling2D())

model.add(Dense(len(encoder.classes_), activation='softmax'))
model.summary()

# Compile
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

"""Fit"""

# Commented out IPython magic to ensure Python compatibility.
%%time
history = model.fit(x_train, y_train,
              batch_size=100,
              epochs=300,
              validation_data=(x_test, y_test),
              shuffle=True)

# Save model and weights
model_name = "birds_classify_v3.h5"
model.save(model_name)
print('Saved trained model at %s ' % model_name)

# Evaluate the model
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# classification report
predictions = model.predict(x_test, verbose=1)

y_true, y_pred = [],[]
classes = encoder.classes_
for idx, prediction in enumerate(predictions): 
    y_true.append(classes[np.argmax(y_test[idx])])
    y_pred.append(classes[np.argmax(prediction)])

print(classification_report(y_pred, y_true))
cnf_matrix = confusion_matrix(y_pred, y_true)
cnf_matrix = cnf_matrix.astype(float) / cnf_matrix.sum(axis=1)[:, np.newaxis]
plot_confusion_matrix(cnf_matrix, classes)





