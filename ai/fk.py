import librosa
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("data_path.csv")
print(df.head())


def extract_mfcc(filename):
    print(filename)
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


X_mfcc = df['Path'].apply(lambda x: extract_mfcc(x))

X = [x for x in X_mfcc]
X = np.array(X)
print(X)

## input split
X = np.expand_dims(X, -1)

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
y = enc.fit_transform(df[['Emotions']])
y = y.toarray()
"""fk_x = pd.DataFrame(X, columns=["x"])
fk_y = pd.DataFrame(y, columns=["y"])
fk=pd.concat([fk_x,fk_y],axis=1)
fk.head()
fk.to_csv("fk.csv")"""
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential([
    LSTM(128, return_sequences=False, input_shape=(40, 1)),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dropout(0.4),
    Dense(8, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# Train the model
history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=256)

epochs = list(range(50))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, label='train accuracy')
plt.plot(epochs, val_acc, label='val accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, label='train loss')
plt.plot(epochs, val_loss, label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
