import os
import pathlib
import warnings
from functools import partial

import joblib
import librosa
import librosa.display

import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.layers import MaxPooling2D
from keras.legacy_tf_layers.core import flatten
from numpy.random import shuffle
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, \
    make_scorer, calinski_harabasz_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.layers import Dense, Dropout, MaxPooling1D, Conv1D, Flatten, LSTM, Conv2D
from tensorflow.python.keras.utils.np_utils import to_categorical

warnings.filterwarnings('ignore')

print("working")


# tess

def data_load():
    Crema = r"crema\AudioWAV"
    crema_directory_list = os.listdir(Crema)

    crema_emotion = []
    crema_path = []

    for file in crema_directory_list:
        # storing file paths
        crema_path.append(Crema + f"\{file}")
        # storing file emotions
        part = file.split('_')
        if part[2] == 'SAD':
            crema_emotion.append('sad')
        elif part[2] == 'ANG':
            crema_emotion.append('angry')
        elif part[2] == 'DIS':
            crema_emotion.append('disgust')
        elif part[2] == 'FEA':
            crema_emotion.append('fear')
        elif part[2] == 'HAP':
            crema_emotion.append('happy')
        elif part[2] == 'NEU':
            crema_emotion.append('neutral')
        else:
            print(part)
            crema_emotion.append('Unknown')
    emotion_df = pd.DataFrame(crema_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(crema_path, columns=['Path'])
    Crema_df = pd.concat([emotion_df, path_df], axis=1)
    print(Crema_df.head())
    print(crema_emotion.__len__())
    # creating Dataframe using all the 4 dataframes we created so far.
    Crema_df.to_csv("data_path.csv", index=False)
    print(Crema_df.head())
    plt.title('Count of Emotions', size=16)
    sns.countplot(Crema_df.Emotions)
    plt.ylabel('Count', size=12)
    plt.xlabel('Emotions', size=12)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()


# data_load()
data_load()


def create_waveplot(data, sr, e):
    plt.figure(figsize=(10, 3))
    plt.title('Waveplot for audio with {} emotion'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.show()


def create_spectrogram(data, sr, e):
    # stft function converts the data into short term fourier transform
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()


data_path = pd.read_csv("data_path.csv")


def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)


def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


path = np.array(data_path.Path)[1]
data, sample_rate = librosa.load(path)


def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen
    # above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3))  # stacking vertically

    return result


def data_imbalance_shuffle(X, Y):
    # all classes
    x_return = []
    y_return = []
    di = {}
    for x, y in zip(X, Y):
        try:
            di[y].append(x)
        except KeyError:
            di[y] = [x]
    min_val = min([len(di[i]) for i in di.keys()])
    for i in di.keys():
        shuffle(di[i])
        for u in range(min_val):
            x_return.append(di[i][u])
            y_return.append(i)

    return x_return, y_return


def work():
    X, Y = [], []
    for path, emotion in zip(data_path.Path, data_path.Emotions):
        feature = get_features(path)
        print(path)
        for ele in feature:
            X.append(ele)
            # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
            Y.append(emotion)
    print(len(X), len(Y), data_path.Path.shape)
    Features = pd.read_csv('features.csv')
    Features = Features.dropna()
    Features = Features.replace("ps", "surprise")
    Y = Features['labels']
    X = Features.iloc[:, :-1].values
    Y = Features['labels']
    print(X)
    print(Y.unique())
    y_ = Y.unique()
    print(len(y_))
    # As this is a multiclass classification problem noncomprehending our Y.
    encoder = LabelEncoder()
    encoder.fit(Y)
    print(len(list(encoder.classes_)))
    Y = encoder.transform(Y)
    # y = e.transform(Y)

    # splitting data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)
    # scaling our data with sklearn Standard scaler
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # making our data compatible to model.

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


def own(x_train,y_train,x_test,y_test,y_):
    model = Sequential()
    model.add(Flatten(input_shape=[x_train.shape[1]]))
    model.add(
        Dense(units=256))
    model.add(Dropout(0.2))
    model.add(Dense(units=len(y_), activation="softmax"))
    print(model)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=64, epochs=7, validation_data=(x_test, y_test))
    print("Accuracy of our model on test data : ", model.evaluate(x_test, y_test)[1] * 100, "%")
    print(model.summary())
    model.save("m.h5")
    fig, ax = plt.subplots(1, 2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    test_acc = history.history['val_accuracy']
    test_loss = history.history['val_loss']


def custom(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    return min(8 * m.trace() * accuracy_score(y_true, y_pred) / 576)


def mlp():
    X, Y = [], []
    """for path, emotion in zip(data_path.Path, data_path.Emotions):
        feature = get_features(path)
        print(path)
        for ele in feature:
            X.append(ele)
            # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
            Y.append(emotion)
    print(len(X), len(Y), data_path.Path.shape)"""
    Features = pd.read_csv('features.csv')
    Features = Features.dropna()
    Features = Features.replace("ps", "surprise")
    Y = Features['labels']
    X = Features.iloc[:, :-1].values
    Y = Features['labels']
    X, Y = data_imbalance_shuffle(X, Y)
    # As this is a multiclass classification problem noncomprehending our Y.
    encoder = LabelEncoder()
    encoder.fit(Y)
    print(len(list(encoder.classes_)))
    Y = encoder.transform(Y)
    # y = e.transform(Y)

    # splitting data
    print(confusion_matrix(Y, Y))
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)
    # scaling our data with sklearn Standard scaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(X=x_train)
    x_test = scaler.transform(x_test)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # making our data compatible to model.
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    m = MLPClassifier(hidden_layer_sizes=[256, 128], verbose=4, max_iter=700)
    g = m
    """
    s = m.fit(x_train, y_train)
    print(s)
    y_pred = m.predict(x_test)

    print(m.classes_)

    print(m.loss)
    print(m.get_params())
    plt.imshow(confusion_matrix(y_test, y_pred), )

    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    print(f1_score(y_test, y_pred, average='macro'))"""

    s = g.fit(X, Y)

    y_pred = m.predict(X)
    print(classification_report(Y, y_pred))
    print(accuracy_score(Y, y_pred))
    print(f1_score(Y, y_pred, average='macro'))
    plt.imshow(confusion_matrix(Y, y_pred))
    print(confusion_matrix(Y, y_pred))
    print(confusion_matrix(Y, y_pred).trace())
    plt.waitforbuttonpress()


def dnn():
    Features = pd.read_csv('features.csv')
    Features = Features.dropna()
    Features = Features.replace("ps", "surprise")
    f = Features.iloc[:, :-1]
    Y = Features['labels']
    X = Features.iloc[:, :-1].values
    Y = Features['labels']

    fea = [tf.feature_column.numeric_column(key=i) for i in f.keys()]
    X, Y = data_imbalance_shuffle(X, Y)
    # As this is a multiclass classification problem noncomprehending our Y.
    encoder = LabelEncoder()
    encoder.fit(Y)
    print(len(list(encoder.classes_)))
    Y = encoder.transform(Y)
    # y = e.transform(Y)

    # splitting data
    print(confusion_matrix(Y, Y))
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)
    # scaling our data with sklearn Standard scaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(X=x_train)
    x_test = scaler.transform(x_test)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # making our data compatible to model.
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    t = tf.estimator.DNNClassifier(
        feature_columns=fea,
        hidden_units=[64, 8],
        n_classes=8)

    def input_fn(features, labels, Training=True, batch_size=64):
        dataset = tf.data.Dataset.from_tensor_slices(features, labels)
        if Training:
            dataset = dataset.shuffle(100).repeat()
        return dataset.batch(batch_size)

    t.train(
        input_fn=lambda: input_fn(fea, y_train, True),
        steps=700

    )


# print(recall_score(y_test,y_pred))
"""    g = GridSearchCV(
        estimator=m,
        param_grid=[{"hidden_layer_sizes": [2048 // i, 256]} for i in range(2, 8, 2)],
        n_jobs=-1,

        scoring={"f1": make_scorer(accuracy_score)},
        refit="f1",
        return_train_score=True

    )"""


def last():
    Features = pd.read_csv('features.csv')
    Features = Features.dropna()
    Features = Features.replace("ps", "surprise")
    f = Features.iloc[:, :-1]
    Y = Features['labels']
    X = Features.iloc[:, :-1].values
    Y = Features['labels']

    # As this is a multiclass classification problem noncomprehending our Y.
    encoder = LabelEncoder()
    encoder.fit(Y)
    print(len(list(encoder.classes_)))
    Y = encoder.transform(Y)
    # y = e.transform(Y)

    # splitting data
    print(confusion_matrix(Y, Y))
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)
    # scaling our data with sklearn Standard scaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(X=x_train)
    x_test = scaler.transform(x_test)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train, y_train = data_imbalance_shuffle(x_train, y_train)
    x_test, y_test = data_imbalance_shuffle(x_test, y_test)
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)
    print(confusion_matrix(y_train, y_train))
    print(confusion_matrix(y_test, y_test))
    m = Sequential([
        Flatten(input_shape=[x_train.shape[-1]]),
        Dense(units=180, activation="relu"),
        Dense(units=8, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(x_train, y_train, batch_size=164, epochs=7000)
    l = m.predict_classes(x_test)
    print(l)
    print(m.count_params())
    y_pred = m.predict(x_test)
    print(confusion_matrix(y_test, l))
    print(classification_report(y_test, l))
    print(accuracy_score(y_test, l))


def conv():
    Features = pd.read_csv('features.csv')
    Features = Features.dropna()
    Features = Features.replace("ps", "surprise")
    f = Features.iloc[:, :-1]
    Y = Features['labels']
    X = Features.iloc[:, :-1].values
    Y = Features['labels']

    # As this is a multiclass classification problem noncomprehending our Y.
    encoder = LabelEncoder()
    encoder.fit(Y)
    print(len(list(encoder.classes_)))
    Y = encoder.transform(Y)
    # y = e.transform(Y)

    # splitting data
    print(confusion_matrix(Y, Y))
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)
    # scaling our data with sklearn Standard scaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(X=x_train)
    x_test = scaler.transform(x_test)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train, y_train = data_imbalance_shuffle(x_train, y_train)
    x_test, y_test = data_imbalance_shuffle(x_test, y_test)
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)
    print(confusion_matrix(y_train, y_train))
    print(confusion_matrix(y_test, y_test))
    print(x_train.shape)
    x_train = x_train.reshape([x_train.shape[0], 9, 6, 3])
    x_test = x_test.reshape([x_test.shape[0], 9, 6, 3])

    print(x_train.shape)
    m = Sequential()
    m.add(Conv2D(32, (3, 3), activation="relu", input_shape=[9, 6, 3]))
    m.add(MaxPooling2D((2, 2)))
    m.add(Conv2D(64, (1, 1), activation="relu"))

    m.add(Flatten())
    m.add(Dense(units=180, activation="relu"))
    m.add(Dense(units=8, activation="softmax"))
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(x_train, y_train, batch_size=64, epochs=300)
    l = m.predict_classes(x_test)
    print(l)
    print(m.count_params())
    y_pred = m.predict(x_test)
    print(confusion_matrix(y_test, l))
    print(classification_report(y_test, l))
    print(accuracy_score(y_test, l))


# conv()
def rnn():
    Features = pd.read_csv('features.csv')
    Features = Features.dropna()
    Features = Features.replace("ps", "surprise")
    f = Features.iloc[:, :-1]
    Y = Features['labels']
    X = Features.iloc[:, :-1].values
    Y = Features['labels']

    # As this is a multiclass classification problem noncomprehending our Y.
    encoder = LabelEncoder()
    encoder.fit(Y)
    print(len(list(encoder.classes_)))
    Y = encoder.transform(Y)
    # y = e.transform(Y)

    # splitting data
    print(confusion_matrix(Y, Y))
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)
    # scaling our data with sklearn Standard scaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(X=x_train)
    x_test = scaler.transform(x_test)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train, y_train = data_imbalance_shuffle(x_train, y_train)
    x_test, y_test = data_imbalance_shuffle(x_test, y_test)
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)
    print(confusion_matrix(y_train, y_train))
    print(confusion_matrix(y_test, y_test))
    print(x_train.shape)
    x_train = x_train.reshape([x_train.shape[0], 81, 2])
    x_test = x_test.reshape([x_test.shape[0], 81, 2])

    print(x_train.shape)
    m = Sequential([
        LSTM(128, return_sequences=False, input_shape=[81, 2]),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dropout(0.4),
        Dense(8, activation='softmax')
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(x_train, y_train, batch_size=64, epochs=70)
    l = m.predict_classes(x_test)
    print(l)
    print(m.count_params())
    y_pred = m.predict(x_test)
    print(confusion_matrix(y_test, l))
    print(classification_report(y_test, l))
    print(accuracy_score(y_test, l))


"""    Ravdess = "ravdess"
    paths = []
    labels = []
    for dirname, _, filenames in os.walk('../speech/results/tess'):

        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            print(os.path.join(dirname, filename))
            label = filename.split('_')[-1]
            label = label.split('.')[0]
            labels.append(label.lower())
    # dataframe for emotion of files
    emotion_df = pd.DataFrame(labels, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(paths, columns=['Path'])
    Tess_df = pd.concat([emotion_df, path_df], axis=1)

    ravdess_directory_list = os.listdir(Ravdess)
    file_emotion = []
    file_path = []
    for dir in ravdess_directory_list:
        # as their are 20 different actors in our previous directory we need to extract files for each actor.
        actor = os.listdir(Ravdess + f"\{dir}")
        for file in actor:

            if pathlib.Path(Ravdess + f"\{dir}" + f"\{file}").is_file():
                part = file.split('.')[0]
                part = part.split('-')

                # third part in each file represents the emotion associated to that file.
                file_emotion.append(int(part[2]))
                file_path.append(Ravdess + f'\{dir}' + f'\{file}')
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

    # changing integers to actual emotions.
    Ravdess_df.Emotions.replace(
        {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'},
        inplace=True)"""
