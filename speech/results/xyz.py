import glob
import os

import librosa
import numpy as np
import soundfile
from tensorflow.python.keras.models import load_model

model1 = load_model("ABCDFHNPS-c-LSTM-layers-2-2-units-128-128-dropout-0.3_0.3_0.3_0.3.h5")
model2 = load_model("AHNPS-c-LSTM-layers-2-2-units-128-128-dropout-0.3_0.3_0.3_0.3.h5")


def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
    return result


# all emotions on RAVDESS dataset
int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# we allow only these emotions ( feel free to tune this on your need )

def load_data(file):
    X = ""
    # extract speech features
    features = extract_feature(file, mfcc=True, chroma=True, mel=True)
    # add to data
    features = np.expand_dims(features, axis=-2)
    X = np.expand_dims(features, axis=-2)
    print(X.ndim)
    print(model2.summary())
    p1 = model1.predict(features)
    p2 = model2.predict(features)
    # print(p1)
    print(p2)


load_data("tess/TESS Toronto emotional speech set data/OAF_Sad/OAF_said_sad.wav")
