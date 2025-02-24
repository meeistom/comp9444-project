###########
# IMPORTS #
###########
import os
# import torch
# import torch.nn as nn
# import torchaudio
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import librosa
from IPython.display import Audio

#########
# PATHS #
#########
crema_path = './archive/Crema'
ravdess_path = './archive/Ravdess/audio_speech_actors_01-24'
savee_path = './archive/Savee'
tess_path = './archive/Tess'

################
# DICTIONARIES # probably move these to be with the retrieve data code in the notebook
################
crema_to_emotion_dct = {
    'ANG': 'anger',
    'DIS': 'disgust',
    'FEA': 'fear',
    'HAP': 'happiness',
    'NEU': 'neutral',
    'SAD': 'sadness',
}

ravdess_to_emotion_dct = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happiness',
    '04': 'sadness',
    '05': 'anger',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprise',
}

savee_to_emotion_dct = {
    'a': 'anger',
    'd': 'disgust',
    'f': 'fear',
    'h': 'happiness',
    'n': 'neutral',
    'sa': 'sadness',
    'su': 'surprise',
}

tess_to_emotion_dct = {
    'angry': 'anger',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happiness',
    'neutral': 'neutral',
    'ps': 'surprise', # Pleasant Surprise
    'sad': 'sadness',
}

#################
# RETRIEVE DATA #
#################

# print(os.getcwd())

# CREMA #
crema_paths = []

for file in os.listdir(crema_path):
    emotion = crema_to_emotion_dct[file.split('_')[2]]
    crema_paths.append((emotion, crema_path+'/'+file))

crema_df = pd.DataFrame.from_dict(crema_paths)
crema_df.rename(columns={0:'emotion',1:'path'}, inplace=True)
print(crema_df.head())

# RAVDESS #
ravdess_paths = []

for folder in os.listdir(ravdess_path):
    for file in os.listdir(ravdess_path+'/'+folder):
        emotion = ravdess_to_emotion_dct[file.split('-')[2]]
        ravdess_paths.append((emotion, ravdess_path+'/'+folder+'/'+file))

ravdess_df = pd.DataFrame.from_dict(ravdess_paths)
ravdess_df.rename(columns={0:'emotion',1:'path'}, inplace=True)
print(ravdess_df.head())

# SAVEE #
savee_paths = []

for file in os.listdir(savee_path):
    x = file.split('_')
    y = x[1][0:2] if x[1][0] == 's' else x[1][0]
    emotion = savee_to_emotion_dct[y]
    savee_paths.append((emotion, savee_path+'/'+file))

savee_df = pd.DataFrame.from_dict(savee_paths)
savee_df.rename(columns={0:'emotion',1:'path'}, inplace=True)
print(savee_df.head())

# TESS #
tess_paths = []

for folder in os.listdir(tess_path):
    for file in os.listdir(tess_path+'/'+folder):
        emotion = tess_to_emotion_dct[file.split('.')[0].split('_')[2]]
        tess_paths.append((emotion, tess_path+'/'+folder+'/'+file))

tess_df = pd.DataFrame.from_dict(tess_paths)
tess_df.rename(columns={0:'emotion',1:'path'}, inplace=True)
print(tess_df.head())

# JOIN DATASETS #
df = pd.concat([crema_df, ravdess_df, savee_df, tess_df], axis=0)
print(df.head())

######################
# DATA PREPROCESSING #
######################

# librosa stuff

# audio characterisitics i care about
# pitch
# frequency
# amplitude
# timbre
# duration
# noise



# for graphics
# create waveform
# create spectrogram


def generate_waveform(emotion, filename, data, sr):
    plt.figure(figsize=(9, 4))
    plt.title(f'{emotion.title()} waveform | {filename}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (% Scale)') # time - floating point
    librosa.display.waveshow(y=data, sr=sr)
    return plt

def generate_spectrogram(emotion, filename, data, sr):
    x = librosa.stft(data) # time - frequency domain
    xdB = librosa.amplitude_to_db(abs(x))

    plt.figure(figsize=(9, 4))
    plt.title(f'{emotion.title()} spectrogram | {filename}')
    librosa.display.specshow(xdB, sr=sr, x_axis='time', y_axis='hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    cb = plt.colorbar()
    cb.set_label('Amplitude (dB)', rotation=270, labelpad=20)
    return plt




# for preprocessing
# data augmentation
# add noise - add noise to speech to perhaps muffle something unnecessary
# shift pitch - since some people talk at a higher or lower pitch, altering this can create a greater range
# stretch - stretching audio can provide more data to analyse
# can combine some of these, like add noise on stretched pitch

def add_noise(data):
    '''Signal to noise ratio is as follows: SNR = 2 * (A_signal - A_noise)
    according to the derivation on https://en.wikipedia.org/wiki/Signal-to-noise_ratio.'''
    A_signal = math.sqrt(np.mean(data**2))
    noise = np.random.normal(0, A_signal/3, data.shape)
    A_noise = math.sqrt(np.mean(noise**2))
    SNR = 2 * (A_signal - A_noise)
    return data + noise * SNR

def pitch_audio(data, sr):
    '''Shift the pitch a certain amount.'''
    amount = np.random.random()
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=amount)

def stretch_audio(data):
    '''Stretch the data a certain amount.'''
    amount = np.random.random()
    amount = amount if amount > 0.45 else amount + 0.45 # want to make sure audio isn't too long
    return librosa.effects.time_stretch(y=data, rate=amount)

filename = df.iat[0,1]
emotion = df.iat[0,0]

print(filename)

data, sr = librosa.load(filename) # REPEATED CODE
noised_data = add_noise(data)
pitched_data = pitch_audio(data, sr)
stretched_data = stretch_audio(data)

# generate_waveform(emotion, filename, data, sr).show()
# generate_waveform(emotion, filename, noised_data, sr).show()
# generate_waveform(emotion, filename, pitched_data, sr).show()
# generate_waveform(emotion, filename, stretched_data, sr).show()
# generate_spectrogram(emotion, filename, data, sr).show()
# generate_spectrogram(emotion, filename, noised_data, sr).show()
# generate_spectrogram(emotion, filename, pitched_data, sr).show()
# generate_spectrogram(emotion, filename, stretched_data, sr).show()

# feature extraction
# timbre - mfcc - emotions often alter tone in speech which is relate to timbre
# amplitude - rms i think - emotions can cause heightened volume 
# smoothness - zcr - various emotions can be smooth or rapid and jumping all over the place

def extract_mfcc(data, sr):
    return librosa.feature.mfcc(y=data, sr=sr)

def extract_rms(data):
    return librosa.feature.rms(y=data)

def extract_zcr(data):
    return librosa.feature.zero_crossing_rate(data)

# can turn 1 audio into all of these:
# original
# noise added
# pitch higher
# pitch lower
# stretch
# pitch higher noise added
# pitch lower noise added
# stretch noise added
# pitch higher stretch
# pitch higher stretch noise added
# pitch lower stretch
# pitch lower stretch recuded noise added
# all together 12 audios



def augment_data(df):

    # extract features straight away

    return

