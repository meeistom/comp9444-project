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
import parselmouth

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
print(df.shape)

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

def pitch_audio(data, sr,pitch_factor):
    # A positive value increases the pitch, a negative value decreases it.
    # For example, pitch_factor = 2.0 increases the pitch by 2 semitones (one octave).
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=pitch_factor)

def stretch_audio(data):
    '''Stretch the data a certain amount.'''
    amount = np.random.random()
    amount = amount if amount > 0.45 else amount + 0.45 # want to make sure audio isn't too long
    stretched_data = librosa.effects.time_stretch(y=data, rate=amount)
    return librosa.util.fix_length(data=stretched_data,size=len(data))


filename = df.iat[0,1]
emotion = df.iat[0,0]

print(filename)

# data, sr = librosa.load(filename) # REPEATED CODE
# noised_data = add_noise(data)
# pitched_data = pitch_audio(data, sr,pitch_factor=2.0)
# stretched_data = stretch_audio(data)
# stretched_data = librosa.util.fix_length(data=stretched_data,size=len(data))
# onsets = librosa.onset.onset_detect(y=stretched_data, sr=sr)
# print(onsets)
# start_sample = librosa.frames_to_samples(onsets[0])
# end_sample = librosa.frames_to_samples(onsets[-1])
# print(len(stretched_data))
# data_selected = stretched_data[start_sample:end_sample]
# print(data_selected.shape)

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
# F0 (Fundamental Frequency) - The basic frequency of a sound - Emotions often alter the pitch of the voice, which is related to the fundamental frequency
# Jitter (Pitch Jitter) - Variations in the pitch or frequency - Emotions can cause fluctuations in the stability of pitch, which is captured by jitter.
# Shimmer (Amplitude Shimmer) - Variations in the loudness or amplitude - Emotions can cause changes in the stability of volume, which is measured by shimmer.
# Speech Rate - The speed at which words are spoken - Emotions can influence the pace of speech, making it faster or slower, which relates to the speech rate.
def extract_mfcc(data, sr,flatten: bool = True):
    mfcc_feature = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc_feature.T) if not flatten else np.ravel(mfcc_feature.T)

def mfcc_show(data,sr):
    '''Spectrogram corresponding to the audio.'''
    plt.figure(figsize=(10, 5))
    mfccs = librosa.feature.mfcc(y=data, sr=sr)
    print(mfccs.shape)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')  # Displaying  the MFCCs
    plt.show()

# mfcc_show(data,sr)

def extract_rms(data):
    return np.squeeze(librosa.feature.rms(y=data))

def rms_show(data,sr,hop_length):
    '''Plot the audio waveform and the Root Mean Square (RMS) energy over time.'''
    rms = extract_rms(data)
    plt.figure(figsize=(10,5))
    librosa.display.waveshow(data,sr=sr,alpha=0.8)
    times = librosa.frames_to_time(np.arange(len(rms.T)), sr=sr, hop_length=hop_length)
    plt.plot(times, rms.T, label='RMS Energy')
    plt.legend(loc='best')
    plt.show()

# rms_show(data,sr,hop_length=512)

def extract_zcr(data):
    return np.squeeze(librosa.feature.zero_crossing_rate(data))

def extract_F0(data,sr):
    # F0 (fundamental frequency)
    pitches, magnitudes = librosa.piptrack(y=data, sr=sr)
    pitch_track = pitches[np.argmax(magnitudes, axis=0), np.arange(magnitudes.shape[1])]
    return pitch_track

def extract_jitter(filename):
    sound = parselmouth.Sound(filename)
    pointProcess = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
    jitter = parselmouth.praat.call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    return jitter

def extract_shimmer(filename):
    sound = parselmouth.Sound(filename)
    pointProcess = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
    shimmer = parselmouth.praat.call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return shimmer

# print(extract_jitter(filename))
# print(extract_shimmer(filename))

def extract_speech_rate(data):
    zero_crossings = librosa.zero_crossings(data)
    speaking_rate = np.sum(zero_crossings) / len(data)
    return speaking_rate

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
# pitch lower stretch
# pitch higher stretch noise added
# pitch lower stretch recuded noise added
# all together 12 audios

# print(extract_mfcc(data,sr))
# print(extract_rms(data))
# print(extract_zcr(data).shape)

def extract_features(data, sr,path, frame_length=2048, hop_length=512):
    #Compute the Root Mean Square Energy (RMSE) for each frame of the audio with a length of 2048(frame_length) by directly calling the `rmse` function from librosa.
    result = np.array([])
    result = np.hstack((result,
                        extract_zcr(data),
                        extract_rms(data),
                        extract_mfcc(data, sr),
                        extract_F0(data,sr),
                        extract_jitter(path),
                        extract_shimmer(path),
                        extract_speech_rate(data)
                                    ))
    return result


def augment_data(df):
    X, Y = [], []
    print("Feature processing...")
    for path, emotion, ind in zip(df.path, df.emotion, range(df.path.shape[0])):

        duration=2.5
        offset=0.6
        data, sr = librosa.load(path, duration=duration, offset=offset)

        # original data
        res1 = extract_features(data, sr,path)
        features = np.array(res1)

        # noise added
        noise_data = add_noise(data)
        res2 = extract_features(noise_data, sr,path)
        features = np.vstack((features, res2))  # stacking vertically

        # pitch higher
        pitched_higher_data = pitch_audio(data, sr,pitch_factor=2.0)
        res3 = extract_features(pitched_higher_data, sr,path)
        features = np.vstack((features, res3))  # stacking vertically

        # pitch lower
        pitched_lower_data = pitch_audio(data, sr,pitch_factor=-2.0)
        res4 = extract_features(pitched_lower_data, sr,path)
        features = np.vstack((features, res4))  # stacking vertically
        # print('res4=',res4.shape)
        # print('features=',features.shape)

        # stretch
        stretched_data = stretch_audio(data)
        res5 = extract_features(stretched_data,sr,path)
        # print('res5=',res5.shape)
        features = np.vstack((features,res5))  # stacking vertically

        # pitch higher noise added
        pitched_data = pitch_audio(data, sr,pitch_factor=2.0)
        data_noise_pitch_higher = add_noise(pitched_data)
        res6 = extract_features(data_noise_pitch_higher, sr,path)
        features = np.vstack((features, res6))  # stacking vertically

        # pitch higher noise added
        pitched_data = pitch_audio(data, sr,pitch_factor=-2.0)
        data_noise_pitch_lower = add_noise(pitched_data)
        res7 = extract_features(data_noise_pitch_lower, sr,path)
        features = np.vstack((features, res7))  # stacking vertically

        # stretch noise added
        stretched_data = stretch_audio(data)
        data_noise_streched = add_noise(stretched_data)
        res8 = extract_features(data_noise_streched, sr,path)
        features = np.vstack((features, res8))  # stacking vertically

        # pitch higher stretch
        stretched_data = stretch_audio(data)
        data_streched_pitched_higher = pitch_audio(stretched_data,sr,pitch_factor=2.0)
        res9 = extract_features(data_streched_pitched_higher, sr,path)
        features = np.vstack((features, res9))  # stacking vertically

        # pitch lower stretch
        stretched_data = stretch_audio(data)
        data_streched_pitched_lower = pitch_audio(stretched_data,sr,pitch_factor=-2.0)
        res10 = extract_features(data_streched_pitched_lower, sr,path)
        features = np.vstack((features, res10))  # stacking vertically

        # pitch higher stretch noise added
        stretched_data = stretch_audio(data)
        data_noise_streched = add_noise(stretched_data)
        data_noise_streched_pitched_higher = pitch_audio(data_noise_streched,sr,pitch_factor=2.0)
        res11 = extract_features(data_noise_streched_pitched_higher,sr,path)
        features = np.vstack((features,res11))  # stacking vertically

        # pitch lower stretch noise added
        stretched_data = stretch_audio(data)
        data_noise_streched = add_noise(stretched_data)
        data_noise_streched_pitched_lower = pitch_audio(data_noise_streched,sr,pitch_factor=-2.0)
        res12 = extract_features(data_noise_streched_pitched_lower,sr,path)
        features = np.vstack((features,res12))  # stacking vertically

        if ind % 100 == 0:
            print(f"{ind} samples has been processed...")

        for ele in features:
            X.append(ele)
            # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
            Y.append(emotion)

    print("Done.")

    # save features
    features_path = "./archive/features.csv"
    extracted_df = pd.DataFrame(X)
    extracted_df["emotion"] = Y
    extracted_df.to_csv(features_path, index=False)
    extracted_df.head()
    extracted_df = pd.read_csv(features_path)
    print(extracted_df.shape)
    # Fill NaN with 0
    extracted_df = extracted_df.fillna(0)
    print(extracted_df.isna().any())
    print(extracted_df.shape)
    extracted_df.head()

augment_data(df)

