###########
# IMPORTS #
###########
import os
# import torch
# import torch.nn as nn
# import torchaudio
import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import librosa

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