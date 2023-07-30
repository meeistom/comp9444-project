# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, GRU, LSTM, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import optimizers
from keras.layers import Dropout
from keras.optimizers import adam_v2
from keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("features_simple.csv")

### data preparation
# clear NaN
df = df.fillna(0)

# encoder = OneHotEncoder()
# Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

num_features = 4
num_samples = len(df)//num_features
num_parameters = len(df.columns) # emotion included

X = []
Y = []
for i in range(0, len(df), num_features):
    current_X = []
    for j in range(i, i+4):
        current_X.append(df.iloc[j][:-1])
        current_Y = df.iloc[j][-1]
    X.append(current_X)
    Y.append(current_Y)

X = np.array(X)
Y = np.array(Y)

label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7)

def create_LSTM():
    model=Sequential()
    model.add(Conv1D(16, 3, padding='same', activation='relu', input_shape=(X.shape)))
    model.add(BatchNormalization())
    
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=8, activation='softmax'))
    
    model.summary()
    return model
    

model = create_LSTM()
optimizer = adam_v2.Adam()
model.compile(optimizer=optimizer,
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

# rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=4, min_lr=0.0000001)

epochs = 100
history = model.fit(x_train.astype(np.float32), y_train.astype(np.float32), batch_size=128, epochs=epochs, validation_split=0.3, verbose=1)#, validation_data=(x_test, y_test))#, callbacks=[rlrp])

pred_test = model.predict(x_test)
# y_pred = encoder.inverse_transform(pred_test)
# y_test = encoder.inverse_transform(y_test)    
    