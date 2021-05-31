# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from keras import models, layers
from keras.utils import to_categorical, normalize
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

IDS2017_data = pd.read_csv("AppDDoS.csv", header = 0)
IDS2017_data = IDS2017_data.dropna()

print("IDS 2017 Data Shape: " + str(IDS2017_data.shape))

IDS2017_data = IDS2017_data.dropna()
IDS2017_data['class'].replace({'normal' : 'NORMAL'}, inplace=True)
IDS2017_data['class'].replace({'slowbody2' : 'ATTACK', 'slowheaders' : 'ATTACK', 'slowloris' : 'ATTACK'}, inplace=True)
IDS2017_data['class'].replace({'ddossim' : 'ATTACK', 'hulk' : 'ATTACK', 'rudy' : 'ATTACK'}, inplace=True)
IDS2017_data['class'].replace({'slowread' : 'ATTACK', 'goldeneye' : 'ATTACK'}, inplace=True)
# IDS2017_data.replace({'class' : 'Class'})
IDS2017_data['class'].replace({'NORMAL' : 0, 'ATTACK' : 1}, inplace=True)
print(IDS2017_data['class'].value_counts())
print()
IDS2017_data.groupby('class').mean()

ignore_features = [
    'srcip',
    'srcport',
    'dstip',
    'dstport',
    'proto',
    'total_fpackets',
    'total_fvolume',
    'total_bpackets',
    'total_bvolume',
    # 'min_fpktl',
    'mean_fpktl',
    # 'max_fpktl',
    'std_fpktl',
    # 'min_bpktl',
    'mean_bpktl',
    # 'max_bpktl',
    'std_bpktl',
    # 'min_fiat',
    'mean_fiat',
    # 'max_fiat',
    'std_fiat',
    # 'min_biat',
    'mean_biat',
    # 'max_biat',
    'std_biat',
    # 'duration',
    # 'min_active',
    'mean_active',
    # 'max_active',
    'std_active',
    'min_idle',
    'mean_idle',
    'max_idle',
    'std_idle',
    'sflow_fpackets',
    'sflow_fbytes',
    'sflow_bpackets',
    'sflow_bbytes',
    'fpsh_cnt',
    # 'bpsh_cnt',
    'furg_cnt',
    'burg_cnt',
    'total_fhlen',
    'total_bhlen',
    'dscp',
    'firstTime',
    'flast',
    'blast',
    'class'
]

data = IDS2017_data
y_ = data['class']
# x_ = data.drop(['class', 'srcip', 'dstip', 'srcport', 'dstport'], axis=1)
x_ = data.drop(columns=ignore_features, axis=1)
x_ = x_.astype(float)
x_ = normalize(x_)

y_ = y_.to_numpy()
x_ = x_.to_numpy()

y_ = to_categorical(y_)

X_train, X_test, y_train, y_test = train_test_split(x_, y_, test_size = 0.125)
print(X_train.shape)
print(X_test.shape)

# Transforming into One-Hot Encoding
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

print(y_train.shape)
print(y_test.shape)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(X_train.shape)
print(X_test.shape)

# Model Definition
model = models.Sequential()

model.add(layers.Conv1D(32, 3, activation='relu', padding="same", input_shape=(12,1)))
model.add(layers.Conv1D(32, 3, padding="same", activation='relu'))
model.add(layers.MaxPooling1D(pool_size = 2, strides = 1, padding = 'same'))

model.add(layers.Conv1D(64, 3, activation = 'relu', padding = 'same'))
model.add(layers.Conv1D(64, 3, activation = 'relu', padding = 'same'))
model.add(layers.MaxPooling1D(pool_size = 2, strides = 1, padding = 'same'))

model.add(layers.Conv1D(96, 3, activation = 'relu', padding = 'same'))
model.add(layers.Conv1D(96, 3, activation = 'relu', padding = 'same'))
model.add(layers.MaxPooling1D(pool_size = 2, strides = 1, padding = 'same'))

model.add(layers.Flatten())
model.add(layers.Dense(90, activation = 'relu', input_shape = (10,)))

model.add(layers.Dropout(0.4))
model.add(layers.Dense(2, activation = 'softmax'))

model.summary()

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='loss',
    factor=0.2,
    patience=5,
    verbose=0,
    min_delta=0.0001
)

checkpoint = ModelCheckpoint(
    filepath='IDS2017-RT.h5',
    monitor='accuracy',
    mode='max',
    save_best_only=True,
    verbose=0
)

callbacks = [checkpoint, reduce_lr]

model.compile(
    loss='categorical_crossentropy',
    optimizer = Adam(lr = 0.001),
    metrics=['accuracy'])

model.fit(X_train, y_train, epochs = 5, batch_size = 128, verbose = 1, callbacks = callbacks)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(test_acc)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(test_acc)

