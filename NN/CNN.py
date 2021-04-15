import pandas as pd
import numpy as np
import csv
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPool2D
from keras.layers.embeddings import Embedding
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
from sklearn.preprocessing import OneHotEncoder

data = []
with open('train_embedded.csv') as f:
    csvReader = csv.reader(f)
    for row in csvReader:
        if row != []:
            data.append(row)

for i in range(len(data)):
    for j in range(len(data[0])):
        data[i][j] = float(data[i][j])

data = np.array(data)

train_data = data[:, 2:]
train_id = data[:, 0]
train_label = data[:, 1]
# print(len(train_data))
train_label = train_label.reshape(len(train_label), 1)
onehot = OneHotEncoder(sparse=False)
train_label = onehot.fit_transform(train_label)

wordSet = set()
for text in train_data:
    for word in text:
        wordSet.add(word)

top_words = len(wordSet) + 1

model = Sequential()
model.add(Dense(64, activation = 'relu', input_shape=(127678, 32)))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

model.compile(optimizer=SGD(lr=0.01), loss=categorical_crossentropy, metrics=['accuracy'])

model_fit = model.fit(train_data, train_label, epochs = 10, validation_split=0.2)


# testing
data = []
with open('test_embedded.csv') as f:
    csvReader = csv.reader(f)
    for row in csvReader:
        if row != []:
            data.append(row)

for i in range(len(data)):
    for j in range(len(data[0])):
        data[i][j] = float(data[i][j])
        
data = np.array(data)

test_data = data[:, 2:]
test_id = data[:, 0]
test_label = data[:, 1]


test_label = test_label.reshape(len(test_label), 1)
onehot = OneHotEncoder(sparse=False)
test_label = onehot.fit_transform(test_label)


score = model.evaluate(test_data, test_label, verbose=0)
print(score)