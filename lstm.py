# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13HWnSWVciUz_DCi04UtSByvyzQtxCgZs
"""

# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dense
from keras.layers.convolutional import Conv1D
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = len(vocab_w)+1
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
# model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(LSTM(100))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print(scores)

import xml.etree.ElementTree as ET

def parseXML(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    posts = []
    texts=  (root.findall("WRITING/TEXT"))
    for text in texts:
        posts.append(text.text)
    return posts

def read_labels(labelfile):
    f = open(labelfile, "r")
    ret = {}
    while True:
        line = f.readline()
        if len(line)==0:
            break
        xml,label = line.split()[0].strip(),line.split()[1].strip()
        label = int(label)
        ret[xml] = label
    # print(ret)
    return ret

def load_data_and_label(labelfile):
    xml_label = read_labels(labelfile)
    data = []
    labels = []
    for k,v in xml_label.items():
        datafile = "/"+k+".xml"
        posts = parseXML(datafile)
        data.append(posts)
        labels.append(v)
    # print (data,labels)
    return data,labels
train_data, train_label = load_data_and_label("./golden_truth.txt")
test_data, test_label = load_data_and_label("/T1_erisk_golden_truth.txt")

#train
X_train=[]
y_train=[]
import preprocessor as p
p.set_options(p.OPT.URL, p.OPT.RESERVED, p.OPT.NUMBER, p.OPT.HASHTAG, p.OPT.EMOJI, p.OPT.SMILEY)
for i, listv in enumerate(train_data):
  for eg in listv:
    if eg==None:
      continue
    eg=p.clean(eg)
    X_train.append(eg.split())
    y_train.append(train_label[i])
X_mid_train=X_train    
def build_dictionary(X_train): 

        dic = {}
        for one in X_train:
          for word in one:
            word = word.lower()
            dic[word] = dic.get(word, 0) + 1
        vocab_w = [w for w in dic if dic[w] >= 20]
        print(len(vocab_w))
        vocab={}
        for i, v in enumerate(vocab_w):
          vocab[v]=i
        for i, one in enumerate(X_train):
          sen=[]
          for word in one:
            word = word.lower()
            if word in vocab_w:
              sen.append(vocab[word]+1)
            else:
              sen.append(0)
          X_train[i]=sen
        return X_train, vocab_w, dic
X_train, vocab_w, dic = build_dictionary(X_mid_train)
X_train=np.array(X_train)
y_train=np.array(y_train)

!pip install tweet-preprocessor

#test
X_test=[]
y_test=[]
import preprocessor as p
p.set_options(p.OPT.URL, p.OPT.RESERVED, p.OPT.NUMBER, p.OPT.HASHTAG, p.OPT.EMOJI, p.OPT.SMILEY)
for i, listv in enumerate(test_data):
  for eg in listv:
    if eg==None:
      continue
    eg=p.clean(eg)
    X_test.append(eg.split())
    y_test.append(test_label[i])
X_mid_test=X_test

    
def build_dictionary_test(X_train, vocab_w): 

        vocab={}
        for i, v in enumerate(vocab_w):
          vocab[v]=i
        for i, one in enumerate(X_train):
          sen=[]
          for word in one:
            word = word.lower()
            if word in vocab_w:
              sen.append(vocab[word]+1)
            else:
              sen.append(0)
          X_train[i]=sen
        return X_train
X_test = build_dictionary_test(X_test, vocab_w)
X_test=np.array(X_test)
y_test=np.array(y_test)

print("Accuracy: %.2f%%" % (scores[1]*100))