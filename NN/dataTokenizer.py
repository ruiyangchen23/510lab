from dataloader import load_data_and_label
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

def tokenize(file):
    text, label, idx = load_data_and_label(file)
    data = []
    for i in range(len(text)):
        for j in range(len(text[i])):
            if isinstance(text[i][j], str):
                data.append([idx[i], label[i], text[i][j]])

    stop_words = set(stopwords.words("english"))

    for i in range(len(data)):
        tmp = word_tokenize(data[i][2])
        data[i][2] = []
        for word in tmp:
            if word not in stop_words:
                newWord = re.sub(r'[^\w\s]',"",word)
                if newWord != "":
                    data[i][2].append(newWord)

    return data
