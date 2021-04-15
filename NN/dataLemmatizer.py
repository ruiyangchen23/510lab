from dataTokenizer import tokenize 
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

def lemmatize(file):
    ps = PorterStemmer()
    lem = WordNetLemmatizer()

    data = tokenize(file)
    for i in range(len(data)):
        for j in range(len(data[i][2])):
            data[i][2][j] = ps.stem(lem.lemmatize(data[i][2][j]))

    return data