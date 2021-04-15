from afinn import Afinn
import nltk 
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from dataloader import load_data_and_label
from collections import defaultdict
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
import pickle

import numpy as np
def get_features(paragraph):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(paragraph)
    ret = []
    ret.append(sentiment_score['neg'])
    ret.append(sentiment_score['neu'])
    ret.append(sentiment_score['pos'])
    ret.append(sentiment_score['compound'])
    return ret

train_subject,train_labels = load_data_and_label("./train/golden_truth.txt","./train/data/")
test_subjects,test_labels = load_data_and_label("./test/T1_erisk_golden_truth.txt","./test/DATA/")


features = {}

# def calculate_sentiment(subject):
#     ret = 0.0
#     for i in range(len(subject)):
#         post = subject[i]
#         feature = get_features(post)
#         if abs(feature['compound'])>0.5:
#             ret += feature['compound']
#     return ret

# def calculate_avg_sentiment(subjects,labels):
#     sentiment_dict = defaultdict(lambda:0)
#     # the avearge sentiment
#     for i in range(len(subjects)):
#         subject = subjects[i]
#         label = labels[i]
#         sentiment = 0
#         for post in subject:
#             si = get_features(post)
#             if si['compound']<-0.5:
#                 sentiment += si['compound']
#             elif si['compound']>0.5:
#                 sentiment += si['compound']
#         sentiment_dict[label]+=sentiment
#     return sentiment_dict

# def get_avg_sentiment(all_subjects,all_labels):
#     subjects = all_subjects[:40]
#     labels = all_labels[:40]
#     i=0
#     while len(labels)<80:
#         if all_labels[i]==0:
#             subjects.append(all_subjects[i])
#             labels.append(all_labels[i])
#         i+=1

#     harm = sum(labels)
#     noharm = len(labels)-harm

#     sentiment_dict = calculate_avg_sentiment(subjects,labels)
#     sentiment_dict[0] = sentiment_dict[0]/noharm
#     sentiment_dict[1] = sentiment_dict[1]/harm
#     print(sentiment_dict)


# def predict(all_subjects,all_labels):
#     right= 0
#     for i in  range(len(all_subjects)):
#         subject = all_subjects[i]
#         sentiment = calculate_sentiment(subject)
#         if sentiment > 30:
#             if all_labels[i]==0:
#                 right +=1.0
#                 print("right")
#             else:
#                 print("wrong")
#         else:
#             if all_labels[i]==1:
#                 right+=1.0
#                 print("right")
#             else:
#                 print("wrong")
#     print (right/len(all_labels))
text_clf = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('svc', SVC(class_weight="balanced"))
                        ])
def train(all_subjects,all_labels):
    features = []
    
    data = []
    label = []
    for i in range(0,200,4):
        subject = all_subjects[i]
        for post in subject:
            data.append(post)
            label.append(all_labels[i])
    # indexes= np.random.choice(120000, 12000)
    d  = []
    l = []
    # for index in indexes:
        # d.append(data[index])
        # l.append(label[index])
    print("begin fit")
    text_clf.fit(data,label)
    print("fitting finished")
    pickle.dump(text_clf, open( "model", "wb" ) )

def predict(subjects,labels):
    s = 0
    right = 0.0

    for i in range(len(subjects)):
        subject = subjects[i]
        data= []
        label  = []
        for post in subject:
            data.append(post)
        prediction = text_clf.predict(data)
        for j in range(len(prediction)):
            if prediction[j]==labels[i]:
                right+=1
            s +=1.0
    print (right/s)
    # print (np.mean(prediction))
    # print(np.sum(prediction))

train(train_subject,train_labels)
predict(test_subjects,test_labels)
# afinn = Afinn()
# score = afinn.score('I want to kill myself!')
# score = afinn.score('help me!')
# predict(all_subjects,all_labels)
# print(score)