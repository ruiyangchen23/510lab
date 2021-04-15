import csv
import json
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

model = Doc2Vec.load("Doc2Vec.model")

jsonFile = open("test_data.json", "r")
objs = json.load(jsonFile)
data = []
label = []
idx = []

for i, obj in enumerate(objs):
    data.append(obj["data"])
    label.append(obj["label"])
    idx.append(obj["id"])
    
dataEmbedded = []
for i,text in enumerate(data):
    embedded = model.infer_vector(text).tolist()
    embedded = [idx[i], label[i]] + embedded
    dataEmbedded.append(embedded)

with open("test_embedded.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(dataEmbedded)