import json
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
  

jsonFile = open("train_data.json", "r")
objs = json.load(jsonFile)
data = []
label = []
idx = []

for i, obj in enumerate(objs):
    data.append(obj["data"])
    label.append(obj["label"])
    idx.append(obj["id"])

taggedData = [TaggedDocument(data[i], [label[i]]) for i in range(len(data))]
model = Doc2Vec(taggedData, vector_size = 32, window = 2, min_count = 1, epochs = 100)
model.save("Doc2Vec.model")
