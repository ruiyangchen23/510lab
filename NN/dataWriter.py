from dataLemmatizer import lemmatize
import json

data = lemmatize("./test/T1_erisk_golden_truth.txt")

jsonFile = open("test_data.json", "w")
aList = [{"id":data[i][0], "label":data[i][1], "data":data[i][2]} for i in range(len(data))]
jsonString = json.dumps(aList)
jsonFile.write(jsonString)
jsonFile.close()
