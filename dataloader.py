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
    print(ret)
    return ret

def load_data_and_label(labelfile):
    xml_label = read_labels(labelfile)
    data = []
    labels = []
    for k,v in xml_label.items():
        datafile = "./train/data/"+k+".xml"
        posts = parseXML(datafile)
        data.append(posts)
        labels.append(v)
    # print (data,labels)
    return data,labels
load_data_and_label("./train/golden_truth.txt")