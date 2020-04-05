import jsonlines
import os
from tqdm import tqdm

def getfile(folder, label):
    ls = []
    all_file = os.listdir(folder)
    for f_ in tqdm(all_file):
        with open(folder + '/' + f_, 'r',encoding='utf-8') as f:
            content = f.readlines()
            assert len(content) == 1
            ls.append({'text':content[0], 'label': label})
    return ls

path = '../aclimdb'
trainfolder = path + '/train'
testfolder = path + '/test'
trainls = getfile(trainfolder + '/pos', 'pos') + getfile(trainfolder + '/neg', 'neg')
testls = getfile(testfolder + '/pos', 'pos') + getfile(testfolder + '/neg', 'neg')
'''
train_json = json.dumps(trainls,ensure_ascii=False)
test_json = json.dumps(testls,ensure_ascii=False)

with open('train.json','w',encoding='utf-8') as f:
    f.write(train_json)
with open('test.json', 'w', encoding='utf-8') as f:
    f.write(test_json)
'''
with jsonlines.open('train.jsonl', mode='w') as writer:
    for i in trainls:
        writer.write(i)
with jsonlines.open('test.jsonl', mode='w') as writer:
    for i in testls:
        writer.write(i)

'''
train_data, test_data = data.TabularDataset.splits(
                            path = 'data',
                            train = 'train.json',
                            test = 'test.json',
                            format = 'json',
                            fields = fields
)
'''