import copy

import datetime
import json
from pprint import pprint
import time

def getTrainExamples():  
    trainf = '../data/train_file.txt'
    json_data_tr = open(trainf)
    tr_data = json.loads(json_data_tr.readline())
    ret = []
    for line in tr_data:
        ret.append((line,line['retweets']))
    return ret
def getTestExamples():
    testf = '../data/test_file.txt'
    json_data_te = open(testf)
    te_data = json.loads(json_data_te.readline())
    ret=[]
    for line in te_data:
        ret.append((line,line['retweets']))
    return ret

# loop through SearchResult objects in search and print to file 'tweet_file.txt'
def output_tweets(data):
    # file to write to
    data_file = open('../data/stanford_tweets.txt','w')
    data_file.write(json.dumps(data))
    data_file.close()

train = getTrainExamples()
test = getTestExamples()
data = []
data.extend(train)
data.extend(test)
output_tweets(data)

