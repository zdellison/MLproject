import copy

import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from numpy import arange
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
