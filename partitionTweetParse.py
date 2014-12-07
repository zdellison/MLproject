import copy

import datetime
from numpy import arange
import json
from pprint import pprint
import time

def ParseData(k):
    data_file = '../data/stanford_tweets.txt'
    json_data = open(data_file)
    data = json.loads(json_data.readline())
    partitions = [[] for i in range(k)]
    i = 0
    for line in data:
        partitions[i].append((line[0],line[1]))
        i = (i+1)%k
    return partitions
