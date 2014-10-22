#!/usr/bin/env python

# This file takes in the text file of JSON formatted dumps from twitter and parses them into
# feature vectors

file = 'tweets.txt'

import json
from pprint import pprint

json_data = open(file)

data = json.load(json_data)
pprint(data)
json_data.close()
