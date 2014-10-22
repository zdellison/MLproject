#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *
import train_tweets

for tweet in train_tweets.examples:
    print "Tweets: ",tweet

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor():
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, return the weight vector (sparse feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (around 15 lines of code expected)
    eta = .02
    numIters = 20 
    for i in range(numIters):
        for i,x in enumerate(train_tweets.examples):
            score = dotProduct(x,weights)
            print "Score: ",score
            loss = (score-train_tweets.values[i])**2
            update = {}
            increment(update,loss,x)
            increment(weights,-eta,update)
            # else: there is no cost, so we do not update the weights
    # END_YOUR_CODE
    print "Weights: ",weights
    return weights

def evaluate(weights):
    for i,x in enumerate(train_tweets.test1):
       score = dotProduct(x,weights)
       print "Prediction for: ",x," is: ",score
       print "We are {0} from the goal of {1}".format(abs(train_tweets.testVal[i]-score),train_tweets.testVal[i])


evaluate(learnPredictor())


