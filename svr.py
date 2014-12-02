from sklearn.svm import SVR
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from numpy import *
import json
from pprint import pprint
import time
import tweetParse
from textblob import TextBlob
import collections


userModel = True
languageModel = True
timeModel = True

secsInDay = 24*60*60
shift = 8*60*60

#pprint(data)[
timeBuckets = {}
########################## seconds in each bucket. 900=15min, 1800=30min, 3600=1hr
step = 3600
##########################
numiters=5
alpha=.01
#################
followerBuckets=10
followerStep=400
followerExp = 4
################
favBuckets=15
favStep=20
favExp=2
################
numfriendBuckets = 15
friendStep = 5
friendExp=2
################
stExp=4
statusBuckets = 10
statusStep = 40
################
numlistBuckets = 6
listStep = 2
listExp=6
################

def tweetFeatureExtractor(line):
    features = []

    if timeModel:
        time = [0 for i in range(24)]
        pstSecs = line['created_at']-shift
        secs= pstSecs%secsInDay
        time[(int)(secs/step)]=1
        features.extend(time)

    if userModel:
        listed = [0 for i in range(numlistBuckets)]
        listIndex = 0
        if (numlistBuckets)**listExp*listStep<=line['user_listed']:
            listIndex=numlistBuckets-1
        else:
            listIndex=(int)((line['user_listed']/listStep)**(1/(listExp+0.0)))
        listed[listIndex]=1
        features.extend(listed)


        friendBuckets=[0 for i in range(numfriendBuckets)]
        friendIndex=0
        if (numfriendBuckets)**friendExp*friendStep<=line['user_friends']:
            friendIndex=numfriendBuckets-1
        else:
            friendIndex=(int)((line['user_friends']/friendStep)**(1/(friendExp+0.0)))
        friendBuckets[friendIndex]=1
        features.extend(friendBuckets)


        followerB=[0 for i in range(followerBuckets)]
        followIndex=0
        if (followerBuckets)**followerExp*followerStep<line['user_followers']:
            followIndex=followerBuckets-1
        else:
            followIndex=(int)((line['user_followers']/followerStep)**(1/(followerExp+0.0)))
        followerB[followIndex]=1
        features.extend(followerB)


        favB = [0 for i in range(favBuckets)]
        favIndex=0
        if (favBuckets)**favExp*favStep<line['user_favourites']:
            favIndex=favBuckets-1
        else:
            favIndex=(int)((line['user_favourites']/favStep)**(1/(favExp+0.0)))
        favB[followIndex]=1
        features.extend(favB)
   
              
        statB = [0 for i in range(statusBuckets)]
        statI = 0
        if (statusBuckets)**stExp*statusStep<line['user_statuses_count']:
            statI=statusBuckets-1
        else:
            statI = (int)((line['user_statuses_count']/statusStep)**(1/(stExp+0.0)))

    return features
    

#def tweetFeatureExtractor(line):
#    features = {}
#
#    if languageModel:    
#        text = TextBlob(line['text'])
#    
#        sent = text.sentiment.polarity
#        if sent==0: features['sentiment_zero']=1
#        if 0<sent<=0.5: features['sentiment_positive']=1
#        if 0.5<sent: features['sentiment_strong_positive']=1
#        if 0>sent>=-0.5: features['sentiment_negative']=1
#        if -0.5>sent: features['sentiment_strong_negative']=1
#        subj =text.sentiment.subjectivity
#        if subj==0: features['subjectivity_zero']=1
#        if 0<subj<=0.5: features['subjectivity_objective']=1
##        tag_list = []
##        for word,tag in text.tags:
##            if tag not in tag_list: tag_list.append(tag)
##        for tag in tag_list: features['tag_'+tag]=1
#    
#        tag_list = collections.Counter()
#        for word,tag in text.tags:
#            tag_list[tag]+=1.0
#        for tag,count in tag_list.items():
#            if count==0: features['tag_'+tag+'_'+'0']=1
#            if 0<count<=5: features['tag_'+tag+'_1-5']=1
#            if 5<count<=10: features['tag_'+tag+'_6-10']=1
#            if 10<count: features['tag_'+tag+'10+']=1
#    return features

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

    
trainExamples = getTrainExamples()
testExamples = getTestExamples()  

feature_list = []
y_list = []
classifier_feature_list = []
classifier_y_list = []
non_zero_feature_list = []
non_zero_y_list = []
for x,y in trainExamples:
    feature = tweetFeatureExtractor(x)
    feature_list.append(feature)
    y_list.append(y)
    classifier_feature_list.append(feature)
    if y==0:
        classifier_y_list.append(0)
    else:
        classifier_y_list.append(1)
        non_zero_feature_list.append(feature)
        non_zero_y_list.append(y)

print "Non Zero Y Values: ",non_zero_y_list

test_feature_list = []
test_y_list = []
for x,y in testExamples:
    test_feature_list.append(tweetFeatureExtractor(x))
    test_y_list.append(y)

test = False
if test:
    n_samples, n_features = 10,5
    np.random.seed(0)
    y = np.random.randn(n_samples)
    X = np.random.randn(n_samples, n_features)
    print y
    print X
    clf = SVR(C=1.0,epsilon=0.2)
    clf.fit(X,y)
    print clf.score(X,y)

## Straight SVR
#y = np.array(y_list)
#X = np.matrix(feature_list)
#test_y = np.array(test_y_list)
#test_X = np.matrix(test_feature_list)
#print y
#print X
#clf = SVR(C=1.0,epsilon=0.4)
#clf.fit(X,y)
#print clf.score(test_X,test_y)
#print test_y_list

#pred_output = [int(round(y)) for y in clf.predict(test_X).tolist()]
#print pred_output

# SVC to determine 0's, SVR on non-zeros
# Training
y = np.array(classifier_y_list)
X = np.matrix(classifier_feature_list)
test_y = np.array(test_y_list)
test_X = np.matrix(test_feature_list)
classifier = SVC(C=2.0)
classifier.fit(X,y)

# Train SVR for non-zero tweets
reg_y = np.array(non_zero_y_list)
reg_X = np.matrix(non_zero_feature_list)
regression = SVR(C=1.0,epsilon=0.2)
regression.fit(reg_X,reg_y)

# Actual y values
print "Actual Values: ",test_y_list

# Run classifier on Test Set
predictions = classifier.predict(test_X).tolist()
print 'Classifier Predictions: ',predictions

# If Classifier outputs non-zero, run SVR
for idx,val in enumerate(predictions):
    if val!=0:
        pred = int(round(regression.predict(np.matrix(classifier_feature_list[idx]))))
        predictions[idx] = pred
print 'Full Predictions: ',predictions

#totalSquareError = 0.0
#count = 0
#for idx,pred in enumerate(pred_output):
#    count+=1
#    totalSquareError += (pred-test_y_list[idx])**2
#print 'Regular SVR Error: ',totalSquareError/count

totalSquareError = 0.0
count = 0
for idx,pred in enumerate(predictions):
    count+=1
    totalSquareError += (pred-test_y_list[idx])**2
print 'SVC/SVR Hybrid Error: ',totalSquareError/count


