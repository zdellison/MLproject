import copy

import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from numpy import arange
import json
from pprint import pprint
import time
import tweetParse
from textblob import TextBlob
import collections


languageModel = True
userModel = True
timeModel = True

id_num= 0
cache = {}

# Print which models are being used:

if timeModel: print "Using Time Model"
if userModel: print "Using User Model"
if languageModel: print "Using Language Model"

def evaluatePredictor(examples, predictor):
    totalSquaredError = 0.0
    count=0.0
    for x, y in examples:
        count+=1
        totalSquaredError+=(predictor(x,weights)-y)**2
        if predictor(x,weights)>20:
           print "Predicted ",predictor(x,weights)," when truth is ",y
        if (predictor(x,weights)-y)**2>100:
            print "Error Greater than 30, Predicted: ",predictor(x,weights)," when truth is ",y
        # if abs(predictor(x,weights)-y)>1:
        #     error += 1
            #print "--------ERROR------------"
    return totalSquaredError/count


def increment(d1, scale, d2):
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale


def dotProduct(d1, d2):
    ret=0.0
    
    if len(d1) < len(d2):
        
        ret=dotProduct(d2, d1)
    else:
        ret=sum(d1.get(f, 0) * v for f, v in d2.items())

    return ret

def tweetFeatureExtractor(line,id_num):
    id_num +=1
    features = {}

    if userModel:    
        if (numlistBuckets)**listExp*listStep<=line['user_listed']:
            features['list_bucket_MAX']=1
        else:
            features['list_bucket_'+str((int)((line['user_listed']/listStep)**(1/(listExp+0.0))))]=1
    
    
    
        if (numfriendBuckets)**friendExp*friendStep<=line['user_friends']:
            features['friend_bucket_MAX']=1
        else:
            features['friend_bucket_'+str((int)((line['user_friends']/friendStep)**(1/(friendExp+0.0))))]=1
    
    
    
        if (followerBuckets)**followerExp*followerStep<line['user_followers']:
            features['follower_bucket_MAX']=1
        else:
            features['follower_bucket_'+str((int)((line['user_followers']/followerStep)**(1/(followerExp+0.0))))]=1
    
    
                
        if (favBuckets)**favExp*favStep<line['user_favourites']:
            features['fav_bucket_MAX']=1
        else:
            features['fav_bucket_'+str((int)((line['user_favourites']/favStep)**(1/(favExp+0.0))))]=1
       
                
    
        if (statusBuckets)**stExp*statusStep<line['user_statuses_count']:
            features['user_statuses_count_MAX']=1
        else:
            features['user_statuses_count_'+str((int)((line['user_statuses_count']/statusStep)**(1/(stExp+0.0))))]=1

    if languageModel:    
        text = TextBlob(line['text'])
    
        sent = text.sentiment.polarity
        if sent==0: features['sentiment_zero']=1
        if 0<sent<=0.5: features['sentiment_positive']=1
        if 0.5<sent: features['sentiment_strong_positive']=1
        if 0>sent>=-0.5: features['sentiment_negative']=1
        if -0.5>sent: features['sentiment_strong_negative']=1
        subj =text.sentiment.subjectivity
        if subj==0: features['subjectivity_zero']=1
        if 0<subj<=0.5: features['subjectivity_objective']=1
        if 0.5<subj: features['subjectivity_subjective']=1
    
#        tag_list = []
#        for word,tag in text.tags:
#            if tag not in tag_list: tag_list.append(tag)
#        for tag in tag_list: features['tag_'+tag]=1
    
        tag_list = collections.Counter()
        for word,tag in text.tags:
            tag_list[tag]+=1.0
        for tag,count in tag_list.items():
            if count==0: features['tag_'+tag+'_'+'0']=1
            if 0<count<=5: features['tag_'+tag+'_1-5']=1
            if 5<count<=10: features['tag_'+tag+'_6-10']=1
            if 10<count: features['tag_'+tag+'10+']=1
    
    
    if timeModel:
        pstSecs = line['created_at']-shift
        secs= pstSecs%secsInDay
        features['time_bucket_'+str((int)(secs/step))]=1
    
#    if id_num in cache: 
#        return cache[id_num]
#    else:
#        cache[id_num]=features
#        return cache[id_num]
    return features

def predictor(feat,weights): 
   
    dot= dotProduct(tweetFeatureExtractor(feat,id_num),weights)
    if dot<0: 
        return 0
    else:
        return round(dot)
    #return 0

def learnPredictor(trainExamples, testExamples, featureExtractor):
    iternum=0
    while iternum<numiters: 
#        non_zero_sent,sent_total = 0.0,0.0
        for x,y in trainExamples:
            features = featureExtractor(x,id_num)         
            score = dotProduct(features,weights)
            residual = (y-score)
            #print residual 
            #print residual**3
            for key in features:
#                if key is 'sentiment':
#                    if features[key] != 0.0:
#                        non_zero_sent += 1.0
#                    sent_total += 1.0
                if key in weights:
                    weights[key]+=alpha*residual*features[key]
                else:
                    weights[key]=alpha*residual*features[key]
        iternum+=1
#        print "Non-Zero Sentiment Percentage:",non_zero_sent/sent_total
        print '================ TRAIN SET: Iteration '+str(iternum)+' ================'
        print 'Iteration number: '+str(iternum)+', train mean squared error = '+str(evaluatePredictor(trainExamples,predictor))
        print '================ TEST SET ================='
        print 'Test error = '+str(evaluatePredictor(testExamples,predictor))

    return weights



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
trainfile = '../data/train_file.txt'
json_data_train = open(trainfile)
train_data = json.loads(json_data_train.readline())
testfile = '../data/test_file.txt'
json_data_test = open(testfile)
test_data = json.loads(json_data_test.readline())



    
trainExamples=tweetParse.getTrainExamples()
testExamples= tweetParse.getTestExamples()    
weights = {}
learnPredictor(trainExamples,testExamples,tweetFeatureExtractor)



totRetweetsTimeBuckets = {}

for line in test_data:
    pstSecs = line['created_at']-shift
    secs= pstSecs%secsInDay
    for i in range(secsInDay/step):
        if i*step<secs and secs<(i+1)*step:
            if i in totRetweetsTimeBuckets:
                totRetweetsTimeBuckets[i]+=predictor(line,weights)
            else:
                totRetweetsTimeBuckets[i]=predictor(line,weights)
# print "TEST ERROR RATE: ",evaluatePredictor(testExamples,predictor)
print weights
print ""
print tweetFeatureExtractor(testExamples[0][0],id_num)
print ""
print predictor(test_data[0],weights)
print ""
print test_data[0]
print ""
print testExamples[0]

listversionTotTweets = [0]*(secsInDay/step)
for key in totRetweetsTimeBuckets:
    listversionTotTweets[key]=totRetweetsTimeBuckets[key]
date1 = datetime.datetime( 2000, 3, 2)
date2 = datetime.datetime( 2000, 3, 3)
if step==3600:
    delta = datetime.timedelta(hours=1)
elif step==1800:
    delta = datetime.timedelta(minutes=30)
elif step==900:
    delta = datetime.timedelta(minutes=15)
dates = drange(date1, date2, delta)



fig, ax1 = plt.subplots()
ax1.plot(dates,listversionTotTweets, 'g.-')
ax1.set_xlabel('Time')


ax1.xaxis.set_major_locator( HourLocator() )
ax1.xaxis.set_minor_locator( HourLocator(arange(0,25,6)) )
ax1.xaxis.set_major_formatter( DateFormatter('%H:%M') )

ax1.fmt_xdata = DateFormatter('%H')
fig.autofmt_xdate()
plt.suptitle('Total Predicted Retweets per Time Bucket\nTesting Data', fontsize=16)
plt.show()
