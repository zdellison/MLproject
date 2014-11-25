import copy

import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from numpy import arange
import json
from pprint import pprint
import time
import tweetParse

  

def evaluatePredictor(examples, predictor):
    totalSquaredError = 0.0
    count=0.0
    for x, y in examples:
        count+=1
        totalSquaredError+=(predictor(x,weights)-y)**2
        if predictor(x,weights)>20:
           print "Predicted ",predictor(x,weights)," when truth is ",y
        
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

def tweetFeatureExtractor(line):

    features = {}

    
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


    pstSecs = line['created_at']-shift
    secs= pstSecs%secsInDay
    features['time_bucket_'+str((int)(secs/step))]=1
    
    return features

def predictor(feat,weights): 
   
    dot= dotProduct(tweetFeatureExtractor(feat),weights)
    if dot<0: 
        return 0
    else:
        return round(dot)
    #return 0

def learnPredictor(trainExamples, testExamples, featureExtractor):
    iternum=0
    while iternum<numiters: 
        for x,y in trainExamples:
            features = featureExtractor(x)         
            score = dotProduct(features,weights)
            residual = (y-score)
            #print residual 
            #print residual**3
            for key in features:
                if key in weights:
                    weights[key]+=alpha*residual*features[key]
                else:
                    weights[key]=alpha*residual*features[key]
        iternum+=1
        print 'Iteration number: '+str(iternum)+', train mean squared error = '+str(evaluatePredictor(trainExamples,predictor))+', test error = '+str(evaluatePredictor(testExamples,predictor))

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
trainfile = 'data_copy/train_file.txt'
json_data_train = open(trainfile)
train_data = json.loads(json_data_train.readline())
testfile = 'data_copy/test_file.txt'
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
print tweetFeatureExtractor(testExamples[0][0])
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
