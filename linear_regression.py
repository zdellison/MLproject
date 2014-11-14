import copy

import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from numpy import arange
import json
from pprint import pprint
import time

def evaluatePredictor(examples, predictor):
    '''
    predictor: a function that takes an x and returns a predicted y.
    Given a list of examples (x, y), makes predictions based on |predict| and returns the fraction
    of misclassiied examples.
    '''
    error = 0.0
    for x, y in examples:
    	print "Predicted ",predictor(x)," when truth is ",y
        if abs(predictor(x)-y)>3:
            error += 1
            print "--------ERROR------------"
    return error / len(examples)


def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale


def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """

    ret=0.0
    
    if len(d1) < len(d2):
        
        ret=dotProduct(d2, d1)
    else:
        ret=sum(d1.get(f, 0) * v for f, v in d2.items())

    return ret

def tweetFeatureExtractor(line):
    pstSecs = line['created_at']-shift
    secs= pstSecs%secsInDay
    features = {}
    for i in range(secsInDay/step):
        if i*step<secs and secs<(i+1)*step:
            features['time_bucket_'+str(i)]=1#(totalRetweetsTimeBuckets[i]+0.0)/(timeBuckets[i]+0.0)
    
    for i in range(followerBuckets):
        if i*followerStep<line['user_followers'] and (i+1)*followerStep>line['user_followers']:
            features['follower_bucket_'+str(i)]=1
    if (followerBuckets-1)*followerStep<line['user_followers']:
        features['follower_bucket_MAX']=1

    for i in range(favBuckets):
        if i*favStep<line['user_favourites_count'] and (i+1)*favStep>line['user_favourites_count']:
            features['fav_bucket_'+str(i)]=1
    if (favBuckets-1)*favStep<line['user_favourites_count']:
        features['fav_bucket_MAX']=1

    for i in range(friendBuckets):
        if i*friendStep<line['user_friends'] and line['user_friends']<(i+1)*friendStep:
            features['friend_bucket_'+str(i)]=1
    if (friendBuckets-1)*friendStep<line['user_friends']:
        features['friend_bucket_MAX']=1
    
    return features

def learnPredictor(trainExamples, testExamples, featureExtractor):

      # feature => weight


    iternum=0
    def predictor(feat): 
        return dotProduct(featureExtractor(feat),weights)

    while iternum<numiters:
        
        for x,y in trainExamples:
            #print ""
            #print "WEIGHTS", weights
            features = featureExtractor(x)
            score =0.0
            
            score += dotProduct(features,weights)
            #print "SCORE", score
            #print "Y",y
            residual = y-score
            #print residual
            
            #print "RESIDUAL", residual
            for key in features:
                if key in weights:
                    weights[key]=weights[key]+alpha*residual*features[key]

                else:
                    weights[key]=alpha*residual*features[key]
                #print weights
                #print x
               # time.sleep(3)
            
            #weights=copy.deepcopy(newWeights)
            #print newWeights
        iternum+=1
        #print weights
        print 'Iteration number: '+str(iternum)+', train error = '+str(evaluatePredictor(trainExamples,predictor))#+', test error = '+str(evaluatePredictor(testExamples,predictor))

    # END_YOUR_CODE
    return weights

trainExamples=[]
testExamples=[]


file = 'tweet_file_3.txt'
secsInDay = 24*60*60
shift = 8*60*60
json_data = open(file)
data = json.loads(json_data.readline())
#pprint(data)[
timeBuckets = {}
totalRetweetsTimeBuckets={}
########################## seconds in each bucket. 900=15min, 1800=30min, 3600=1hr
step = 1800
##########################
numiters=5
alpha=.001
followerBuckets=10
followerStep=50
favBuckets=10
favStep=50
friendBuckets=10
friendStep=50
for line in data:
    pstSecs = line['created_at']-shift
    secs= pstSecs%secsInDay

    for i in range(secsInDay/step):
        
        if i*step<secs and secs<(i+1)*step:
     
            
            if i in totalRetweetsTimeBuckets:
                totalRetweetsTimeBuckets[i]+=line['retweets']
            else:
                totalRetweetsTimeBuckets[i]=line['retweets']
            if i in timeBuckets:
                #timeBuckets[i]+=line['retweets']
                timeBuckets[i]+=1
            else:
                #timeBuckets[i]=line['retweets']
                timeBuckets[i]=1

    
    
    
for line in data:
    
    trainExamples.append((line,line['retweets']))
      
weights = {}
learnPredictor(trainExamples,testExamples,tweetFeatureExtractor)



totalRetweetsTimeBuckets = {}
for line in data:
    pstSecs = line['created_at']-shift
    secs= pstSecs%secsInDay
    for i in range(secsInDay/step):
        if i*step<secs and secs<(i+1)*step:
            if i in totalRetweetsTimeBuckets:
                totalRetweetsTimeBuckets[i]+=dotProduct(tweetFeatureExtractor(line),weights)
            else:
                totalRetweetsTimeBuckets[i]=dotProduct(tweetFeatureExtractor(line),weights)

listversionTotTweets = [0]*(secsInDay/step)
for key in timeBuckets:
    listversionTotTweets[key]=totalRetweetsTimeBuckets[key]
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
ax1.plot(dates,listversionTotTweets, 'g.-',label = 'Predicted Retweets per Time')
ax1.set_xlabel('Time')


ax1.xaxis.set_major_locator( HourLocator() )
ax1.xaxis.set_minor_locator( HourLocator(arange(0,25,6)) )
ax1.xaxis.set_major_formatter( DateFormatter('%H:%M') )

ax1.fmt_xdata = DateFormatter('%H')
fig.autofmt_xdate()
ax1.legend()
plt.show()













