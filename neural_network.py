import copy
import numpy as np
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
        # if predictor(x,weights)>20:
        #    print "Predicted ",predictor(x,weights)," when truth is ",y
        
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

def listDotProduct(l1,l2):
    dp = 0.0
    if len(l1)!=len(l2): return None
    for i in range(len(l1)):
        dp+=l1[i]*l2[i]
    return dp

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
    toBeAdded = {}
    for key in features:
        toBeAdded['time_bucket_'+str((int)(secs/step))+'_'+key]=1
    for key in toBeAdded:
        features[key]=toBeAdded[key]
    features['time_bucket_'+str((int)(secs/step))]=1
    
    return features

def predictor(feat,weights): 
   
    dot= dotProduct(tweetFeatureExtractor(feat),weights)
    if dot<0: 
        return 0
    else:
        return round(dot)


def learnPredictor(trainExamples, testExamples, featureExtractor):
    iternum=0
    #We use a sparse matrix notation for weight matricies W^l:
    #       Let W^l be a dictionary of dictionaries, W^l={}
    #       W^l_{i,j}=W^l[i][j]
    #       Note that W^l is in R^{numFeatures x numFeatures} or in practice has numFeatures^2 key/value pairs
    #Thus, W is a list of dictionaries of dictionaries of length of the number of layers
    #
    #b is a list of bias terms corresponding to the bias of each layer
    #       b=[]*numberOfLayers
    #
    #Note that all values of W and b must be initialized to random values near 0

    while iternum<numiters: 
        for x,y in trainExamples:
            features = featureExtractor(x)  
            #compute activations for all levels:
            activations = []
            for l in range(numLevels):
                levelActivations=[]
                for i in range(numNeuronsPerLevel):
                    if l==0:
                        activation=dotProduct(W_1[i],features)
                #     else:
                #         activation = 0.0
                #         for j in range(numNeuronsPerLevel):
                #             activation+=W[l][i][j]
                    activation+=b[l][i]
                    levelActivations.append(activation)
                activations.append(levelActivations)
                if l>0:
                    activations.append(W[l]*activations[l-1])
            #compute final activation (hypothesis)
            activation = 0.0
            for i in range(numNeuronsPerLevel):
                 activation+=W[numLevels][0][i]
            activations.append([activation])

            #compute error corresponding to each level (backward step) these errors are stored in delta
            delta = [[0 for j in range(numNeuronsPerLevel)]for i in range(numLevels+1)]
            for l in range(numLevels,-1,-1): #iterate over levels backwards
                if l==numLevels:
                    delta[l]=y-activations[numLevels]
                    if activations[numLevels]!=activations[len(activations)-1]: x=3/0 #explode
                if l<numLevels:
                    neuronDelta = []
                    for j in range(numNeuronsPerLevel):
                        neuronDelta.append(listDotProduct(delta[l+1],W))

    
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
numiters=10
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
numLevels=2
numNeuronsPerLevel = 10
b=[[np.random.normal(0, 0.01) for i in range(numNeuronsPerLevel)] for j in range(numLevels+1)]
W_1 = [{} for i in range(numNeuronsPerLevel)]
W = [numpy.random.normal(0,.01,(numNeuronsPerLevel, numNeuronsPerLevel)) for k in range(numLevels)]
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
