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
        totalSquaredError+=(predictor(x,W,b,y)-y)**2
        # if y>20:
        #    print "Predicted ",predictor(x,W,b)," when truth is ",y
        
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

    # features = [0 for i in range(24)]

    # listed = [0 for i in range(numlistBuckets)]
    # listIndex = 0
    # if (numlistBuckets)**listExp*listStep<=line['user_listed']:
    #     listIndex=numlistBuckets-1
    # else:
    #     listIndex=(int)((line['user_listed']/listStep)**(1/(listExp+0.0)))
    # listed[listIndex]=1
    # features.extend(listed)


    # friendBuckets=[0 for i in range(numfriendBuckets)]
    # friendIndex=0
    # if (numfriendBuckets)**friendExp*friendStep<=line['user_friends']:
    #     friendIndex=numfriendBuckets-1
    # else:
    #     friendIndex=(int)((line['user_friends']/friendStep)**(1/(friendExp+0.0)))
    # friendBuckets[friendIndex]=1
    # features.extend(friendBuckets)


    # followerB=[0 for i in range(followerBuckets)]
    # followIndex=0
    # if (followerBuckets)**followerExp*followerStep<line['user_followers']:
    #     followIndex=followerBuckets-1
    # else:
    #     followIndex=(int)((line['user_followers']/followerStep)**(1/(followerExp+0.0)))
    # followerB[followIndex]=1
    # features.extend(followerB)


    # favB = [0 for i in range(favBuckets)]
    # favIndex=0
    # if (favBuckets)**favExp*favStep<line['user_favourites']:
    #     favIndex=favBuckets-1
    # else:
    #     favIndex=(int)((line['user_favourites']/favStep)**(1/(favExp+0.0)))
    # favB[followIndex]=1
    # features.extend(favB)
   
            
    # statB = [0 for i in range(statusBuckets)]
    # statI = 0
    # if (statusBuckets)**stExp*statusStep<line['user_statuses_count']:
    #     statI=statusBuckets-1
    # else:
    #     statI = (int)((line['user_statuses_count']/statusStep)**(1/(stExp+0.0)))
    # statB[statI]=1
    # features.extend(statB)


    # pstSecs = line['created_at']-shift
    # secs= pstSecs%secsInDay
    

    # features[(int)(secs/step)]=1

    # return features
    features=[0 for i in range(2)]
    if line['user_listed']>0:
        features[1]=1
    else:
        features[0]=1
    return features
def predictor(feat,W,b,y): 
   
    features = tweetFeatureExtractor(feat)
    ################
    #compute activations for all levels:
    z=[[] for j in range(0,numLevels+1)]
    activations = [[] for j in range(0,numLevels+1)]
        
    for l in range(0,numLevels):
        if l==0:
            z[l]=np.matrix(W[l]*np.matrix(features).T+np.matrix(b[l]).T)
            activations[l]=sigmoid(z[l])

        else:  
            z[l]=np.matrix(W[l]*activations[l-1]+np.matrix(b[l]).T)
            activations[l]=sigmoid(z[l])
            #compute final activation (hypothesis)
    z[numLevels]=np.matrix(W[numLevels]*activations[numLevels-1]+np.matrix(b[numLevels]))
    activations[numLevels]=sigmoid(z[numLevels])
    hypothesis=activations[numLevels]
    print "Predicted ",hypothesis, " when truth is ",features, " for features: "

    print ""
    if hypothesis<0: 
        return 0
    else:
        return round(hypothesis)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def dsigmoid(x):
    return np.multiply(sigmoid(x),(1.0-sigmoid(x)))
def learnPredictor(trainExamples, testExamples, featureExtractor):
    #We use a dense matrix notation for weight matricies W^l:
    #       Let W^l be numpy matrix
    #       W^l[i][j] = weight  associated with the connection between unit j in layer l, and unit i in layer l+1
    #       Note that W^l is in R^{numFeatures x numFeatures} 
    #
    #b is a list of bias vectors 
    #       b^l_i=the bias associated with unit i in layer l+1
    #
    #Note that all values of W and b must be initialized to random values near 0


    #allow function to alter global b,W
    global W
    global b
    iternum=0

    while iternum<numiters: 
        #initialize deltaW to all zeros
        deltaW = [np.matrix(np.zeros((numNeuronsPerLevel, numNeuronsPerLevel))) for k in range(numLevels)]
        deltaW.append(np.matrix(np.zeros(numNeuronsPerLevel)))
        
        #initialize deltab to all zeros
        deltab=[np.matrix(np.zeros(numNeuronsPerLevel)) for i in range(numLevels)]
        deltab.append(np.matrix(0))

        #for all features 1,...,m
        for x,y in trainExamples:
            
            features = featureExtractor(x)  

            ################
            #compute activations for all levels:
            z=[[] for j in range(0,numLevels+1)]
            activations = [[] for j in range(0,numLevels+1)]
        
            for l in range(0,numLevels):
                if l==0:
                    z[l]=np.matrix(W[l]*np.matrix(features).T+np.matrix(b[l]).T)
                    activations[l]=sigmoid(z[l])

                else:  
                    z[l]=np.matrix(W[l]*activations[l-1]+np.matrix(b[l]).T)
                    activations[l]=sigmoid(z[l])
            #compute final activation (hypothesis)
            z[numLevels]=np.matrix(W[numLevels]*activations[numLevels-1]+np.matrix(b[numLevels]))
            activations[numLevels]=sigmoid(z[numLevels])

            #################
            #compute error corresponding to each level (backward step) these errors are stored in delta
            #compute final delta
            delta = [[0 for j in range(numNeuronsPerLevel)]for i in range(numLevels)]
            delta.append([-(y-activations[numLevels].item(0))*dsigmoid(z[numLevels]).item(0)])
            #compute preceeding deltas
            for l in range(numLevels,0,-1): #iterate over levels backwards
                delta[l-1]=(np.multiply(W[l].T*np.matrix(delta[l]),dsigmoid(z[l])))
           
            ##################
            #compute all gradients W_l and gradients b_l
            gradW =[]
            for l in range(0,numLevels+1):
                gradW.append(delta[l]*activations[l].T)
            ##This following line is my experiment, I could not find in the tutorial information about
            ##changing dimensionality on this level.
            gradW[numLevels]=gradW[numLevels]*features

            #calculate gradb
            gradb = [activations[i].T for i in range(0,numLevels+1)]

            #################
            #apply gradient to deltaW
            deltaW=[deltaW[l]+gradW[l] for l in range(numLevels+1)]
            
            #apply gradient to deltab
            deltab=[deltab[l]+gradb[l] for l in range(numLevels+1)]


        #increment iterations
        iternum+=1
        
        m=len(trainExamples)
        #update all levels of W
        for l in range(numLevels+1):
            W[l]-=learningRate*((1.0/m)*deltaW[l]+lambdaDecay*np.matrix(W[l]))

        #update all levels of b
        b=[b[l]-learningRate*((1.0/m)*deltab[l]) for l in range(numLevels+1)]
        
        print 'Iteration number: '+str(iternum)+', train mean squared error = '+str(evaluatePredictor(trainExamples,predictor))+', test error = '+str(evaluatePredictor(testExamples,predictor))

    return (W,b)



secsInDay = 24*60*60
shift = 8*60*60

# timeBuckets = {}
########################## seconds in each bucket. 900=15min, 1800=30min, 3600=1hr
step = 3600
##########################
numiters=1000

# #################
# followerBuckets=8
# followerStep=400
# followerExp = 4
# ################
# favBuckets=10
# favStep=20
# favExp=2
# ################
# numfriendBuckets = 10
# friendStep = 5
# friendExp=2
# ################
# stExp=4
# statusBuckets = 8
# statusStep = 40
# ################
# numlistBuckets = 5
# listStep = 2
# listExp=6
################
trainfile = 'data_copy/experimental_tweets.txt'
json_data_train = open(trainfile)
train_data = json.loads(json_data_train.readline())
testfile = 'data_copy/test_file.txt'
json_data_test = open(testfile)
test_data = json.loads(json_data_test.readline())
#################
#neural network costants
numLevels=2
numNeuronsPerLevel = 2#24+followerBuckets+favBuckets+numfriendBuckets+statusBuckets+numlistBuckets
# ##b=np.random.normal(0,.01,(numLevels+1, numNeuronsPerLevel)) 
b=[np.random.normal(0,.1,numNeuronsPerLevel) for i in range(numLevels)]
b.append(np.random.normal(0,.1))
# b=[np.ones(numNeuronsPerLevel) for i in range(numLevels)]
# b.append(np.ones(1))
# ##W = [np.random.normal(0,.01,(numNeuronsPerLevel, numNeuronsPerLevel)) for k in range(numLevels+1)]
W = [np.random.normal(0,.1,(numNeuronsPerLevel, numNeuronsPerLevel)) for k in range(numLevels)]
W.append(np.matrix(np.random.normal(0,.1,numNeuronsPerLevel)))
# W=[]
# W.append(np.matrix([[2.0,1.0],[1.0,1.0]]))
# W.append(np.matrix([[1.0,1.0],[1.0,1.0]]))
# W.append(np.matrix([[1.0],[1.0]]))

lambdaDecay=0.01
learningRate=0.1
##################


    
trainExamples=tweetParse.getTrainExamples()
testExamples= tweetParse.getTestExamples()    

learnPredictor(trainExamples,testExamples,tweetFeatureExtractor)



# totRetweetsTimeBuckets = {}

# for line in test_data:
#     pstSecs = line['created_at']-shift
#     secs= pstSecs%secsInDay
#     for i in range(secsInDay/step):
#         if i*step<secs and secs<(i+1)*step:
#             if i in totRetweetsTimeBuckets:
#                 totRetweetsTimeBuckets[i]+=predictor(line,weights)
#             else:
#                 totRetweetsTimeBuckets[i]=predictor(line,weights)


# listversionTotTweets = [0]*(secsInDay/step)
# for key in totRetweetsTimeBuckets:
#     listversionTotTweets[key]=totRetweetsTimeBuckets[key]
# date1 = datetime.datetime( 2000, 3, 2)
# date2 = datetime.datetime( 2000, 3, 3)
# if step==3600:
#     delta = datetime.timedelta(hours=1)
# elif step==1800:
#     delta = datetime.timedelta(minutes=30)
# elif step==900:
#     delta = datetime.timedelta(minutes=15)
# dates = drange(date1, date2, delta)



# fig, ax1 = plt.subplots()
# ax1.plot(dates,listversionTotTweets, 'g.-')
# ax1.set_xlabel('Time')


# ax1.xaxis.set_major_locator( HourLocator() )
# ax1.xaxis.set_minor_locator( HourLocator(arange(0,25,6)) )
# ax1.xaxis.set_major_formatter( DateFormatter('%H:%M') )

# ax1.fmt_xdata = DateFormatter('%H')
# fig.autofmt_xdate()
# plt.suptitle('Total Predicted Retweets per Time Bucket\nTesting Data', fontsize=16)
# plt.show()
