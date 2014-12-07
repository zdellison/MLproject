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
import partitionTweetParse

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

tag_list = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']

def FeatureExtractor(k,languageModel=True,userModel=True,timeModel=True):
    if languageModel: print 'Using language model'
    if userModel: print 'Using user model'
    if timeModel: print 'Using time model'
    
    partitions = partitionTweetParse.ParseData(k)

    partitioned_data = [[] for i in range(k)]

    for idx in range(k):
        
        features = []
        y = []
        
        for pos in range(len(partitions[idx])):
            
            y.append(partitions[idx][pos][1])
            f = []

            if timeModel:
                time = [0 for i in range(24)]
                pstSecs = partitions[idx][pos][0]['created_at']-shift
                secs= pstSecs%secsInDay
                time[(int)(secs/step)]=1
                f.extend(time)

            if userModel:
                listed = [0 for i in range(numlistBuckets)]
                listIndex = 0
                if (numlistBuckets)**listExp*listStep<=partitions[idx][pos][0]['user_listed']:
                    listIndex=numlistBuckets-1
                else:
                    listIndex=(int)((partitions[idx][pos][0]['user_listed']/listStep)**(1/(listExp+0.0)))
                listed[listIndex]=1
                f.extend(listed)


                friendBuckets=[0 for i in range(numfriendBuckets)]
                friendIndex=0
                if (numfriendBuckets)**friendExp*friendStep<=partitions[idx][pos][0]['user_friends']:
                    friendIndex=numfriendBuckets-1
                else:
                    friendIndex=(int)((partitions[idx][pos][0]['user_friends']/friendStep)**(1/(friendExp+0.0)))
                friendBuckets[friendIndex]=1
                f.extend(friendBuckets)


                followerB=[0 for i in range(followerBuckets)]
                followIndex=0
                if (followerBuckets)**followerExp*followerStep<partitions[idx][pos][0]['user_followers']:
                    followIndex=followerBuckets-1
                else:
                    followIndex=(int)((partitions[idx][pos][0]['user_followers']/followerStep)**(1/(followerExp+0.0)))
                followerB[followIndex]=1
                f.extend(followerB)


                favB = [0 for i in range(favBuckets)]
                favIndex=0
                if (favBuckets)**favExp*favStep<partitions[idx][pos][0]['user_favourites']:
                    favIndex=favBuckets-1
                else:
                    favIndex=(int)((partitions[idx][pos][0]['user_favourites']/favStep)**(1/(favExp+0.0)))
                favB[followIndex]=1
                f.extend(favB)

                statB = [0 for i in range(statusBuckets)]
                statI = 0
                if (statusBuckets)**stExp*statusStep<partitions[idx][pos][0]['user_statuses_count']:
                    statI=statusBuckets-1
                else:
                    statI = (int)((partitions[idx][pos][0]['user_statuses_count']/statusStep)**(1/(stExp+0.0)))
                statB[statI]=1
                f.extend(statB)

             
            if languageModel:    
                text = TextBlob(partitions[idx][pos][0]['text'])
                language = []
            
                sent = text.sentiment.polarity
                language.append(1 if sent==0 else 0)
                language.append(1 if 0<sent<=0.5 else 0)
                language.append(1 if 0.5<sent else 0)
                language.append(1 if 0>sent>=-0.5 else 0)
                language.append(1 if -0.5>sent else 0)
                subj = text.sentiment.subjectivity
                language.append(1 if subj==0 else 0)
                language.append(1 if 0<subj<=0.5 else 0)
                language.append(1 if 0.5<subj<=1 else 0)
        #        tag_list = []
        #        for word,tag in text.tags:
        #            if tag not in tag_list: tag_list.append(tag)
        #        for tag in tag_list: features['tag_'+tag]=1
                
                tagList = collections.Counter()
                for word,tag in text.tags:
                    tagList[tag]+=1.0
                for tag in tag_list:
                    language.append(1 if (tag in tagList and tagList[tag]==0) else 0)
                    language.append(1 if (tag in tagList and 0<tagList[tag]<=5) else 0)
                    language.append(1 if (tag in tagList and 5<tagList[tag]<=10) else 0)
                    language.append(1 if (tag in tagList and 10<tagList[tag]) else 0)

                f.extend(language)

            features.append(f)

    # We now have feature lists in all k slots of partitioned_data
    # for a given fold (i.e. given i), we can separate the features and y values into:
        # 1. full feature and full y lists
        # 2. classifier feature list and classifier y values - convert non-zero retweet counts to 1
        # 3. non-zero feature list and non-zero y list - make a list of only non-zero retweets
    
        classifier_y_list = []
        non_zero_feature_list = []
        non_zero_y_list = []
        for line in range(len(features)):
            if y[line]==0:
                classifier_y_list.append(0)
            else:
                classifier_y_list.append(1)

                non_zero_feature_list.append([line])
                non_zero_y_list.append(y[line])
        partitioned_data[idx] = ((features,y),(features,classifier_y_list),(non_zero_feature_list,non_zero_y_list))
   
    # ============ RETURN partitioned_data ==============
    return partitioned_data





# TODO: Implement languageModel feature extraction:


    #def tweetFeatureExtractor(partitions[idx]):
    #    features = {}
    #
    #    if languageModel:    
    #        text = TextBlob(partitions[idx]['text'])
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
    
      
                      
