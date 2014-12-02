#!/usr/bin/env python 

import json
import time,datetime

def output_tweets(tr,te):
    train_file = open('../data/train_file.txt','w')
    train_file.write(json.dumps(tr))
    train_file.close()
    test_file = open('../data/test_tweets.txt','w')
    test_file.write(json.dumps(te))
    test_file.close()

f = open('../data/full_tweets2.txt')
train = []
test = []
tweet_num = 0
for tw in f:
    t = eval(tw)
    tweet = {}
    tweet['created_at'] = time.mktime(t.created_at.timetuple())
    tweet['text'] = t.text
    tweet['user_followers'] = t.user.followers_count
    tweet['user_friends'] = t.user.friends_count
    tweet['user_listed'] = t.user.listed_count
    tweet['user_favourites'] = t.user.favourites_count
    tweet['user_statuses_count'] = t.user.statuses_count
    tweet['hashtags_count'] = len(t.entities['hashtags'])
    tweet['hashtags'] = t.entities['hashtags']
    tweet['mentions'] = len(t.entities['user_mentions'])
    tweet['retweets'] = t.retweet_count
    tweet['favorites'] = t.favorite_count
    if tweet_num%10==9:
        test.append(t)
    else:
        train.append(t)
    print t.created_at
    tweet_num+=1

output_tweets(train,test)
print "Number of tweets: {0}".format(tweet_num)
f.close() 
