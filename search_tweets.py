#!/usr/bin/env python

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy import API
from tweepy import TweepError
from tweepy.streaming import StreamListener
import json
import argparse
import time,datetime

# ---------------- Query Parameters ----------------------
query = 'california -RT until:2014-11-13'
count = 100
max_tweets = 1000000
train_file = 'train_file4.txt'
test_file = 'test_file4.txt'
full_tweet_file = 'full_tweets4.txt'
keyfile = 'twitter_keyfile.txt'
logfile = 'tweet_pull_metadata.txt'
# --------------------------------------------------------

log = open(logfile,'a')

log.write("\n\n")
log.write("New Twitter API Query: \n")
log.write("Query: {0}\n".format(query))
log.write("Count: {0}\n".format(count))
log.write("max_tweets: {0}\n".format(max_tweets))
log.write("Train File: {0}\n".format(train_file))
log.write("Test File: {0}\n".format(test_file))
log.write("Full Tweet File: {0}\n".format(full_tweet_file))

# loop through SearchResult objects in search and print to file 'tweet_file.txt'
def output_tweets(tr,te,full_tweets):
    # file to write to
    tr_file = open('../data/'+train_file,'w')
    tr_file.write(json.dumps(tr))
    tr_file.close()
    te_file = open('../data/'+test_file,'w')
    te_file.write(json.dumps(te))
    te_file.close()

    for tweet in full_tweets:
        print str(tweet)

# process tweets
def process_tweets(s):
    train = []
    test = []
    tweet_num = 0
    last_created = ''
    for t in s:
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
            test.append(tweet)
        else:
            train.append(tweet)
        last_created = t.created_at
        tweet_num+=1
    log.write("Training Size: {0}\n".format(len(train)))
    log.write("Testing Size: {0}\n".format(len(test)))
    log.write("Last Created: {0}\n".format(last_created))
    output_tweets(train,test,s)
   
# set up the argument parser

parser = argparse.ArgumentParser(description='Fetch data with Twitter Streaming API')
parser.add_argument('--keyfile', help='file with user credentials')
parser.add_argument('--filter', metavar='W', nargs='*', help='space-separated list of words; tweets are returned that match any word in the list')
args = parser.parse_args()

# read twitter app credentials

if args.keyfile: keyfile=args.keyfile
creds = {}
for line in open(keyfile, 'r'):
    key, value = line.rstrip().split()
    creds[key] = value

# set up authentication
auth = OAuthHandler(creds['api_key'], creds['api_secret'])
auth.set_access_token(creds['token'], creds['token_secret'])

# set up search
api = API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)


searched_tweets = []
last_id = -1
while len(searched_tweets) < max_tweets:
    try:
        new_tweets = api.search(q=query,count=count,max_id=str(last_id-1))
        if not new_tweets:
            break
        searched_tweets.extend(new_tweets)
        last_id = new_tweets[-1].id
    except TweepError as error:
        log.write(error.message)
        break
process_tweets(searched_tweets)


log.close()
