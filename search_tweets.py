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
query = 'stanford until:2014-11-12'
count = 100
max_tweets = 500000
# --------------------------------------------------------

# loop through SearchResult objects in search and print to file 'tweet_file.txt'
def output_tweets(s,full_tweets):
    # file to write to
    f = open('tweet_file2.txt','w')
    f.write(json.dumps(s))
    f.close()
    f2 = open('full_tweets2.txt','w')
    f2.write(str(full_tweets))
    f2.close()

# process tweets
def process_tweets(s):
    list_tweets = []
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
        list_tweets.append(tweet)
#        print "Modified Tweet is: ",tweet
        print t.created_at
    output_tweets(list_tweets,s)
   
# set up the argument parser

print "Setting up arg parser"

parser = argparse.ArgumentParser(description='Fetch data with Twitter Streaming API')
parser.add_argument('--keyfile', help='file with user credentials', required=True)
parser.add_argument('--filter', metavar='W', nargs='*', help='space-separated list of words; tweets are returned that match any word in the list')
args = parser.parse_args()

# read twitter app credentials

print "Read app credentials"

creds = {}
for line in open(args.keyfile, 'r'):
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
    print "Querying tweets: {0}".format(len(searched_tweets))
    try:
        new_tweets = api.search(q=query,count=count,max_id=str(last_id-1))
        if not new_tweets:
            break
        searched_tweets.extend(new_tweets)
        last_id = new_tweets[-1].id
    except TweepError as error:
        print error.message
        break
process_tweets(searched_tweets)


