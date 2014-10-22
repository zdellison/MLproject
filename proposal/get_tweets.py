#!/usr/bin/env python

# Use the tweepy package in Python to connect to the Twitter RESTApi search endpoint and get a dump of
# tweets than we can then turn into feature vectors

import tweepy

username = 'zach.ellison@gmail.com'
password = 'public10'
auth = tweepy.auth.BasicAuthHandler(username, password)
api = tweepy.API(auth)

search = api.search('football',rpp=10)
print search


#consumer_key = 'NPhCSKM3miiID9DpMPUO0dbks'
#consumer_secret = 'cjLaGBCX1KF9wjIWP4TKQkQf4ivTQGpZx6PeexszyCD1CTckJh'
#
#
#
#auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#
#try:
#    redirect_url = auth.get_authorization_url()
#except tweepy.TweepError:
#    print 'Error! Failed to get request token.'
#
#session.set('request_token', (auth.request_token.key, auth.request_token.secret))
#
#verifier = request.GET.get('oauth_verifier')
#
#auth = tweepy.OAuthHnadler(consumer_key, consumer_secret)
#token = session.get('request_token')
#session.delete('request_token')
#auth.set_request_token(token[0],token[1])
#
#try:
#    auth.get_access_token(verifier)
#except tweepy.TweepError:
#    print 'Error! Failed to get request token.'
#
#key = auth.access_token.key
#secret = auth.access_token.secret
#
#api = tweepy.API(auth)
#
#search = api.search('football',rpp=10)
#
#print search
