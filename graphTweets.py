file = 'data_copy/test_file.txt'
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from numpy import arange
import math

import json
from pprint import pprint
secsInDay = 24*60*60
shift = 8*60*60
fltr = 10000
json_data = open(file)
numRetweeted = 0.0
numTweets = 0.0
data = json.loads(json_data.readline())
#pprint(data)[
numTweetsTimeBuckets = {}
timeBuckets = {}
totalRetweetsTimeBuckets = {}
maxtweet=0
numRetweetedLarge =0
sumRetweets=0
sumRetweetsLarge=0
########################## seconds in each bucket. 900=15min, 1800=30min, 3600=1hr
step = 3600

numfollowBuckets=10
followerBuckets=[0]*(numfollowBuckets+1)
fBCounts = [0]*(numfollowBuckets+1)
followerStep=400
##########################
numfriendBuckets = 15
friendStep = 5
friendExp=2
friendBuckets=[0]*(numfriendBuckets+1)
frCounts=[0]*(numfriendBuckets+1)
######
numstBuckets = 10
stStep = 40
stExp=4
stBuckets=[0]*(numstBuckets+1)
stCounts=[0]*(numstBuckets+1)
########
numlistBuckets = 6
listStep = 2
listExp=6
listBuckets=[0]*(numlistBuckets+1)
listCounts=[0]*(numlistBuckets+1)

for line in data:
	numTweets+=1




	for i in range(numlistBuckets):
		if (i)**listExp*listStep<=line['user_listed'] and (i+1)**listExp*listStep>line['user_listed']:
			listBuckets[i]+=line['retweets']
			listCounts[i]+=1.0
	if (numlistBuckets-1)**listExp*listStep<=line['user_listed']:
		listBuckets[numlistBuckets]+=line['retweets']
		listCounts[numlistBuckets]+=1.0






	for i in range(numstBuckets):
		if (i)**stExp*stStep<=line['user_statuses_count'] and (i+1)**stExp*stStep>line['user_statuses_count']:
			stBuckets[i]+=line['retweets']
			stCounts[i]+=1.0
	if (numstBuckets-1)**stExp*stStep<=line['user_statuses_count']:
		stBuckets[numstBuckets]+=line['retweets']
		stCounts[numstBuckets]+=1.0


	for i in range(numfriendBuckets):
		if (i)**friendExp*friendStep<=line['user_friends'] and (i+1)**friendExp*friendStep>line['user_friends']:
			friendBuckets[i]+=line['retweets']
			frCounts[i]+=1.0
	if (numfriendBuckets-1)**friendExp*friendStep<=line['user_friends']:
		friendBuckets[numfriendBuckets]+=line['retweets']
		frCounts[numfriendBuckets]+=1.0


	for i in range(numfollowBuckets):
		if i**4*followerStep<=line['user_followers'] and (i+1)**4*followerStep>line['user_followers']:
			followerBuckets[i]+=line['retweets']
			fBCounts[i]+=1.0
	if (numfollowBuckets-1)**4*followerStep<=line['user_followers']:
		followerBuckets[numfollowBuckets]+=line['retweets']
		fBCounts[numfollowBuckets]+=1.0
	


	#print ""
	#print "Line is: ",line
	pstSecs = line['created_at']-shift
	secs= pstSecs%secsInDay
	# if line['retweets']>fltr:
	# 	continue
	if line['retweets']>0:
		numRetweeted+=1
		sumRetweets+=line['retweets']

	if line['retweets']>10:
		numRetweetedLarge+=1
		sumRetweetsLarge+=line['retweets']
	# if line['retweets']>maxtweet:
	# 	maxtweet=line['retweets']
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
listversionAveTweets = [0]*(secsInDay/step)
listversionTotTweets = [0]*(secsInDay/step)
listversionTraffic = [0]*(secsInDay/step)
for key in timeBuckets:
	listversionAveTweets[key]=(totalRetweetsTimeBuckets[key]+0.0)/(timeBuckets[key]+0.0)
	listversionTotTweets[key]=totalRetweetsTimeBuckets[key]
	listversionTraffic[key]=timeBuckets[key]


 

for i in range(numlistBuckets+1):
	if listCounts[i]>0:
		listBuckets[i]=(listBuckets[i]+0.0)/listCounts[i]
print listBuckets

plt.plot(listBuckets,label='Average Retweets VS Lists');

plt.suptitle('Average Retweets VS Lists: \nNumber of Buckets: '+str(numlistBuckets)+', Step Size: '+str(listStep)+', Exponent: '+str(listExp), fontsize=12)
plt.show()



for i in range(numstBuckets+1):
	if stCounts[i]>0:
		stBuckets[i]=(stBuckets[i]+0.0)/stCounts[i]
print stBuckets
plt.plot(stBuckets,label='Average Retweets VS Friends');
plt.suptitle('Average Retweets VS Statuses: \nNumber of Buckets: '+str(numstBuckets)+', Step Size: '+str(stStep)+', Exponent: '+str(stExp), fontsize=12)
plt.show()



for i in range(numfriendBuckets+1):
	if frCounts[i]>0:
		friendBuckets[i]=(friendBuckets[i]+0.0)/frCounts[i]
print friendBuckets
plt.plot(friendBuckets,label='Average Retweets VS Friends');
plt.suptitle('Average Retweets VS Friends: \nNumber of Buckets: '+str(numfriendBuckets)+', Step Size: '+str(friendStep)+', Exponent: '+str(friendExp), fontsize=12)
plt.show()



for i in range(numfollowBuckets+1):
	if fBCounts[i]>0:
		followerBuckets[i]=(followerBuckets[i]+0.0)/fBCounts[i]
print followerBuckets
plt.plot(followerBuckets,label='Average Retweets VS Followers');
plt.suptitle('Average Retweets VS Followers: \nNumber of Buckets: '+str(numfollowBuckets)+', Step Size: '+str(followerStep)+', Exponent: '+str(4), fontsize=12)
plt.show()




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
#ax1.plot(t, s1, 'b-')
lns1 = ax1.plot(dates,listversionAveTweets, 'b-', label='Average Retweets per Time')

ax1.set_xlabel('Time')
# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel('Average Retweets', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')


ax2 = ax1.twinx()
#ax2.plot(t, s2, 'r.')
lns2 =ax2.plot(dates,listversionTotTweets, 'r--',label='Total Retweets per Time')
ax2.set_ylabel('Retweets', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')

#ax3 = ax2.twinx()#.twiny()
#ax2.plot(t, s2, 'r.')
#ax3.plot(dates,listversionTraffic, 'g.-')
# ax3.set_ylabel('Total Tweets per Time (traffic)', color='g')
# for tl in ax2.get_yticklabels():
#     tl.set_color('g')


ax1.xaxis.set_major_locator( HourLocator() )
ax1.xaxis.set_minor_locator( HourLocator(arange(0,25,6)) )
ax1.xaxis.set_major_formatter( DateFormatter('%H:%M') )

ax1.fmt_xdata = DateFormatter('%H')
fig.autofmt_xdate()

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
plt.suptitle('Total True Retweets per Time Bucket\nTest Data', fontsize=16)
plt.show()
fig, ax1 = plt.subplots()
ax1.plot(dates,listversionTraffic, 'g.-',label = 'Total Traffic')
ax1.set_xlabel('Time')
ax2.set_ylabel('Retweets', color='r')

ax1.xaxis.set_major_locator( HourLocator() )
ax1.xaxis.set_minor_locator( HourLocator(arange(0,25,6)) )
ax1.xaxis.set_major_formatter( DateFormatter('%H:%M') )

ax1.fmt_xdata = DateFormatter('%H')
fig.autofmt_xdate()
ax1.legend()
plt.show()
print '############'
print 'num tweets', numTweets
print "num retweeted ",numRetweeted 
print 'num retweeted large',numRetweetedLarge
print "RETWEET RATE: ",numRetweeted/numTweets
print "sum retweets", sumRetweets
print "sum retweets large", sumRetweetsLarge
json_data.close()