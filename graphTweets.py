file = 'tweet_file_3.txt'
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from numpy import arange

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
########################## seconds in each bucket. 900=15min, 1800=30min, 3600=1hr
step = 1800
##########################
for line in data:
	numTweets+=1
	#print ""
	#print "Line is: ",line
	pstSecs = line['created_at']-shift
	secs= pstSecs%secsInDay
	# if line['retweets']>fltr:
	# 	continue
	if line['retweets']>0:
		numRetweeted+=1
		print line
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
print "RETWEET RATE: ",numRetweeted/numTweets
json_data.close()