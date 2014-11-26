#
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from numpy import arange
import json
from pprint import pprint
import time
import tweetParse
import linear_regression

trExamples=tweetParse.getTrainExamples()
teExamples=tweetParse.getTestExamples()
weights = linear_regression.learnPredictor(trExamples,teExamples,linear_regression.tweetFeatureExtractor)
step = linear_regression.step
predicted1 =  [0 for i in range(linear_regression.secsInDay/step)]
predicted2= [0 for i in range(linear_regression.secsInDay/step)]
feat1 = {"user_listed": 41, "user_followers": 5286, "text": "Mini Med School: The Heart - Stanford Continuing Studies Program | http://t.co/XCZfAYEKkm | Anatomy &amp; Physiology #free #Anatomy #Physiology", "created_at": 1415778490.0, "hashtags": [{"indices": [117, 122], "text": "free"}, {"indices": [123, 131], "text": "Anatomy"}, {"indices": [132, 143], "text": "Physiology"}], "hashtags_count": 3, "user_favourites": 4, "user_friends": 3019, "favorites": 0, "mentions": 0, "user_statuses_count": 187601, "retweets": 6}
feat2 = {"user_listed": 13, "user_followers": 153, "text": "@StanfordCISAC Congrats on writing a great Stanford Tweet! http://t.co/6InMD83cVO (Ranked 31st for Nov 10.)", "created_at": 1415778948.0, "hashtags": [], "hashtags_count": 0, "user_favourites": 41, "user_friends": 36, "favorites": 0, "mentions": 1, "user_statuses_count": 14648, "retweets": 0}

for i in range(linear_regression.secsInDay/step):
    feat1['created_at']=feat2['created_at']=linear_regression.shift+1+i*step
    predicted1[i]=linear_regression.dotProduct(linear_regression.tweetFeatureExtractor(feat1),weights)
    predicted2[i]=linear_regression.dotProduct(linear_regression.tweetFeatureExtractor(feat2),weights)


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
lns1 = ax1.plot(dates,predicted1, 'b-', label='Mini Med School:')

ax1.set_xlabel('Time')
# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel('Predicted Retweets', color='b')

ax2 = ax1.twinx()
#ax2.plot(t, s2, 'r.')
lns2 =ax2.plot(dates,predicted2, 'r--',label='Congrats')





ax1.xaxis.set_major_locator( HourLocator() )
ax1.xaxis.set_minor_locator( HourLocator(arange(0,25,6)) )
ax1.xaxis.set_major_formatter( DateFormatter('%H:%M') )

ax1.fmt_xdata = DateFormatter('%H')
fig.autofmt_xdate()

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)

plt.show()

