# Python file containing oects that hold features of a few tweets that we can run
# some initial tests on

# value = retweet_count

# Note this dataset is generate from a query that sets for before 2014-10-15 and a query
# string of including college and football

import time,datetime


example1 = {
        "created_at":.000000001*float(time.mktime(datetime.datetime.strptime("Tue Oct 14 23:59:56 +0000 2014",'%a %b %d %H:%M:%S +0000 %Y').timetuple())),
        #"text": "What time is the USMNT game actually - ESPN still talking college football at 7:59:30",
        'user_followers': 8064,
        'user_friends': 614,
        'user_listed': 338,
        'user_favourites': 42,
        'user_statuses_count': 98326,
        #'hashtags':{},
        'user_mentions':0,
        'media':0
    }
    

value1 = 0

example2 = {
        "created_at":.000000001*float(time.mktime(datetime.datetime.strptime("Tue Oct 14 23:59:51 +0000 2014",'%a %b %d %H:%M:%S +0000 %Y').timetuple())),
        #"text": "RT @USATODAY: Who has the best #CollegeFootball field? You tell us! Vote here: http://t.co/bG9VW6DATO http://t.co/XbFK0qEwz2",
        'user_followers': 61,
        'user_friends': 103,
        'user_listed': 0,
        'user_favourites': 670,
        'user_statuses_count': 303,
        #'hashtags':{
        #    "College Football":1
        #},
        'user_mentions': 1,
        'media':1
}

value2 = 244

example3 = {
        "created_at":.000000001*float(time.mktime(datetime.datetime.strptime("Tue Oct 14 23:59:51 +0000 2014",'%a %b %d %H:%M:%S +0000 %Y').timetuple())),
        #"text": "#College Football UCLA Football: 3 Biggest X-Factors for Bruins vs. Cal: The UCLA footb... http://t.co/4jNwmjSevD http://t.co/xJJrw4pKVM",
        'user_followers': 515,
        'user_friends': 1533,
        'user_listed': 2,
        'user_favourites': 30,
        'user_statuses_count': 19566,
        #'hashtags':{
        #    "College":1
        #    },
        'user_mentions':0,
        'media':0
}

value3 = 1

example4 = {
        "created_at":.000000001*float(time.mktime(datetime.datetime.strptime("Tue Oct 14 23:59:51 +0000 2014",'%a %b %d %H:%M:%S +0000 %Y').timetuple())),
        #"text":"college sec-announces-2015-football\nhttp://t.co/2uI2YRg3me",
        'user_followers': 448,
        'user_friends': 1994,
        'user_listed': 6,
        'user_favourites': 1051,
        'user_statuses_count': 9056,
        #'hashtags':{},
        'user_mentions':0,
        'media':0
}

value4 = 0

example5 = {
        "created_at":.000000001*float(time.mktime(datetime.datetime.strptime("Tue Oct 14 23:59:48 +0000 2014",'%a %b %d %H:%M:%S +0000 %Y').timetuple())),
        #"text":"RT @BUFootball: "(Baylor's) win against TCU is more impressive than Ole Miss' wins against 'Bama & Texax A&M." @MattLeinartQB, FOX 4: htt:...",
        'user_followers': 308,
        'user_friends': 268,
        'user_listed': 2,
        'user_favourites': 7906,
        'user_statuses_count': 2817,
        #'hashtags':{},
        'user_mentions':2,
        'media':0
}

value5 = 36

example6 = {
        "created_at":.000000001*float(time.mktime(datetime.datetime.strptime("Tue Oct 14 23:59:48 +0000 2014",'%a %b %d %H:%M:%S +0000 %Y').timetuple())),
        #"text":"RT @SportsCenter: ICYMI: FSU looking at how 950+ Jameis Winston autographs were authenticated by single company. http://t.co/Igx1wV2jH1 htt...",
        'user_followers': 1468,
        'user_friends': 1735,
        'user_listed': 3,
        'user_favourites': 29280,
        'user_statuses_count': 34682,
        #'hashtags':{},
        'user_mentions':1,
        'media':1
}

value6 = 1458

example6 = {
        "created_at":.000000001*float(time.mktime(datetime.datetime.strptime("Tue Oct 14 23:59:48 +0000 2014",'%a %b %d %H:%M:%S +0000 %Y').timetuple())),
        #"text":"RT @SportsCenter: ICYMI: FSU looking at how 950+ Jameis Winston autographs were authenticated by single company. http://t.co/Igx1wV2jH1 htt...",
        'user_followers': 1468,
        'user_friends': 1735,
        'user_listed': 3,
        'user_favourites': 29280,
        'user_statuses_count': 34682,
        #'hashtags':{},
        'user_mentions':1,
        'media':1
}

value6 = 1458

example7 = {
        "created_at":.000000001*float(time.mktime(datetime.datetime.strptime("Tue Oct 14 23:59:48 +0000 2014",'%a %b %d %H:%M:%S +0000 %Y').timetuple())),
        #"text":"RT @USC_Athletics: USC RB Buck Allen is 3rd in the nation in yards from scrimmage and getting some Heisman buzz. http://t.co/6nVoI3Zuf8 htt...",
        'user_followers': 1826,
        'user_friends': 1919,
        'user_listed': 24,
        'user_favourites': 8024,
        'user_statuses_count': 80359,
        #'hashtags':{},
        'user_mentions':1,
        'media':0
}

value7 = 43

example8 = {
        "created_at":.000000001*float(time.mktime(datetime.datetime.strptime("Tue Oct 14 23:59:45 +0000 2014",'%a %b %d %H:%M:%S +0000 %Y').timetuple())),
        #"text":"Winston's attorney challenges FSU on hearing http://t.co/yR92egiobi",
        'user_followers': 236,
        'user_friends': 193,
        'user_listed': 7,
        'user_favourites': 127,
        'user_statuses_count': 47413,
        #'hashtags':{},
        'user_mentions':1,
        'media':1
}

value8 = 0



# ------------------- Test Values ----------------------

test1 = [
        example7,
        example8
        ]

testVal = [
        value7,
        value8
        ]

examples = [
        example1,
        example2,
        example3,
        example4,
        example5,
        example6
        ]

values = [
        value1,
        value2,
        value3,
        value4,
        value5,
        value6,
        ]
