

# Consumer Key (API Key)		mmYbAE3dwhL58soPC6S3yXmrF
# Consumer Secret (API Secret)	QCMZ0YxjLZIQOgmWvGxfhTB1ElZt7qVDY3uMQimqQrm74sP5jh
# Access Token					3080070659-VSSWarRZuaQ9CS8JPjCGNiY1R0Qoelj2pV4B903
# Access Token Secret			n16mpi7m7spd6F1m3XacxOIdrlWALFYdUSajaGzL1H7X6
# Owner							boldbrandywine
# Owner ID						3080070659


# CONNECT to Twitter Streaming API and download data 
# Using Tweepy to connect to Twitter Streaming API / dl data

# Import the necessary methods from tweepy library

import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import csv
import re 
import time 

import urllib
#url = 'http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=1'


import requests 
from bs4 import BeautifulSoup as bs 

from requests.exceptions import ConnectionError


#from twython import Twython # pip install twython
#import time # standard lib

# Variables that contain the user creds to access Twitter API
access_key = "3080070659-VSSWarRZuaQ9CS8JPjCGNiY1R0Qoelj2pV4B903"
access_secret = "n16mpi7m7spd6F1m3XacxOIdrlWALFYdUSajaGzL1H7X6"
consumer_key = "mmYbAE3dwhL58soPC6S3yXmrF"
consumer_secret = "QCMZ0YxjLZIQOgmWvGxfhTB1ElZt7qVDY3uMQimqQrm74sP5jh"


def extract_link(text):
    regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    match = re.search(regex, text)
    if match:
        return match.group()
    return ''

    #tweets['link'] = tweets['text'].apply(lambda tweet: extract_link(tweet))

# def get_abstract(a_tweet):
# 	try:
# 		outlink = extract_link(a_tweet.text.encode("utf-8"))
# 		r = requests.get(outlink)

# 		soup = bs(r.content, "lxml")
# 		letters = soup.find_all("blockquote", class_ ='abstract mathjax')
# 		abstract = letters[0].get_text().replace('\n', ' ')
# 		return abstract
		
# 	except IndexError:
# 		time.sleep(2)
# 		return "INDEXERROR"

# def extract_arxiv_url(text):
#     regex = r'arXiv:?[^\s]+'
#     match = re.search(regex, text)
#     if match:
#         url = "https://arxiv.org/abs/" + str(match.group()[6:])
#         url = url.replace('...', '')
#         return url
#     return ''
		

def get_all_tweets(screen_name):
	#Twitter only allows access to a users most recent 3240 tweets with this method
	
	#authorize twitter, initialize tweepy
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_key, access_secret)
	api = tweepy.API(auth)

	#initialize a list to hold all the tweepy Tweets
	alltweets = []	
	
	#make initial request for most recent tweets (200 is the maximum allowed count)
	new_tweets = api.user_timeline(screen_name = screen_name, count=200)
	
	#save most recent tweets
	alltweets.extend(new_tweets)
	
	#save the id of the oldest tweet less one
	oldest = alltweets[-1].id - 1
	
	#keep grabbing tweets until there are no tweets left to grab
	while len(new_tweets) > 0:
		print "getting tweets before %s" % (oldest)
		
		#all subsiquent requests use the max_id param to prevent duplicates
		new_tweets = api.user_timeline(screen_name = screen_name, count=200,max_id=oldest)
		
		#save most recent tweets
		alltweets.extend(new_tweets)
		
		#update the id of the oldest tweet less one
		oldest = alltweets[-1].id - 1
		
		print "...%s tweets downloaded so far" % (len(alltweets))
	
	#transform the tweepy tweets into a 2D array that will populate the csv	
	outtweets = [[tweet.id_str, tweet.created_at, 
				  extract_link(tweet.text.encode("utf-8")), 
				  tweet.text.encode("utf-8"), tweet.retweet_count, 
				  tweet.favorite_count] for tweet in alltweets]

	# get abstracts
	abstract_list = []
	outlinks = [extract_link(tweet.text.encode("utf-8")) for tweet in alltweets]
	for url in outlinks:
		# print url 
		print "Before 1.5"
		time.sleep(1.5)
		print "After 1.5"
		#r = requests.get(url)
		r = urllib.urlopen(url).read()
		print "Request gotten"
		try:
			soup = bs(r, "lxml") #r.content for requests
			letters = soup.find_all("blockquote", class_ ='abstract mathjax')
			abstract = letters[0].get_text().replace('\n', ' ')
			abstract_list.append(abstract)
		except:
			abstract_list.append("INDEXERROR")


	# concat the two together

	for i in range(0, len(outtweets)):
		outtweets[i].append(abstract_list[i])
	
	#write the csv	
	with open('%s_tweets.csv' % screen_name, 'wb') as f:
		writer = csv.writer(f)
		writer.writerow(["id", "created_at", "link", "text", "retweets", "favorites", "abstract"])
		writer.writerows(outtweets)
	
	pass




if __name__ == '__main__':
	#pass in the username of the account you want to download
	get_all_tweets("StatMLPapers")
















