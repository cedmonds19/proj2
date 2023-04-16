#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Packages and Imports
import tweepy
import pandas as pd
from twitter_scraper import get_tweets
from multiprocessing import Pool
import twitter
import json
import networkx as nx # networkx - graph 
import matplotlib
import numpy
import sys
import time
from functools import partial
from sys import maxsize as maxint
from urllib.error import URLError
from http.client import BadStatusLine
import matplotlib.pyplot as plt # For plotting graph
import pickle # Handling output?


# In[7]:


# Twitter dev account login for access keys/tokens/secrets - Example 1 Cookbook
    
# Elevated account access
CONSUMER_KEY = '12Cyj4UGAlzhqicly5CCQgpK0'
CONSUMER_SECRET = 'UpyuspnNzo0YTpXWy7h9qLJAZSfCXsDmQlQGQpk6TMCdPwooH8'
OAUTH_TOKEN = '3304948091-0iu8Qs2xOlmP99dyRB4STJlxYIz09GTgZqaUDN4'
OAUTH_TOKEN_SECRET = 'p08sdleLTpZLb9476QqbHNb5U5bndus8jxO3bwcmcAz9j'
    
# authenticate with the Twitter API
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

# create the API object
api = tweepy.API(auth)


# In[8]:


# Keywords - word bank for tweet scraping

tweets = []

weeknd_keywords = ["Weeknd", "Abel Makkonen Tesfaye"]
chili_pepper_keywords = ["red hot chili peppers", "chili peppers", "rhcp"]
soulja_boy_keywords = ["DeAndre Cortez Way", "Soulja Boy"]

artist_keywords = ["weeknd", "Weeknd", "Red Hot Chili Peppers", 
            "red hot chili peppers", "the weeknd", "the Weeknd",
            "chili peppers", "Soulja Boy", "soulja boy", 
            "DeAndre Cortez Way", "Abel Makkonen Tesfaye"]


# In[9]:


def weeknd_tweets(keyword):
    weeknd_tweet_data = pd.DataFrame(columns=['tweet_id', 'Username', 'text', 'Artist', 'created_at'])
    num_tweets = 50
    
    for keyword in weeknd_keywords:
        for tweet in tweepy.Cursor(api.search_tweets, q=keyword, lang='en').items(num_tweets):
            tweets.append(tweet)
    
    for tweet in tweets:
        weeknd_tweet_data = weeknd_tweet_data.append({'tweet_id': tweet.id,
                                                      'Username': tweet.user.screen_name,
                                                      'text': tweet.text,
                                                      'Artist': "The Weeknd",
                                                      'created_at': tweet.created_at},
                                                     ignore_index=True)
    return weeknd_tweet_data


# In[10]:


weeknd_tweets("Weeknd")


# In[ ]:





# In[ ]:




