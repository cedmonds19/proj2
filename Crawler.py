# Packages and Imports
import tweepy
import pandas as pd
import twitter
from textblob import TextBlob
import json
import networkx as nx # networkx - graph 
import matplotlib
import numpy
import sys
import time
import re
import preprocessor as p
from functools import partial
from sys import maxsize as maxint
from urllib.error import URLError
from http.client import BadStatusLine
import matplotlib.pyplot as plt # For plotting graph

# Twitter dev account login for access keys/tokens/secrets
# Elevated account access
CONSUMER_KEY = '12Cyj4UGAlzhqicly5CCQgpK0'
CONSUMER_SECRET = 'UpyuspnNzo0YTpXWy7h9qLJAZSfCXsDmQlQGQpk6TMCdPwooH8'
OAUTH_TOKEN = '3304948091-0iu8Qs2xOlmP99dyRB4STJlxYIz09GTgZqaUDN4'
OAUTH_TOKEN_SECRET = 'p08sdleLTpZLb9476QqbHNb5U5bndus8jxO3bwcmcAz9j'
    
# authenticate with the Tweepy API
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

# Create  API object
# Parameters: Access keys and tokens
# wait_on_rate_limit set to true to handle exceeding rate limit with correct errors
api = tweepy.API(auth, wait_on_rate_limit=True, retry_count=10, retry_delay=5, retry_errors=set([503]))

# Keywords for tweets mentioning the Weeknd, Red Hot Chili Peppers, Soulja Boy, and Miley Cyrus
weeknd_keywords = ["Weeknd", "Abel Makkonen Tesfaye"]
chili_pepper_keywords = ["red hot chili peppers", "chili peppers", "rhcp"]
miley_cyrus_keywords = ["Miley Cyrus", "Hannah Montana"]

# Emoticons (in case of old style)
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
emoticons = emoticons_happy.union(emoticons_sad)
# Emojis
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

# Clean tweets
# Uses preprocessor package: cleans emojies, hashtags, and URLs
def clean_tweets(tweet):
    clean_tweet = p.clean(tweet)
    return clean_tweet

# Crawl tweets regarding The Weeknd and store to dataframe and csv file
def weeknd_tweets(output_file): 
    weeknd_tweets = []
    num_tweets = 8000
    
    # Add each tweet (excluding retweets) to our tweet list
    for keyword in weeknd_keywords:
        for tweet in tweepy.Cursor(api.search_tweets, q=keyword, lang='en', tweet_mode='extended').items(num_tweets):
            # Filter out retweets
            if not hasattr(tweet, 'retweeted_status'):
                weeknd_tweets.append(tweet)

    # Add tweets to list
    weeknd_tweet_data = []
    for tweet in weeknd_tweets:
        weeknd_tweet_dict = {'tweet_id': tweet.id, # Dataframe columns being filled
                                                      'Username': tweet.user.screen_name,
                                                      'text': tweet.full_text,
                                                      'Artist': "The Weeknd",
                                                      'created_at': tweet.created_at}
        weeknd_tweet_data.append(weeknd_tweet_dict)
    
    # Create DataFrame from list of dictionaries using pd.concat
    weeknd_tweet_data_df = pd.concat([pd.DataFrame(data=[tweet]) for tweet in weeknd_tweet_data], ignore_index=True)
    
    # Drop duplicate and NaN values and print/save the dataframe to a new file
    weeknd_tweet_data_df.drop_duplicates()
    weeknd_tweet_data_df.dropna()
    weeknd_tweet_data_df['text'] = weeknd_tweet_data_df['text'].apply(clean_tweets, str) # Clean tweets and make sure they are str
    weeknd_tweet_data_df.to_csv(output_file, index=False)
    
    return weeknd_tweet_data_df # csv file

# TextBlob framework for sentiment analysis
# Classifies each tweet as either positive, negative, or neutral
def classify_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Crawl tweets regarding RHCP and store to dataframe and csv file
# Same as initial weekend crawler but with different keywords
def chili_pepper_tweets(output_file):
    chili_pepper_tweets = []
    num_tweets = 8000
    
    for keyword in chili_pepper_keywords:
        for cp_tweet in tweepy.Cursor(api.search_tweets, q=keyword, lang='en').items(num_tweets):
            # Filter out retweets
            if not hasattr(cp_tweet, 'retweeted_status'):
                chili_pepper_tweets.append(cp_tweet)
    
    # Add tweets to dataframe
    chili_pepper_tweet_data = []
    for cp_tweet in chili_pepper_tweets:
        chili_pepper_tweet_dict = {'tweet_id': cp_tweet.id,
                                                      'Username': cp_tweet.user.screen_name,
                                                      'text': cp_tweet.text,
                                                      'Artist': "Red Hot Chili Peppers",
                                                      'created_at': cp_tweet.created_at}
        chili_pepper_tweet_data.append(chili_pepper_tweet_dict)
    # Create DataFrame from list of dictionaries using pd.concat
    chili_pepper_tweet_data_df = pd.concat([pd.DataFrame(data=[cp_tweet]) for cp_tweet in chili_pepper_tweet_data], ignore_index=True)
    
    # Clean tweets and print/save the dataframe to a new file
    chili_pepper_tweet_data_df.drop_duplicates()
    chili_pepper_tweet_data_df.dropna()
    chili_pepper_tweet_data_df['text'] = chili_pepper_tweet_data['text'].apply(clean_tweets, str)
    chili_pepper_tweet_data_df.to_csv(output_file, index=False)
    return chili_pepper_tweet_data_df

# Miley Cyrus crawler - Same as previous two with different keybank
# This was initially our Soulja Boy crawler
# Switched names and keywords after running it for Soulja Boy
def miley_cyrus_tweets(output_file):
    miley_cyrus_tweets = []
    num_tweets = 8000
    
    # Keyword detection for soulja boy
    for keyword in miley_cyrus_keywords:
        for mc_tweet in tweepy.Cursor(api.search_tweets, q=keyword, lang='en').items(num_tweets):
            # Filter out retweets
            if not hasattr(mc_tweet, 'retweeted_status'):
                miley_cyrus_tweets.append(mc_tweet)

    # Add tweets to dataframe
    miley_cyrus_tweet_data = []
    for mc_tweet in miley_cyrus_tweets:
        miley_cyrus_tweet_dict = {'tweet_id': mc_tweet.id,
                                                      'Username': mc_tweet.user.screen_name,
                                                      'text': mc_tweet.text,
                                                      'Artist': "Miley Cyrus",
                                                      'created_at': mc_tweet.created_at}
        miley_cyrus_tweet_data.append(miley_cyrus_tweet_dict)
    
    # Create DataFrame from list of dictionaries using pd.concat
    miley_cyrus_tweet_data_df = pd.concat([pd.DataFrame(data=[mc_tweet]) for mc_tweet in miley_cyrus_tweet_data], ignore_index=True)
    
    # Clean tweets and print/save the dataframe to a new file
    miley_cyrus_tweet_data_df.drop_duplicates()
    miley_cyrus_tweet_data_df.dropna()
    miley_cyrus_tweet_data_df['text'] = miley_cyrus_tweet_data_df['text'].apply(clean_tweets, str)    
    miley_cyrus_tweet_data_df.to_csv(output_file, index=False)
    
    return miley_cyrus_tweet_data_df

# TextBlob framework used again here
# Adds polarity value for each tweet
def add_polarity_score(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Creates new dataset with added columns after tweet Sentiment Analysis
def sentiment_analysis(dataframeFile):
    # Load the data frame
    df = pd.read_csv(dataframeFile)
    df['text'] = df['text'].apply(clean_tweets, str)
    
    # Create a new column for sentiment classification
    df['sentiment'] = df['text'].apply(classify_sentiment)
   
    # Create a new column for polarity score
    df['polarity'] = df['text'].apply(add_polarity_score)
    
    # Save the updated data frame as a new file
    dataFrameAsString = dataframeFile
    parts = dataFrameAsString.split(".csv")
    dataFrameWithPolarity = parts[0] + "WithPolarity.csv"
    df.to_csv(dataFrameWithPolarity, index=False)


if __name__ == "__main__":
    # Weeknd crawler function call and sentiment analysis tweet data
    weeknd_tweets('weekendOutput.csv')
    sentiment_analysis('weekendOutput.csv')

    # Chili Peppers crawler function call and sentiment analysis tweet data
    chili_pepper_tweets('chiliPepperOutput.csv')
    sentiment_analysis('chiliPepperOutput.csv')

    # Soulja Boy crawler function call and sentiment analysis tweet data
    miley_cyrus_tweets('mileyCyrusOutput.csv')
    sentiment_analysis('mileyCyrusOutput.csv')