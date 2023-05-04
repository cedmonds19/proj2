import pandas as pd
import twitter
import json
import matplotlib
import networkx as nx # networkx - graph
import numpy
import pickle # For output handling purposes.
import sys
import datetime
import time
from textblob import TextBlob
import matplotlib.pyplot as plt # Graph plotting purposes.
from sys import maxsize as maxint
from functools import partial
from http.client import BadStatusLine
from urllib.error import URLError

import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('omw-1.4')

import tweepy
from nltk.corpus import wordnet


# In[67]:


CONSUMER_KEY = '12Cyj4UGAlzhqicly5CCQgpK0'
CONSUMER_SECRET = 'UpyuspnNzo0YTpXWy7h9qLJAZSfCXsDmQlQGQpk6TMCdPwooH8'
OAUTH_TOKEN = '3304948091-0iu8Qs2xOlmP99dyRB4STJlxYIz09GTgZqaUDN4'
OAUTH_TOKEN_SECRET = 'p08sdleLTpZLb9476QqbHNb5U5bndus8jxO3bwcmcAz9j'

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

twitter_api = tweepy.API(auth)


# In[68]:


# This function was taken from the Chapter #9 Twitter Cookbook program.
def make_twitter_request(twitter_api_func, max_errors=10, *args, **kw):
    # A nested helper function that handles common HTTPErrors. Return an updated
    # value for wait_period if the problem is a 500 level error. Block until the
    # rate limit is reset if it's a rate limiting issue (429 error). Returns None
    # for 401 and 404 errors, which requires special handling by the caller.
    def handle_twitter_http_error(e, wait_period=2, sleep_when_rate_limited=True):
        if wait_period > 3600: # Seconds
            print('Too many retries, Quitting.', file=sys.stderr)
            raise e
            
        # See https://developer.twitter.com/en/docs/basics/response-codes
        # for common codes
        
        if e.e.code == 401:
            print('Encountered 401 Error (Not Authorized)', file=sys.stderr)
            return None
        elif e.e.code == 404:
            print('Encountered 404 Error (Not Found)', file=sys.stderr)
            return None
        elif e.e.code == 429:
            print('Encountered 429 Error (Rate Limit Exceeded)', file=sys.stderr)
            if sleep_when_rate_limited:
                print('Retrying in 15 minutes...ZzZ...', file=sys.stderr)
                sys.stderr.flush()
                time.sleep(60 * 15 + 5)
                print('...ZzZ...Awake now and trying again.', file=sys.stderr)
                return 2
            else: 
                raise e # Caller is required to handle the rate limiting issue.
        elif e.e.code in (500, 502, 503, 504):
            print('Encountered {0} Error. Retrying in {1} seconds.'.format(e.e.code, wait_period), file=sys.stderr)
            time.sleep(wait_period)
            wait_period *= 1.5
            return wait_period
        else:
            raise e
            
     # End of nested helper function

    wait_period = 2
    error_count = 0

    while True:
        try:
            return twitter_api_func(*args, **kw)
        except twitter.api.TwitterHTTPError as e:
            error_count = 0
            wait_period = handle_twitter_http_error(e, wait_period)
            if wait_period is None:
                return
        except URLError as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            printOut("URLError encountered. Continuing.", file=sys.stderr)
            if error_count > max_errors:
                printOut("Too many consecutive errors...bailing out.", file=sys.stderr)
                raise
        except BadStatusLine as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            printOut("BadStatusLine encountered. Continuing.", file=sys.stderr)
            if error_count > max_errors:
                printOut("Too many consecutive errors...bailing out.", file=sys.stderr)
                raise


# In[69]:


# Word bank for the Red Hot Chili Peppers.
word_bank_RH = ["Red Hot Chili Peppers", "RHCP", "Chili Peppers", "Rhcp", "rhcp", "red hot chili peppers"]

# Word bank for The Weeknd.
word_bank_TW = ['The Weeknd', 'Weeknd', 'the weeknd', 'weeknd', 'starboy', 'Starboy']

# Word bank for Soulja Boy.
word_bank_SB = ['Soulja Boy', 'soulja boy', 'Soulja Boi', 'soulja boi', 'Soulja', 'soulja']


# In[60]:


######## RHCP word bank synonyms. ########

# Crawl tweets and append similar words to word bank
for kw in word_bank_RH:
    for tweet in tweepy.Cursor(twitter_api.search_tweets, q=kw, lang='en', tweet_mode='extended').items(10):
        # Get tweet text
        tweet_text = tweet.full_text.lower()

        # Find similar words using NLTK WordNet
        for word in tweet_text.split():
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
            synonyms = list(set(synonyms))

            # Append similar words to word bank
            for syn in synonyms:
                if syn not in word_bank_RH:
                    word_bank_RH.append(syn)
                
# Print updated word bank
print(word_bank_RH)

######## TW word bank synonyms. ########

# Crawl tweets and append similar words to word bank
for kw in word_bank_TW:
    for tweet in tweepy.Cursor(twitter_api.search_tweets, q=kw, lang='en', tweet_mode='extended').items(100):
        # Get tweet text
        tweet_text = tweet.full_text.lower()

        # Find similar words using NLTK WordNet
        for word in tweet_text.split():
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
            synonyms = list(set(synonyms))

            # Append similar words to word bank
            for syn in synonyms:
                if syn not in word_bank_TW:
                    word_bank_TW.append(syn)
                
# Print updated word bank
print(word_bank_TW)

######## SJ word bank synonyms. ######## 

# Crawl tweets and append similar words to word bank
for kw in word_bank_SB:
    for tweet in tweepy.Cursor(twitter_api.search_tweets, q=kw, lang='en', tweet_mode='extended').items(5000):
        # Get tweet text
        tweet_text = tweet.full_text.lower()

        # Find similar words using NLTK WordNet
        for word in tweet_text.split():
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
            synonyms = list(set(synonyms))

            # Append similar words to word bank
            for syn in synonyms:
                if syn not in word_bank_SB:
                    word_bank_SB.append(syn)
                
# Print updated word bank
print(word_bank_SB)


# In[65]:


# Define the columns of the DataFrame.
columns_RH = ['tweet_text', 'user_name', 'user_location', 'user_followers_text']

# Create the empty Pandas DataFrame for this musician.
df_RH = pd.DataFrame(columns=columns_RH)

# Crawl tweets and match with word_bank for the Red Hot Chili Peppers (RHCP).
for kw in word_bank_RH:
    for tweet in tweepy.Cursor(twitter_api.search_tweets, q=kw, lang='en', tweet_mode='extended').items(5000):
        # Obtain the tweet text.
        tweet_text = tweet.full_text.lower()

        # Check for matches with the RHCP word bank.
        for word in word_bank_RH:
            if word in tweet_text:
                # Add tweet data into the DataFrame.
                row = {'tweet_text': tweet_text, 'user_name': tweet.user.screen_name,
                       'user_location': tweet.user.location, 'user_followers_count': tweet.user.followers_count}
                df_RH = df_RH.append(row, ignore_index=True)
            
# Print DataFrame.
print(df_RH)


# In[ ]:


# Define the columns of the DataFrame.
columns_TW = ['tweet_text', 'user_name', 'user_location', 'user_followers_count']

# Create the empty Pandas DataFrame for this musician.
df_TW = pd.DataFrame(columns=columns_TW)

# Crawl tweets and match with word_bank for The Weeknd (TW).
for kw in word_bank_TW:
    for tweet in tweepy.Cursor(twitter_api.search_tweets, q=kw, lang='en', tweet_mode='extended').items(5000):
        # Obtain the tweet text.
        tweet_text = tweet.full_text.lower()

        # Check for matches with the TW word bank.
        for word in word_bank_TW:
            if word in tweet_text:
                # Add tweet data into the DataFrame.
                row = {'tweet_text': tweet_text, 'user_name': tweet.user.screen_name,
                       'user_location': tweet.user.location, 'user_followers_count': tweet.user.followers_count}
                df_TW = df_TW.append(row, ignore_index=True)
            
# Print DataFrame.
print(df_TW)


# In[ ]:


# Define the columns of the DataFrame.
columns_SB = ['tweet_text', 'user_name', 'user_location', 'user_followers_count']

# Create the empty Pandas DataFrame for this musician.
df_SB = pd.DataFrame(columns=columns_SB)

# Crawl tweets and match with word_bank for Soulja Boy (SB).
for kw in word_bank_SB:
    for tweet in tweepy.Cursor(twitter_api.search_tweets, q=kw, lang='en', tweet_mode='extended').items(5000):
        # Obtain the tweet text.
        tweet_text = tweet.full_text.lower()

        # Check for matches with the SB word bank.
        for word in word_bank_SB:
            if word in tweet_text:
                # Add tweet data into the DataFrame.
                row = {'tweet_text': tweet_text, 'user_name': tweet.user.screen_name,
                       'user_location': tweet.user.location, 'user_followers_count': tweet.user.followers_count}
                df_SB = df_SB.append(row, ignore_index=True)
            
# Print DataFrame.
print(df_SB)


# In[70]:


def classify_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'


# In[71]:


def add_polarity_score(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


# In[72]:


def sentiment_analysis(dataframeFile):
    # Load the DataFrame.
    df = pd.read_csv(dataframeFile)
    # Create a new column for the user sentiment classification process.
    df['sentiment'] = df['text'].apply(classify_sentiment)
    # Create a new column for the polarity score.
    df['polarity'] = df['text'].apply(add_polarity_score)
    # Save the updated DataFrame as a new file.
    dataFrameAsString = dataframeFile
    parts = dataFrameAsString.split('csv')
    dataFrameWithPolarity = parts[0] + "WithPolarity.csv"
    df.to_csv(dataFrameWithPolarity, index=False)


# In[ ]:




