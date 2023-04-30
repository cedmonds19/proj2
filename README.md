# Community Detection with Sentiment Analysis in Musical Preference with Twitter
![made-with-python](https://user-images.githubusercontent.com/56307593/235365242-e5901619-44c0-4b9c-8cde-cb423d12279b.svg)

A Python project aimed at community detection for tweets about four different artists, network modeling, classification, and prediction.
Our project contains two main portions of code:
* Tweet Crawling and Sentiment Analysis using:
  - Tweepy API
  - Preprocessor package for langauge processing and cleaning
  - TextBlob framework for sentiment analysis tools

<img width="757" alt="Screenshot 2023-04-30 at 12 55 10 PM" src="https://user-images.githubusercontent.com/56307593/235365893-378433ae-725c-4d6d-a781-44552224282b.png">

* Machine Learning Models and Algorithms using multiple classification and optimization algorithms inlcuding:
  - Clustering (K-Means and Agglomerative)
  - Feed Forward Neural Network (Keras sequential model)
  - MLP Classifier Backpropagated FFNN
  - Genetic Algorithm (PyGad)
  - Differential Evolution (SciPy)


# Installation
Prerequesites:
* sklearn (we use multiple algorithms from scikit)
* TextBlob
* TensorFlow (with Keras)
* Pandas
* Tweepy
* preprocessor
* Elevated account access for Twitter Dev Account
* SciPy

```
pip install textblob
pip install tensorflow
pip install tweepy
pip install preprocessor
pip install keras
pip install scikit-learn
pip install pandas
pip install scipy
```

Once all of the above dependencies have been properly installed, it's time to install our project files.
The file used for tweet scraping, data preprocessing/cleaning, and data collection is Crawler.py

# Crawler.py
After authenticating unique access keys and tokens, Crawler.py is ready to collect tweets and run sentiment analysis. 

Running "Crawler.py" as is will:
* Collect multiple thousand user tweets containing artist keywords defined in keyword lists for our three different crawling functions
  - The number of tweets collected depends on the time you allow the program to run. Waiting on rate limits and allowing the program to stop naturally will result in the most number of tweets being collected.
* Filter our retweets, hashtags, emojies, etc.
* Add each ID, username, the tokenized tweet text, the artist involved, and the date screated to a pandas dataframe
* Clean tweets further unsuring no NaN values, duplicates, or non-string values
* Run TextBlob sentiment analysis on every tweet collected and add sentiment classification and polarity score columns to a new dataframe


Below is our implementation of our Weeknd tweet crawling function in Crawler.py where tweets are collected, cleaned, and added to a dataframe.
```python
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
```
The number of tweets collected for each artist can be changed inside the file, as well as the keywords used for each artist. We designed Crawler.py for easy name swapping. It's incredibly easy to search for tweets relating to any name you wish; this can be achieved by modifying the artist keyword lists at the beginning of the program. The other artist crawling functions are essentially identical to our ```weeknd_tweets``` crawler, bearing different names and keywords. Tweet features that are added to the dataframe can also be modified by the user, and retweets can be left in the data as well.


```python
if __name__ == "__main__":
    # Weeknd crawler function call and sentiment analysis tweet data
    weeknd_tweets('weekendOutput.csv')
    sentiment_analysis('weekendOutput.csv')

    # Chili Peppers crawler function call and sentiment analysis tweet data
    chili_pepper_tweets('chiliPepperOutput.csv')
    sentiment_analysis('chiliPepperOutput.csv')

    # Miley Cyrus crawler function call and sentiment analysis tweet data
    miley_cyrus_tweets('mileyCyrusOutput.csv')
    sentiment_analysis('mileyCyrusOutput.csv')
```
Crawler.py's main function runs a crawler for each artist, collecting as many tweets allowed by the runtime/rate limit. Our ```sentiment_analysis``` function adds a column for tweet sentiment and polarity score that are assigned and calculated by the TextBlob framework. "Positive", "Negative", or "Neutral" is assigned to every tweet along with a polarity score between (-1, 1) representing how positive or negative each tweet is. New csv files with these columns are created and saved for combination, leading us to our next file.

# mergeData.ipynb (.py)
mergeData.ipynb is a very simple file that reads in our saved tweet csv with sentiment analysis files and combines them into one. 

```python
#Merge all Data
merged_df = pd.concat([chili_data, soulja_data, weekend_data, miley_data], axis=0)

#Write the merged dataframe to a new CSV file
merged_df.to_csv('all_music_data.csv', index=False)
```
Clustering and other algorithms require data from one source, which is the purpose of this file. The files are read in as ```chili_data, soulja_data, weekend_data, miley_data``` before the merging.

