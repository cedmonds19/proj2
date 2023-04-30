# Community Detection with Sentiment Analysis in Musical Preference with Twitter
![made-with-python](https://user-images.githubusercontent.com/56307593/235365242-e5901619-44c0-4b9c-8cde-cb423d12279b.svg)

A Python project aimed at community detection for tweets about four different artists, network modeling, classification, and prediction.
Our project contains two main portions of code:
* Tweet Crawling and Sentiment Analysis using Tweepy API and the TextBlob framework for language processing

* Tweepy Instantiation
<img width="757" alt="Screenshot 2023-04-30 at 12 55 10 PM" src="https://user-images.githubusercontent.com/56307593/235365893-378433ae-725c-4d6d-a781-44552224282b.png">

* TextBlob function
<img width="474" alt="Screenshot 2023-04-30 at 12 55 47 PM" src="https://user-images.githubusercontent.com/56307593/235365915-92959f2c-9086-43c7-9165-48673bf336f9.png">



* Machine Learning Models and Algorithms using multiple classification and optimization algorithms





# INSTALLATION
Prerequesites:
* sklearn (we use multiple algorithms from scikit)
* TextBlob
* TensorFlow (with Keras)
* Pandas
* Tweepy
* preprocessor
* Elevated account access for Twitter Dev Account

```
pip install textblob
pip install tensorflow
pip install tweepy
pip install preprocessor
pip install keras
pip install scikit-learn
pip install pandas
```
Once all of the above dependencies have been properly installed, it's time to install our project files.
The file used for tweet scraping, data preprocessing/cleaning, and data collection is Crawler.py

* After authenticating unique access keys and tokens, Crawler.py is ready to collect tweets and run SA. 
The screenshot of our code below is one of our three crawler functions.

Clicking run as is will:
* Collect multiple thousand user tweets containing artist keywords defined in keyword lists
* Filter our retweets
* Add each tweets ID, username, the tokenized tweet text, the artist involved, and the date screated to a pandas dataframe
* Clean tweets further unsuring no NaN values, duplicates, or non-string values
* Run TextBlob sentiment analysis on every tweet collected and add sentiment classification and polarity score columns to a new dataframe

Below is our implementation of our Weeknd tweet crawling function in Crawler.py where tweets are collected, cleaned, and added to a dataframe.
<img width="795" alt="Screenshot 2023-04-30 at 1 12 00 PM" src="https://user-images.githubusercontent.com/56307593/235366694-f130f02d-12c0-48ea-a310-898145b68bb0.png">
