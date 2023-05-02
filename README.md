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
# Merge all Data
merged_df = pd.concat([chili_data, soulja_data, weekend_data, miley_data], axis=0)

# Write the merged dataframe to a new CSV file
merged_df.to_csv('all_music_data.csv', index=False)
```
Clustering and other algorithms require data from one source, which is the purpose of this file. The files are read in as ```chili_data, soulja_data, weekend_data, miley_data``` before the merging.


# Machine Learning Implmentation
After data collection is complete and our CSV files are saved with artist tweet data, the machine learning aspect of our project is primed and ready.

Running Clustering-NN-On-CSV.py as is will:
* Normalize/Standardize our numerical data for clustering purposes
* Executes the K-Means clustering algorithm upon the new dataframe with all collected tweets and their sentiments/polarities
  - k = 6 in our project. This can be changed by the user. Max iterations and and n_init are also modifiable parameters
* Executes the Agglomerative clustering algorithm
  - The algorithm produces 0 clusters and employs euclidean distance as a measure of cluster affinity. These are all modifiable parameters that can change the result of our agglomeration
* Visualize our clustering algorithms, including distortions, number of clusters, and where each cluster center is
* Create a feed forward neural network model and train on our tweet data using:
  - Backpropagation
  - MLP Classifier
  - Genetic Algorithm (PyGad)
  - Differential Evolution (SciPy)
  
## Clustering - how to run

K-Means: fit subset solely on polarity and sentiment data columns. This is run on the total dataset and also a subset of only positive tweets as well. K-Means clustering can be implemented and run on many different subsets of our data. (ie. only positive or negative tweets about certain artists)
  - Modifiable parameters: n_clusters, init, n_init, max_iter, random_state -> any of these parameters can be changed as the user desires

Agglomerative: very similar implementation to K-Means and used mainly for comparison
  - Modifiable parameters: see code below
  
  ```python
  from sklearn.cluster import AgglomerativeClustering

  ac = AgglomerativeClustering(n_clusters=None,
                             distance_threshold=1.0,
                             affinity='euclidean',
                             linkage='complete')
 ```

## Neural Network Implementation - how it works
* Encode the data values of the 'polarity' data column to be:
    - -1: Negative sentiment about the artist.
    - 0: Neutral sentiment about the artist.
    - +1: Positive sentiment about the artist.
* Split the newly created Pandas DataFrame into corresponding training and testing datasets for ML purposes.
```python
X_train, X_test, y_train, y_test = train_test_split(nn_df.drop(columns=['polarity']), nn_df['polarity'], test_size=0.30, random_state=13)
```

* Create a Keras Sequential FFNN neural network model with architecture of:
    - Input nodes: 2 (artist and sentiment)
    - Hidden layer nodes: 10.
    - Output nodes: 1 (sentiment classification)
* Compile the Sequential model with Adam() optimizer, categorical_crossentropy loss, and accuracy metric.

```python
sa_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
```


## MLP Classifier
The difference with MLPClassifier and Sequential model is that with MLPClassifier the FFNN architecture is inferred. Predictions are made after MLPClassifier runs and compiles. 

Parameters:
```python
mlp_sa = MLPClassifier(max_iter=100,
                       alpha=1e-4,
                       solver='adam',
                       batch_size=32,
                       verbose=10,
                       tol=1e-4,
                       random_state=13,
                       learning_rate_init=0.1,
                       early_stopping=True)
```
This MLPClassifier is then fit to the dataset for training. MLPClassifier is extraordinarily easy to implement and quite effective as well.


## Genetic Algorithm Implementation
Using the same neural network architecture as our neural network implementation using Keras, we also trained our model using a Genetic Algorithm. Here are a few tips to properly implement and install a GA using the PyGad framework.

Neural Network:
```python
ga_model = Sequential()
ga_model.add(keras.Input(shape=(2,)))
ga_model.add(Dense(10, activation='sigmoid'))
ga_model.add(Dense(1))
```

Define fitness function: binary accuracy as metric for fitness, update network weights after each generation
```python
def fitness_func(solution, sol_idx):
    global X_train, y_train, keras_ga, ga_model
    
    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=ga_model,
                                                                 weights_vector=solution)
    
    ga_model.set_weights(weights=model_weights_matrix)
    
    predictions = ga_model.predict(X_train)
    
    ba = tensorflow.keras.metrics.CategoricalAccuracy()
    ba.update_state(y_train, predictions)
    ret_acc = ba.result().numpy()
    
    return ret_acc
```

Instantiate a kerasGA:
```python
keras_ga = pygad.kerasga.KerasGA(model = ga_model,
                                 num_solutions = 20)
```

Parameters: initial population are the inital keras population weights, runs for 10 epochs, 5 parents reproduce every epoch, two point crossover is used (can be easily changed). 
* Mutation type is a self defined function we implemented that swaps a randomly non-maked weight and a randomly masked weight

```python
initial_population = keras_ga.population_weights
num_generations = 10
num_parents_mating = 5
crossover_type = "two_points"
crossover_probability = 0.6
mutation_type = mutation_exec(masked_weights, ga_model)
parent_selection_type = "rws"
keep_elitism = 1

ga_fin = pygad.GA(num_generations=num_generations,
                  num_parents_mating=num_parents_mating,
                  initial_population=initial_population,
                  crossover_type=crossover_type,
                  crossover_probability=crossover_probability,
                  mutation_type=mutation_type,
                  parent_selection_type=parent_selection_type,
                  keep_elitism=keep_elitism,
                  fitness_func=fitness_func,
                  on_generation=callback_generation)
```

After running and compiling, our GA makes predictions based on our best solution weights on our trained neural network model


## Differential Evolution Implementation: how to implement and use
* Again, use the same neural network architecture as before (2, 10, 1)
* Implement a fitness function (very similar to our GA implementation)

```python
def fitness_func(params):
    global history, train_acc_de, test_acc_de
    
    loss = tensorflow.keras.losses.BinaryCrossentropy()
    de_model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])
    history = de_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    
    accuracy = history.history['accuracy'][-1]
    test_acc = history.history['val_accuracy']
    train_acc_de.append(accuracy)
    test_acc_de.append(test_acc)
    
    return accuracy
```
* Set bounds for parameter search and instantiate other parameters
  - tol: relative tolerance for convergence
  - maxiter: maximum number of generations over which the entire population is evolved
  - mutation: mutation constant - random mutation rate between 0.5-1 assigned after every generation
  - recombination: same as crossover probability - 70% of crossover

```python
bounds = [(10, 100), (10, 100)]
result = differential_evolution(fitness_func,
                                bounds,
                                tol=0.01,
                                maxiter=10,
                                mutation=(0.5, 1),
                                recombination=0.7)
````
* Predict outcomes using trained model

```python
de_pred_y = de_model.predict(X_test).round()
de_pred_x = de_model.predict(X_train).round()
```

where de_model is the same implementation as 

```python
ga_model = Sequential()
ga_model.add(keras.Input(shape=(2,)))
ga_model.add(Dense(10, activation='sigmoid'))
ga_model.add(Dense(1))
```


