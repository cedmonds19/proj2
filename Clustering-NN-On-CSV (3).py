#!/usr/bin/env python
# coding: utf-8

# # ****
# # Data Pre-processing & K-Means Clustering Program Sectors.
# # ****

# In[1132]:


# Import relevant Python packages for program-specific purposes with ML --> K-Means Clustering sections.
# Pandas is the primary package of choice throughout this program for preprocessing and handling all datasets.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

pd.options.display.float_format = '{:,.2f}'.format

# Setup the interactive notebook mode.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from IPython.display import display, HTML


# In[1133]:


# Downloading of relevant CSV file/files for data processing and program implementation.
#sen_data = pd.read_csv("weekendOutputWithPolarity.csv")
#sen_data = pd.read_csv("chiliPepperOutputWithPolarity.csv")
sen_data = pd.read_csv("all_music_data.csv")


# In[1134]:


# Print out first five lines of the specified Pandas DataFrame. 
sen_data.head()

# Print length of sen_data Pandas DataFrame.
len(sen_data)


# In[1135]:


# Check for NaN values within each column of the DataFrame.
sen_data.isna().sum()


# In[1136]:


# Drop all NaN values from each specific column of the entire DataFrame.
sen_data.dropna(inplace=True)


# In[1137]:


# Reset indexes of the DataFrame after removal of all NaN values contained within the DataFrame.
sen_data.reset_index().head()


# In[1138]:


# Set Kval to six --> Kval specifies the number of clusters to be produced when executing the 
# K-Means clustering algorithm.
Kval = 6

# Print out each columns datatype within the DataFrame.
sen_data.dtypes


# In[1139]:


# Mapping and Encoding data values within the sen_data Pandas DataFrame.
artist_map = {'The Weeknd' : 0, 'Red Hot Chili Peppers' : 1, 'Soulja Boy' : 2, 'Miley Cyrus' : 3} # Mapping per artist.
sen_data = sen_data.applymap(lambda x : artist_map.get(x) if x in artist_map else x) # Application per artist.

sentiment_map = {'positive' : 0, 'neutral' : 1, 'negative' : 2} # Mapping per sentiment type.
sen_data = sen_data.applymap(lambda x : sentiment_map.get(x) if x in sentiment_map else x) # Application per sentiment type.

# Printing out the first five lines of the data-encoded DataFrame.
sen_data.head()


# In[1140]:


# Altering the datetime column of created_at to %YYYY formatting.
sen_data['created_at'] = pd.to_datetime(sen_data['created_at']).dt.strftime('%Y')

# Printing out the first five lines of the DataFrame.
sen_data.head()


# In[1141]:


# For good measure, check the datatypes of each column within this DataFrame.
sen_data.dtypes


# In[1142]:


# Change the datatype of the 'created_at' column from 'object' datatype --> 'int' datatype.
sen_data['created_at'] = sen_data['created_at'].astype(int)


# In[1143]:


# Print datatype of columns in DataFrame to ensure that 'created_at' is changed from 'object' to 'int' datatype.
sen_data.dtypes


# In[1144]:


# Import viz libraries for data visualization of clustered data values through Scatter plots.  
import plotly
plotly.offline.init_notebook_mode(connected=True)
from plotly.graph_objs import *
from plotly import tools
import plotly.graph_objects as go
import seaborn as sns


# In[1145]:


# Compute correlations between values within the DataFrame.
correl = sen_data.corr()

# Compute and print a heatmap of all correlations between data values within the DataFrame.
trace = go.Heatmap(z=correl.values,
                   x=correl.index.values,
                   y=correl.columns.values)
data=[trace]
plotly.offline.iplot(data, filename='basic-heatmap')


# In[1146]:


# Print out all columns and first five lines of the DataFrame.
sen_data.columns

sen_data.head()


# In[1147]:


# Split data columns of the DataFrame into text columns and numerical-valued columns for clustering purposes.
cols1 = ['tweet_id', 'Username', 'created_at']
cols2 = ['polarity', 'sentiment', 'Artist']

# Standardize the numerically-valued data columns and create a DataFrame with these normalized numerical columns. 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

subset_data = pd.DataFrame(sc.fit_transform(sen_data[cols2]), columns = sen_data[cols2].columns, index = sen_data.index)

# Print first five lines of the new DataFrame containing normalized numerical values. 
subset_data.head()


# # K-Means Clustering.

# In[1148]:


# Execute the K-Means clustering algorithm upon the new DataFrame.
# Algorithm produces 6 clusters and a new DataFrame containing which cluster each data value is contained within.
from sklearn.cluster import KMeans

km = KMeans(n_clusters = Kval,
            init = 'k-means++',
            n_init = 10,
            max_iter = 500,
            random_state = 50)
data_km = pd.DataFrame(km.fit_predict(subset_data), index = subset_data.index)

data_km.head()


# In[1149]:


# Merge the cluster column of the above DataFrame onto the original DataFrame.
data_km.rename(columns = {0:'cluster'}, inplace = True)

# Merge DataFrame produced after K-Means clustering onto the original DataFrame.
data_km = data_km.merge(sen_data, left_index = True, right_index = True)

data_km.head()


# In[1150]:


# Plot the clusters onto a graph through a Scatter Plot that displays specific cluster data points.
plot_data = []

for clus in set(data_km['cluster']):
    df = data_km[data_km['cluster'] == clus]
    plot_data.append(go.Scatter(x=df['sentiment'], y=df['polarity'],
                                text=df['Artist'],
                                name='cluster' + str(clus), mode='markers'))
    
# Add the following for cluster centroids.
# df_cc = pd.DataFrame(km.cluster_centers_)
# plot_data.append(go.Scatter(x=df_cc[1], y=df_cc[0],
#                           # text=df['name'],
#                             name='cluster center', mode='markers'))

layout = go.Layout(xaxis=dict(title='Sentiment'), yaxis=dict(title='Polarity'),
                   title='Clustering')
fig = go.Figure(data=plot_data, layout=layout)
plotly.offline.iplot(fig)


# In[1151]:


# Plot K-Means clustering distortions.
# This plot displays the iterations required to minimize the euclidean distance of points within a cluster to
# the centroid of that specific cluster. 
distortions = []
for i in range(1, 30):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=500,
                random_state=50)
    km.fit(subset_data[['sentiment', 'polarity']]) # Fitting solely 'polarity' and 'sentiment' data columns.
    distortions.append(km.inertia_)
plt.plot(range(1, 30), distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()


# In[1152]:


# Plot K-Means clustering distortions.
# This plot displays the iterations required to minimize the euclidean distance of points within a cluster to
# the centroid of that specific cluster.
# Difference between this plot and above plot is that this plot computes distortions for all columns
# of the DataFrame, not solely for 'polarity' and for 'sentiment'. Thus a generalized distortions plot.
distortions = []
for i in range(1, 30):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=500,
                random_state=50)
    km.fit(subset_data) # Fitting all data columns of the 'subset_data' DataFrame.
    distortions.append(km.inertia_)
plt.plot(range(1, 30), distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()


# # Agglomerative Clustering.

# In[1153]:


# Implementation and execution of the Agglomerative Clustering algorithm. 
# Primarily implemented for comparison with the K-Means clustering algorithm.
# The algorithm produces 0 clusters and employs euclidean distance as a measure of cluster affinity.
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=None,
                             distance_threshold=1.0,
                             affinity='euclidean',
                             linkage='complete')

# Create and display a new DataFrame containing which cluster each specific data point is contained within.
data_ac = pd.DataFrame(ac.fit_predict(subset_data), index=subset_data.index)
data_ac.head()

# Merge the 'cluster' column of the 'data_ac' DataFrame onto the original DataFrame and diaplay the DataFrame.
data_ac.rename(columns={0 : 'cluster'}, inplace=True)
data_ac = data_ac.merge(sen_data, left_index=True, right_index=True)
data_ac.head()


# In[1154]:


# Plot the clusters onto a graph through a Scatter Plot that displays specific cluster data points.
plot_data = []
for clus in set(data_ac['cluster']):
    df = data_ac[data_ac['cluster'] == clus]
    plot_data.append(go.Scatter(x=df['sentiment'], y=df['polarity'],
                                text=df['Artist'],
                                name='cluster' + str(clus), mode='markers'))

layout = go.Layout(xaxis=dict(title='Sentiment'), yaxis=dict(title='Polarity'),
                   title='Clustering')
fig = go.Figure(data=plot_data, layout=layout)
plotly.offline.iplot(fig)


# # K-Means Clustering --> Positive Polarity - Sentiment Analysis.

# In[1155]:


# Filter the original DataFrame to contain solely polarity values that are greater than 0.
# Thus, filtering the DataFrame for tweets and users that share positive sentiment about the artist.
pos_sen_data = sen_data[sen_data['polarity'] > 0.0]
pos_sen_data.head()


# In[1156]:


# Standardize the numerically-valued data columns and create a DataFrame with these normalized numerical columns.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

pos_pol = pd.DataFrame(sc.fit_transform(pos_sen_data[cols2]), columns=pos_sen_data[cols2].columns, index=pos_sen_data.index)

# Print the first five lines of the newly created DataFrame and the length of this DataFrame.
pos_pol.head()

len(pos_pol)


# In[1157]:


# Execute the K-Means clustering algorithm upon the new DataFrame.
# Algorithm produces 6 clusters and a new DataFrame containing which cluster each data value is contained within.
from sklearn.cluster import KMeans

km_pos = KMeans(n_clusters = 6,
            init='k-means++',
            n_init=10,
            max_iter=500,
            random_state=50)
data_km = pd.DataFrame(km_pos.fit_predict(pos_pol[cols2]), index=pos_pol[cols2].index)


# In[1158]:


# Merge the cluster column of the above DataFrame onto the original DataFrame.
data_km.rename(columns={0 : 'cluster'}, inplace=True)

# Merge DataFrame produced after K-Means clustering onto the original DataFrame.
data_km = data_km.merge(sen_data, left_index=True, right_index=True)

data_km.head()


# In[1159]:


# Plot the clusters onto a graph through a Scatter Plot that displays specific cluster data points.
plot_data = []
for clus in set(data_km['cluster']):
    df = data_km[data_km['cluster'] == clus]
    plot_data.append(go.Scatter(x=df['sentiment'], y=df['polarity'],
                                text=df['Artist'],
                                name='cluster' + str(clus), mode='markers'))
    
# Added the succeeding source-code for cluster centers.
# df_cc = pd.DataFrame(km_pos.cluster_centers_)
# plot_data.append(go.Scatter(x=df_cc[1], y=df_cc[0],
#                             text=df['name'],
#                             name='cluster center', mode='markers'))

layout = go.Layout(xaxis=dict(title='Sentiment'), yaxis=dict(title='Polarity'),
                   title='Clustering')
fig = go.Figure(data=plot_data, layout=layout)
plotly.offline.iplot(fig)


# In[1160]:


# Plot K-Means clustering distortions.
# This plot displays the iterations required to minimize the euclidean distance of points within a cluster to
# the centroid of that specific cluster.
distortions = []
for i in range(1, 30):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=500,
                random_state=50)
    km.fit(pos_pol[['sentiment', 'polarity']]) # Fitting solely 'polarity' and 'sentiment' data columns.
    distortions.append(km.inertia_)
plt.plot(range(1, 30), distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()


# In[1161]:


# Plot K-Means clustering distortions.
# This plot displays the iterations required to minimize the euclidean distance of points within a cluster to
# the centroid of that specific cluster.
# Difference between this plot and above plot is that this plot computes distortions for all columns
# of the DataFrame, not solely for 'polarity' and for 'sentiment'. Thus a generalized distortions plot.
distortions = []
for i in range(1, 30):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=500,
                random_state=50)
    km.fit(pos_pol) # Fitting all data columns of the 'pos_pol' DataFrame.
    distortions.append(km.inertia_)
plt.plot(range(1, 30), distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()


# # ****
# # Neural Network Creation Sector Of Program.
# # ****

# In[1162]:


# Import relevant Python packages for building and utilizing of Feed Forward Neural Networks (FFNN) in Python.
# The Python packages of Keras and of TensorFlow are the packages of choice for FFNN creation and utilization.
import tensorflow as tf
import keras
import tensorflow.keras
from tensorflow.keras import layers
from keras import optimizers
import random
from random import randrange
from math import exp
from tensorflow.keras.optimizers.legacy import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense

# Import relevant packages for the assessment of FFNN optimization.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score
from sklearn.metrics import classification_report, roc_auc_score


# In[1163]:


# Create a new Pandas DataFrame with solely numerical data columns from the original DataFrame
# containing both text columns and numerical columns.
nn_df = sen_data.drop(columns=['tweet_id', 'Username', 'text', 'created_at'])

nn_df.head()


# In[1164]:


# Encode the data values of the 'polarity' data column to be:
    # -1: Negative sentiment about the artist.
    # 0: Neutral sentiment about the artist.
    # +1: Positive sentiment about the artist.
nn_df.loc[nn_df['polarity'] < 0.0] = -1

nn_df.loc[nn_df['polarity'] == 0.0] = 0

nn_df.loc[nn_df['polarity'] > 0.0] = 1

nn_df.head()


# In[1165]:


# Split the newly created Pandas DataFrame into corresponding training and testing datasets for ML purposes.
X_train, X_test, y_train, y_test = train_test_split(nn_df.drop(columns=['polarity']), nn_df['polarity'], test_size=0.30, random_state=13)

X_train
X_test
y_train
y_test


# In[1166]:


# Create a Keras Sequential FFNN neural network model with architecture of:
    # Input nodes: 2.
    # Hidden layer nodes: 10.
    # Output nodes: 1.
sa_model = Sequential()

sa_model.add(keras.Input(shape=(2,)))
sa_model.add(Dense(10, activation='sigmoid'))
sa_model.add(Dense(1))


# In[1167]:


# Compile the Sequential model with Adam() optimizer, categorical_crossentropy loss, and accuracy metric.
sa_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


# In[1168]:


# Implement two Sequential model callbacks for overall model optimization of node weights.
callback00 = ModelCheckpoint(filepath='best_model.hdf5', monitor='val_loss', save_best_only=True, save_weights_only=True)

callback01 = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)


# In[1169]:


# Fit the Sequential model onto the training data and check with testing data.
# The Sequential model updates and optimizes node weight values through an embedded Backpropagation algorithm.
sa_model_history = sa_model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[callback00, callback01], validation_data=(X_test, y_test))


# In[1170]:


# Plot train data vs. test data accuracy scores per epoch of Backpropagated FFNN model.

plt.plot(sa_model_history.history['accuracy'])
plt.plot(sa_model_history.history['val_accuracy'])
plt.title('Backpropagated FFNN Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[1171]:


# Predict X_train values and X_test values for usage in confusion matrices and classification reports.
sa_pred_x = sa_model.predict(X_train).round()
sa_pred_y = sa_model.predict(X_test).round()


# In[1172]:


# Create a confusion matrix and subsequent classification report on the training dataset.
sa_train_cm = confusion_matrix(y_train, sa_pred_x)
sa_train_cr = classification_report(y_train, sa_pred_x)

# Print confusion matrix and classification report.
print(sa_train_cm)
print(sa_train_cr)


# In[1173]:


# Create confusion matrix and subsequent classification report on the testing dataset.
sa_test_cm = confusion_matrix(y_test, sa_pred_y)
sa_test_cr = classification_report(y_test, sa_pred_y)

# Print confusion matrix and classification report.
print(sa_test_cm)
print(sa_test_cr)


# In[1174]:


# Import seaborn package for visualizing confusion matrices.
import seaborn as sns


# In[1175]:


# FFNN Train Dataset Confusion Matrix.
plt.figure(figsize=(5,4))
sns.heatmap(sa_train_cm, annot=True)
plt.title('FFNN Train Data Confusion Matrix')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


# In[1176]:


# FFNN Test Dataset Confusion Matrix.
plt.figure(figsize=(5,4))
sns.heatmap(sa_test_cm, annot=True)
plt.title('FFNN Test Data Confusion Matrix')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


# # MLP Classifier Backpropagated FFNN.

# In[1177]:


# Import relevant packages for implementing an MLPClassifier FFNN with Backpropagation.
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import warnings 


# In[1178]:


# Suppress all warning messages for clarity when inspecting and analyzing program outputs.
# Reduces clutter.
warnings.filterwarnings('ignore')


# In[1179]:


# Create an MLPClassifier FFNN with architecture of:
    # Input nodes: 2.
    # Hidden Layer Nodes: 100.
    # Output Nodes: 3.
# Difference with MLPClassifier and Sequential model is that with MLPClassifier the FFNN architecture is inferred.
mlp_sa = MLPClassifier(max_iter=100,
                       alpha=1e-4,
                       solver='adam',
                       batch_size=32,
                       verbose=10,
                       tol=1e-4,
                       random_state=13,
                       learning_rate_init=0.1,
                       early_stopping=True)


# In[1180]:


# Fit the MLPClassifier onto the training datasets.
mlp_sa.fit(X_train, y_train)


# In[1181]:


# Print the accuracy of the MLPClassifier in predicting the values of training and testing datasets.
print('Training set score: {0}'.format(mlp_sa.score(X_train, y_train)))
print('Testing set score: {0}'.format(mlp_sa.score(X_test, y_test)))


# In[1182]:


# Predict training and testing dataset values.
train_pred = mlp_sa.predict(X_train)
test_pred = mlp_sa.predict(X_test)


# In[1183]:


# Compute accuracy score of the MLPClassifier on the training and testing datasets.
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)


# In[1184]:


# Print the accuracy of the MLPClassifier in predicting the values of training and testing datasets.
print(train_acc)
print(test_acc)


# In[1185]:


# Plot the loss curve per epoch when executing the MLPClassifier FFNN with embedded Backpropagation algorithm.
plt.plot(mlp_sa.loss_curve_)
plt.show()


# In[1186]:


# Plot the accuracy curve per epoch when executing the MLPClassifier FFNN with embedded Backpropagation algorithm.
plt.plot(mlp_sa.validation_scores_)
plt.show()


# In[1187]:


# Create confusion matrix and subsequent classification report for training dataset with MLPClassifier.
cm_mlp_tr = confusion_matrix(y_train, train_pred)
cr_mlp_tr = classification_report(y_train, train_pred)

# Print the confusion matrix and classification report --> Training Dataset.
print(cm_mlp_tr)
print(cr_mlp_tr)


# In[1188]:


# Create confusion matrix and subsequent classification report for testing dataset with MLPClassifier.
cm_mlp_ts = confusion_matrix(y_test, test_pred)
cr_mlp_ts = classification_report(y_test, test_pred)

# Print the confusion matrix and classification report --> Testing Dataset.
print(cm_mlp_ts)
print(cr_mlp_ts)


# In[1189]:


# Train Dataset Confusion Matrix.
plt.figure(figsize=(5,4))
sns.heatmap(cm_mlp_tr, annot=True)
plt.title('Train Dataset Confusion Matrix')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


# In[1190]:


# Test Dataset Confusion Matrix.
plt.figure(figsize=(5,4))
sns.heatmap(cm_mlp_ts, annot=True)
plt.title('Test Dataset Confusion Matrix')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


# In[1191]:


# Print out all coefficients of the MLPClassifier Backpropagated FFNN.
print(mlp_sa.coefs_)


# In[1192]:


# Print out all intercepts of the MLPClassifier Backpropagated FFNN.
print(mlp_sa.intercepts_)


# In[1193]:


# Print out the number of input features being computed through the MLPClassifier Backpropagated FFNN.
print(mlp_sa.n_features_in_)


# In[1194]:


# Plot accuracy score of Sequential model with accuracy score of MLPClassifier for comparison of models methods.
plt.plot(mlp_sa.validation_scores_)
plt.plot(sa_model_history.history['accuracy'])
plt.title('Sequential vs. MLPClassifier Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['MLPClassifier', 'Sequential'], loc='upper left')
plt.show()


# # ****
# # Beginning of Genetic Algorithm For Execution Upon Dataset.
# # ****

# In[1195]:


import pygad
import pygad.nn
import pygad.gann
import pygad.kerasga


# In[1196]:


# Create a Keras Sequential FFNN neural network model with architecture of:
    # Input nodes: 2.
    # Hidden layer nodes: 10.
    # Output nodes: 1.
ga_model = Sequential()

ga_model.add(keras.Input(shape=(2,)))
ga_model.add(Dense(10, activation='sigmoid'))
ga_model.add(Dense(1))


# In[1197]:


ga_model.compile(optimizer='adam', loss=tensorflow.keras.losses.CategoricalCrossentropy(), metrics=tensorflow.keras.metrics.Accuracy())


# In[1198]:


print(ga_model.get_weights())


# In[1199]:


gann_weights = np.concatenate([layer.flatten() for layer in ga_model.get_weights()])

masked_weights = np.zeros_like(gann_weights)
org_weights = int(0.1 * gann_weights.size)
org_indexes = np.random.choice(gann_weights.size, size=org_weights, replace=False)

masked_weights[org_indexes] = np.random.choice(gann_weights[org_indexes], size=org_weights)

print(masked_weights)


# In[1200]:


keras_ga = pygad.kerasga.KerasGA(model = ga_model,
                                 num_solutions = 20)


# In[1201]:


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


# In[1202]:


print("X_train size: ", X_train.size)
print("y_train size: ", y_train.size)


# In[1203]:


# Global array variables for storing train data and test data accuracy scores per GA epoch.
ga_arr_train_acc = []
ga_arr_test_acc = []


# In[1204]:


def callback_generation(ga_instance):
    
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    
    ################ Storing values of the train dataset prediction accuracy. ################
    
    y_pred_tr_ga = pygad.kerasga.predict(model=ga_model,
                                      solution=solution,
                                      data=X_train)
    
    train_ba_ga = tensorflow.keras.metrics.BinaryAccuracy()
    train_ba_ga.update_state(y_train, y_pred_tr_ga)
    train_acc_ga = train_ba_ga.result().numpy()
    ga_arr_train_acc.append(train_acc_ga)
    print("Train Accuracy Score: {0:0.4f}".format(train_acc_ga))
    
    ################ Dividing line - Computing CA of Test Data Below. ################
    
    y_pred_ts_ga = pygad.kerasga.predict(model=ga_model,
                                         solution=solution,
                                         data=X_test)
    
    test_ba_ga = tensorflow.keras.metrics.BinaryAccuracy()
    test_ba_ga.update_state(y_test, y_pred_ts_ga)
    test_acc_ga = test_ba_ga.result().numpy()
    ga_arr_test_acc.append(test_acc_ga)
    print("Test Accuracy Score: {0:0.4f}".format(test_acc_ga))


# In[1205]:


def mutation_exec(masked_weights, ga_model):
    
    global gann_weights
    
    non_masked_indexes = np.where(masked_weights == 1)[0]
    masked_indexes = np.where(masked_weights == 0)[0]
    
    if len(non_masked_indexes) > 0 and len(masked_weights) > 0:
        rand_nonmasked_index = np.random.choice(non_masked_indexes)
        rand_masked_index = np.where(masked_weights[rand_nonmasked_index] == 0)[0][0]
        
        p_val = ga_model[rand_nonmasked_index].copy()
        
        gann_weights[rand_nonmasked_index] = gann_weights[rand_masked_index]
        gann_weights[rand_masked_index] = p_val
        
        ga_model.set_weights(gann_weights)
        
        return ga_model


# In[1206]:


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


# In[1207]:


ga_fin.run()


# In[1208]:


ga_fin.plot_fitness(title = "PyGAD & Keras/Tensorflow Epoch vs. Model Fitness", linewidth=5)


# In[1209]:


solution, solution_fitness, solution_idx = ga_fin.best_solution()

print("Fitness value of the best solution: {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution: {solution_idx}".format(solution_idx=solution_idx))


# In[1210]:


predictions = pygad.kerasga.predict(model=ga_model,
                                    solution=solution,
                                    data=X_train)

print("Predictions: \n", predictions)


# In[1211]:


bce = tensorflow.keras.losses.BinaryCrossentropy()
print("Binary Crossentropy: ", bce(y_train, predictions).numpy())


# In[1212]:


ba = tensorflow.keras.metrics.BinaryAccuracy() 
ba.update_state(y_train, predictions) 
ga_accuracy = ba.result().numpy() * 100
print("Genetic Algorithm Overall Accuracy: %", ga_accuracy)


# In[1213]:


y_pred_tr_ga = pygad.kerasga.predict(model=ga_model, 
                                     solution=solution, 
                                     data=X_train) 

train_ba_ga = tensorflow.keras.metrics.BinaryAccuracy() 
train_ba_ga.update_state(y_train, y_pred_tr_ga) 
train_acc_ga = train_ba_ga.result().numpy() 
print("Train Accuracy Score: {0:0.4f}".format(train_acc_ga)) 

y_pred_ts_ga = pygad.kerasga.predict(model=ga_model, 
                                     solution=solution, 
                                     data=X_test) 

test_ba_ga = tensorflow.keras.metrics.BinaryAccuracy() 
test_ba_ga.update_state(y_test, y_pred_ts_ga) 
test_acc_ga = test_ba_ga.result().numpy() 
print("Test Accuracy Score: {0:0.4f}".format(test_acc_ga))


# In[1214]:


plt.plot(ga_arr_train_acc) 
plt.plot(ga_arr_test_acc)
plt.title("Genetic Algorithm Model - Accuracy vs. Log Weight Updates")
plt.xlabel("Log Weight Updates") 
plt.ylabel("GA Accuracy") 
plt.legend(["Train", "Test"], loc = "upper left") 
plt.show()


# # ****
# # Beginning Of Differential Evolution For Execution Upon The Dataset.
# # ****

# In[ ]:


from scipy.optimize import differential_evolution


# In[ ]:


# Variable to store DE train data and test data model.fit progress data.
history = None
train_acc_de = []
test_acc_de = []


# In[ ]:


# Create a Keras Sequential FFNN neural network model with architecture of:
    # Input nodes: 2.
    # Hidden layer nodes: 10.
    # Output nodes: 1.
de_model = Sequential()

de_model.add(keras.Input(shape=(2,)))
de_model.add(Dense(10, activation='sigmoid'))
de_model.add(Dense(1))


# In[ ]:


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


# In[ ]:


# Set bounds for the parameter search.

bounds = [(10, 100), (10, 100)]


# In[ ]:


result = differential_evolution(fitness_func,
                                bounds,
                                tol=0.01,
                                maxiter=10,
                                mutation=(0.5, 1),
                                recombination=0.7)


# In[1129]:


print('Optimal Accuracy: ', result.fun)
print('Optimal parameters for weight optimization: ', result.x)


# In[1130]:


# Plot train data and test data accuracy scores over DE algorithm execution.

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy vs. Log Weight Updates")
plt.xlabel("Log Weight Updates")
plt.ylabel("Model Accuracy")
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[1131]:


de_pred_y = de_model.predict(X_test).round()
de_pred_x = de_model.predict(X_train).round()

