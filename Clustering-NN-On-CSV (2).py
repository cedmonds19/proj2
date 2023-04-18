#!/usr/bin/env python
# coding: utf-8

# # ****
# # Data Pre-processing & K-Means Clustering Program Sectors.
# # ****

# In[384]:


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


# In[385]:


# Downloading of relevant CSV file/files for data processing and program implementation.
sen_data = pd.read_csv("chiliPepperOutputWithPolarity.csv")


# In[386]:


# Print out first five lines of the specified Pandas DataFrame. 
sen_data.head()


# In[387]:


# Check for NaN values within each column of the DataFrame.
sen_data.isna().sum()


# In[388]:


# Drop all NaN values from each specific column of the entire DataFrame.
sen_data.dropna(inplace=True)


# In[389]:


# Reset indexes of the DataFrame after removal of all NaN values contained within the DataFrame.
sen_data.reset_index().head()


# In[390]:


# Set Kval to six --> Kval specifies the number of clusters to be produced when executing the 
# K-Means clustering algorithm.
Kval = 6

# Print out each columns datatype within the DataFrame.
sen_data.dtypes


# In[391]:


# Mapping and Encoding data values within the sen_data Pandas DataFrame.
artist_map = {'The Weeknd' : 0, 'Red Hot Chili Peppers' : 1, 'Soulja Boy' : 2} # Mapping per artist.
sen_data = sen_data.applymap(lambda x : artist_map.get(x) if x in artist_map else x) # Application per artist.

sentiment_map = {'positive' : 0, 'neutral' : 1, 'negative' : 2} # Mapping per sentiment type.
sen_data = sen_data.applymap(lambda x : sentiment_map.get(x) if x in sentiment_map else x) # Application per sentiment type.

# Printing out the first five lines of the data-encoded DataFrame.
sen_data.head()


# In[392]:


# Altering the datetime column of created_at to %YYYY formatting.
sen_data['created_at'] = pd.to_datetime(sen_data['created_at']).dt.strftime('%Y')

# Printing out the first five lines of the DataFrame.
sen_data.head()


# In[393]:


# For good measure, check the datatypes of each column within this DataFrame.
sen_data.dtypes


# In[394]:


# Change the datatype of the 'created_at' column from 'object' datatype --> 'int' datatype.
sen_data['created_at'] = sen_data['created_at'].astype(int)


# In[395]:


# Print datatype of columns in DataFrame to ensure that 'created_at' is changed from 'object' to 'int' datatype.
sen_data.dtypes


# In[396]:


# Import viz libraries for data visualization of clustered data values through Scatter plots.  
import plotly
plotly.offline.init_notebook_mode(connected=True)
from plotly.graph_objs import *
from plotly import tools
import plotly.graph_objects as go
import seaborn as sns


# In[397]:


# Compute correlations between values within the DataFrame.
correl = sen_data.corr()

# Compute and print a heatmap of all correlations between data values within the DataFrame.
trace = go.Heatmap(z=correl.values,
                   x=correl.index.values,
                   y=correl.columns.values)
data=[trace]
plotly.offline.iplot(data, filename='basic-heatmap')


# In[398]:


# Print out all columns and first five lines of the DataFrame.
sen_data.columns

sen_data.head()


# In[399]:


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

# In[400]:


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


# In[401]:


# Merge the cluster column of the above DataFrame onto the original DataFrame.
data_km.rename(columns = {0:'cluster'}, inplace = True)

# Merge DataFrame produced after K-Means clustering onto the original DataFrame.
data_km = data_km.merge(sen_data, left_index = True, right_index = True)

data_km.head()


# In[402]:


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


# In[403]:


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


# In[404]:


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

# In[405]:


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


# In[406]:


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

# In[407]:


# Filter the original DataFrame to contain solely polarity values that are greater than 0.
# Thus, filtering the DataFrame for tweets and users that share positive sentiment about the artist.
pos_sen_data = sen_data[sen_data['polarity'] > 0.0]
pos_sen_data.head()


# In[408]:


# Standardize the numerically-valued data columns and create a DataFrame with these normalized numerical columns.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

pos_pol = pd.DataFrame(sc.fit_transform(pos_sen_data[cols2]), columns=pos_sen_data[cols2].columns, index=pos_sen_data.index)

# Print the first five lines of the newly created DataFrame and the length of this DataFrame.
pos_pol.head()

len(pos_pol)


# In[409]:


# Execute the K-Means clustering algorithm upon the new DataFrame.
# Algorithm produces 6 clusters and a new DataFrame containing which cluster each data value is contained within.
from sklearn.cluster import KMeans

km_pos = KMeans(n_clusters = 6,
            init='k-means++',
            n_init=10,
            max_iter=500,
            random_state=50)
data_km = pd.DataFrame(km_pos.fit_predict(pos_pol[cols2]), index=pos_pol[cols2].index)


# In[410]:


# Merge the cluster column of the above DataFrame onto the original DataFrame.
data_km.rename(columns={0 : 'cluster'}, inplace=True)

# Merge DataFrame produced after K-Means clustering onto the original DataFrame.
data_km = data_km.merge(sen_data, left_index=True, right_index=True)

data_km.head()


# In[411]:


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


# In[412]:


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


# In[413]:


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

# In[414]:


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


# In[415]:


# Create a new Pandas DataFrame with solely numerical data columns from the original DataFrame
# containing both text columns and numerical columns.
nn_df = sen_data.drop(columns=['tweet_id', 'Username', 'text', 'created_at'])

nn_df.head()


# In[416]:


# Encode the data values of the 'polarity' data column to be:
    # -1: Negative sentiment about the artist.
    # 0: Neutral sentiment about the artist.
    # +1: Positive sentiment about the artist.
nn_df.loc[nn_df['polarity'] < 0.0] = -1

nn_df.loc[nn_df['polarity'] == 0.0] = 0

nn_df.loc[nn_df['polarity'] > 0.0] = 1

nn_df.head()


# In[417]:


# Split the newly created Pandas DataFrame into corresponding training and testing datasets for ML purposes.
X_train, X_test, y_train, y_test = train_test_split(nn_df.drop(columns=['polarity']), nn_df['polarity'], test_size=0.30, random_state=13)

X_train
X_test
y_train
y_test


# In[418]:


# Create a Keras Sequential FFNN neural network model with architecture of:
    # Input nodes: 2.
    # Hidden layer nodes: 10.
    # Output nodes: 1.
sa_model = Sequential()

sa_model.add(keras.Input(shape=(2,)))
sa_model.add(Dense(10, activation='sigmoid'))
sa_model.add(Dense(1))


# In[419]:


# Compile the Sequential model with Adam() optimizer, categorical_crossentropy loss, and accuracy metric.
sa_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


# In[420]:


# Implement two Sequential model callbacks for overall model optimization of node weights.
callback00 = ModelCheckpoint(filepath='best_model.hdf5', monitor='val_loss', save_best_only=True, save_weights_only=True)

callback01 = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)


# In[421]:


# Fit the Sequential model onto the training data and check with testing data.
# The Sequential model updates and optimizes node weight values through an embedded Backpropagation algorithm.
sa_model_history = sa_model.fit(X_train, y_train, epochs=10, batch_size=10, callbacks=[callback00, callback01], validation_data=(X_test, y_test))


# In[422]:


# Plot train data vs. test data accuracy scores per epoch of Backpropagated FFNN model.

plt.plot(sa_model_history.history['accuracy'])
plt.plot(sa_model_history.history['val_accuracy'])
plt.title('Backpropagated FFNN Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[423]:


# Predict X_train values and X_test values for usage in confusion matrices and classification reports.
sa_pred_x = sa_model.predict(X_train).round()
sa_pred_y = sa_model.predict(X_test).round()


# In[424]:


# Create a confusion matrix and subsequent classification report on the training dataset.
sa_train_cm = confusion_matrix(y_train, sa_pred_x)
sa_train_cr = classification_report(y_train, sa_pred_x)

# Print confusion matrix and classification report.
print(sa_train_cm)
print(sa_train_cr)


# In[425]:


# Create confusion matrix and subsequent classification report on the testing dataset.
sa_test_cm = confusion_matrix(y_test, sa_pred_y)
sa_test_cr = classification_report(y_test, sa_pred_y)

# Print confusion matrix and classification report.
print(sa_test_cm)
print(sa_test_cr)


# In[426]:


# Import seaborn package for visualizing confusion matrices.
import seaborn as sns


# In[427]:


# FFNN Train Dataset Confusion Matrix.
plt.figure(figsize=(5,4))
sns.heatmap(sa_train_cm, annot=True)
plt.title('FFNN Train Data Confusion Matrix')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


# In[428]:


# FFNN Test Dataset Confusion Matrix.
plt.figure(figsize=(5,4))
sns.heatmap(sa_test_cm, annot=True)
plt.title('FFNN Test Data Confusion Matrix')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


# # MLP Classifier Backpropagated FFNN.

# In[429]:


# Import relevant packages for implementing an MLPClassifier FFNN with Backpropagation.
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import warnings 


# In[430]:


# Suppress all warning messages for clarity when inspecting and analyzing program outputs.
# Reduces clutter.
warnings.filterwarnings('ignore')


# In[431]:


# Create an MLPClassifier FFNN with architecture of:
    # Input nodes: 2.
    # Hidden Layer Nodes: 100.
    # Output Nodes: 3.
# Difference with MLPClassifier and Sequential model is that with MLPClassifier the FFNN architecture is inferred.
mlp_sa = MLPClassifier(max_iter=10,
                       alpha=1e-4,
                       solver='adam',
                       verbose=10,
                       tol=1e-4,
                       random_state=13,
                       learning_rate_init=0.1,
                       early_stopping=True)


# In[432]:


# Fit the MLPClassifier onto the training datasets.
mlp_sa.fit(X_train, y_train)


# In[433]:


# Print the accuracy of the MLPClassifier in predicting the values of training and testing datasets.
print('Training set score: {0}'.format(mlp_sa.score(X_train, y_train)))
print('Testing set score: {0}'.format(mlp_sa.score(X_test, y_test)))


# In[434]:


# Predict training and testing dataset values.
train_pred = mlp_sa.predict(X_train)
test_pred = mlp_sa.predict(X_test)


# In[435]:


# Compute accuracy score of the MLPClassifier on the training and testing datasets.
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)


# In[436]:


# Print the accuracy of the MLPClassifier in predicting the values of training and testing datasets.
print(train_acc)
print(test_acc)


# In[437]:


# Plot the loss curve per epoch when executing the MLPClassifier FFNN with embedded Backpropagation algorithm.
plt.plot(mlp_sa.loss_curve_)
plt.show()


# In[438]:


# Plot the accuracy curve per epoch when executing the MLPClassifier FFNN with embedded Backpropagation algorithm.
plt.plot(mlp_sa.validation_scores_)
plt.show()


# In[439]:


# Create confusion matrix and subsequent classification report for training dataset with MLPClassifier.
cm_mlp_tr = confusion_matrix(y_train, train_pred)
cr_mlp_tr = classification_report(y_train, train_pred)

# Print the confusion matrix and classification report --> Training Dataset.
print(cm_mlp_tr)
print(cr_mlp_tr)


# In[440]:


# Create confusion matrix and subsequent classification report for testing dataset with MLPClassifier.
cm_mlp_ts = confusion_matrix(y_test, test_pred)
cr_mlp_ts = classification_report(y_test, test_pred)

# Print the confusion matrix and classification report --> Testing Dataset.
print(cm_mlp_ts)
print(cr_mlp_ts)


# In[441]:


# Train Dataset Confusion Matrix.
plt.figure(figsize=(5,4))
sns.heatmap(cm_mlp_tr, annot=True)
plt.title('Train Dataset Confusion Matrix')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


# In[442]:


# Test Dataset Confusion Matrix.
plt.figure(figsize=(5,4))
sns.heatmap(cm_mlp_ts, annot=True)
plt.title('Test Dataset Confusion Matrix')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


# In[443]:


# Print out all coefficients of the MLPClassifier Backpropagated FFNN.
print(mlp_sa.coefs_)


# In[444]:


# Print out all intercepts of the MLPClassifier Backpropagated FFNN.
print(mlp_sa.intercepts_)


# In[445]:


# Print out the number of input features being computed through the MLPClassifier Backpropagated FFNN.
print(mlp_sa.n_features_in_)

