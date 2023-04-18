#!/usr/bin/env python
# coding: utf-8

# # ****
# # Data Pre-processing & K-Means Clustering Program Sectors.
# # ****

# In[106]:


# Import relevant Python packages for program-specific purposes with ML --> K-Means Clustering sections.
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


# In[107]:


sen_data = pd.read_csv("weekendOutputWithPolarity.csv")


# In[108]:


sen_data.head()


# In[109]:


sen_data.isna().sum()


# In[110]:


sen_data.dropna(inplace=True)


# In[111]:


sen_data.reset_index().head()


# In[112]:


Kval = 6

sen_data.dtypes


# In[113]:


# Mapping and Encoding data values within the sen_data Pandas DataFrame.
artist_map = {'The Weeknd' : 0, 'Red Hot Chili Peppers' : 1, 'Soulja Boy' : 2} # Mapping per artist.
sen_data = sen_data.applymap(lambda x : artist_map.get(x) if x in artist_map else x) # Application per artist.

sentiment_map = {'positive' : 0, 'neutral' : 1, 'negative' : 2} # Mapping per sentiment type.
sen_data = sen_data.applymap(lambda x : sentiment_map.get(x) if x in sentiment_map else x) # Application per sentiment type.

sen_data.head()


# In[114]:


# Altering the datetime column of created_at to %YYYY formatting.
sen_data['created_at'] = pd.to_datetime(sen_data['created_at']).dt.strftime('%Y')

sen_data.head()


# In[115]:


sen_data.dtypes


# In[116]:


sen_data['created_at'] = sen_data['created_at'].astype(int)


# In[117]:


sen_data.dtypes


# In[118]:


# Define function required for the importing of viz libraries. 
import plotly
plotly.offline.init_notebook_mode(connected=True)
from plotly.graph_objs import *
from plotly import tools
import plotly.graph_objects as go
import seaborn as sns


# In[119]:


correl = sen_data.corr()

trace = go.Heatmap(z=correl.values,
                   x=correl.index.values,
                   y=correl.columns.values)
data=[trace]
plotly.offline.iplot(data, filename='basic-heatmap')


# In[120]:


sen_data.columns

sen_data.head()


# In[121]:


cols1 = ['tweet_id', 'Username', 'created_at']
cols2 = ['polarity', 'sentiment', 'Artist']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

subset_data = pd.DataFrame(sc.fit_transform(sen_data[cols2]), columns = sen_data[cols2].columns, index = sen_data.index)

subset_data.head()


# # K-Means Clustering.

# In[122]:


from sklearn.cluster import KMeans

km = KMeans(n_clusters = Kval,
            init = 'k-means++',
            n_init = 10,
            max_iter = 500,
            random_state = 50)
data_km = pd.DataFrame(km.fit_predict(subset_data), index = subset_data.index)

data_km.head()


# In[123]:


data_km.rename(columns = {0:'cluster'}, inplace = True)

data_km = data_km.merge(sen_data, left_index = True, right_index = True)

data_km.head()


# In[126]:


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


# In[132]:


distortions = []
for i in range(1, 13):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=500,
                random_state=50)
    km.fit(subset_data[['sentiment', 'polarity']])
    distortions.append(km.inertia_)
plt.plot(range(1, 13), distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()


# In[142]:


distortions = []
for i in range(1,13):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=500,
                random_state=50)
    km.fit(subset_data)
    distortions.append(km.inertia_)
plt.plot(range(1,13), distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()


# # Agglomerative Clustering.

# In[140]:


from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=None,
                             distance_threshold=1.0,
                             affinity='euclidean',
                             linkage='complete')
data_ac = pd.DataFrame(ac.fit_predict(subset_data), index=subset_data.index)
data_ac.head()

data_ac.rename(columns={0 : 'cluster'}, inplace=True)
data_ac = data_ac.merge(sen_data, left_index=True, right_index=True)
data_ac.head()


# In[141]:


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

# In[143]:


pos_sen_data = sen_data[sen_data['polarity'] > 0.0]
pos_sen_data.head()


# In[144]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

pos_pol = pd.DataFrame(sc.fit_transform(pos_sen_data[cols2]), columns=pos_sen_data[cols2].columns, index=pos_sen_data.index)

pos_pol.head()

len(pos_pol)


# In[145]:


from sklearn.cluster import KMeans

km_pos = KMeans(n_clusters = 6,
            init='k-means++',
            n_init=10,
            max_iter=500,
            random_state=50)
data_km = pd.DataFrame(km_pos.fit_predict(pos_pol[cols2]), index=pos_pol[cols2].index)


# In[146]:


data_km.rename(columns={0 : 'cluster'}, inplace=True)
data_km = data_km.merge(sen_data, left_index=True, right_index=True)
data_km.head()


# In[147]:


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


# In[150]:


distortions = []
for i in range(1,7):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=500,
                random_state=50)
    km.fit(pos_pol[['sentiment', 'polarity']])
    distortions.append(km.inertia_)
plt.plot(range(1,7), distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()


# In[151]:


distortions = []
for i in range(1,7):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=500,
                random_state=50)
    km.fit(pos_pol)
    distortions.append(km.inertia_)
plt.plot(range(1,7), distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()


# # ****
# # Neural Network Creation Sector Of Program.
# # ****

# In[154]:


# Import relevant Python packages for building and utilizing of Feed Forward Neural Networks (FFNN) in Python.
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score
from sklearn.metrics import classification_report, roc_auc_score


# In[224]:


nn_df = sen_data.drop(columns=['tweet_id', 'Username', 'text', 'created_at'])

nn_df.head()


# In[262]:


nn_df.loc[nn_df['polarity'] < 0.0] = -1

nn_df.loc[nn_df['polarity'] == 0.0] = 0

nn_df.loc[nn_df['polarity'] > 0.0] = 1

nn_df.head()


# In[263]:


X_train, X_test, y_train, y_test = train_test_split(nn_df.drop(columns=['polarity']), nn_df['polarity'], test_size=0.30, random_state=13)

X_train
X_test
y_train
y_test


# In[282]:


sa_model = Sequential()

sa_model.add(keras.Input(shape=(2,)))
sa_model.add(Dense(10, activation='sigmoid'))
sa_model.add(Dense(1))


# In[283]:


sa_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


# In[284]:


callback00 = ModelCheckpoint(filepath='best_model.hdf5', monitor='val_loss', save_best_only=True, save_weights_only=True)

callback01 = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)


# In[285]:


sa_model_history = sa_model.fit(X_train, y_train, epochs=10, batch_size=10, callbacks=[callback00, callback01], validation_data=(X_test, y_test))


# In[286]:


# Plot train data vs. test data accuracy scores per epoch of Backpropagated FFNN model.

plt.plot(sa_model_history.history['accuracy'])
plt.plot(sa_model_history.history['val_accuracy'])
plt.title('Backpropagated FFNN Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[287]:


sa_pred_x = sa_model.predict(X_train).round()
sa_pred_y = sa_model.predict(X_test).round()


# In[288]:


sa_train_cm = confusion_matrix(y_train, sa_pred_x)
sa_train_cr = classification_report(y_train, sa_pred_x)

print(sa_train_cm)
print(sa_train_cr)


# In[289]:


sa_test_cm = confusion_matrix(y_test, sa_pred_y)
sa_test_cr = classification_report(y_test, sa_pred_y)

print(sa_test_cm)
print(sa_test_cr)


# In[290]:


import seaborn as sns


# In[291]:


# FFNN Train Dataset Confusion Matrix.

plt.figure(figsize=(5,4))
sns.heatmap(sa_train_cm, annot=True)
plt.title('FFNN Train Data Confusion Matrix')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


# In[292]:


# FFNN Test Dataset Confusion Matrix.

plt.figure(figsize=(5,4))
sns.heatmap(sa_test_cm, annot=True)
plt.title('FFNN Test Data Confusion Matrix')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()

