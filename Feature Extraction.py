#!/usr/bin/env python
# coding: utf-8


%cd -your path-

# # **Importing All Libraries**

# In[ ]:


import librosa, os, soundfile, numpy as np, pandas as pd, matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
from IPython.core.display import HTML
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
import glob, os
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import OneClassSVM
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import scipy.stats as stats


# # **Selecting Data with Similar Features**

# In[ ]:




totalLangData = [] #variable for all the 10 languages features
totalLabelData = [] #variable for all the labels

# Get all .npy files
npy_files = glob.glob("*.npy")

# Select the files you want to process using a list of indices
indices = [1,2,6,9,10,12,13,14,15,17]
for index in indices:
    if index < len(npy_files):
        filename = npy_files[index]
        oneLangData = np.load(filename, allow_pickle=True)
        langTrainData = oneLangData[0:4999, 4]
        totalLangData += [x.flatten().tolist() for x in langTrainData]
        label = oneLangData[0:4999, 1].tolist()
        totalLabelData += label

#convert both lists to numpy array to feed to the classifier
totalLangDataAsArray = np.asarray(totalLangData)
totalLabelDataAsArray = np.asarray(totalLabelData)

#data scaling
scaler = StandardScaler()
# keep our unscaled features just in case we need to process them alternatively
features_scaled = totalLangDataAsArray
features_scaled = scaler.fit_transform(features_scaled)

#dimensionality reduction
pca = PCA(n_components=190)
pca.fit(features_scaled)
components = pca.fit_transform(features_scaled)

total_var = pca.explained_variance_ratio_.sum() * 100
print(total_var)


#data splitting
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    components,
    totalLabelDataAsArray,
    test_size=0.3,
    random_state=69
)


# ### **Plotting Data**

# In[ ]:


df = pd.DataFrame(totalLangDataAsArray)


#dimensionality reduction by PCA
pca = PCA(n_components=2)
pca.fit(features_scaled)
components = pca.fit_transform(features_scaled)

#Explained Variance Ratio
total_var = pca.explained_variance_ratio_.sum() * 100

#plotting
fig = px.scatter(components, x=0, y=1,title=f'Total Explained Variance: {total_var:.2f}%', color=totalLabelDataAsArray, labels={'0': 'PCA Component 1', '1': 'PCA Component 2'})
fig.update_layout(legend_title_text='Language')
fig.show()


# In[ ]:


#3D plotting
df = pd.DataFrame(totalLangDataAsArray)
pca = PCA(n_components=3)
components = pca.fit_transform(df)

total_var = pca.explained_variance_ratio_.sum() * 100

fig = px.scatter_3d(
    components, x=0, y=1, z=2, color=totalLabelDataAsArray,
    title=f'Total Explained Variance: {total_var:.2f}%',
    labels={'0': 'PCA Component 1', '1': 'PCA Component 2', '2': 'PCA Component 3'}
)
fig.update_layout(legend_title_text='Language')
fig.show()




