# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 18:13:24 2016

@author: and_ma
"""

import pandas as pd
import os
import re
import numpy as np

path_resultsFolder = "/Users/and_ma/Documents/DataScience/UB_DataScience/DataScience_Project/gitHub/ADHD_Project/resultsProject/"
path_gmm = path_resultsFolder+'gmmClustering/'
path_hierarchichal = path_resultsFolder+'hierarchicalClustering/'
path_kmeans = path_resultsFolder + 'kmeansClustering/'


patternFile = r'[\w]+\.csv'


## 1. Loading gmm clustering results dataframe
df_gmm = []
filenames =  filenames = os.listdir(path_gmm)  
for experiment in filenames:
    if re.match(patternFile,experiment) != None:
        df_temp = pd.read_csv(path_gmm+experiment)
        df_temp = df_temp[['patientName','experiment','cluster']]
        df_gmm.append(df_temp)        
df_gmm = pd.concat(df_gmm)
df_gmm.sort_values(by='patientName', ascending= True, inplace=True) # Dataframe is sorted by cluster type: 0 or 1
df_gmm = df_gmm.reset_index(drop = True)

## 2. Loading hierchical clustering results dataframe
df_hier = []
filenames =  filenames = os.listdir(path_hierarchichal)  
for experiment in filenames:
    if re.match(patternFile,experiment) != None:
        df_temp = pd.read_csv(path_hierarchichal+experiment)
        df_temp = df_temp[['patientName','experiment','cluster']]
        df_hier.append(df_temp)        
df_hier = pd.concat(df_hier)
df_hier.sort_values(by='patientName', ascending= True, inplace=True) # Dataframe is sorted by cluster type: 0 or 1
df_hier = df_hier.reset_index(drop = True)

## 3. Loading kmeans clustering results dataframe
df_kmeans = []
filenames =  filenames = os.listdir(path_kmeans)  
for experiment in filenames:
    if re.match(patternFile,experiment) != None:
        df_temp = pd.read_csv(path_kmeans+experiment)
        df_temp = df_temp[['patientName','experiment','cluster']]
        df_kmeans.append(df_temp)        
df_kmeans = pd.concat(df_kmeans)
df_kmeans.sort_values(by='patientName', ascending= True, inplace=True) # Dataframe is sorted by cluster type: 0 or 1
df_kmeans = df_kmeans.reset_index(drop = True)

# Testing is always the same patient
#df_results = pd.DataFrame({})
#df_results['patientName_kmeans'] = df_kmeans['patientName']
#df_results['patientName_gmm'] = df_gmm['patientName']
#df_results['patientName_hier'] = df_hier['patientName']
#df_results['experiment'] = df_kmeans['experiment']

df_results = df_kmeans[['patientName','experiment']]
df_results['cluster_GMM'] = df_gmm['cluster']
df_results['cluster_Hierarchichal'] = df_hier['cluster']
df_results['cluster_Kmeans'] = df_kmeans['cluster']

#df_results = df_results.sort_values(by='experiment')
#df_results = df_results.reset_index(drop = True)

gmm_cluster = lambda x: 'k1' if x == 0  else 'k2'
hier_cluster = lambda x: 'k1' if x == 1 else 'k2'
kmeans_cluster = lambda x: 'k1' if x == 1 else 'k2'
df_results['cluster_GMM']=df_results['cluster_GMM'].apply(gmm_cluster)
df_results['cluster_Hierarchichal']=df_results['cluster_Hierarchichal'].apply(hier_cluster)
df_results['cluster_Kmeans']=df_results['cluster_Kmeans'].apply(kmeans_cluster)

#df = pd.read_csv("resultsClustering.csv")
list_clustering = ['cluster_GMM','cluster_Hierarchichal','cluster_Kmeans']
def mostCommon(df,list_clustering):
    best = []    
    for i in range(df.shape[0]):
        dicc = {'k1':0,'k2':0}
        for j in list_clustering:
            dicc[df.loc[i][j]]+=1
        if dicc['k1']>dicc['k2']:
            best.append('k1')
        else:
            best.append('k2')
    
    return best    
            
col_best = mostCommon(df_results,list_clustering)    
df_results['Best_Cluster']=np.array(col_best)


## Saving dataframe
filename = 'resultsClustering.csv'
if not os.path.exists(filename):
    df_results.to_csv(filename, sep=',', encoding='utf-8')



## Creating File for Supervised Learning
## 1. Loading gmm clustering results dataframe
supervisedLearning = []
filename_slearning = 'supervisedLearningDataSet.csv'
filenames =  filenames = os.listdir(path_gmm)  
for experiment in filenames:
    if re.match(patternFile,experiment) != None:
        df_temp = pd.read_csv(path_gmm+experiment)
        supervisedLearning.append(df_temp)        
        
supervisedLearning = pd.concat(supervisedLearning)
supervisedLearning.sort_values(by='patientName', ascending= True, inplace=True) # Dataframe is sorted by cluster type: 0 or 1
supervisedLearning = supervisedLearning.reset_index(drop = True)
supervisedLearning = supervisedLearning.drop(['cluster','Unnamed: 0'],axis=1)
supervisedLearning['Best_Cluster'] = df_results['Best_Cluster']

if not os.path.exists(filename_slearning):
    supervisedLearning.to_csv(filename_slearning, sep=',', encoding='utf-8')


### Visually checked
#gmm_cluster 0 ->'k1'
#gmm_cluster 1 ->'k2'
#hier_cluster 1 ->'k1'
#hier_cluster 2 ->'k2'
#kmeans_cluster 0 ->'k2'
#kmenas_cluster 1 ->'k1'
 

