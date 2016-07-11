# -*- coding: utf-8 -*-
"""

Created on Sat Jul  9 01:14:40 2016
@author: and_ma

"""

import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
path = "/Users/and_ma/Documents/DataScience/UB_DataScience/DataScience_Project/gitHub/last_sabadoGMM/"


df_A = pd.read_csv(path+"Apatients.csv")
df_B = pd.read_csv(path+"Bpatients.csv")
df_C = pd.read_csv(path+"Cpatients.csv")

df_A['experiment']='A'
df_B['experiment']='B'
df_C['experiment']='C'

df_merge = pd.concat([df_A,df_B,df_C])
df_merge.sort_values(by='patientName', ascending= True, inplace=True) # Dataframe is sorted by cluster type: 0 or 1
df_merge = df_merge.reset_index(drop = True)

#path_data = '/Users/and_ma/Documents/DataScience/UB_DataScience/DataScience_Project/gitHub/'
#train = pd.read_csv(path_data+'supervisedLearningDataSet.csv')

path_gmm = "/Users/and_ma/Documents/DataScience/UB_DataScience/DataScience_Project/gitHub/last_sabadoGMM/gmm/"
## 1. Loading gmm clustering results dataframe
patternFile = r'[\w]+\.csv'

######  Loadnig  SOM y GMM 
filenames =  filenames = os.listdir(path_gmm)
df_gmm = []
for experiment in filenames:
    if re.match(patternFile,experiment) != None:
        df_temp = pd.read_csv(path_gmm+experiment)
        df_gmm.append(df_temp)        
df_gmm = pd.concat(df_gmm)
df_gmm.sort_values(by='patientName', ascending= True, inplace=True) # Dataframe is sorted by cluster type: 0 or 1
df_gmm = df_gmm.reset_index(drop = True)


path_som = '/Users/and_ma/Documents/DataScience/UB_DataScience/DataScience_Project/gitHub/ADHD_Project/UnsupervisedLearning/'
filenames = os.listdir(path_som)
df_som = []
for experiment in filenames:
    if re.match(patternFile,experiment) != None:
        df_temp = pd.read_csv(path_som+experiment)
        df_som.append(df_temp)        
df_som = pd.concat(df_som)
df_som.sort_values(by='patientName', ascending= True, inplace=True) # Dataframe is sorted by cluster type: 0 or 1
df_som = df_som.reset_index(drop = True)
df_som_invert = lambda x: 1 if x==0 else 0
df_som['cluster'] = df_som['cluster'].apply(df_som_invert)
########

df_merge['GMM'] = df_gmm['cluster']
df_merge['patientsGMM'] = df_gmm['patientName']
df_merge['SOM'] = df_som['cluster']
df_merge['patientsSOM'] = df_som['patientName']




df_results =pd.DataFrame({})

df_results = df_merge[['patientName','experiment']]
df_results['PCA_X_gmm'] = df_gmm['PCA_x']  
df_results['PCA_Y_gmm'] = df_gmm['PCA_y']
df_results['PCA_X_som'] = df_som['PCA_x']  
df_results['PCA_Y_som'] = df_som['PCA_y']

df_results['PCA_X'] = df_merge['norm64comp_PCA_x']  
df_results['PCA_Y'] = df_merge['norm64comp_PCA_y']
df_results['PCA_Z'] = df_merge['norm56comp_PCA_z']

df_results['cluster_SOM'] = df_som['cluster']
df_results['cluster_Hierarchichal'] = df_merge['norm64hierarchical']
df_results['cluster_GMM'] = df_gmm['cluster']
df_results['cluster_Kmeans'] = df_merge['norm64kmeans++']
df_results['cluster_Spectral'] = df_merge['norm64spectral']
list_clustering = ['cluster_SOM','cluster_GMM','cluster_Kmeans','cluster_Hierarchichal','cluster_Spectral']
df_results[list_clustering].hist(sharex=True)


def mostCommon(df,list_clustering):
    best = []    
    for i in range(df.shape[0]):
        dicc = {'k1':0,'k0':0}
        for j in list_clustering:
            key = 'k'+str(df.loc[i][j])
            dicc[key]+=1
        if dicc['k1']>dicc['k0']:
            best.append(1)
        else:
            best.append(0)    
    return best 

def solapamiento(array1,array2):
    
    return np.count_nonzero([array1 == array2])
        

def plotCluster_tSNE(data,labels):
    
    
    return 0    

drop_columns1 = ['norm56comp_PCA_x', 'norm56comp_PCA_y',
       'norm56comp_PCA_z', 'norm56hierarchical', 'norm56kmeans++',
       'norm56spectral', 'norm64hierarchical', 'norm64kmeans++',
       'norm64spectral', 'GMM', 'patientsGMM', 'SOM',
       'patientsSOM']
drop_columns2=['56comp_PCA_x', '56comp_PCA_y', '56comp_PCA_z', '56hierarchical',
       '56kmeans++', '56spectral', '64_PCA_x', '64_PCA_y', '64_PCA_z',
       '64comp_PCA_x', '64comp_PCA_y', '64comp_PCA_z', '64hierarchical',
       '64kmeans++', '64spectral']

       
col_best = mostCommon(df_results,list_clustering)    
df_results['Best_Cluster'] = np.array(col_best)


df_merge['Best_Cluster'] = np.array(col_best)
df_merge = df_merge.drop(drop_columns1,axis=1)
df_merge = df_merge.drop(drop_columns2,axis=1)

print("The PCA  values are different Check! we save both PCA")
df_gmm[['PCA_x','PCA_y','patientName']].head()
df_results[['PCA_X','PCA_Y','patientName']].head()

# Saving dataframe
filename_unsupervised = 'resultsClustering_lunes11_PCA_gmm.csv'
filename_supervised = 'supervisedLearningDataSet_Lunes11.csv'

df_results.to_csv(filename_unsupervised, sep=',', encoding='utf-8')
df_merge.to_csv(filename_supervised, sep=',', encoding='utf-8')

#plotCluster_matPlot(df_results[df_results['experiment']=='B'],'PCA_X','PCA_Y','cluster_Spectral')
