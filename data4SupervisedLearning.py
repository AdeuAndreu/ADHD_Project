# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 20:39:16 2016

@author: and_ma
"""

import pandas as pd
import numpy as np

path ="/Users/and_ma/Documents/DataScience/UB_DataScience/DataScience_Project/TeamWork/Data_Results_Unsupervised/"

df = pd.read_csv(path+"Apatients.csv")
df_GMMA = pd.read_csv(path+'gmmApatients.csv')
#list_clustering = ['cluster_GMM','cluster_Hierarchichal','cluster_Kmeans']
#def mostCommon(df,list_clustering):
#    best = []    
#    for i in range(df.shape[0]):
#        dicc = {'k1':0,'k2':0}
#        for j in list_clustering:
#            dicc[df.loc[i][j]]+=1
#        if dicc['k1']>dicc['k2']:
#            best.append('k1')
#        else:
#            best.append('k2')
#    
#    return best    
#            
#col_best = mostCommon(df,list_clustering)    
#df['best']=np.array(col_best)

