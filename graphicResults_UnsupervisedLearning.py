# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 01:31:17 2016

@author: and_ma
"""
import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/and_ma/Documents/DataScience/UB_DataScience/DataScience_Project/gitHub/ADHD_Project/'
df_results = pd.read_csv(path+"resultsClustering_lunes11_PCA_gmm.csv")

list_clustering = ['cluster_SOM','cluster_GMM','cluster_Kmeans','cluster_Hierarchichal','cluster_Spectral']

fig = plt.figure()

def plotCluster_matPlot(df,name_x,name_y,name_cluster, title):
    
    df_plot = pd.DataFrame({})
    df_plot['PCA_x'] = df[name_x]
    df_plot['PCA_y'] = df[name_y]
    df_plot['cluster'] = df[name_cluster]
    
    number1 = len(df_plot[df_plot['cluster']==0])
    #number2 = len(df_plot[df_plot['cluster']==1])
    
    df_plot.sort_values(by='cluster', ascending= True, inplace=True) # Dataframe is sorted by cluster type: 0 or 1
    ### Plot result clustering in two dimensions
    fig = plt.figure(figsize=(8,8))
    plt.rcParams['legend.fontsize'] = 10
    plt.plot(df_plot['PCA_x'].values[0:number1], df_plot['PCA_y'].values[0:number1], 'o', markersize=8, color='green', alpha=0.5, label='0')
    plt.plot(df_plot['PCA_x'].values[number1:], df_plot['PCA_y'].values[number1:], '^', markersize=8, alpha=0.5, color='red', label='1')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()    
   
    return 
# 1
N_plots = len(list_clustering)
f,ax = plt.subplots(1, N_plots , sharey=True,figsize=(40,8))
for i in np.arange(N_plots):
    df_plot = df_results[['PCA_X', 'PCA_Y',list_clustering[i]]]
    number1 = len(df_plot[df_plot[list_clustering[i]]==0])
    df_plot.sort_values(by=list_clustering[i], ascending= True, inplace=True)
    ax[i].plot(df_plot['PCA_X'].values[0:number1], df_plot['PCA_Y'].values[0:number1], 'o', markersize=8, color='green', alpha=0.5, label='0')
    ax[i].plot(df_plot['PCA_X'].values[number1:], df_plot['PCA_Y'].values[number1:], '^', markersize=8, alpha=0.5, color='red', label='1')
    ax[i].set_title(list_clustering[i])  
plt.show() 




N_plots = len(list_clustering)
f,ax = plt.subplots(1, N_plots , sharey=False,figsize=(40,8))
for i in np.arange(N_plots):
    df_plot = df_results[['PCA_X', 'PCA_Y','PCA_X_gmm', 'PCA_Y_gmm',
       'PCA_X_som', 'PCA_Y_som',list_clustering[i]]]
    number1 = len(df_plot[df_plot[list_clustering[i]]==0])
    df_plot.sort_values(by=list_clustering[i], ascending= True, inplace=True)
    
#    if list_clustering[i] == 'cluster_SOM':
#        ax[i].plot(df_plot['PCA_X_som'].values[0:number1], df_plot['PCA_Y_som'].values[0:number1], 'o', markersize=8, color='green', alpha=0.5, label='0')
#        ax[i].plot(df_plot['PCA_X_som'].values[number1:], df_plot['PCA_Y_som'].values[number1:], '^', markersize=8, alpha=0.5, color='red', label='1')
#        ax[i].set_title(list_clustering[i])        
#        
#    elif list_clustering[i] =='cluster_GMM':
#        ax[i].plot(df_plot['PCA_X_gmm'].values[0:number1], df_plot['PCA_Y_gmm'].values[0:number1], 'o', markersize=8, color='green', alpha=0.5, label='0')
#        ax[i].plot(df_plot['PCA_X_gmm'].values[number1:], df_plot['PCA_Y_gmm'].values[number1:], '^', markersize=8, alpha=0.5, color='red', label='1')
#        ax[i].set_title(list_clustering[i]) 
#        
#    else:
    ax[i].plot(df_plot['PCA_X'].values[0:number1], df_plot['PCA_Y'].values[0:number1], 'o', markersize=8, color='green', alpha=0.5, label='0')
    ax[i].plot(df_plot['PCA_X'].values[number1:], df_plot['PCA_Y'].values[number1:], '^', markersize=8, alpha=0.5, color='red', label='1')
    ax[i].set_title(list_clustering[i])  

plt.show() 




# 2
f,ax = plt.subplots(1, N_plots , sharey=False,figsize=(20,5))
for i in np.arange(N_plots):
    df_plot = df_results[list_clustering[i]]
    df_plot.plot.hist(ax=ax[i],title=list_clustering[i])
#    ax[i].xlabel('PCA Component 1')
#    ax[i].ylabel('PCA Component 2')

plt.show()


# 3
N_plots = len(list_clustering)
f,ax = plt.subplots(2, N_plots , sharey=True,figsize=(40,8))
for i in np.arange(N_plots):
    df_plot = df_results[['PCA_X', 'PCA_Y',list_clustering[i]]]
    number1 = len(df_plot[df_plot[list_clustering[i]]==0])
    df_plot.sort_values(by=list_clustering[i], ascending= True, inplace=True)
    ax[0,i].plot(df_plot['PCA_X'].values[0:number1], df_plot['PCA_Y'].values[0:number1], 'o', markersize=8, color='green', alpha=0.5, label='0')
    ax[0,i].plot(df_plot['PCA_X'].values[number1:], df_plot['PCA_Y'].values[number1:], '^', markersize=8, alpha=0.5, color='red', label='1')
    df_plot[list_clustering[i]].hist(ax=ax[1,i])    
#    ax[i].xlabel('PCA Component 1')
#    ax[i].ylabel('PCA Component 2')
#    ax[i].set_title(list_clustering[i])
plt.show() 

def solapamiento(array1,array2):
    
    return np.count_nonzero([array1 == array2])
    
tam = len(list_clustering)
matrix_solapamiento = np.zeros((tam,tam))    
for fila in range(tam):
    for colum in range(tam):
        matrix_solapamiento[fila,colum] = solapamiento(df_results[list_clustering[fila]].values,df_results[list_clustering[colum]].values)