# -*- coding: utf-8 -*-
"""

Created on Tue Jun 28 20:44:46 2016

@author: and_ma

"""


import pandas as pd
import scipy.io as sio
import os
import re
import matplotlib.pyplot as plt

def load(path = "/Users/and_ma/Documents/DataScience/UB_DataScience/DataScience_Project/Features/"):
    #input: path to the data
    #output: dataframe with all the data.

    #set the path for the filenames
    filenames = os.listdir(path)
    
    #electrodes names
    electrodes = ['Fp1', 'F3', 'C3', 'Fz', 'Cz', 'Fp2', 'F4', 'C4']
    
    #bandPower names
    bandPower = ['Theta2+Alpha1', 'Theta', 'Alpha', 'Beta_Global', 'Beta_Alta', 'Beta_Baja', 'Gamma']
    
    #let us create the real dataframe
    #columns for the band powers
    columns_df=[]
    for i in electrodes:
        for j in bandPower:
            combination = i + '_(' + j + ')'
            columns_df.append(combination)
    
    #columns for the band power ratios
    columns_df2 = ['BPR_Fp1', 'BPR_F3', 'BPR_C3', 'BPR_Fz', 'BPR_Cz', 'BPR_Fp2', 'BPR_F4', 'BPR_C4']
    
    #patterns to determine experiments
    patternA = r'[\w]+A\.mat'
    patternB = r'[\w]+B\.mat'
    patternC = r'[\w]+C\.mat'
    
    #define the list of dataframes
    list_df=[]
    for i in filenames:
        patient = sio.loadmat(path + i)
        bandPowerDF = pd.DataFrame(patient['BandPower'].reshape(1,56), columns = columns_df)
        bandPowerRatioDF = pd.DataFrame(patient['BandPowerRatio'].reshape(1,8), columns = columns_df2)
        patientDF =  pd.concat([bandPowerDF, bandPowerRatioDF], axis=1)
        patientDF['patientName']=i.split('.mat')[0]
        if re.match(patternA,i) != None:
            patientDF['experiment'] = 'A'
        elif re.match(patternB,i) != None:
            patientDF['experiment'] = 'B'
        elif re.match(patternC,i) != None:
            patientDF['experiment'] = 'C'
        list_df.append(patientDF)
    
    #via the concat all the dataframes are concatenated
    patientsDF = pd.concat(list_df)
    
    #nans are dropped
    patientsDF = patientsDF.dropna(how='any')
    
    
    #dataframe is sorted by experiment type: order A, B and C
    patientsDF.sort_values(by='experiment', ascending= True, inplace=True)
    
    #indexs are reset
    patientsDF = patientsDF.reset_index(drop = True)
     
    #let us separate the data of the three experiments

    #Dropping the data from the patientsDF
    #dropPatientsDF = patientsDF.drop(['experiment','patientName'],1)

    return (patientsDF)
    
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