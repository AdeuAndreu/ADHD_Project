# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 21:33:41 2016

@author: and_ma
"""

import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def plotCluster_matPlot(df,name_x='PCA_x',name_y='PCA_y',name_cluster='cluster'):
    
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
    plt.xlabel('PCA Componente 1')
    plt.ylabel('PCA Componente 2')
    plt.title('Samples for %s'%name_cluster)
    plt.legend(loc='upper right')
    plt.show()    
   
    return None      

import matplotlib.pyplot as plt
import numpy as np

# Simple data to display in various forms
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

plt.close('all')

# Just a figure and one subplot
f, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')

# Two subplots, the axes array is 1-d
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(x, y)
axarr[0].set_title('Sharing X axis')
axarr[1].scatter(x, y)

# Two subplots, unpack the axes array immediately
f, ax1= plt.subplots(1, 2, sharey=True)
ax1[0].plot(x, y)
ax1[0].set_title('Sharing Y axis')
ax1[1].scatter(x, y)

# Three subplots sharing both x/y axes
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing both axes')
ax2.scatter(x, y)
ax3.scatter(x, 2 * y ** 2 - 1, color='r')
# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
