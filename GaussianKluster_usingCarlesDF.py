# -*- coding: utf-8 -*-
"""

Created on Wed Jun 22 13:06:19 2016
@author: and_ma

"""

import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn import mixture
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import loadingADHD_Data
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

patientsDF = loadingADHD_Data.load()
ApatientsDF = patientsDF[patientsDF['experiment']=='A']
dropApatientsDF = ApatientsDF.drop(['experiment', 'patientName'],1)
BpatientsDF = patientsDF[patientsDF['experiment']=='B']
dropBpatientsDF = BpatientsDF.drop(['experiment', 'patientName'],1)
CpatientsDF = patientsDF[patientsDF['experiment']=='C']
dropCpatientsDF = CpatientsDF.drop(['experiment', 'patientName'],1)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

dropPatientsDF = patientsDF.drop(['experiment','patientName'],1)
numberA = len(patientsDF[patientsDF['experiment']=='A'])
numberB = len(patientsDF[patientsDF['experiment']=='B'])
numberC = len(patientsDF[patientsDF['experiment']=='C'])


### Do the PCA decomposition! for the sake of visualisation, only 3 components are considered
pca = PCA(n_components=3)
newdataPCA = pca.fit_transform(dropPatientsDF.values)  # concatenates vectors row by row
pca.explained_variance_ratio_

# the sum of the variance ratios for three components of the PCA is 0.928521380016
#plot of the data on the PCA space: 3D
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10
ax.plot(newdataPCA[0:numberA, 0], newdataPCA[0:numberA, 1],\
    newdataPCA[0:numberA, 2], 'o', markersize=8, color='blue', alpha=0.5, label='A')    
ax.plot(newdataPCA[numberA:(numberA+numberB), 0], newdataPCA[numberA:(numberA+numberB), 1],\
    newdataPCA[numberA:(numberA+numberB), 2], '^', markersize=8, color='red', alpha=0.5, label='B')    
ax.plot(newdataPCA[(numberA+numberB):, 0], newdataPCA[(numberA+numberB):, 1],\
    newdataPCA[(numberA+numberB):, 2], '+', markersize=8, color='green', alpha=0.5, label='C')


plt.title('Samples for experiment A, B and C in the PCA space')
ax.legend(loc='upper right')
ax.set_xlabel('First PCA base vector')
ax.set_ylabel('Second PCA base vector')
ax.set_zlabel('Third PCA base vector')
plt.show()
#no conlusion from that: only visualization


### Plot in two dimensions
fig = plt.figure(figsize=(8,8))
plt.rcParams['legend.fontsize'] = 10
plt.plot(newdataPCA[0:numberA, 0], newdataPCA[0:numberA, 1], 'o', markersize=8, color='blue', alpha=0.5, label='A')
plt.plot(newdataPCA[numberA:(numberA+numberB), 0], newdataPCA[numberA:(numberA+numberB), 1], '^', markersize=8, alpha=0.5, color='red', label='B')
plt.plot(newdataPCA[(numberA+numberB):, 0], newdataPCA[(numberA+numberB):, 1], '+', markersize=8, color='green', alpha=0.5, label='C')
plt.title('Samples for experiment A, B and C in the PCA space')
plt.legend(loc='upper right')
plt.show()


#### Gaussian Clustering : Gaussian mixture models
#GMM methods
#aic(X)	Akaike information criterion for the current model fit
#bic(X)	Bayesian information criterion for the current model fit
#fit(X[, y])	Estimate model parameters with the EM algorithm.
#fit_predict(X[, y])	Fit and then predict labels for data.
#get_params([deep])	Get parameters for this estimator.
#predict(X)	Predict label for data.
#predict_proba(X)	Predict posterior probability of data under each Gaussian in the model.
#sample([n_samples, random_state])	Generate random samples from the model.
#score(X[, y])	Compute the log probability under the model.
#score_samples(X)	Return the per-sample likelihood of the data under the model.
#set_params(**params)	Set the parameters of this estimator.

def getBestGMM(X,n_components,cv_types):
    
    '''
    Function that finds the best GMM cluster trying different gaussians and 
    different number of clusters    
    '''
    
    lowest_bic = np.infty
    bic = []
    silhouette = []
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            Y_predicted=gmm.predict(X)
            if cv_type =='tied':
                silhouette.append(metrics.silhouette_score(X, Y_predicted,  metric='euclidean'))
                #I only save the values for tied, because i know from the first run that its the best gaussian
            if n_components>=1:
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm
                    
    bic = np.array(bic)
    color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm','y'])
    return best_gmm,color_iter,bic,silhouette

def plotBIC_Scores(color_iter,n_components,cv_types,bic):
    spl = plt.subplot(1, 1, 1)
    bars = []
    
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
                            
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('Bayesian Information Criteria score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)   

### 1. Get best model for A patients
X_A = dropApatientsDF.values
n_components_range = range(2,15)
cv_types = ['spherical', 'tied', 'diag', 'full']
clf_A,color_iter,bic,silhouette_A = getBestGMM(X_A,n_components_range,cv_types)  

### 2. Save the Clf winner results
Y_A = clf_A.predict(X_A)
# Very interesting clf_A.predict_proba(X_A), it gives back the actual likelihood


# Value metrics silhouette score
best_A = metrics.silhouette_score(X_A, Y_A,  metric='euclidean')
print ("Best sillhouette A patients ",best_A)

### 3. Plotting sillhouette for cluster A
plt.figure() 
plt.plot(np.arange(2,15),np.array(silhouette_A),color='b')
plt.title("Group A Sillhouette Values for GMM Klustering, Best = %s"%(best_A))
plt.xlabel("K value: number of clusters")
plt.ylabel("Silhouette Score")


### 4. Ploting GMM clustering result using PCA Components
pca = PCA(n_components=3)
newdataPCA = pca.fit_transform(X_A)  # concatenates vectors row by row
pca.explained_variance_ratio_
plotApatientsGMM = ApatientsDF.copy() 
plotApatientsGMM['PCA_x'] = newdataPCA.T[0]
plotApatientsGMM['PCA_y'] = newdataPCA.T[1]
plotApatientsGMM['cluster'] = Y_A

plotApatientsGMM.sort_values(by='cluster', ascending= True, inplace=True) # Dataframe is sorted by cluster type: 0 or 1

### Indexes for ploting
number1 = len(plotApatientsGMM[plotApatientsGMM['cluster']==0])
number2 = len(plotApatientsGMM[plotApatientsGMM['cluster']==1])

### Plot result clustering in two dimensions
fig = plt.figure(figsize=(8,8))
plt.rcParams['legend.fontsize'] = 10
plt.plot(plotApatientsGMM['PCA_x'].values[0:number1], plotApatientsGMM['PCA_y'].values[0:number1], 'o', markersize=8, color='blue', alpha=0.5, label='0')
plt.plot(plotApatientsGMM['PCA_x'].values[number1:], plotApatientsGMM['PCA_y'].values[number1:], '^', markersize=8, alpha=0.5, color='red', label='1')
plt.xlabel('PCA eingenvector 1')
plt.ylabel('PCA eingenvector 2')
plt.title('Samples for experiment A, GMM Clustering')
plt.legend(loc='upper right')
plt.show()

plotApatientsGMM.to_csv('gmmApatients.csv', sep=',', encoding='utf-8')
###################################### END Patients A

#### Gaussian Clustering: Experiment B

### 1. Get best model for A patients
X_B = dropBpatientsDF.values
n_components_range = range(2,15)
cv_types = ['spherical', 'tied', 'diag', 'full']
clf_B,color_iter,bic,silhouette_B = getBestGMM(X_B,n_components_range,cv_types)  

### 2. Save the Clf winner results
Y_B= clf_B.predict(X_B)

# Value metrics silhouette score 
best_B = metrics.silhouette_score(X_B, Y_B,  metric='euclidean')
print ("Best sillhouette B patients ",best_B)

### 3. Plotting sillhouette for cluster B
plt.figure() 
plt.plot(np.arange(2,15),np.array(silhouette_B),color='b')
plt.title("Group B Sillhouette Values for GMM Klustering, Best = %s"%best_B)
plt.xlabel("K value: number of clusters")
plt.ylabel("Silhouette Score")


### 4. Ploting GMM clustering result using PCA Components
pca = PCA(n_components=3)
newdataPCA = pca.fit_transform(X_B)  # concatenates vectors row by row
pca.explained_variance_ratio_
plotBpatientsGMM = BpatientsDF.copy()
plotBpatientsGMM['PCA_x'] = newdataPCA.T[0]
plotBpatientsGMM['PCA_y'] = newdataPCA.T[1]
plotBpatientsGMM['cluster'] = Y_B

plotBpatientsGMM.sort_values(by='cluster', ascending= True, inplace=True) # Dataframe is sorted by cluster type: 0 or 1

### Indexes for ploting
number1 = len(plotBpatientsGMM[plotBpatientsGMM['cluster']==0])
number2 = len(plotBpatientsGMM[plotBpatientsGMM['cluster']==1])

### Plot result clustering in two dimensions
fig = plt.figure(figsize=(8,8))
plt.rcParams['legend.fontsize'] = 10
plt.plot(plotBpatientsGMM['PCA_x'].values[0:number1], plotBpatientsGMM['PCA_y'].values[0:number1], 'o', markersize=8, color='blue', alpha=0.5, label='0')
plt.plot(plotBpatientsGMM['PCA_x'].values[number1:], plotBpatientsGMM['PCA_y'].values[number1:], '^', markersize=8, alpha=0.5, color='red', label='1')
plt.xlabel('PCA eingenvector 1')
plt.ylabel('PCA eingenvector 2')
plt.title('Samples for experiment B, GMM Clustering')
plt.legend(loc='upper right')
plt.show()
plotBpatientsGMM.to_csv('gmmBpatients.csv', sep=',', encoding='utf-8')
###################################### END Patients B

#### Gaussian Clustering: Experiment C
### 1. Get best model for A patients
X_C = dropCpatientsDF.values
n_components_range = range(2,15)
cv_types = ['spherical', 'tied', 'diag', 'full']
clf_C,color_iter,bic,silhouette_C = getBestGMM(X_C,n_components_range,cv_types)  

### 2. Save the Clf winner results
Y_C = clf_C.predict(X_C)

# Value metrics silhouette score 
best_C = metrics.silhouette_score(X_C, Y_C,  metric='euclidean')
print ("Best sillhouette C patients ",best_C)

### 3. Plotting sillhouette for cluster A
plt.figure() 
plt.plot(np.arange(2,15),np.array(silhouette_C),color='b')
plt.title("Group C Sillhouette Values for GMM Klustering, Best = %s"%best_C)
plt.xlabel("K value: number of clusters")
plt.ylabel("Silhouette Score")


### 4. Ploting GMM clustering result using PCA Components
pca = PCA(n_components=3)
newdataPCA = pca.fit_transform(X_C)  # concatenates vectors row by row
pca.explained_variance_ratio_
plotCpatientsGMM = CpatientsDF.copy()
plotCpatientsGMM['PCA_x'] = newdataPCA.T[0]
plotCpatientsGMM['PCA_y'] = newdataPCA.T[1]
plotCpatientsGMM['cluster'] = Y_C

plotCpatientsGMM.sort_values(by='cluster', ascending= True, inplace=True) # Dataframe is sorted by cluster type: 0 or 1

### Indexes for ploting
number1 = len(plotCpatientsGMM[plotCpatientsGMM['cluster']==0])
number2 = len(plotCpatientsGMM[plotCpatientsGMM['cluster']==1])

### Plot result clustering in two dimensions
fig = plt.figure(figsize=(8,8))
plt.rcParams['legend.fontsize'] = 10
plt.plot(plotCpatientsGMM['PCA_x'].values[0:number1], plotCpatientsGMM['PCA_y'].values[0:number1], 'o', markersize=8, color='blue', alpha=0.5, label='0')
plt.plot(plotCpatientsGMM['PCA_x'].values[number1:], plotCpatientsGMM['PCA_y'].values[number1:], '^', markersize=8, alpha=0.5, color='red', label='1')
plt.xlabel('PCA eingenvector 1')
plt.ylabel('PCA eingenvector 2')
plt.title('Samples for experiment C, GMM Clustering')
plt.legend(loc='upper right')
plt.show()
plotCpatientsGMM.to_csv('gmmCpatients.csv', sep=',', encoding='utf-8')
#imagePath_C = '/Users/and_ma/Documents/DataScience/UB_DataScience/DataScience_Project/TeamWork/imagesGMM/C_GMM_Clustering.png'
#plt.savefig(imagePath_C, format='png', dpi=900)












