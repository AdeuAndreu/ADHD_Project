# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 22:24:08 2016

@author: and_ma
"""


import pandas as pd
import numpy as np
#from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn import svm,tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import supervisedLearning_commons


#1.Read DAta
path_data = '/Users/and_ma/Documents/DataScience/UB_DataScience/DataScience_Project/gitHub/ADHD_Project/'

train = pd.read_csv(path_data+'supervisedLearningDataSet.csv')


label_cluster = lambda x: 0 if x == 'k1'  else 1
train['Best_Cluster'] = train['Best_Cluster'].apply(label_cluster)
y_train = train['Best_Cluster'].values
#y_test = test['Best_Cluster'].values

###################################################drop Cols



numeric_cols=['Fp1_(Theta2+Alpha1)', 'Fp1_(Theta)', 'Fp1_(Alpha)',
       'Fp1_(Beta_Global)', 'Fp1_(Beta_Alta)', 'Fp1_(Beta_Baja)',
       'Fp1_(Gamma)', 'F3_(Theta2+Alpha1)', 'F3_(Theta)', 'F3_(Alpha)',
       'F3_(Beta_Global)', 'F3_(Beta_Alta)', 'F3_(Beta_Baja)', 'F3_(Gamma)',
       'C3_(Theta2+Alpha1)', 'C3_(Theta)', 'C3_(Alpha)', 'C3_(Beta_Global)',
       'C3_(Beta_Alta)', 'C3_(Beta_Baja)', 'C3_(Gamma)', 'Fz_(Theta2+Alpha1)',
       'Fz_(Theta)', 'Fz_(Alpha)', 'Fz_(Beta_Global)', 'Fz_(Beta_Alta)',
       'Fz_(Beta_Baja)', 'Fz_(Gamma)', 'Cz_(Theta2+Alpha1)', 'Cz_(Theta)',
       'Cz_(Alpha)', 'Cz_(Beta_Global)', 'Cz_(Beta_Alta)', 'Cz_(Beta_Baja)',
       'Cz_(Gamma)', 'Fp2_(Theta2+Alpha1)', 'Fp2_(Theta)', 'Fp2_(Alpha)',
       'Fp2_(Beta_Global)', 'Fp2_(Beta_Alta)', 'Fp2_(Beta_Baja)',
       'Fp2_(Gamma)', 'F4_(Theta2+Alpha1)', 'F4_(Theta)', 'F4_(Alpha)',
       'F4_(Beta_Global)', 'F4_(Beta_Alta)', 'F4_(Beta_Baja)', 'F4_(Gamma)',
       'C4_(Theta2+Alpha1)', 'C4_(Theta)', 'C4_(Alpha)', 'C4_(Beta_Global)',
       'C4_(Beta_Alta)', 'C4_(Beta_Baja)', 'C4_(Gamma)', 'BPR_Fp1', 'BPR_F3',
       'BPR_C3', 'BPR_Fz', 'BPR_Cz', 'BPR_Fp2', 'BPR_F4', 'BPR_C4','PCA_x', 'PCA_y'
       ]


#2.2 Categorical features
categorical_cols=['experiment']

#3. Discarded features
drop_cols=['Unnamed: 0','patientName','experiment','Best_Cluster']

#Extrating numercial values
x_num_train = train[numeric_cols].as_matrix()


#4.Droping and processing columns
train = train.drop(drop_cols,axis=1)
#test = test.drop(drop_cols,axis=1)


# 4.2 Categorical
x_train = oneHotEncoding(train, numeric_cols) 
#x_test = oneHotEncoding(test, numeric_cols) 

# Without oneHotEncoding
#x_train = train.as_matrix()
#x_test = test.as_matrix()

#4.3 Separate train and test
PRC=0.7
x_train, y_train, x_test, y_test = randomization_train_2_twoSet(x_train,y_train,PRC=PRC)

#Achtung: this do the same as randomization_train_2_twoSet but it fails!
#x_train, y_train, x_test, y_test = train_test_split(x_train,y_train,test_size=PRC)

#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PRC)

# 5.1 RandomForest
print ('Training Random Forest ..')
n_estimators = 51
max_depth = 12
min_samples_split = 2
random_state = 1
max_features = 'auto'
verbose = 1
n_jobs = 1

w_0 = y_train[np.where(y_train==0)].shape[0] / y_train.shape[0]
w_1 = y_train[np.where(y_train==1)].shape[0] / y_train.shape[0]

#class_weight = {0:w_1,1:w_0}
class_weight = 'balanced'

##normalization
#scaler = StandardScaler()
#x_train = scaler.fit_transform(x_train)
#forest = RandomForestClassifier(max_depth = max_depth, min_samples_split=min_samples_split, n_estimators=n_estimators, random_state = random_state,class_weight=class_weight)
##forest = RandomForestClassifier(n_estimators=n_estimators,max_depth = max_depth,min_samples_split=min_samples_split,class_weight=class_weight)
#my_forest = forest.fit(x_train,y_train)
#print ('Random forest training score ',my_forest.score(x_train,y_train))
#
##6.2 Predicting
#x_test = scaler.transform(x_test)
#my_prediction = my_forest.predict(x_test)
#acc = metrics.accuracy_score(my_prediction, y_test)
#
#f1 = metrics.f1_score(my_prediction, y_test)
#precision, recall, thresholds = metrics.precision_recall_curve(my_prediction, y_test)
#
#print ('Random Forest mean accuracy: '+ str(acc))
#print ('Random Forest mean F1-Score: '+ str(f1))
#
##print ('Random Forest mean precision: '+ str(precision))
##print ('Random Forest mean recall: '+ str(recall))
##print ('Random Forest mean thresholds: '+ str(thresholds))
#
#print ('Performance Evaluation')
##precision_recall(my_prediction,y_test)
#print (metrics.classification_report(y_test,my_prediction))
########################################################  End Random-Forest


## Using k-fold cross validation to measure performance
n_folds = 10
kf=cross_validation.KFold(n=y_train.shape[0], n_folds=n_folds, shuffle=False, random_state=0)

acc = np.zeros((n_folds,))
precision = np.zeros((n_folds,))
recall = np.zeros((n_folds,))
#thresholds = np.zeros((n_folds,))
f1 = np.zeros((n_folds,))
i = 0
X = x_train
y = y_train
yhat = y_train.copy()
for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    dt = RandomForestClassifier(max_depth = max_depth, min_samples_split=min_samples_split, n_estimators=n_estimators, random_state = random_state,class_weight=class_weight)
    
    dt.fit(X_train,y_train)
    X_test = scaler.transform(X_test)
    yhat[test_index] = dt.predict(X_test)
    acc[i] = metrics.accuracy_score(yhat[test_index], y_test)
    precision[i] = metrics.precision_score(yhat[test_index], y_test)
    recall[i] = metrics.recall_score(yhat[test_index], y_test)
    f1[i]  = metrics.f1_score(yhat[test_index], y_test)
    i=i+1

print ('Random Forest mean accuracy: '+ str(np.mean(acc)))
print ('Random Forest mean F1-Score: '+ str(np.mean(f1)))
print ('Random Forest mean precision: '+ str(np.mean(precision)))
print ('Random Forest mean recall: '+ str(np.mean(recall)))
