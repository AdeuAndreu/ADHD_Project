# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 00:18:38 2016

@author: and_ma
"""

import numpy as np
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn import metrics
import matplotlib.pyplot as plt

def oneHotEncoding(train, numeric_cols):
    # receives the clean tain and test data
    # in: train and test numpy matrix
    x_num_train = train[numeric_cols].as_matrix()
    #x_num_test = test[numeric_cols].as_matrix()
    cat_train = train.drop(numeric_cols, axis=1)
    #cat_test = test.drop(numeric_cols, axis=1)
    x_cat_train = cat_train.T.to_dict().values()
    #x_cat_test = cat_test.T.to_dict().values()
    # 5.1 vectorize
    vectorizer = DV(sparse=False)
    vec_x_cat_train = vectorizer.fit_transform(x_cat_train)
    #vec_x_cat_test = vectorizer.transform(x_cat_test)
    # complete x
    x_train = np.hstack((x_num_train, vec_x_cat_train))
    #x_test = np.hstack((x_num_test, vec_x_cat_test))
    return x_train

def randomization_train_2_twoSet(x_train,y_train,PRC=0.7):
    #Alternative:
    #from sklearn.cross_validation import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PRC)
    perm = np.random.permutation(x_train.shape[0])
    split_point = int(np.ceil(y_train.shape[0]*PRC))

    X_train = x_train[perm[:split_point].ravel(),:]
    Y_train = y_train[perm[:split_point].ravel()]

    X_test = x_train[perm[split_point:].ravel(),:]
    Y_test = y_train[perm[split_point:].ravel()]

    return (X_train, Y_train, X_test, Y_test)

def precision_recall(y, yhat):
    #from sklearn import metrics
    #Alternative: (metrics.classification_report(y,y_pred))
    TP = np.sum(np.logical_and(yhat==1,y==1))
    TN = np.sum(np.logical_and(yhat==0,y==0))
    FP = np.sum(np.logical_and(yhat==1,y==0))
    FN = np.sum(np.logical_and(yhat==0,y==1))
    
    accuracy = (TP+TN) / y.shape[0]
    precision = TP/(TP+FN)
    recall = TP/(TP+FP)
    F1 = 2*precision*recall/(precision+recall)    
    
    w0 = y[np.where(y==0)].shape[0] / y.shape[0]
    w1 = y[np.where(y==1)].shape[0] / y.shape[0]
    MacroF1 = w0*F1+w1*F1    
    
    print ('TP: ' + str(TP))
    print ('TN: ' + str(TN))
    print ('FP: ' + str(FP))
    print ('FN: ' + str(FN))
    print ('Accuracy: ' + str(accuracy))
    print ('sensitivity/recall: '+ str(precision))
    print ('precision: '+ str(recall))
    print ('F1: '+ str(F1))
    print ('Macro F1 '+str(MacroF1))

def draw_confusion(y,yhat,labels):
    cm = metrics.confusion_matrix(y, yhat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(cm)
    plt.title('Confusion matrix',size=20)
    ax.set_xticklabels([''] + labels, size=20)
    ax.set_yticklabels([''] + labels, size=20)
    plt.ylabel('Predicted',size=20)
    plt.xlabel('True',size=20)
    for i in range(2):
        for j in range(2):
            ax.text(i, j, cm[i,j], va='center', ha='center',color='white',size=20)
    fig.set_size_inches(7,7)
    plt.show()
