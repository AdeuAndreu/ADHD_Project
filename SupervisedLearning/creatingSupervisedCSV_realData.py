# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:51:52 2016

@author: and_ma
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:42:06 2016

@author: and_ma
"""

from sklearn.feature_extraction import DictVectorizer as DV
#from sklearn import svm,tree
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC
#from sklearn import metrics
#import matplotlib.pyplot as plt
#from sklearn import cross_validation
#from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import train_test_split
#import supervisedLearning_commons
import pandas  as pd
import numpy as np

path = "/Users/and_ma/Documents/DataScience/UB_DataScience/DataScience_Project/gitHub/last_sabadoGMM/"

df_unsupervised = pd.read_csv('/Users/and_ma/Documents/DataScience/UB_DataScience/DataScience_Project/gitHub/ADHD_Project/resultsClustering_lunes11_PCA_gmm.csv')
df_supervised = pd.read_csv('/Users/and_ma/Documents/DataScience/UB_DataScience/DataScience_Project/gitHub/ADHD_Project/supervisedLearningDataSet_Lunes11.csv')
df_true = pd.read_csv(path+'Subjects_Table_Sent_UB.csv')

   
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
    
def solapamiento(array1,array2):    
    return np.count_nonzero([array1 == array2])/array1.shape[0]


cols_to_keep = ['Renamed_Subject','Age', 'Discard']

cluster_columns=['cluster_SOM',
       'cluster_Hierarchichal', 'cluster_GMM', 'cluster_Kmeans',
       'cluster_Spectral', 'Best_Cluster']



df_supervised['Best_Cluster'] = df_unsupervised['Best_Cluster'].apply(lambda x: 1 if x==0 else 0)

df_true['Renamed_Subject'] = df_true['Renamed_Subject'].apply(lambda x: x.strip())
df_true['Diagnose'] = df_true['Diagnose'].apply(lambda x: x.strip())
df_true['Discard'] = df_true['Discard'].apply(lambda x: x.strip())

rename_list =[]
ADHD = []
Discard = []
Age = []
Sex = []
ADHD_TAG = []
for unsup_patient in df_supervised['patientName']:
    real_name = unsup_patient.split("_")[0]
    
    #found_row = []
    found_row = df_true[ df_true['Renamed_Subject']== real_name]  
  
    if len(found_row)>0:
        rename_list.append(found_row.iloc[0]['Renamed_Subject'])
        
        if found_row.iloc[0]['Diagnose'] == 'CTRL':
            ADHD.append(0)
        else:
            ADHD.append(1)
        Discard.append(found_row.iloc[0]['Discard'])        
        Age.append(found_row.iloc[0]['Age'])
        Sex.append(found_row.iloc[0]['Sex'])
        ADHD_TAG.append(found_row.iloc[0]['Diagnose'])
        
df_supervised['Rename'] = rename_list         
df_supervised['Discard'] = Discard        
df_supervised['ADHD'] = ADHD         
df_supervised['ADHD_TAG'] = ADHD_TAG

print ("Checking right balance")
df_supervised[['ADHD','Best_Cluster']].hist()
  
  
filename_unsupervised = 'resultsClustering_reall.csv'

filename_supervised = 'supervisedLearningDataSet_Martes12_reales.csv'

#df_results.to_csv(filename_unsupervised, sep=',', encoding='utf-8')
df_supervised.to_csv(filename_supervised, sep=',', encoding='utf-8')
  