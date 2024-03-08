# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:17:51 2020

@author: AbdelAzeem
"""

import warnings
warnings.filterwarnings('ignore') 
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
#----------------------------------------------------
#reading data

data = pd.read_csv('heart.csv')
#data.describe()

#X Data
X = data.iloc[:,:-1]
#y Data
y = data.iloc[:,-1]
print('X Data is \n' , X.head())
print('X shape is ' , X.shape)

# -------------- MinMaxScaler for Data --------------
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)


#---------Feature Selection = Logistic Regression 13=>7 -------------------

from sklearn.linear_model import  LogisticRegression

thismodel = LogisticRegression()


FeatureSelection = SelectFromModel(estimator = thismodel, max_features = None) # make sure that thismodel is well-defined
X = FeatureSelection.fit_transform(X, y)

#showing X Dimension 
print('X Shape is ' , X.shape)
print('Selected Features are : ' , FeatureSelection.get_support())


#------------ Splitting data ---33% Test  67% Training -----------------------

#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

#Splitted Data
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)



#------- ensemble Bagging Classifier --- 87 % -- n_estimators=50 & 100 ---


from sklearn.ensemble import BaggingClassifier

'''
sklearn.ensemble.BaggingClassifier(base_estimator=None, n_estimators=10,
                                   max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False,
                                   oob_score=False, warm_start=False,
                                   n_jobs=None, random_state=None, verbose=0)
'''

model = BaggingClassifier(n_estimators=70)
import time
t0 = time.clock()
model.fit(X_train,y_train)
tr = (time.clock()-t0)

print('BaggingClassifier Train Score is : ' , model.score(X_train, y_train))
print('BaggingClassifier Test Score is : ' , model.score(X_test, y_test))
#print('BaggingClassifier features importances are : ' , model.feature_importances_)
print('----------------------------------------------------')
print('time in msec', tr*1000)
print('----------------------------------------------------')

#Calculating Prediction
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)

#-----------------------------------------------
# ------------------ Metrics ----------------------
# ---------- confusion_matrix ----------

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)



from sklearn.metrics import roc_auc_score
#5 roc_auc_score(y_true, y_score, average=’macro’, sample_weight=None,max_fpr=None)
TP = 47
TN = 38
FN = 7
FP = 8
accuracy_score = ((TP + TN) / float(TP + TN + FP + FN))*100
precision_score = (TP /float(TP + FP ))*100
recall_score = (TP / float(TP + FN))*100
f1_score = (2 * (precision_score * recall_score) / (precision_score + recall_score))
print('accuracy_score is :' , accuracy_score)
print('Precision Score is : ', precision_score)
print('recall_score is : ', recall_score)
print('f1_score is :' , f1_score)
ROCAUCScore = roc_auc_score(y_test,y_pred, average='micro') #it can be : macro,weighted,samples
print('ROCAUC Score : ', ROCAUCScore*100)

# --------------------------------------------------



##################### ((Grid Search)) ##############

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd


SelectedModel = BaggingClassifier()

'''
sklearn.ensemble.BaggingClassifier(base_estimator=None, n_estimators=10,
                                   max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False,
                                   oob_score=False, warm_start=False,
                                   n_jobs=None, random_state=None, verbose=0)
'''

SelectedParameters = {'n_estimators':[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,150,200,300]}

GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters, cv = 10,return_train_score=True)

GridSearchModel.fit(X_train, y_train)

sorted(GridSearchModel.cv_results_.keys())

GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]

# Showing Results
print('All Results are :\n', GridSearchResults )
print('Best Score is :', GridSearchModel.best_score_)
print('Best Parameters are :', GridSearchModel.best_params_)
print('Best Estimator is :', GridSearchModel.best_estimator_)


#     ---- try 2 -- Grid Search -- 86 % -- n_estimators=70


from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier(n_estimators=99)
import time
t0 = time.clock()
model.fit(X_train,y_train)
tr = (time.clock()-t0)
print('BaggingClassifier Train Score is : ' , model.score(X_train, y_train))
print('BaggingClassifier Test Score is : ' , model.score(X_test, y_test))
#print('BaggingClassifier features importances are : ' , model.feature_importances_)
print('----------------------------------------------------')
print('time in msec', tr*1000)
print('----------------------------------------------------')
# ---------- confusion_matrix ----------

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)



from sklearn.metrics import roc_auc_score
#5 roc_auc_score(y_true, y_score, average=’macro’, sample_weight=None,max_fpr=None)
TP = 50
TN = 39
FN = 4
FP = 7
accuracy_score = ((TP + TN) / float(TP + TN + FP + FN))*100
precision_score = (TP /float(TP + FP ))*100
recall_score = (TP / float(TP + FN))*100
f1_score = (2 * (precision_score * recall_score) / (precision_score + recall_score))
print('accuracy_score is :' , accuracy_score)
print('Precision Score is : ', precision_score)
print('recall_score is : ', recall_score)
print('f1_score is :' , f1_score)
ROCAUCScore = roc_auc_score(y_test,y_pred, average='micro') #it can be : macro,weighted,samples
print('ROCAUC Score : ', ROCAUCScore*100)

# --------------------------------------------------
