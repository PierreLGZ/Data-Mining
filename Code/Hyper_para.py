#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 23:16:20 2022

@author: pierre
"""

import pandas as pd
import os
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

#Train / Test
os.chdir('/Users/pierre/Documents/M2/Cours_M2/Fouille donne/Projet_fouille/données')
X_train = pd.read_pickle("X_train.pkl")
y_train = pd.read_pickle("y_train.pkl")
X_test = pd.read_pickle("X_test.pkl")
y_test = pd.read_pickle("y_test.pkl")


from sklearn.metrics import make_scorer
f1_scorer = make_scorer(f1_score, average='binary', pos_label='1') #gridsearch

#loaded_model = pickle.load(open(filename, 'rb'))
os.chdir('/Users/pierre/Documents/M2/Cours_M2/Fouille donne/Projet_fouille/Hyper_para')
#############################################################################

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0, n_jobs=4)

param_search = {'class_weight':[{'0':1,'1':1}, {'0':1,'1':2}, {'0':1,'1':3}]}

tscv = TimeSeriesSplit(n_splits=5)
gsearch_rfc = GridSearchCV(estimator=rfc, cv=tscv, 
                        param_grid=param_search, n_jobs = 4, scoring = f1_scorer)
gsearch_rfc.fit(X_train, y_train)

print(gsearch_rfc.best_params_)
print(gsearch_rfc.best_score_)

gsearch_rfc.best_estimator_.fit(X_train,y_train)

y_pred = gsearch_rfc.best_estimator_.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

filename = 'gsearch_rfc.sav'
pickle.dump(gsearch_rfc, open(filename, 'wb'))

# {'class_weight': {'0': 1, '1': 1}}
# 0.770805089564871
# [[740803     34]
#  [  2219   4354]]
# 0.794453060852112


#############################################################################
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0, max_iter = 500, n_jobs = 1)

param_search = {'class_weight':[{'0':1,'1':1}, {'0':1,'1':2}, {'0':1,'1':3}, {'0':1,'1':4}, {'0':1,'1':5}]}

tscv = TimeSeriesSplit(n_splits=5)
gsearch_lr = GridSearchCV(estimator=lr, cv=tscv, 
                        param_grid=param_search, n_jobs = 1, scoring = f1_scorer)
gsearch_lr.fit(X_train, y_train)

print(gsearch_lr.best_params_)
print(gsearch_lr.best_score_)

gsearch_lr.best_estimator_.fit(X_train,y_train)

y_pred = gsearch_lr.best_estimator_.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

filename = 'gsearch_lr.sav'
pickle.dump(gsearch_lr, open(filename, 'wb'))

# {'class_weight': {'0': 1, '1': 2}}
# 0.7549957924557548
# [[740586    251]
#  [  2228   4345]]
# 0.778046378368699

##############################################################################
import xgboost as xgb
import numpy as np
xgboost = xgb.XGBClassifier(eval_metric='mlogloss',n_jobs = 16) 

scale_pos_weight = round(np.sqrt((2961189) /(16115)))
#sum(negative instances) / sum(positive instances)
param_search = {'scale_pos_weight':[1, 5, 10, scale_pos_weight, 20]}

tscv = TimeSeriesSplit(n_splits=3)
gsearch_xgboost = GridSearchCV(estimator=xgboost, cv=tscv, 
                        param_grid=param_search, n_jobs = 16, scoring = f1_scorer)
gsearch_xgboost.fit(X_train, y_train)

print(gsearch_xgboost.best_params_)
print(gsearch_xgboost.best_score_)

gsearch_xgboost.best_estimator_.fit(X_train,y_train)

y_pred = gsearch_xgboost.best_estimator_.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

# {'scale_pos_weight': 1}
# 0.7746593461664486

filename = 'gsearch_xgboost.sav'
pickle.dump(gsearch_xgboost, open(filename, 'wb'))
