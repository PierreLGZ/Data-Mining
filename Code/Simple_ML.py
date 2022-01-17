#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 19:27:07 2022

@author: pierre
"""
import os
os.chdir('/Users/pierre/Documents/M2/Cours_M2/Fouille donne/Projet_fouille/données/')
import pandas as pd
from collections import Counter
import pickle
import numpy as np 
import matplotlib.pyplot as plt

# load
# D = pd.read_pickle("Data_clean.pkl")
X_train = pd.read_pickle("X_train.pkl")
y_train = pd.read_pickle("y_train.pkl")
X_test = pd.read_pickle("X_test.pkl")
y_test = pd.read_pickle("y_test.pkl")

X_train.dtypes

###################### Test des modèles sur les données ################################
#Si veut grid serach, juste a dé commenter

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
#from sklearn.metrics import make_scorer
#f1_scorer = make_scorer(f1_score, average='binary', pos_label='1') #gridsearch

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

# param_search = {'C' : [1,10,100],
#                 'penalty': ['l2']}

# tscv = TimeSeriesSplit(n_splits=5)
# gsearch_lr = GridSearchCV(estimator=lr, cv=tscv, 
#                         param_grid=param_search, n_jobs = 1, scoring = f1_scorer)
# gsearch_lr.fit(X_train, y_train)
# print(gsearch_lr.best_params_)
# print(gsearch_lr.best_score_)

# gsearch_lr.best_estimator_.fit(X_train,y_train)

#y_pred = gsearch_lr.best_estimator_.predict(X_test)
y_pred = lr.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

# [[740638    199]
#  [  2259   4314]]
# 0.7782789103373624

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)
# param_search = {'n_estimators' : [50, 100, 500],
#                 'learning_rate': [0.01, 0.1, 1.0]}
# tscv = TimeSeriesSplit(n_splits=5)
# gsearch_gbc = GridSearchCV(estimator=gbc, cv=tscv, 
#                         param_grid=param_search, n_jobs = 1, scoring = f1_scorer)
# gsearch_gbc.fit(X_train, y_train)
# print(gsearch_gbc.best_params_)
# print(gsearch_gbc.best_score_)
# gsearch_gbc.best_estimator_.fit(X_train,y_train)

# y_pred = gsearch_gbc.best_estimator_.predict(X_test)
y_pred = rfc.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

# [[740803     34]
#  [  2219   4354]]
# 0.794453060852112

from sklearn.ensemble import AdaBoostClassifier
adab = AdaBoostClassifier(random_state=0)
adab.fit(X_train, y_train)
# param_search = {'n_estimators' : [50, 100, 500],
#                 'learning_rate': [0.01, 0.1, 1.0]}
# tscv = TimeSeriesSplit(n_splits=5)
# gsearch_adab = GridSearchCV(estimator=adab, cv=tscv, 
#                         param_grid=param_search, n_jobs = 1,scoring=f1_scorer)
# gsearch_adab.fit(X_train, y_train)
# print(gsearch_adab.best_params_)

y_pred = adab.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

# [[740792     45]
#  [  2228   4345]]
# 0.7926662409924292

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
# param_search = {'solver' : ['svd', 'lsqr', 'eigen'],
#                 'shrinkage': ['auto', 'float']}
# tscv = TimeSeriesSplit(n_splits=5)
# gsearch_lda = GridSearchCV(estimator=lda, cv=tscv, 
#                         param_grid=param_search, n_jobs = 1, scoring = f1_scorer)
# gsearch_lda.fit(X_train, y_train)
# print(gsearch_lda.best_params_)

y_pred = lda.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

# [[740628    209]
#  [  2242   4331]]
# 0.7794474939260326

import xgboost as xgb
import numpy as np
xgboost = xgb.XGBClassifier(eval_metric='mlogloss',n_jobs = 1) #, scale_pos_weight = np.sqrt((740795+2204) /(42+4369))
#scale_pos_weight [default=1]
#https://stats.stackexchange.com/questions/243207/what-is-the-proper-usage-of-scale-pos-weight-in-xgboost-for-imbalanced-datasets
#import numpy as np
#np.sqrt((740795+2204) /(42+4369))
#[[739421   1416]
# [  1936   4637]]
#0.7345160779344212
X_train.head()
xgboost.fit(X_train, y_train)
# param_search = {'max_depth' : [1,2,3],
#                 'learning_rate' : [0.1,0.5,1],
#                 'n_estimators' : [50,100]}
# tscv = TimeSeriesSplit(n_splits=5)
# gsearch_xgb = GridSearchCV(estimator=xgboost, cv=tscv, 
#                         param_grid=param_search, n_jobs = 1, scoring=f1_scorer)
# gsearch_xgb.fit(X_train, y_train)
# print(gsearch_xgb.best_params_)

y_pred = xgboost.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

# [[740792     45]
#  [  2195   4378]]
# 0.7962895598399419

####################  ROC
import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = lda.predict_proba(X_test)
preds = probs[:,1]
fpr_lda, tpr_lda, threshold = metrics.roc_curve(y_test, preds, pos_label='1')
roc_auc_lda = metrics.auc(fpr_lda, tpr_lda)

probs = xgboost.predict_proba(X_test)
preds = probs[:,1]
fpr_xgboost, tpr_xgboost, threshold = metrics.roc_curve(y_test, preds, pos_label='1')
roc_auc_xgboost = metrics.auc(fpr_xgboost, tpr_xgboost)

probs = rfc.predict_proba(X_test)
preds = probs[:,1]
fpr_rfc, tpr_rfc, threshold = metrics.roc_curve(y_test, preds, pos_label='1')
roc_auc_rfc = metrics.auc(fpr_rfc, tpr_rfc)

probs = adab.predict_proba(X_test)
preds = probs[:,1]
fpr_adab, tpr_adab, threshold = metrics.roc_curve(y_test, preds, pos_label='1')
roc_auc_adab = metrics.auc(fpr_adab, tpr_adab)

probs = lr.predict_proba(X_test)
preds = probs[:,1]
fpr_lr, tpr_lr, threshold = metrics.roc_curve(y_test, preds, pos_label='1')
roc_auc_lr = metrics.auc(fpr_lr, tpr_lr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Courbes ROC des différentes modèles')
plt.plot(fpr_lda, tpr_lda, 'blue', label = 'AUC LDA = %0.2f' % roc_auc_lda)
plt.plot(fpr_xgboost, tpr_xgboost, 'green', label = 'AUC XGBoost = %0.2f' % roc_auc_xgboost)
plt.plot(fpr_rfc, tpr_rfc, 'pink', label = 'AUC RandomForest = %0.2f' % roc_auc_rfc)
plt.plot(fpr_adab, tpr_adab, 'purple', label = 'AUC AdaBoost = %0.2f' % roc_auc_adab)
plt.plot(fpr_lr, tpr_lr, 'yellow', label = 'AUC Logistic Regression = %0.2f' % roc_auc_lr)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'black')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

os.chdir('/Users/pierre/Documents/M2/Cours_M2/Fouille donne/Projet_fouille/Simple/')
########## Save model
# save the model to disk
filename = 'lda_simple.sav'
pickle.dump(lda, open(filename, 'wb'))

filename = 'xgboost_simple.sav'
pickle.dump(xgboost, open(filename, 'wb'))

filename = 'rfc_simple.sav'
pickle.dump(rfc, open(filename, 'wb'))

filename = 'adab_simple.sav'
pickle.dump(adab, open(filename, 'wb'))

filename = 'lr_simple.sav'
pickle.dump(lr, open(filename, 'wb'))

#loaded_model = pickle.load(open(filename, 'rb'))
