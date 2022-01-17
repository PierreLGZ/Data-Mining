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

#Crée dataframe spécialement pour recupérer scores modèles
D = pd.read_pickle("Data_clean.pkl")

df_score = D[['Montant', 'VerifianceCPT1', 'VerifianceCPT2',
        'VerifianceCPT3', 'D2CB', 'ScoringFP1', 'ScoringFP2', 'ScoringFP3',
        'TauxImpNb_RB', 'TauxImpNB_CPM', 'EcartNumCheq', 'NbrMagasin3J',
        'DiffDateTr1', 'DiffDateTr2', 'DiffDateTr3', 'CA3TRetMtt', 'CA3TR',
        'Heure', 'Month', 'Weekday', 'CodeDecision_0','CodeDecision_1','CodeDecision_2','CodeDecision_3']]

os.chdir('/Users/pierre/Documents/M2/Cours_M2/Fouille donne/Projet_fouille/Simple/')
rfc = pickle.load(open('rfc_simple.sav', 'rb'))
lda = pickle.load(open('lda_simple.sav', 'rb'))
lr = pickle.load(open('lr_simple.sav', 'rb'))
adab = pickle.load(open('adab_simple.sav', 'rb'))
xgboost = pickle.load(open('xgboost_simple.sav', 'rb'))

#predict_log_proba(X)
D['rfc_score'] = rfc.predict_proba(df_score)[:,1]
D['lda_score'] = lda.predict_proba(df_score)[:,1]
D['lr_score'] = lr.predict_proba(df_score)[:,1]
D['adab_score'] = adab.predict_proba(df_score)[:,1]
D['xgboost_score'] = xgboost.predict_proba(df_score)[:,1]

# D = pd.DataFrame({'rfc_score': rfc.predict_proba(df_score)[:,1], 'lda_score': lda.predict_proba(df_score)[:,1], 
#                        'lr_score': lr.predict_proba(df_score)[:,1], 'adab_score': adab.predict_proba(df_score)[:,1],
#                        'xgboost_score': xgboost.predict_proba(df_score)[:,1],
#                        'DateTransaction' : D['DateTransaction'],
#                        'FlagImpaye' : D['FlagImpaye']})

#Train 
X_train = D[D['DateTransaction'] < '2017-09-01']
y_train = X_train.FlagImpaye
X_train = X_train.drop("FlagImpaye", 1)
X_train = X_train.drop("DateTransaction", 1)

#Test
X_test = D[D['DateTransaction'] >= "2017-09-01"]
y_test = X_test.FlagImpaye
X_test = X_test.drop("FlagImpaye", 1)
X_test = X_test.drop("DateTransaction", 1)
print('train test')

###### Execution des modèles (mais sans leurs propres predictions)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

os.chdir('/Users/pierre/Documents/M2/Cours_M2/Fouille donne/Projet_fouille/Score/')

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=500, n_jobs = 1)
lr.fit(X_train.loc[:, X_train.columns != 'lr_score'], y_train)
y_pred = lr.predict(X_test.loc[:, X_test.columns != 'lr_score'])
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

# [[740771     66]
#  [  2172   4401]]
# 0.7972826086956522
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train.loc[:, X_train.columns != 'rfc_score'], y_train)
y_pred = rfc.predict(X_test.loc[:, X_test.columns != 'rfc_score'])
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))
# [[740743     94]
#  [  2168   4405]]
# 0.7957008670520233
from sklearn.ensemble import AdaBoostClassifier
adab = AdaBoostClassifier(random_state=0)
adab.fit(X_train.loc[:, X_train.columns != 'adab_score'], y_train)
y_pred = adab.predict(X_test.loc[:, X_test.columns != 'adab_score'])
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))
# [[740760     77]
#  [  2168   4405]]
# 0.7969244685662595
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train.loc[:, X_train.columns != 'lda_score'], y_train)
y_pred = lda.predict(X_test.loc[:, X_test.columns != 'lda_score'])
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))
# [[740758     79]
#  [  2192   4381]]
# 0.7941629656485091
import xgboost as xgb
xgboost = xgb.XGBClassifier(eval_metric='mlogloss',n_jobs = 1) #, scale_pos_weight = np.sqrt((740795+2204) /(42+4369))
xgboost.fit(X_train.loc[:, X_train.columns != 'xgboost_score'], y_train)
y_pred = xgboost.predict(X_test.loc[:, X_test.columns != 'xgboost_score'])
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))
# [[740760     77]
#  [  2168   4405]]
# 0.7969244685662595

#Save
filename = 'lr_fusion.sav'
pickle.dump(lr, open(filename, 'wb'))

filename = 'rfc_fusion.sav'
pickle.dump(rfc, open(filename, 'wb'))

filename = 'adab_fusion.sav'
pickle.dump(adab, open(filename, 'wb'))

filename = 'lda_fusion.sav'
pickle.dump(lda, open(filename, 'wb'))

filename = 'xgboost_fusion.sav'
pickle.dump(xgboost, open(filename, 'wb'))


