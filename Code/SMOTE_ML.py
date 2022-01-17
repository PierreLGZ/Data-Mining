#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from collections import Counter
import pickle
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

#Train / Test
os.chdir('/Users/pierre/Documents/M2/Cours_M2/Fouille donne/Projet_fouille/données')
X_train = pd.read_pickle("X_train.pkl")
y_train = pd.read_pickle("y_train.pkl")
X_test = pd.read_pickle("X_test.pkl")
y_test = pd.read_pickle("y_test.pkl")

X_train.dtypes
import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())

os.chdir('/Users/pierre/Documents/M2/Cours_M2/Fouille donne/Projet_fouille/SMOTE')
################################ Resampling ############################################## 
#Only SMOTE
from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=0 ,sampling_strategy=0.1,k_neighbors=1700)
#round(np.sqrt(X_train.shape[0])) #k_neigbors
X_smt, y_smt = smt.fit_resample(X_train, y_train)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=500)
lr.fit(X_smt, y_smt)
y_pred = lr.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_smt, y_smt)
y_pred = rfc.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

from sklearn.ensemble import AdaBoostClassifier
adab = AdaBoostClassifier(random_state=0)
adab.fit(X_smt, y_smt)
y_pred = adab.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_smt, y_smt)
y_pred = lda.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

import xgboost as xgb
import numpy as np
xgboost = xgb.XGBClassifier(eval_metric='mlogloss',n_jobs = 1) #, scale_pos_weight = np.sqrt((740795+2204) /(42+4369))
xgboost.fit(X_smt, y_smt)
y_pred = xgboost.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

filename = 'lda_SMOTE.sav'
pickle.dump(lda, open(filename, 'wb'))

filename = 'xgboost_SMOTE.sav'
pickle.dump(xgboost, open(filename, 'wb'))

filename = 'rfc_SMOTE.sav'
pickle.dump(rfc, open(filename, 'wb'))

filename = 'adab_SMOTE.sav'
pickle.dump(adab, open(filename, 'wb'))

filename = 'lr_SMOTE.sav'
pickle.dump(lr, open(filename, 'wb'))

################################################################################
#SMOTE + Under
from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=0 ,sampling_strategy=0.1,k_neighbors=1700)
#round(np.sqrt(X_train.shape[0])) #k_neigbors
X_smt, y_smt = smt.fit_resample(X_train, y_train)
Counter(y_smt)

from imblearn.under_sampling import RandomUnderSampler
rs = RandomUnderSampler(sampling_strategy=0.5)
X_rs, y_rs = rs.fit_resample(X_smt, y_smt)
X_rs.shape
Counter(y_rs)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=500)
lr.fit(X_rs, y_rs)
y_pred = lr.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_rs, y_rs)
y_pred = rfc.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

from sklearn.ensemble import AdaBoostClassifier
adab = AdaBoostClassifier(random_state=0)
adab.fit(X_rs, y_rs)
y_pred = adab.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_rs, y_rs)
y_pred = lda.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

import xgboost as xgb
import numpy as np
xgboost = xgb.XGBClassifier(eval_metric='mlogloss',n_jobs = 1) #, scale_pos_weight = np.sqrt((740795+2204) /(42+4369))
xgboost.fit(X_rs, y_rs)
y_pred = xgboost.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

filename = 'lda_SMOTEund.sav'
pickle.dump(lda, open(filename, 'wb'))

filename = 'xgboost_SMOTEund.sav'
pickle.dump(xgboost, open(filename, 'wb'))

filename = 'rfc_SMOTEund.sav'
pickle.dump(rfc, open(filename, 'wb'))

filename = 'adab_SMOTEund.sav'
pickle.dump(adab, open(filename, 'wb'))

filename = 'lr_SMOTEund.sav'
pickle.dump(lr, open(filename, 'wb'))

################################################################################
#randomOver + Under
from imblearn.over_sampling import RandomOverSampler
# define oversampling strategy
ro = RandomOverSampler(sampling_strategy=0.1)
# fit and apply the transform
X_ro, y_ro = ro.fit_resample(X_train, y_train)


from imblearn.under_sampling import RandomUnderSampler
rs = RandomUnderSampler(sampling_strategy=0.5)
X_rs, y_rs = rs.fit_resample(X_ro, y_ro)
X_rs.shape
Counter(y_rs)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=500)
lr.fit(X_rs, y_rs)
y_pred = lr.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_rs, y_rs)
y_pred = rfc.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

from sklearn.ensemble import AdaBoostClassifier
adab = AdaBoostClassifier(random_state=0)
adab.fit(X_rs, y_rs)
y_pred = adab.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_rs, y_rs)
y_pred = lda.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))

import xgboost as xgb
import numpy as np
xgboost = xgb.XGBClassifier(eval_metric='mlogloss',n_jobs = 1) #, scale_pos_weight = np.sqrt((740795+2204) /(42+4369))
xgboost.fit(X_rs, y_rs)
y_pred = xgboost.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='binary', pos_label='1'))


filename = 'lda_ovund.sav'
pickle.dump(lda, open(filename, 'wb'))

filename = 'xgboost_ovund.sav'
pickle.dump(xgboost, open(filename, 'wb'))

filename = 'rfc_ovund.sav'
pickle.dump(rfc, open(filename, 'wb'))

filename = 'adab_ovund.sav'
pickle.dump(adab, open(filename, 'wb'))

filename = 'lr_ovund.sav'
pickle.dump(lr, open(filename, 'wb'))

################################ Pipeline


# from imblearn.pipeline import Pipeline
# from imblearn.over_sampling import SMOTE
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
# from imblearn.under_sampling import ClusterCentroids


# from imblearn.over_sampling import SMOTENC

# pipe = Pipeline(
#     [#('resample2', RandomUnderSampler(replacement = False)),
#      ('resample1', SMOTENC(categorical_features=[1], random_state=0)),
#     ('model', LinearDiscriminantAnalysis())])

# p_grid = dict(resample1__k_neighbors=[1050,1100,1150,1200])#Fait des test avec 5,10 mais se rend compte que doit faire plus grd

# tscv = TimeSeriesSplit(n_splits=5)

# from sklearn.metrics import make_scorer
# f1_scorer = make_scorer(f1_score, average='binary', pos_label='1')

# grid_search = GridSearchCV(
#     estimator=pipe, param_grid=p_grid, cv=tscv, scoring=f1_scorer,
#                        return_train_score=True
# )
# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)
# print(grid_search.best_score_)
# print(grid_search.cv_results_)

# #only smote
# # {'resample__k_neighbors': 1790}
# # 0.7314228400536547
# #Expected n_neighbors <= n_samples
# #The score on this train-test partition for these parameters will be set to nan
# #DONC test avec random

# #Only randomundersample : 0.14842637904572892

# #randomundersample + smote
# # {'resample1__k_neighbors': 1100}
# # 0.1517334221514806

# ######## Modele appliquer juste après pré-traitement (sans Grid Search donc para par default) #######

# #RandomClassifier :
# # [[740837      0]
# #  [  6573      0]]
# #0.00
# #LDA :
# # [[740628    209]
# #  [  2242   4331]]
# #0.7794474939260326

# #XGBoost :
# # [[740837      0]
# #  [  2311   4262]]
# #0.7962895598399419

# #sans verifiance
# #RandomClassifier : 0.003038128512836093
# #LDA : 0.781095406360424
# #XGBoost :0.7976470588235294

# ######## Modele appliquer juste après pré-traitement (avec Grid Search) #######

# # save the model to disk
# #filename = 'ElasticNet.sav'
# #pickle.dump(Elasnet, open(filename, 'wb'))

