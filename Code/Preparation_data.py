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

D = pd.read_csv("guillaume.txt", delimiter=";", dtype='string')

#D = pd.concat([D.head(), D.tail()])

#Drop useless col
D = D.drop("ZIBZIN", 1)
D = D.drop("IDAvisAutorisationCheque", 1)

#Duplicate
D = D.drop_duplicates()

#Delete line with juste names of columns
D = D[D.CodeDecision != 'CodeDecision']
D = D[D.CodeDecision != '4']

#To factpr
D['CodeDecision'] = D['CodeDecision'].astype('category')
D['FlagImpaye'] = D['FlagImpaye'].astype('category')
print("factor")

#To date
D["DateTransaction"] = pd.to_datetime(D["DateTransaction"])
print("date")

#Add new col
D["Month"] = D["DateTransaction"].dt.month
D["Weekday"] = D["DateTransaction"].dt.weekday
D["Heure"] = D["DateTransaction"].dt.hour
print('new col')

#To numeric
col_num = ['Montant',
          'VerifianceCPT1',
          'VerifianceCPT2',
          'VerifianceCPT3',
          'D2CB',
          'ScoringFP1',
          'ScoringFP2',
          'ScoringFP3',
          'TauxImpNb_RB',
          'TauxImpNB_CPM',
          'EcartNumCheq',
          'NbrMagasin3J',
          'DiffDateTr1',
          'DiffDateTr2',
          'DiffDateTr3',
          'CA3TRetMtt',
          'CA3TR',
          'Heure',
          'Month',
          'Weekday']
for col in col_num:
    if col not in ['Heure', 'Month', 'Weekday']:
        D[col] = D[col].str.replace(',','.')
    D[col] = D[col].astype('float')

#Skewed
# import numpy as np
# logcolumns = ['EcartNumCheq', 'CA3TRetMtt', 'CA3TR', 'ScoringFP3', 'ScoringFP1'
#          , 'TauxImpNb_RB', 'EcartNumCheq', 'Montant']
# for col in logcolumns:
#     D[col] = np.log10(D[col]+1) #log(x+1)

##Code decision
D = pd.get_dummies(D, columns=['CodeDecision'])

#Scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
D[col_num] = scaler.fit_transform(D[col_num])
print("num")

#Train 
#X_train = Z[Z['DateTransaction'] < '2017-09-01']   #for pca
X_train = D[D['DateTransaction'] < '2017-09-01']
y_train = X_train.FlagImpaye
X_train = X_train.drop("FlagImpaye", 1)
X_train = X_train.drop("DateTransaction", 1)
X_train.head()

#Test
#X_test = Z[Z['DateTransaction'] >= "2017-09-01"]   # for pca
X_test = D[D['DateTransaction'] >= "2017-09-01"]
y_test = X_test.FlagImpaye
X_test = X_test.drop("FlagImpaye", 1)
X_test = X_test.drop("DateTransaction", 1)
X_test.head()
print('train test')

import pickle

# save
# os.chdir('/Users/pierre/Documents/M2/Cours_M2/Fouille donne/Projet_fouille/données')
# D.to_pickle("Data_clean.pkl")
# X_train.to_pickle("X_train.pkl")
# y_train.to_pickle("y_train.pkl")
# X_test.to_pickle("X_test.pkl")
# y_test.to_pickle("y_test.pkl")
