#!/usr/bin/env python
# coding: utf-8

import os
os.chdir('/Users/pierre/Documents/M2/Cours_M2/Fouille donne/Projet_fouille/données/')

import pandas as pd
from collections import Counter
import pickle
import numpy as np 
import matplotlib.pyplot as plt

################## Préparation des données
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


import calendar
#creation des colonnes année,mois,jours
D['year']=D["DateTransaction"].dt.year 
D['month']=D["DateTransaction"].dt.month
D['month'] = D['month'].apply(lambda x: calendar.month_abbr[x])
#Les jours sont numérotés de 0 à 6 on créer un dictionnaire pour leur affecter leurs noms
dayOfWeek={0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
D['day'] = D['DateTransaction'].dt.dayofweek.map(dayOfWeek)


# On utilisera les colonnes temporelles seulement pour les statistiques descriptives, on les passent donc en "string" pour en faire des variables catégorielles
D['year']=D['year'].astype('str') 
D['month']=D['month'].astype('str') 


D['day']=D['day'].astype('str') 
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
          'year',
          'month',
          'day']

################## Statistiques descriptives

#Montant dépensé par mois selon les jours de la semaine
import seaborn as sns
from matplotlib import pyplot as plt
f, ax = plt.subplots(figsize=(15, 8))

sns.lineplot(x=D['month'], y=D['Montant'],data=D, hue='day', ax=ax)

#Nombre de transactions refusées par mois selon les jours de la semaine 
f, ax = plt.subplots(figsize=(15, 8))
sns.lineplot(x=D['month'], y=D['FlagImpaye'],data=D, hue='day', ax=ax)

#Analyse de la matrice des corrélations
f, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(ax=ax,data=D.corr(), annot = True)

#Subset des acceptés vs refus
accept=D.loc[D['FlagImpaye'] =='0',:]
refus=D.loc[D['FlagImpaye'] =='1',:]
accept  =  accept.reset_index(drop = True)
refus  =  refus.reset_index(drop = True)
nrow_acc = len(accept.index)
nrow_ref = len(refus.index)
print("Pourcentage de refus : "+ str(nrow_ref/(nrow_ref + nrow_acc)*100)+"%")

#Distribution des montants de transactions
accept["Montant"].plot(kind='kde',
                   xlim=(0,100),
             stacked=False,
             figsize=(15, 5),
             )
refus["Montant"].plot(kind='kde',
                   xlim=(0,100),
             stacked=False,
             figsize=(15, 5),
             )

plt.xlabel("Montant des transactions")
plt.ylabel("Nombre des transactions")
plt.title("Distribution des montants de transactions")
plt.legend(['transactions acceptées','transactions refusées'])

plt.show()

#Etude du code de Décision par JOUR
f, ax = plt.subplots(figsize=(15, 8))
sns.boxplot(x=D["day"], y=np.log(D['Montant']), hue="CodeDecision",data=D, palette="Set3",ax=ax).set_title('Distribution des montants selon le code décision par jours de la semaine')

#Etude du code de Décision par MOIS

f, ax = plt.subplots(figsize=(15, 8))
sns.boxplot(x=D["month"], y=np.log(D['Montant']), hue="CodeDecision",data=D, palette="Set3",ax=ax).set_title('Distribution des montants selon le code décision par mois')

#Cross table flagImpaye / CodeDecision
pd.crosstab(D["FlagImpaye"],D['CodeDecision'], margins=True, normalize=True)


