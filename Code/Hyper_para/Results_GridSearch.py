#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 22:28:58 2022

@author: pierre
"""
import os
os.chdir('/Users/pierre/Documents/M2/Cours_M2/Fouille donne/Projet_fouille/Hyper_para/')
import pandas as pd
from collections import Counter
import pickle

rfc = pickle.load(open('gsearch_rfc.sav', 'rb'))

lr = pickle.load(open('gsearch_lr.sav', 'rb'))

xgboost = pickle.load(open('gsearch_xgboost.sav', 'rb'))

rfc = pd.DataFrame(rfc.cv_results_)

lr = pd.DataFrame(lr.cv_results_)

xgb = pd.DataFrame(xgboost.cv_results_)