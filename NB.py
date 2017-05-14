#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
import gc
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

import xgboost as xgb
#s = pd.read_csv('submission_score16.txt', sep='\t')

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

df = pd.read_csv('data_clean.csv', sep=';')
df=df[['DATE', 'WEEK_END', 'DAY_WE_DS', 'TPER_TEAM', 'TPER_HOUR', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS', 'month', 'year', 'day', 'hour',
    'minute']]

submission = pd.read_csv('submission_clean.csv', sep=',')
submission=submission[['DATE', 'WEEK_END', 'DAY_WE_DS', 'TPER_TEAM', 'TPER_HOUR', 'ASS_ASSIGNMENT', 'month', 'year', 'day', 'hour',
                      'minute']]

from sklearn import preprocessing

le1 = preprocessing.LabelEncoder()
le1.fit(df.DAY_WE_DS)
df.DAY_WE_DS = le1.transform(df.DAY_WE_DS)
submission.DAY_WE_DS = le1.transform(submission.DAY_WE_DS)

le2 = preprocessing.LabelEncoder()
le2.fit(df.ASS_ASSIGNMENT)
df.ASS_ASSIGNMENT = le2.transform(df.ASS_ASSIGNMENT)
submission.ASS_ASSIGNMENT = le2.transform(submission.ASS_ASSIGNMENT)

assignement = le2.classes_
assignement = np.delete(assignement, [4,7, 24])

import datetime

def transform_data(data):
    month = []
    year = []
    day = []
    tper_hour = []
    minute = []
    tper_team = []
    DAY_WE_DS = []
    WEEK_END = []

    for i in data.index:
        #date = pd.datetime.strptime(i, '%Y-%m-%d %H:%M:%S.%f')
        date = i
        month.append(date.month)
        year.append(date.year)
        day.append(date.day)
        tper_hour.append(date.hour)
        minute.append(date.minute)
        if (date.hour==7 and date.minute==30):
            tper_team.append("Nuit")
        elif (date.hour==23 and date.minute==30):
            tper_team.append("Nuit")
        elif (0<=date.hour<=7):
            tper_team.append("Nuit")
        else:
            tper_team.append("Jours")
        d = datetime.date(date.year, date.month, date.day)
        d = d.strftime("%A")
        if (d=="Monday"):
            DAY_WE_DS.append('Lundi')
            WEEK_END.append(0)
        if (d=="Tuesday"):
            DAY_WE_DS.append('Mardi')
            WEEK_END.append(0)
        if (d=="Wednesday"):
            DAY_WE_DS.append('Mercredi')
            WEEK_END.append(0)
        if (d=="Thursday"):
            DAY_WE_DS.append('Jeudi')
            WEEK_END.append(0)
        if (d=="Friday"):
            DAY_WE_DS.append('Vendredi')
            WEEK_END.append(1)
        if (d=="Saturday"):
            DAY_WE_DS.append('Samedi')
            WEEK_END.append(1)
        if (d=="Sunday"):
            DAY_WE_DS.append('Dimanche')
            WEEK_END.append(1)

    data['TPER_TEAM'] = tper_team
    data['TPER_HOUR'] = tper_hour
    data['month'] = month
    data['year'] = year
    data['day'] = day
    data['DAY_WE_DS'] = DAY_WE_DS
    data['WEEK_END'] = WEEK_END
    data['minute'] = minute
    
    return data

from sklearn import tree
import multiprocessing

#def predict_submission(ass):
#for ass in assignement:
s = pd.read_csv('submission.txt', sep='\t')
#assignement = ['Téléphonie']

for ass in assignement:

    num_to_change = np.where(le2.classes_ == ass)[0][0]
    data = df[df.ASS_ASSIGNMENT==num_to_change] #Gestion - Accueil Telephonique
    data.DATE = map(dateparse, data.DATE)
    data = data.groupby(data.DATE).sum()

    data = data[(data.index.year==2012)].append(data[(data.index.year==2013)])

    data = transform_data(data)

    data_ = data[['WEEK_END', 'DAY_WE_DS', 'TPER_HOUR', 'month', 'year', 'day', 'minute']]
    data_.DAY_WE_DS = le1.transform(data_.DAY_WE_DS)

    calls = data.CSPL_RECEIVED_CALLS * 1.7
    calls = pd.Series.round(calls)

    s_ = submission[submission.ASS_ASSIGNMENT==num_to_change]    
    s_=s_[['WEEK_END', 'DAY_WE_DS', 'TPER_HOUR', 'month', 'year', 'day', 'minute']]
    
    clf = tree.DecisionTreeClassifier(random_state=99)
    clf = clf.fit(data_, calls)
    s_predict = clf.predict(s_)
    
    # clf = RandomForestClassifier(n_estimators=10)
    # clf = clf.fit(data_, calls)
    # s_predict = clf.predict(s_)

    # clf = MultinomialNB()
    # clf = clf.fit(data_, calls)
    # s_predict = clf.predict(s_)

    j=0
    for index, row in s[s.ASS_ASSIGNMENT==ass].iterrows():
        #print str(index)+' '+str(row['prediction'])
        s.set_value((s.index==index), 'prediction', s_predict[j])
        j=j+1

    del data, data_, calls, s_, s_predict, clf
    gc.collect()

s.to_csv('submission.txt', sep='\t', index=False)


# p = multiprocessing.Pool(4)
# p.map(predict_submission, [ass for ass in assignement])
