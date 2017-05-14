# coding: utf8

import gc, sys
import numpy as np
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import datetime as DT
import multiprocessing
import functools

# Reste à traiter
#Téléphonie, Tech. Total, RTC

# Les fumantes
#'CMS', 'Crises', 'Gestion', 'Manager', 'Gestion Clients', 'Gestion DZ' 'Prestataires'

# To Fix: 'Tech. Total', 'RTC'

#Mecanicien de 7 à 18 inclus
#CAT de 8 à 19
#RTC de 9 à 18


df = pd.read_csv('data_clean.csv', sep=';')
df=df[['DATE', 'WEEK_END', 'DAY_WE_DS', 'TPER_TEAM', 'TPER_HOUR', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS','month', 'year','day', 'minute', 'hour']]

submission = pd.read_csv("submission_clean.csv", sep=',')
submission=submission[['DATE', 'WEEK_END', 'DAY_WE_DS', 'TPER_TEAM', 'TPER_HOUR', 'ASS_ASSIGNMENT','month', 'year','day', 'hour', 'minute']]

submit = pd.read_csv('submission.txt', sep='\t')

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
heure = pd.Series.unique(df.TPER_HOUR) 
heure.sort()

assignement = pd.Series.unique(df.ASS_ASSIGNMENT) 
assignement.sort()
assignement = np.delete(assignement,[0,1,2,4,7,20,27])

minute = pd.Series.unique(df.minute) 
minute.sort()

def create_dataset(dataset, look_back=1): 
    dataX, dataY = [], [] 
    for i in range(len(dataset)-look_back-1): 
        a = dataset[i:(i+look_back), 0] 
        dataX.append(a) 
        dataY.append(dataset[i + look_back, 0]) 
        return numpy.array(dataX), numpy.array(dataY)

def create_data_DayByDay(Jour, Heu, Min, Categorie):
    data = df[df.TPER_HOUR==Heu]
    data = data[data.minute == Min]
    data = data[data.ASS_ASSIGNMENT==Categorie]

    #group data by dates for selections
    dateparse1 = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    data.DATE = map(dateparse1, data.DATE)
    data = data.groupby(data.DATE).sum()
    data['DATE'] = data.index
    
    s = submission[(submission.DAY_WE_DS==Jour) & 
                   (submission.ASS_ASSIGNMENT==Categorie) & 
                   (submission.TPER_HOUR==Heu) & 
                   (submission.minute==Min) ]
    
    return create_cool_Data(data.CSPL_RECEIVED_CALLS, False)
    

def create_Data_for_Job(Jour, Heu, Min, Categorie):
    data = df[df.TPER_HOUR==Heu]
    data = data[data.minute == Min]
    data = data[data.ASS_ASSIGNMENT==Categorie]
    data = data[data.DAY_WE_DS==Jour]

    #group data by dates for selections
    dateparse1 = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    data.DATE = map(dateparse1, data.DATE)
    data = data.groupby(data.DATE).sum()
    data['DATE'] = data.index
    
    s = submission[(submission.DAY_WE_DS==Jour) & 
                   (submission.ASS_ASSIGNMENT==Categorie) & 
                   (submission.TPER_HOUR==Heu) & 
                   (submission.minute==Min) ]
    
    return data.CSPL_RECEIVED_CALLS, s

def create_cool_Data(d1, weekly):
    New = pd.Series()
    temp = d1[d1.index.year==2012]
    ttt = temp[temp.index.month==12]
    
    idx = pd.date_range(ttt.index[0], ttt.index[ttt.shape[0]-1])
    ttt.index = pd.DatetimeIndex(ttt.index)
    ttt = ttt.reindex(idx, fill_value=np.mean(ttt.values))
    
    if (weekly==True):
        ttt = ttt[0::7]
    
    New = New.append(ttt)
    
    for Y in [2013]:
        for M in range(1,13):
            temp = d1[d1.index.year==Y]
            ttt = temp[temp.index.month==M]
            
            idx = pd.date_range(ttt.index[0], ttt.index[ttt.shape[0]-1])
            ttt.index = pd.DatetimeIndex(ttt.index)
            ttt = ttt.reindex(idx, fill_value=np.mean(ttt.values))
            
            if (weekly==True):
                ttt = ttt[0::7]

            New = New.append(ttt)
    return New

def get_predictions(s, scaler, model, d):
    to_pred = []
    for i in s.DATE.values:
        u = pd.datetime.strptime(i, '%Y-%m-%d %H:%M:%S.%f') - DT.timedelta(days=7)
        #print d[u]
        to_pred.append(d[u]*2)
    to_pred = np.array(to_pred)
    to_pred = to_pred.reshape(len(to_pred),1) 
    to_pred = scaler.fit_transform(to_pred)
    to_pred = to_pred.reshape(len(to_pred),1,1)
    to_pred = model.predict(to_pred)
    predictions = scaler.inverse_transform(to_pred)
    
    return predictions

def create_model(trainX, trainY, nb_epoch, look_back):   
    model = Sequential()
    model.add(LSTM(4, input_dim=look_back))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, nb_epoch=nb_epoch, batch_size=1, verbose=2)
    return model

def fill_value_in_fileSubmission(s, Categorie, predictions):
    sumbission_dict = {}
    for k in range(len(s.DATE.values)):
        #submit.set_value((submit.DATE==s.DATE.values[k]) & (submit.ASS_ASSIGNMENT==Categorie)
        #                 , 'prediction', predictions[k])
        sumbission_dict[(s.DATE.values[k], Categorie)] = predictions[k]
    return sumbission_dict


#_________________________________________________________________

Min = 0
# C'est ici que tu dois changer les categories. Tu écris le nom de celle que tu veux prédire

# Voici la liste des categories que tu fais:

#Min = 30
# Domicile, Gestion, Gestion - Accueil Telephonique, Gestion Assurances, Gestion Renault, Japon, Mécanicien, Médical, Nuit, RENAULT, 
# Regulation Medicale, SAP, Services, Tech. Axa, Tech. Inter, CAT, Médical, Médecin
#Min = 0
Categorie = 'RENAULT' # Domicile, Tech. Axa, 
                  #Gestion - Accueil Telephonique, Gestion Assurances, Gestion Renault, 
                  #Japon, Médical, Nuit, RENAULT, Régulation Médicale, SAP, Services, Tech. Inter, Mécanicien, CAT

# Ce que j'ai ajouté sur le fichier de ahmed:
# Médical, Mécanicien, CAT

#Jour = ''
#_____________________________________________________________________-


def predict_submission(Heu, Jour):
    print "*** Jour: "+str(Jour)+" Catégorie: "+str(Categorie)+" / heure: "+str(Heu)+" / minute: "+str(Min)
        
    d1, s = create_Data_for_Job(Jour, Heu, Min, Categorie)
    
    #for model creation and training
    New = create_cool_Data(d1, True)

    #for submisson file, to predict 
    d = create_data_DayByDay(Jour, Heu, Min, Categorie)
    
    #moy = np.mean(New.values)
    #New = New - New.shift(1) + np.mean(New.values)
    #New[0] = moy

    New = New.values*2
    New = np.around(New)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    New = scaler.fit_transform(New)
    New = New.reshape(len(New),1)
    
    look_back = 1         
    trainX, trainY = create_dataset(New, look_back)           
    
    #reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    
    model = create_model(trainX, trainY, 1000, look_back)

    predictions = get_predictions(s, scaler, model, d)
    
    final_predictions = fill_value_in_fileSubmission(s, Categorie, predictions)
    
    del d1,s,d,New,scaler,trainX,trainY,model,predictions
    gc.collect()
    sys.stdout.flush()
    
    return final_predictions

def fullfill_submission(output):
    for i in range(len(output)):
        for key, value in output[i]:
            #print output[i][key,values]
            submit.set_value((submit.DATE==key) & (submit.ASS_ASSIGNMENT==value),
                             'prediction', output[i][key,value][0])


# def computation(np, J):
#     """ np is number of processes to fork """
   
#     return output

# if __name__=='__main__':
#     output = computation()

Jours = ['Lundi' , 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']



#Standard
for J in Jours:
    #partial_predict_submission = partial(predict_submission, Jour=J)
    p = multiprocessing.Pool(4)
    output = p.map(functools.partial(predict_submission, Jour=J), [h for h in heure])
    fullfill_submission(output)
    del output
    sys.stdout.flush()
    gc.collect()

# #RTC me sort une erreur de key error Timestamp('2013-02-01 18:00:00')
# for J in ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi']:
#     #partial_predict_submission = partial(predict_submission, Jour=J)
#     p = multiprocessing.Pool(4)
#     output = p.map(functools.partial(predict_submission, Jour=J), [h for h in range(8,20)])
#     fullfill_submission(output)
#     del output
#     sys.stdout.flush()
#     gc.collect()


submit.to_csv('submission.txt', sep='\t', index=False)

