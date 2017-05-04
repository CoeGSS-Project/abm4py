#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:22:32 2017

processing records
@author: geiges, GCF
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns; sns.set()
sns.set_color_codes("dark")

plotRecords     = False
plotCarStockBar = False
prefPerLabel    = False
utilPerLabel    = False
incomePerLabel  = False
meanPrefPerLabel= False
printCellMaps   = False
printPredMeth   = True
path = 'output/sim0053/'
#%% init
    
def loadObj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#%% plotting of the records
if plotRecords:
    #path = path 
    files = os.listdir(path + 'rec/')


    for filename in files:
        if filename.endswith('.csv'):
            df = pd.read_csv(path + 'rec/' + filename, index_col=0)
            plt.figure()
            plt.plot(df)
            plt.title(filename)
            plt.legend(df.columns)

#%%  plot car stock as bar plot
legStr = list()
if plotCarStockBar:
    plt.figure()
    enums   = loadObj(path + 'enumerations')
    df = pd.read_csv(path +  'rec/' + 'carStock.csv', index_col=0)
    nSteps = df.shape[0]
    nCars = np.zeros(nSteps)
    colorPal =  sns.color_palette("Set3", n_colors=len(enums['brands'].values()), desat=.8)
    for i, brand in enumerate(enums['brands'].values()):
       plt.bar(np.arange(nSteps), df.ix[:,i],bottom=nCars, color =colorPal[i], width=1)
       nCars += df.ix[:,i]
       legStr.append(brand)
plt.legend(legStr)
#%% loading household agent file

agMat   = np.load(path + 'agentFile_type2.npy')
propDic = loadObj(path + 'attributeList_type2')
enums   = loadObj(path + 'enumerations')

nSteps, nAgents, nProp = agMat.shape

#%%
step = 20
columns= ['']*agMat.shape[2]
for key in propDic.keys():
    for i in propDic[key]:
        columns[i] = key
df = pd.DataFrame(agMat[step],columns=columns)
#ax = sns.countplot(x=propDic['prefTyp'][0], hue=propDic['label'][0], data=df)


#%% preference types per car label
if prefPerLabel:
    res = dict()
    for carLabel in range(0,8):
        res[carLabel] = np.zeros([nSteps,4])
    
    for step in range(0,nSteps):
        for carLabel in range(0,8):
            idx = agMat[step,:,propDic['label'][0]] == carLabel
            for prefType in range(0,4):
                res[carLabel][step,prefType] = np.sum(agMat[step,idx,propDic['prefTyp'][0]] == prefType)
    
    legStr = list()
    for prefType in range(0,4):
        legStr.append(enums['prefTypes'][prefType])
    for carLabel in range(0,8):
        fig = plt.figure()
        plt.plot(res[carLabel])
        plt.title(enums['brands'][carLabel])
        plt.legend(legStr,loc=0)
    
    fig.suptitle('n preference types per car label')
print 1

#%% utiilty per car label
if utilPerLabel:
    res = np.zeros([nSteps,8])
    for step in range(0,nSteps):
        for carLabel in range(0,8):
            idx = agMat[step,:,propDic['label'][0]] == carLabel
            res[step, carLabel] = np.mean(agMat[step,idx,propDic['util'][0]])
    legStr = list()
    for label in range(0,8):
        legStr.append(enums['brands'][label])        
    fig = plt.figure()
    plt.plot(res)
    plt.title('Average utility by brand')
    plt.legend(legStr,loc=0)

#%% income per car label
if incomePerLabel:
    res = np.zeros([nSteps,8])
    for step in range(0,nSteps):
        for carLabel in range(0,8):
            idx = agMat[step,:,propDic['label'][0]] == carLabel
            res[step, carLabel] = np.mean(agMat[step,idx,propDic['income'][0]])
    legStr = list()
    for label in range(0,8):
        legStr.append(enums['brands'][label])        
    fig = plt.figure()
    plt.plot(res)
    plt.title('Average income by brand')
    plt.legend(legStr,loc=0)    

#%% mean preference per car label
ensembleAverage = np.mean(agMat[0,:,propDic['preferences']], axis = 1)
if meanPrefPerLabel:
    fig = plt.figure()
    res = dict()
    for carLabel in range(0,8):
        res[carLabel] = np.zeros([nSteps,4])
    for step in range(0,nSteps):
        for carLabel in range(0,8):
            idx = np.where(agMat[step,:,propDic['label'][0]] == carLabel)[0]
            res[carLabel][step,:] = np.mean(agMat[np.ix_([step],idx,propDic['preferences'])],axis=1) / ensembleAverage
    legStr = list()
    for prefType in range(0,4):
        legStr.append(enums['prefTypes'][prefType])
    
    
    for carLabel in range(0,8):
        plt.subplot(2,4,carLabel+1)    
        plt.plot(res[carLabel])
        plt.title(enums['brands'][carLabel])
        plt.legend(legStr,loc=0)
        plt.xlim([0,nSteps])

    fig.suptitle('mean preference per car label')
#%%
if printPredMeth:
   sumAvergUtil = np.sum(agMat[:,:,propDic['predMeth']]==1, axis = 1) 
   sumLinRef = np.sum(agMat[:,:,propDic['predMeth']]==2, axis = 1) 
   plt.plot(sumAvergUtil)
   plt.plot(sumLinRef)
   plt.title('Prediction that lead to car buy')
   plt.legend(['Cond Util','Lin. Reg.'],loc=0)
   plt.xlim([0,nSteps])


plt.show()
#%% loading cell agent file

if printCellMaps:
    agMat   = np.load(path + 'agentFile_type1.npy')
    propDic = loadObj(path + 'attributeList_type1')
    enums   = loadObj(path + 'enumerations')
    
    #get extend 
    step = 0
    min_ = [int(x) for x in np.min(agMat[step,:,propDic['pos']],axis=1)]
    max_ = [int(x+1) for x in np.max(agMat[step,:,propDic['pos']],axis=1)]
    cellMap = np.zeros(max_)
    for agID in range(agMat.shape[1]):
        x,y = agMat[step,agID,propDic['pos']]
        cellMap[int(x),int(y)] = 1
        cellIdx = np.where(cellMap)

    step = 45
    max_ = np.max(agMat[step,:,propDic['carsInCell']])
    min_ = np.min(agMat[step,:,propDic['carsInCell']])
    for carLabel in range(0,8):
        plt.subplot(2,4,carLabel+1)   
        cellMap[cellIdx] = agMat[step,:,propDic['carsInCell'][carLabel]]
        plt.pcolor(cellMap)
        plt.clim([min_, max_])
    plt.colorbar()




#%%
plt.show()