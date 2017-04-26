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
plotCarStockBar = True
prefPerLabel    = False
utilPerLabel    = False
meanPrefPerLabel= True

#%% init
    
def loadObj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#%% plotting of the records
path = 'output/rec/'
files = os.listdir(path)

if plotRecords:
    for filename in files:
        if filename.endswith('.csv'):
            df = pd.read_csv(path + filename, index_col=0)
            plt.figure()
            plt.plot(df)
            plt.title(filename)
            plt.legend(df.columns)

#%%  plot car stock as bar plot
legStr = list()
if plotCarStockBar:
    enums   = loadObj('output/enumerations')
    df = pd.read_csv(path + 'carStock.csv', index_col=0)
    nSteps = df.shape[0]
    nCars = np.zeros(nSteps)
    colorPal =  sns.color_palette("Set3", n_colors=len(enums['brands'].values()), desat=.8)
    for i, brand in enumerate(enums['brands'].values()):
       plt.bar(np.arange(nSteps), df.ix[:,i],bottom=nCars, color =colorPal[i], width=1)
       nCars += df.ix[:,i]
       legStr.append(brand)
plt.legend(legStr)
#%% loading agent file

agMat = np.load('output/agentFile.npy')
propDic = loadObj('output/attributeList')
enums   = loadObj('output/enumerations')

nSteps, nAgents, nProp = agMat.shape

#%%
#df = pd.DataFrame(agMat[0])
#ax = sns.countplot(x=propDic['prefTyp'][0], hue=propDic['label'][0], data=df)


#%% preference types per car label
if prefPerLabel:
    res = dict()
    for carLabel in range(0,8):
        res[carLabel] = np.zeros([nSteps,3])
    
    for step in range(0,nSteps):
        for carLabel in range(0,8):
            idx = agMat[step,:,propDic['label'][0]] == carLabel
            for prefType in range(0,3):
                res[carLabel][step,prefType] = np.sum(agMat[step,idx,propDic['prefTyp'][0]] == prefType)
    
    legStr = list()
    for prefType in range(0,3):
        legStr.append(enums['prefTypes'][prefType])
    for carLabel in range(0,8):
        plt.figure()
        plt.plot(res[carLabel])
        plt.title(enums['brands'][carLabel])
        plt.legend(legStr,loc=0)
    
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
    plt.figure()
    plt.plot(res)
    plt.title('Average utility by brand')
    plt.legend(legStr,loc=0)

#%% mean preference per car label
if meanPrefPerLabel:
    res = dict()
    for carLabel in range(0,8):
        res[carLabel] = np.zeros([nSteps,3])
    for step in range(0,nSteps):
        for carLabel in range(0,8):
            idx = np.where(agMat[step,:,propDic['label'][0]] == carLabel)[0]
            res[carLabel][step,:] = np.mean(agMat[np.ix_([step],idx,propDic['preferences'])],axis=1)
    legStr = list()
    for prefType in range(0,3):
        legStr.append(enums['prefTypes'][prefType])
    plt.figure()
    
    for carLabel in range(0,8):
        plt.subplot(2,4,carLabel+1)    
        plt.plot(res[carLabel])
        plt.title(enums['brands'][carLabel])
        plt.legend(legStr,loc=0)
        plt.xlim([0,nSteps])

plt.show()