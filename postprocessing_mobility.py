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
import seaborn as sns; sns.set()
sns.set_color_codes("dark")

plotRecords     = 0
plotCarStockBar = 1
prefPerLabel    = 1
utilPerLabel    = 1
incomePerLabel  = 1
meanPrefPerLabel= 1
printCellMaps   = 0
printPredMeth   = True
path = 'output/sim0178/'
#%% init
    
from class_auxiliary import loadObj

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


#%% loading household agent file

agMat   = np.load(path + 'agentFile_type2.npy')
propDic = loadObj(path + 'attributeList_type2')
enums   = loadObj(path + 'enumerations')

nSteps, nAgents, nProp = agMat.shape

#%%  plot car stock as bar plot
legStr = list()

carMat = np.zeros([agMat.shape[0],3])
for time in range(agMat.shape[0]):
    carMat[time,:]= np.bincount(agMat[time,:,propDic['mobilityType'][0]].astype(int),minlength=3).astype(float)
if plotCarStockBar:
    plt.figure()
    enums   = loadObj(path + 'enumerations')
    #df = pd.read_csv(path +  'rec/' + 'carStock.csv', index_col=0)
    nSteps = agMat.shape[0]
    nCars = np.zeros(nSteps)
    colorPal =  sns.color_palette("Set3", n_colors=len(enums['brands'].values()), desat=.8)
    for i, brand in enumerate(enums['brands'].values()):
       plt.bar(np.arange(nSteps), carMat[:,i],bottom=nCars, color =colorPal[i], width=1)
       nCars += carMat[:,i]
       legStr.append(brand)
#plt.legend(legStr)
plt.legend(legStr,bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

#%% number of different car per agent:
nUniqueCars = list()
for i in range(agMat.shape[1]):
    nUniqueCars.append(len(np.unique(agMat[:,i,propDic['mobilityType'][0]])))

print np.mean(nUniqueCars)

for prefTyp in range(3):
    nUniqueCars = list()
    idx = agMat[0,:,propDic['prefTyp'][0]] == prefTyp
    for i in np.where(idx)[0]:
        nUniqueCars.append(len(np.unique(agMat[:,i,propDic['mobilityType'][0]])))
    print 'Pref Type = ' + enums['prefTypes'][prefTyp] + ' ownes ' + str(np.mean(nUniqueCars)) +'different cars'

#%% number of buys

x = np.diff(agMat,axis=0)
#x = x[:,:,propDic['label']] != 0
carBuys =np.sum(x[:,:,propDic['mobilityType']] > 0,axis = 0)
np.mean(carBuys)
np.max(carBuys)
np.min(carBuys)
for prefTyp in range(3):
    idx = agMat[0,:,propDic['prefTyp'][0]] == prefTyp
    
    print 'Pref Type "' + enums['prefTypes'][prefTyp] + '" buys ' + str(np.mean(carBuys[idx])) +'times'
    
    
print 1
#%% df for one timestep
if False:
    step = 50
    columns= ['']*agMat.shape[2]
    for key in propDic.keys():
        for i in propDic[key]:
            columns[i] = key
    df = pd.DataFrame(agMat[step],columns=columns)

#%% df for one agent
if False:
    agent = 0
    columns= ['']*agMat.shape[2]
    for key in propDic.keys():
        for i in propDic[key]:
            columns[i] = key
    df = pd.DataFrame(agMat[:,agent,:],columns=columns)

#%% preference types per car label
prefTypes = np.zeros(3)
for prefTyp in range(0,3):
    prefTypes[prefTyp] = np.sum(agMat[0,:,propDic['prefTyp'][0]] == prefTyp)

if prefPerLabel:
    res = dict()
    for carLabel in range(0,8):
        res[carLabel] = np.zeros([nSteps,3])
    
    for step in range(0,nSteps):
        for carLabel in range(0,8):
            idx = agMat[step,:,propDic['mobilityType'][0]] == carLabel
            for prefType in range(0,3):
                res[carLabel][step,prefType] = np.sum(agMat[step,idx,propDic['prefTyp'][0]] == prefType) / prefTypes[prefType]
    
    legStr = list()
    for prefType in range(0,3):
        legStr.append(enums['prefTypes'][prefType])
    for carLabel in range(0,len(enums['brands'])):
        fig = plt.figure()
        plt.plot(res[carLabel])
        plt.title(enums['brands'][carLabel])
        plt.legend(legStr,loc=0)
    
        fig.suptitle('n preference types per car label')
print 1

#%% utiilty per car label
if utilPerLabel:
    res = np.zeros([nSteps,len(enums['brands'])])
    for step in range(0,nSteps):
        for carLabel in range(0,len(enums['brands'])):
            idx = agMat[step,:,propDic['mobilityType'][0]] == carLabel
            res[step, carLabel] = np.mean(agMat[step,idx,propDic['util'][0]])
    legStr = list()
    for label in range(0,len(enums['brands'])):
        legStr.append(enums['brands'][label])        
    fig = plt.figure()
    plt.plot(res)
    plt.title('Average utility by brand')
    plt.legend(legStr,loc=0)

#%% income per car label
if incomePerLabel:
    res = np.zeros([nSteps,len(enums['brands'])])
    for step in range(0,nSteps):
        for carLabel in range(0,len(enums['brands'])):
            idx = agMat[step,:,propDic['mobilityType'][0]] == carLabel
            res[step, carLabel] = np.mean(agMat[step,idx,propDic['income'][0]])
    legStr = list()
    for label in range(0,len(enums['brands'])):
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
    for carLabel in range(0,len(enums['brands'])):
        res[carLabel] = np.zeros([nSteps,3])
    for step in range(0,nSteps):
        for carLabel in range(0,len(enums['brands'])):
            idx = np.where(agMat[step,:,propDic['mobilityType'][0]] == carLabel)[0]
            res[carLabel][step,:] = np.mean(agMat[np.ix_([step],idx,propDic['preferences'])],axis=1) / ensembleAverage
    legStr = list()
    for prefType in range(0,3):
        legStr.append(enums['prefTypes'][prefType])
    
    h = list()
    for carLabel in range(0,len(enums['brands'])):
        plt.subplot(2,2,carLabel+1)    
        h.append(plt.plot(res[carLabel]))
        plt.title(enums['brands'][carLabel])
        #plt.legend(legStr,loc=0)
        plt.xlim([0,nSteps])
    plt.legend(legStr,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #fig.legend(h, legStr, loc = (1,1,0,0))
    plt.tight_layout()
    fig.suptitle('mean preference per car label')
print 1
#%% mean consequences per car label

if meanPrefPerLabel:
    fig = plt.figure()
    res = dict()
    for carLabel in range(0,len(enums['brands'])):
        res[carLabel] = np.zeros([nSteps,3])
    for step in range(0,nSteps):
        ensembleAverage = np.mean(agMat[step,:,propDic['x']], axis = 1)
        for carLabel in range(0,len(enums['brands'])):
            idx = np.where(agMat[step,:,propDic['mobilityType'][0]] == carLabel)[0]
            res[carLabel][step,:] = np.mean(agMat[np.ix_([step],idx,propDic['x'])],axis=1) #/ ensembleAverage
    legStr = list()
    for prefType in range(0,3):
        legStr.append(enums['prefTypes'][prefType])
    
    h = list()
    for carLabel in range(0,len(enums['brands'])):
        plt.subplot(2,2,carLabel+1)    
        h.append(plt.plot(res[carLabel]))
        plt.title(enums['brands'][carLabel])
        #plt.legend(legStr,loc=0)
        plt.xlim([0,nSteps])
    plt.legend(legStr,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #fig.legend(h, legStr, loc = (1,1,0,0))
    plt.tight_layout()
    fig.suptitle('mean consequences per car label')
print 1

#%% print predition method

if printPredMeth:
   sumAvergUtil = np.sum(agMat[:,:,propDic['predMeth']]==1, axis = 1) 
   sumLinRef = np.sum(agMat[:,:,propDic['predMeth']]==2, axis = 1) 
   plt.figure()
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



#%% loading cell agent file

cellMat   = np.load(path + 'agentFile_type1.npy')
propDict = loadObj(path + 'attributeList_type1')
enums   = loadObj(path + 'enumerations')

nSteps, nAgents, nProp = cellMat.shape

#%%
meanCon   = np.zeros([nSteps,3])
meanEco   = np.zeros([nSteps,3])
meanPrc   = np.zeros([nSteps,3])
for step in range(nSteps):
    meanVect = np.mean(cellMat[step,:,propDict['cellX']],axis=1)
    meanCon[step,:] = meanVect[:3]
    meanEco[step,:] = meanVect[3:6]
    meanPrc[step,:] = meanVect[6:]

plt.figure()
plt.subplot(2,2,1)
plt.plot(meanCon)
plt.legend(enums['brands'].values())
plt.title('convenience, mean over cells')
plt.subplot(2,2,2)
plt.plot(meanEco)
plt.legend(enums['brands'].values())
plt.title('ecology, mean over cells')
plt.subplot(2,2,3)
plt.plot(meanPrc)
plt.legend(enums['brands'].values())
plt.title('price on cell level')
plt.xlim([0,nSteps])    

#%%
plt.clf()
landLayer = np.zeros(np.max(cellMat[0,:,propDict['pos']]+1,axis=1))
for iCell in range(cellMat.shape[1]):
    landLayer[cellMat[0,iCell,propDict['pos'][0]],cellMat[0,iCell,propDict['pos'][1]]] = 1
#plt.pcolormesh(landLayer)
landLayer = landLayer.astype(bool)
res = landLayer*1.0
step = 80
test = landLayer*0
for iBrand in range(3):
    res = landLayer*1.0
    res[landLayer] = cellMat[0,:,propDict['carsInCell'][iBrand]] / np.sum(cellMat[0,:,propDict['carsInCell']],axis=0)
    test = test + res
    #res[landLayer==False] = np.nan
    plt.subplot(2,2,iBrand+1)
    plt.pcolormesh(res)
    plt.clim([0,1])
    plt.colorbar()
    plt.title(enums['brands'][iBrand] + ' per cells')
print 1

#%%
plt.clf()
landLayer = np.zeros(np.max(cellMat[0,:,propDict['pos']]+1,axis=1))
for iCell in range(cellMat.shape[1]):
    landLayer[cellMat[0,iCell,propDict['pos'][0]],cellMat[0,iCell,propDict['pos'][1]]] = 1
#plt.pcolormesh(landLayer)
landLayer = landLayer.astype(bool)
res = landLayer*1.0
step = 10
for iBrand in range(len(propDict['cellX'])):
    res = landLayer*1.0
    res[landLayer] = cellMat[step,:,propDict['cellX'][iBrand]] 
    #res[landLayer==False] = np.nan
    plt.subplot(3,3,iBrand+1)
    plt.pcolormesh(res)
    plt.clim([np.min(res[landLayer]),np.max(res[landLayer])])
    print [np.min(res[landLayer]),np.max(res[landLayer])]
    plt.colorbar()
    plt.title('consequences per cells')

print 1
#%%
plt.show()