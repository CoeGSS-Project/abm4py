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
import sys
sys.path.append('/media/sf_shared/python/modules/biokit')
import seaborn as sns; sns.set()
sns.set_color_codes("dark")

plotRecords     = 0
plotCarStockBar = 1
prefPerLabel    = 1
utilPerLabel    = 1
incomePerLabel  = 1
meanPrefPerLabel= 1
meanConsequencePerLabel = 1
printCellMaps   = 1
emissionsPerLabel   = 1

path = 'output/sim0018/'

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

nPr = len(enums['priorities'])

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
       plt.bar(np.arange(nSteps), carMat[:,i],bottom=nCars, color =colorPal[[1,0,2][i]], width=1)
       nCars += carMat[:,i]
       legStr.append(brand)
#plt.legend(legStr)
plt.legend(legStr,bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

#%% sales
carSales = np.zeros([agMat.shape[0],3])
for time in range(agMat.shape[0]):
    for brand in range(0,len(enums['brands'])):
        idx = agMat[time,:,propDic['predMeth'][0]] ==1
        carSales[time,:] = np.bincount(agMat[time,idx,propDic['mobilityType'][0]].astype(int),minlength=3).astype(float)
fig = plt.figure()
plt.plot(carSales)
plt.legend(enums['brands'].values(),loc=0)

plt.title('sales per mobility Type')
#%% number of different car per agent:
nUniqueCars = list()
for i in range(agMat.shape[1]):
    nUniqueCars.append(len(np.unique(agMat[:,i,propDic['mobilityType'][0]])))

print np.mean(nUniqueCars)

for prefTyp in range(nPr):
    nUniqueCars = list()
    idx = agMat[0,:,propDic['prefTyp'][0]] == prefTyp
    for i in np.where(idx)[0]:
        nUniqueCars.append(len(np.unique(agMat[:,i,propDic['mobilityType'][0]])))
    print 'Pref Type = ' + enums['priorities'][prefTyp] + ' ownes ' + str(np.mean(nUniqueCars)) +'different cars'

#%% number of buys

x = np.diff(agMat,axis=0)
#x = x[:,:,propDic['label']] != 0
carBuys =np.sum(x[:,:,propDic['mobilityType']] > 0,axis = 0)
np.mean(carBuys)
np.max(carBuys)
np.min(carBuys)
for prefTyp in range(nPr):
    idx = agMat[0,:,propDic['prefTyp'][0]] == prefTyp
    
    print 'Pref Type "' + enums['priorities'][prefTyp] + '" buys ' + str(np.mean(carBuys[idx])) +'times'
    
    
print 1
#%% df for one timestep
fig = plt.figure()


if False:
    step = 1

    columns= ['']*agMat.shape[2]
    for key in propDic.keys():
        for i in propDic[key]:
            columns[i] = key
    for i,idx in enumerate(propDic['prop']):  
        columns[idx] = ['emissions','price'][i]           
    for i,idx in enumerate(propDic['preferences']):
        columns[idx] = 'pref of ' + enums['priorities'][i] 
    for i,idx in enumerate(propDic['consequences']):
        columns[idx] = ['comfort','environmental','remainig money','similarity'][i] 
       
    df = pd.DataFrame(agMat[step],columns=columns)
    del df['time']
    del df['noisyUtil']
    del df['carID']
    del df['predMeth']
    del df['expUtil']
    del df['name']
    del df['type']
    del df['prefTyp']  
    del df['mobilityType']    
             
            

#from biokit.viz import corrplot
#c = corrplot.Corrplot(df)
#c.plot()

#%% df for one agent
if False:
    agent = 0
    columns= ['']*agMat.shape[2]
    for key in propDic.keys():
        for i in propDic[key]:
            columns[i] = key
    df = pd.DataFrame(agMat[:,agent,:],columns=columns)


#%% df for one timestep
if False:
    #%%
    step = 9
    columns= ['']*agMat.shape[2]
    for key in propDic.keys():
        for i in propDic[key]:
            columns[i] = key
    df = pd.DataFrame(agMat[step,:,:],columns=columns)
#%% preference types per car label
prefTypes = np.zeros(nPr)
for prefTyp in range(0,nPr):
    prefTypes[prefTyp] = np.sum(agMat[0,:,propDic['prefTyp'][0]] == prefTyp)

if prefPerLabel:
    res = dict()
    for carLabel in range(0,8):
        res[carLabel] = np.zeros([nSteps,nPr])
    
    for step in range(0,nSteps):
        for carLabel in range(0,8):
            idx = agMat[step,:,propDic['mobilityType'][0]] == carLabel
            for prefType in range(0,nPr):
                res[carLabel][step,prefType] = np.sum(agMat[step,idx,propDic['prefTyp'][0]] == prefType) / prefTypes[prefType]
    
    legStr = list()
    for prefType in range(0,nPr):
        legStr.append(enums['priorities'][prefType])
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
        res[carLabel] = np.zeros([nSteps,nPr])
    for step in range(0,nSteps):
        for carLabel in range(0,len(enums['brands'])):
            idx = np.where(agMat[step,:,propDic['mobilityType'][0]] == carLabel)[0]
            res[carLabel][step,:] = np.mean(agMat[np.ix_([step],idx,propDic['preferences'])],axis=1) / ensembleAverage
    legStr = list()
    for prefType in range(0,nPr):
        legStr.append(enums['priorities'][prefType])
    
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

if meanConsequencePerLabel:
    fig = plt.figure()
    res = dict()
    for carLabel in range(0,len(enums['brands'])):
        res[carLabel] = np.zeros([nSteps,nPr])
    for step in range(0,nSteps):
        ensembleAverage = np.mean(agMat[step,:,propDic['consequences']], axis = 1)
        for carLabel in range(0,len(enums['brands'])):
            idx = np.where(agMat[step,:,propDic['mobilityType'][0]] == carLabel)[0]
            res[carLabel][step,:] = np.mean(agMat[np.ix_([step],idx,propDic['consequences'])],axis=1) #/ ensembleAverage
    legStr = list()
    for prefType in range(0,nPr):
        legStr.append(enums['consequences'][prefType])
    
    h = list()
    for carLabel in range(0,len(enums['brands'])):
        plt.subplot(2,2,carLabel+1)    
        h.append(plt.plot(res[carLabel]))
        plt.title(enums['brands'][carLabel])
        
        plt.xlim([0,nSteps])
    # plt.legend(legStr,loc=0)        
    plt.legend(legStr,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #fig.legend(h, legStr)
    plt.tight_layout()
    fig.suptitle('mean consequences per car label')
print 1

#%% print consequences per brand
if emissionsPerLabel:
    fig = plt.figure()
    res = np.zeros([nSteps,len(enums['brands'])])
    for i in range(2):
        plt.subplot(2,1,i+1)
        for step in range(0,nSteps):
            for carLabel in range(0,len(enums['brands'])):
                idx = agMat[step,:,propDic['mobilityType'][0]] == carLabel
                res[step, carLabel] = np.mean(agMat[step,idx,propDic['prop'][i]])
        legStr = list()
        for label in range(0,len(enums['brands'])):
            legStr.append(enums['brands'][label])        
        
        plt.plot(res)
        if i == 0:
            plt.title('Average emissions by brand')
        else:
            plt.title('Average price by brand')
    plt.legend(legStr,loc=0)    

plt.show()
#%% loading cell agent file


if printCellMaps:
    #%% loading cell agent file
    
    cellMat   = np.load(path + 'agentFile_type1.npy')
    propDict = loadObj(path + 'attributeList_type1')
    enums   = loadObj(path + 'enumerations')
    
    nSteps, nAgents, nProp = cellMat.shape
#%%    
if printCellMaps:

    meanCon   = np.zeros([nSteps,3])
    meanEco   = np.zeros([nSteps,3])
    meanPrc   = np.zeros([nSteps,3])
    for step in range(nSteps):
        meanVect = np.mean(cellMat[step,:,propDict['convenience']],axis=1)
        meanCon[step,:] = meanVect
#        meanEco[step,:] = meanVect[3:6]
#        meanPrc[step,:] = meanVect[6:]
    
    plt.figure()
    #plt.subplot(2,2,1)
    plt.plot(meanCon)
    plt.legend(enums['brands'].values())
    plt.title('convenience, mean over cells')
    
    #%%
    import copy
    plt.clf()
    landLayer = np.zeros(np.max(cellMat[0,:,propDict['pos']]+1,axis=1).astype(int).tolist())
    for iCell in range(cellMat.shape[1]):
        x = cellMat[0,iCell,propDict['pos'][0]].astype(int)
        y = cellMat[0,iCell,propDict['pos'][1]].astype(int)
        landLayer[x,y] = 1
    #plt.pcolormesh(landLayer)
    landLayer = landLayer.astype(bool)
    res = landLayer*1.0
    step = 19
    test = landLayer*0
    for iBrand in range(3):
        res = landLayer*1.0
        res[np.where(landLayer)] = cellMat[step,:,propDict['carsInCell'][iBrand]] / cellMat[step,:,propDict['population']]
        print np.max(res)
        test = test + res
        if iBrand == 1:
            arrayData = copy.copy(res)
        #res[landLayer==False] = np.nan
        plt.subplot(2,2,iBrand+1)
        plt.pcolormesh(res)
        #plt.clim([0,1])
        plt.colorbar()
        plt.title(enums['brands'][iBrand] + ' cars per cells')
    print 1
    sys.path.append('/media/sf_shared/python/database')
    import class_map
    import matplotlib
    foMap = class_map.Map()
    cm = matplotlib.cm.get_cmap('YlGn')
    normed_data = (arrayData - np.nanpercentile(arrayData,5)) / (np.nanpercentile(arrayData,95) - np.nanpercentile(arrayData,5))
    self.minmax = np.nanpercentile(arrayData,5), np.nanpercentile(arrayData,95)
    colored_data = cm(normed_data)
    foMap.addImage(colored_data, mercator=False, latMin=53.9167-62*0.04166666, latMax=53.9167,lonMin=6.625,lonMax=6.625+118*0.04166666,min_=0,max_=0)
    from branca.utilities import color_brewer
    cols = color_brewer('YlGn',6)
    cmap = folium.LinearColormap(cols,index = [np.nanpercentile(arrayData,5), np.nanpercentile(arrayData,95)], caption='test')
    self.map.add_child(cmap)
    foMap.view()
    #%%
    plt.figure()
    plt.clf()
    landLayer = np.zeros(np.max(cellMat[step,:,propDict['pos']]+1,axis=1).astype(int).tolist())
    for iCell in range(cellMat.shape[1]):
        x = cellMat[0,iCell,propDict['pos'][0]].astype(int)
        y = cellMat[0,iCell,propDict['pos'][1]].astype(int)
        landLayer[x,y] = 1
    landLayer = landLayer.astype(bool)
    res = landLayer*1.0
    step = 1
    test = landLayer*0
    for iBrand in range(3):
        res = landLayer*1.0
        res[landLayer] = cellMat[step,:,propDict['convenience'][iBrand]]
        test = test + res
        #res[landLayer==False] = np.nan
        plt.subplot(2,2,iBrand+1)
        plt.pcolormesh(res)
        #plt.clim([0,1])
        plt.colorbar()
        plt.title('convenience of ' + enums['brands'][iBrand] + ' cars per cells')
    print 1
#%%
plt.figure()
res = landLayer*1.0
res[landLayer] = cellMat[0,:,propDict['population']][0]
plt.pcolormesh(res)
#plt.clim([0,1])
plt.colorbar()
plt.title('population')
print 1
plt.show()

sys.path.append('/media/sf_shared/python/database')
import class_map
import matplotlib
foMap = class_map.Map()
cm = matplotlib.cm.get_cmap('YlGn')
arrayData = res
normed_data = (arrayData - np.nanpercentile(arrayData,5)) / (np.nanpercentile(arrayData,95) - np.nanpercentile(arrayData,5))
self.minmax = np.nanpercentile(arrayData,5), np.nanpercentile(arrayData,95)
colored_data = cm(normed_data)
foMap.addImage(colored_data, mercator=False, latMin=53.9167-62*0.04166666, latMax=53.9167,lonMin=6.625,lonMax=6.625+118*0.04166666,min_=0,max_=0)
foMap.view()
