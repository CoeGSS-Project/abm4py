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
sys.path.append('/home/geiges/database/modules/folium/')
sys.path.append('modules/')
import seaborn as sns; sns.set()
sns.set_color_codes("dark")

plotRecords       = 0
plotCarStockBar   = 0
plotCarSales      = 0
prefPerLabel      = 0
utilPerLabel      = 0
incomePerLabel    = 0
meanPrefPerLabel  = 0
meanConsequencePerLabel = 0
printCellMaps     = 0
emissionsPerLabel = 0
printPopulation   = 0


if len(sys.argv) > 2:
    simNo = sys.argv[1]
    if os.path.isfile('output/sim' + str(simNo).zfill(4) + '/nodeOutput.hdf5'):
        path = 'output/sim' + str(simNo).zfill(4) + '/'
    elif os.path.isfile('/mnt/lustre/geiges/output/sim' + str(simNo).zfill(4) + '/nodeOutput.hdf5'):
        path = '/mnt/lustre/geiges/output/sim' + str(simNo).zfill(4) + '/'
else:
    path = 'output/sim0112/'


simParas   = loadObj(path + 'simulation_parameters')

nBurnIn       = simParas['burnIn']
withoutBurnIn = True
years         = True         # only applicable in plots without burn-in

print 'omniscient Agents: ' + str(simParas['omniscientAgents'])
print 'burn-in phase: ' + str(nBurnIn)
print 'of which omniscient burn-in: ' + str(simParas['omniscientBurnIn'])


#%% init
    


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

persMat      = np.load(path + 'agentFile_type3.npy')
persPropDict = loadObj(path + 'attributeList_type3')
hhMat        = np.load(path + 'agentFile_type2.npy')
hhPropDict   = loadObj(path + 'attributeList_type2')

enums        = loadObj(path + 'enumerations')

nSteps, nPers, nPersProp = persMat.shape
nSteps, nHhs,  nHhProp   = hhMat.shape

parameters= loadObj(path + 'simulation_parameters')
nPrior = len(enums['priorities'])

greenHH = list()
for time in range(nSteps):
    nIDsofHH = persMat[time,persMat[time,:,persPropDict['mobType'][0]]==1,persPropDict['hhID'][0]].astype(int)
    hhIDs = [np.where(hhMat[time,:,hhPropDict['name'][0]]==nID)[0][0] for nID in nIDsofHH]
    greenHH.append(np.asarray(hhIDs))

#%%
res = np.zeros([nSteps,3])
for time in range(nSteps):
    for mobType in range(3):
        res[time,mobType] = np.mean(persMat[time,persMat[time,:,persPropDict['mobType'][0]]==mobType,persPropDict['ages'][0]])

fig = plt.figure()
plt.plot(res)
#plt.title(enums['brands'][carLabel])
if withoutBurnIn:
    plt.xlim([nBurnIn,nSteps])
    if years:
        years = (nSteps - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)    
plt.legend(['Combution engine', 'Electic engine', 'other mobility types'],loc=0)
plt.title('Average age of mobility actors')  

#%%

res = np.zeros([nSteps,3])
for time in range(nSteps):
    for mobType in range(3):
        res[time,mobType] = np.mean(persMat[time,persMat[time,:,persPropDict['mobType'][0]]==mobType,persPropDict['genders'][0]])-1

fig = plt.figure()
plt.plot(res)
#plt.title(enums['brands'][carLabel])
if withoutBurnIn:
    plt.xlim([nBurnIn,nSteps])
    if years:
        years = (nSteps - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)    
plt.legend(['Combution engine', 'Electic engine', 'other mobility types'],loc=0)
plt.title('Share of women')  

      
#%%  plot car stock as bar plot
legStr = list()

carMat = np.zeros([nSteps,3])
for time in range(nSteps):
    carMat[time,:]= np.bincount(persMat[time,:,persPropDict['mobType'][0]].astype(int),minlength=3).astype(float)*200

if plotCarStockBar:
    plt.figure()
    enums   = loadObj(path + 'enumerations')
    #df = pd.read_csv(path +  'rec/' + 'carStock.csv', index_col=0)
    #nSteps = agMat.shape[0]
    nCars = np.zeros(nSteps)
    colorPal =  sns.color_palette("Set3", n_colors=len(enums['brands'].values()), desat=.8)
    for i, brand in enumerate(enums['brands'].values()):
       plt.bar(np.arange(nSteps), carMat[:,i],bottom=nCars, color =colorPal[[1,0,2][i]], width=1)
       nCars += carMat[:,i]
       legStr.append(brand)
#plt.legend(legStr)
    if withoutBurnIn:
        plt.xlim([nBurnIn,nSteps])
        if years:
            years = (nSteps - nBurnIn) / 12
            plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
    plt.subplots_adjust(top=0.96,bottom=0.14,left=0.1,right=0.80,hspace=0.45,wspace=0.1)
    plt.legend(['Combution engine', 'Electic engine', 'other mobility types'],bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

#%%
plt.figure()
plt.plot(carMat)
if withoutBurnIn:
    plt.xlim([nBurnIn,nSteps])
    if years:
        years = (nSteps - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
plt.subplots_adjust(top=0.96,bottom=0.14,left=0.1,right=0.80,hspace=0.45,wspace=0.1)
plt.legend(['Combution engine', 'Electic engine', 'other mobility types'],bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

#%% plot sales per mobility type

carSales = np.zeros([nSteps,3])
for time in range(nSteps):
    for brand in range(0,len(enums['brands'])):
        #idx = persMat[time,:,persPropDict['predMeth'][0]] == 1
        #carSales[time,:] = np.bincount(persMat[time,idx,persPropDict['type'][0]].astype(int),minlength=3).astype(float)
        carSales[time,:] = np.bincount(persMat[time,:,persPropDict['mobType'][0]].astype(int),minlength=3).astype(float)
if plotCarSales:
    fig = plt.figure()
    plt.plot(carSales)
    if withoutBurnIn:
        plt.xlim([nBurnIn,nSteps])
        if years:
            years = (nSteps - nBurnIn) / 12
            plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)    
    plt.legend(enums['brands'].values(),loc=0)
    
    plt.title('sales per mobility Type')
    
#%% number of different cars per preference type:
if False: 
    nUniqueCars = list()
    for i in range(agMat.shape[1]):
        nUniqueCars.append(len(np.unique(agMat[:,i,propDic['type'][0]])))
    
    print np.mean(nUniqueCars)
    
    for prefTyp in range(nPrior):
        nUniqueCars = list()
        idx = persMat[0,:,persPropDict['prefTyp'][0]] == prefTyp
        for i in np.where(idx)[0]:
            nUniqueCars.append(len(np.unique(agMat[:,i,propDic['type'][0]])))
        print 'Pref Type = ' + enums['priorities'][prefTyp] + ' ownes ' + str(np.mean(nUniqueCars)) +'different cars'

#%% number of buys
if False:
    x = np.diff(persMat,axis=0)
    #x = x[:,:,propDic['label']] != 0
    carBuys =np.sum(x[:,:,persPropDict['type']] > 0,axis = 0)
    np.mean(carBuys)
    np.max(carBuys)
    np.min(carBuys)
    for prefTyp in range(nPr):
        idx = persMat[0,:,persPropDict['prefTyp'][0]] == prefTyp
        
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
    del df['type']    
                     
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
    step = 9
    columns= ['']*agMat.shape[2]
    for key in propDic.keys():
        for i in propDic[key]:
            columns[i] = key
    df = pd.DataFrame(agMat[step,:,:],columns=columns)
    
#%% priority types per mobility types
    
prefTypes = np.zeros(nPrior)
for prefTyp in range(0,nPrior):
    prefTypes[prefTyp] = np.sum(persMat[0,:,persPropDict['prefTyp'][0]] == prefTyp)

if prefPerLabel:
    res = dict()
    for carLabel in range(len(enums['brands'])):
        res[carLabel] = np.zeros([nSteps,nPrior])
    
    for time in range(nSteps):
        for carLabel in range(len(enums['brands'])):
            idx = persMat[time,:, persPropDict['mobType'][0]] == carLabel
            for prefType in range(nPrior):
                res[carLabel][time,prefType] = np.sum(persMat[time,idx,persPropDict['prefTyp'][0]] == prefType) / prefTypes[prefType]
    
    legStr = list()
    for prefType in range(nPrior):
        legStr.append(enums['priorities'][prefType])
    for carLabel in range(len(enums['brands'])):
        fig = plt.figure()
        plt.plot(res[carLabel])
        plt.title(enums['brands'][carLabel])
        if withoutBurnIn:
            plt.xlim([nBurnIn,nSteps])
            if years:
                years = (nSteps - nBurnIn) / 12
                plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)    
        plt.legend(legStr,loc=0)
        
        fig.suptitle('n priority types per mobility types')
print 1

#%% utiilty per mobility type

if utilPerLabel:
    res = np.zeros([nSteps,len(enums['brands'])])
    for time in range(nSteps):
        for carLabel in range(0,len(enums['brands'])):
            idx = persMat[time,:,persPropDict['mobType'][0]] == carLabel
            res[time, carLabel] = np.mean(persMat[time,idx,persPropDict['util'][0]])
    legStr = list()
    for label in range(len(enums['brands'])):
        legStr.append(enums['brands'][label])        
    fig = plt.figure()
    plt.plot(res)
    if withoutBurnIn:
        plt.xlim([nBurnIn,nSteps])
        if years:
            years = (nSteps - nBurnIn) / 12
            plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)  
    plt.legend(legStr,loc=0)
    
    plt.title('Average utility by mobility type')

#%% income per car label
incomePerLabel= True
if incomePerLabel:
    res = np.zeros([nSteps,2])
    std = np.zeros([nSteps,2])
    for time in range(nSteps):
        for carLabel in range(len(enums['brands'])):
            idx = hhMat[time,:,hhPropDict['type'][0]] == carLabel
            res[time, 0] = np.mean(hhMat[time,:,hhPropDict['income'][0]])
            std[time, 0] = np.std(hhMat[time,:,hhPropDict['income'][0]])
            idx = np.zeros(hhMat.shape[1])
            idx[list(greenHH[time])]
            res[time, 1] = np.mean(hhMat[time,:,hhPropDict['income'][0]][greenHH[time]])
            std[time, 1] = .5* np.std(hhMat[time,:,hhPropDict['income'][0]][greenHH[time]])
    legStr = list()
    for label in range(0,len(enums['brands'])):
        legStr.append(enums['brands'][label])        
    fig = plt.figure()
    plt.plot(res)
    
    plt.fill_between(range(0,nSteps), res[:,0]+ std[:,0], res[:,0]- std[:,0], facecolor='blue', interpolate=True, alpha=0.1,)
    plt.plot(res[:,0]+ std[:,0],'b--', linewidth = 1)
    plt.plot(res[:,0]- std[:,0],'b--', linewidth = 1)
    plt.fill_between(range(0,nSteps), res[:,1]+ std[:,1], res[:,1]- std[:,1], facecolor='green', interpolate=True, alpha=0.1,)
    plt.plot(res[:,1]+ std[:,1],'g--', linewidth = 1)
    plt.plot(res[:,1]- std[:,1],'g--', linewidth = 1)
    #ax.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True)
    
    #plt.plot(res- std,'--')
    plt.title('Equalized household income')
    if withoutBurnIn:
        plt.xlim([nBurnIn,nSteps])
        if years:
            years = (nSteps - nBurnIn) / 12
            plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)  
    plt.legend(['Average Household', 'Household income using an electic car', 'Avg STD', 'Elec STD',''],loc=3) 

print ''


#%% mean priority per car label

ensembleAverage = np.mean(persMat[0,:,persPropDict['preferences']], axis = 1)
if meanPrefPerLabel:
    fig = plt.figure()
    res = dict()
    for carLabel in range(len(enums['brands'])):
        res[carLabel] = np.zeros([nSteps,nPrior])
    for time in range(nSteps):
        for carLabel in range(len(enums['brands'])):
            idx = np.where(persMat[time,:,persPropDict['mobType'][0]] == carLabel)[0]
            res[carLabel][time,:] = np.mean(persMat[np.ix_([time],idx,persPropDict['preferences'])],axis=1) / ensembleAverage
    legStr = list()
    for prefType in range(nPrior):
        legStr.append(enums['priorities'][prefType])
    
    h = list()
    for carLabel in range(len(enums['brands'])):
        plt.subplot(2,2,carLabel+1)    
        h.append(plt.plot(res[carLabel]))
        plt.title(enums['brands'][carLabel])
        #plt.legend(legStr,loc=0)
        if withoutBurnIn: 
            plt.xlim([nBurnIn,nSteps])
            if years:
                years = (nSteps - nBurnIn) / 12
                plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)          
    plt.legend(legStr,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #fig.legend(h, legStr, loc = (1,1,0,0))
    plt.tight_layout()

    fig.suptitle('mean priority per mobility type')

print 1

#%% mean consequences per mobility type
 
enums['consequences'] = {0: 'convenience', 1: 'eco-friendliness', 2: 'remaining money', 3: 'similarity'}

if meanConsequencePerLabel:
    fig = plt.figure()
    res = dict()
    for carLabel in range(len(enums['brands'])):
        res[carLabel] = np.zeros([nSteps,nPrior])
    for time in range(nSteps):
        ensembleAverage = np.mean(persMat[time,:,persPropDict['consequences']], axis = 1)
        for carLabel in range(0,len(enums['brands'])):
            idx = np.where(persMat[time,:,persPropDict['mobType'][0]] == carLabel)[0]
            res[carLabel][time,:] = np.mean(persMat[np.ix_([time],idx,persPropDict['consequences'])],axis=1) #/ ensembleAverage
    legStr = list()
    for prefType in range(nPrior):
        legStr.append(enums['consequences'][prefType])
    
    h = list()
    for carLabel in range(0,len(enums['brands'])):
        plt.subplot(2,2,carLabel+1)    
        h.append(plt.plot(res[carLabel]))
        plt.title(enums['brands'][carLabel])        
        if withoutBurnIn: 
            plt.xlim([nBurnIn,nSteps])
            if years:
                years = (nSteps - nBurnIn) / 12
                plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)          
    # plt.legend(legStr,loc=0)        
    plt.legend(legStr,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #fig.legend(h, legStr)
    plt.tight_layout()
    fig.suptitle('mean consequences per mobility type')
print 1

#%% print consequences per mobility type

if emissionsPerLabel:
    fig = plt.figure()
    res = np.zeros([nSteps,len(enums['brands'])])
    for i in range(2):
        plt.subplot(2,1,i+1)
        for time in range(nSteps):
            for carLabel in range(len(enums['brands'])):
                idx = persMat[time,:,persPropDict['mobType'][0]] == carLabel
                res[time, carLabel] = np.mean(persMat[time,idx,persPropDict['prop'][i]])
        legStr = list()
        for label in range(len(enums['brands'])):
            legStr.append(enums['brands'][label])        
        
        plt.plot(res)
        if i == 0:
            plt.title('Average emissions by mobility type')
        else:
            plt.title('Average price by mobility type')
        if withoutBurnIn: 
            plt.xlim([nBurnIn,nSteps])
            if years:
                years = (nSteps - nBurnIn) / 12
                plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)          
    plt.subplots_adjust(top=0.96,bottom=0.14,left=0.04,right=0.96,hspace=0.45,wspace=0.1)
    plt.legend(legStr,loc=0)    

#plt.show()
#%% loading cell agent file


if printCellMaps:
    #%% loading cell agent file
    
    cellMat      = np.load(path + 'agentFile_type1.npy')
    cellPropDict = loadObj(path + 'attributeList_type1')
    enums        = loadObj(path + 'enumerations')
    
    nSteps, nCells, nProp = cellMat.shape
#%%    
if printCellMaps:

    meanCon   = np.zeros([nSteps,3])
    meanEco   = np.zeros([nSteps,3])
    meanPrc   = np.zeros([nSteps,3])
    for time in range(nSteps):
        meanVect = np.mean(cellMat[time,:,cellPropDict['convenience']],axis=1)
        meanCon[time,:] = meanVect
#        meanEco[step,:] = meanVect[3:6]
#        meanPrc[step,:] = meanVect[6:]
    
    plt.figure()
    #plt.subplot(2,2,1)
    plt.plot(meanCon)
    plt.legend(enums['brands'].values())
    plt.title('convenience, mean over cells')

    #print tous
    time = 339
    #Bremen
    ids =  np.where(cellMat[time,:,cellPropDict['regionId'][0]]== 1518)
    tmp =  np.sum(cellMat[time,ids,cellPropDict['carsInCell'][1]]) /  np.sum(cellMat[time,:,cellPropDict['carsInCell']][:,ids])
    print 'Green cars per 1000 people Bremen: ' + str(tmp*1000)
    
    ids =  np.where(cellMat[time,:,cellPropDict['regionId'][0]]== 6321)
    tmp =  np.sum(cellMat[time,ids,cellPropDict['carsInCell'][1]]) /  np.sum(cellMat[time,:,cellPropDict['carsInCell']][:,ids])
    print 'Green cars per 1000 people Niedersachsen: ' + str(tmp*1000)
    
    #%%
    
    import copy
    plt.clf()
    landLayer = np.zeros(np.max(cellMat[0,:,cellPropDict['pos']]+1,axis=1).astype(int).tolist())
    for iCell in range(cellMat.shape[1]):
        x = cellMat[0,iCell,cellPropDict['pos'][0]].astype(int)
        y = cellMat[0,iCell,cellPropDict['pos'][1]].astype(int)
        landLayer[x,y] = 1
    #plt.pcolormesh(landLayer)
    landLayer = landLayer.astype(bool)
    res = landLayer*1.0
    step = 250
    test = landLayer*0
    for iBrand in range(3):
        res = landLayer*1.0
        res[np.where(landLayer)] = cellMat[step,:,cellPropDict['carsInCell'][iBrand]] / cellMat[step,:,cellPropDict['population']]
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
    sys.path.append('/media/sf_shared/python/modules')
    sys.path.append('/media/sf_shared/python/database')
    import class_map
    import matplotlib
    import folium
    foMap = class_map.Map()
    cm = matplotlib.cm.get_cmap('YlGn')
    normed_data = (arrayData - np.nanpercentile(arrayData,5)) / (np.nanpercentile(arrayData,98) - np.nanpercentile(arrayData,5))
    minmax = np.nanpercentile(arrayData,5), np.nanpercentile(arrayData,98)
    colored_data = cm(normed_data)
    foMap.addImage(colored_data, mercator=False, latMin=53.9167-62*0.04166666, latMax=53.9167,lonMin=6.625,lonMax=6.625+118*0.04166666,min_=0,max_=0)
    from branca.utilities import color_brewer
    cols = color_brewer('YlGn',6)
    cmap = folium.LinearColormap(cols,vmin=float(minmax[0]),vmax=float(minmax[1]), caption='Electric car share')
    
    foMap.map.add_child(cmap)
    foMap.view()
    #%%
    plt.figure()
    plt.clf()
    landLayer = np.zeros(np.max(cellMat[step,:,cellPropDict['pos']]+1,axis=1).astype(int).tolist())
    for iCell in range(cellMat.shape[1]):
        x = cellMat[0,iCell,cellPropDict['pos'][0]].astype(int)
        y = cellMat[0,iCell,cellPropDict['pos'][1]].astype(int)
        landLayer[x,y] = 1
    landLayer = landLayer.astype(bool)
    res = landLayer*1.0
    step = 1
    test = landLayer*0
    for iBrand in range(3):
        res = landLayer*1.0
        res[landLayer] = cellMat[step,:,cellPropDict['convenience'][iBrand]]
        test = test + res
        #res[landLayer==False] = np.nan
        plt.subplot(2,2,iBrand+1)
        plt.pcolormesh(res)
        #plt.clim([0,1])
        plt.colorbar()
        plt.title('convenience of ' + enums['brands'][iBrand] + ' cars per cells')
    print 1
#%%
if printPopulation:
    plt.figure()
    res = landLayer*1.0
    res[landLayer] = cellMat[0,:,cellPropDict['population']][0]
    plt.pcolormesh(res)
    #plt.clim([0,1])
    plt.colorbar()
    plt.title('population')
    print 1





plt.show()

#sys.path.append('/media/sf_shared/python/database')
#import class_map
#import matplotlib
#foMap = class_map.Map()
#cm = matplotlib.cm.get_cmap('YlGn')
#arrayData = res
#normed_data = (arrayData - np.nanpercentile(arrayData,5)) / (np.nanpercentile(arrayData,95) - np.nanpercentile(arrayData,5))
#self.minmax = np.nanpercentile(arrayData,5), np.nanpercentile(arrayData,95)
#colored_data = cm(normed_data)
#foMap.addImage(colored_data, mercator=False, latMin=53.9167-62*0.04166666, latMax=53.9167,lonMin=6.625,lonMax=6.625+118*0.04166666,min_=0,max_=0)
#foMap.view()
