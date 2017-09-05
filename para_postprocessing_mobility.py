#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:22:32 2017

processing records
@author: geiges, GCF
"""

import matplotlib as mpl
mpl.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tables as ta
from os.path import expanduser
home = expanduser("~")
import socket
if socket.gethostname() in ['gcf-VirtualBox', 'ThinkStation-D30']:
    sys.path.append('/media/sf_shared/python/modules/biokit')
    sys.path.append('/home/geiges/database/modules/folium/')
else:
    sys.path.append(home + '/python/modules/folium/')
    
#sys.path.append('/home/geiges/database/')
sys.path.append('modules/')
import seaborn as sns; sns.set()
from class_auxiliary import loadObj, getEnvironment

#sns.set_color_codes("dark")
sns.color_palette("Paired")
#%% init
plotRecords       = 1
plotCarStockBar   = 1
plotCarSales      = 1
salesProperties   = 1
womanSharePerMobType = 1
agePerMobType     = 1
averageCarAge     = 1
prefPerLabel      = 1
utilPerLabel      = 1
incomePerLabel    = 1
greenPerIncome    = 1
expectUtil        = 1
meanPrefPerLabel  = 1
meanESSR          = 1
meanConsequencePerLabel = 1
printCellMaps     = 1
emissionsPerLabel = 1
peerBubbleSize    = 1
doFolium          = 1
cellMovie         = 1



simNo = sys.argv[1]

if socket.gethostname() in ['gcf-VirtualBox', 'ThinkStation-D30']:

    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path = [dir_path + '/h5py/build/lib.linux-x86_64-2.7'] + sys.path 
    sys.path = [dir_path + '/mpi4py/build/lib.linux-x86_64-2.7'] + sys.path 

path = getEnvironment(None,getSimNo = False) +'sim' + str(simNo).zfill(4) + '/'
    

simParas   = loadObj(path + 'simulation_parameters')

nBurnIn       = simParas['burnIn']
withoutBurnIn = False
years         = True         # only applicable in plots without burn-in

print 'omniscient Agents: ' + str(simParas['omniscientAgents'])
print 'burn-in phase: ' + str(nBurnIn)
print 'of which omniscient burn-in: ' + str(simParas['omniscientBurnIn'])







#%% loading agent file
     

h5file = ta.open_file(path + 'nodeOutput.hdf5', mode = "r")
enums        = loadObj(path + 'enumerations')
persPropDict = loadObj(path + 'attributeList_type3')
hhPropDict   = loadObj(path + 'attributeList_type2')
parameters   = loadObj(path + 'simulation_parameters')
parameters['timeStepMag'] = int(np.ceil(np.log10(parameters['nSteps'])))

def getData(parameters, nodeType,timeStep):
    
    dataPath = '/' + str(nodeType)+ '/' + str(timeStep).zfill(parameters['timeStepMag'])
    node = h5file.get_node(dataPath)
    array = node.read()
            
    return array

persMatStep = getData(parameters,3,0)

persMat = np.zeros([parameters['nSteps'], persMatStep.shape[0],persMatStep.shape[1]])
for step in range(parameters['nSteps']):
    persMat[step,:,:] = getData(parameters,3,step)

hhMatStep = getData(parameters,2,0)

hhMat = np.zeros([parameters['nSteps'], hhMatStep.shape[0],hhMatStep.shape[1]])
for step in range(parameters['nSteps']):
    hhMat[step,:,:] = getData(parameters,2,step)

nSteps, nPers, nPersProp = persMat.shape
nSteps, nHhs,  nHhProp   = hhMat.shape    
nPrior = len(enums['priorities'])

del persMatStep, hhMatStep


if plotRecords:
    import tables as ta
    
    h5File  = ta.File(path + '/globals.hdf5', 'r')
    glob    = h5File.get_node('/glob/')
    calData = h5File.get_node('/calData/')
    for data in glob._f_iter_nodes():
        plt.figure()
        plt.plot(data.read())
        plt.title(data.name)
        plt.savefig(path + data.name)

        if data.name in calData._v_children.keys(): 
            group = h5File.get_node('/calData/' + data.name)
            cData = group.read()
            
            plt.gca().set_prop_cycle(None)
            
            for i in range(1,data.shape[1]):
                plt.plot(cData[:,0], cData[:,i],'o')
                print cData[:,i]
                
        if withoutBurnIn:
            plt.xlim([nBurnIn,nSteps])
        if years:
            years = (nSteps - nBurnIn) / 12
            plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)    
            if 'stock' in data.name:
                plt.yscale('log')
        plt.savefig(path + data.name)
# Auxiliary function

if False:
    #%%
    timeStep = 120
    plt.figure()
    idx = 0
    for prefType in range(3):
        boolMask = persMat[timeStep,:,persPropDict['prefTyp'][0]] == prefType
        x = persMat[timeStep,boolMask,persPropDict['consequences'][idx]]
        y = persMat[timeStep,boolMask,persPropDict['preferences'][idx]]

        plt.scatter(x,y,s=10)
    plt.xlabel(['comfort','environmental','remainig money','similarity'][idx])
    plt.ylabel(['convenience','ecology','money','innovation'][idx])
    
    #%%
if False:
    def cobbDouglasUtil(x, alpha):
        utility = 1.
        factor = 100        
        for i in range(len(x)):
            utility *= (factor*x[i])**alpha[i]
        if np.isnan(utility) or np.isinf(utility):
            import pdb
            pdb.set_trace()
        return utility

    timeStep = 0

    idx = 100
    consequences = persMat[timeStep,idx,persPropDict['consequences']]
    priorities   = persMat[timeStep,idx,persPropDict['preferences']]

    print cobbDouglasUtil(consequences, priorities)
    print persMat[timeStep,idx,persPropDict['util']]
    #%%

if averageCarAge:
    res = np.zeros([nSteps,3])
    std = np.zeros([nSteps,3])
    for time in range(nSteps):
        for mobType in range(3):
            res[time,mobType] = np.mean(persMat[time,persMat[time,:,persPropDict['mobType'][0]]==mobType,persPropDict['lastAction'][0]])/12
            std[time,mobType] = np.std(persMat[time,persMat[time,:,persPropDict['mobType'][0]]==mobType,persPropDict['lastAction'][0]]/12)
    
    
    fig = plt.figure()
    plt.fill_between(range(0,nSteps), res[:,0]+ std[:,0], res[:,0]- std[:,0], facecolor='blue', interpolate=True, alpha=0.1,)
    plt.fill_between(range(0,nSteps), res[:,1]+ std[:,1], res[:,1]- std[:,1], facecolor='green', interpolate=True, alpha=0.1,)
    plt.fill_between(range(0,nSteps), res[:,2]+ std[:,2], res[:,2]- std[:,2], facecolor='red', interpolate=True, alpha=0.1,)
    plt.plot(res)
    #plt.title(enums['brands'][carLabel])
    if withoutBurnIn:
        plt.xlim([nBurnIn,nSteps])
    if years:
        years = (nSteps - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)    
    plt.legend(['Combution engine', 'Electic engine', 'other mobility types'],loc=0)
    plt.title('Average fleet age per mobility type [years]')  
    plt.savefig(path + 'fleetAge')

if meanESSR:
    res = np.zeros([nSteps,3])
    for time in range(nSteps):
        for mobType in range(3):
            res[time,mobType] = np.mean(persMat[time,persMat[time,:,persPropDict['mobType'][0]]==mobType,persPropDict['ESSR'][0]])
    
    fig = plt.figure()
    plt.plot(res)
    #plt.title(enums['brands'][carLabel])
    if withoutBurnIn:
        plt.xlim([nBurnIn,nSteps])
    if years:
        years = (nSteps - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)    
    plt.legend(['Combution engine', 'Electic engine', 'other mobility types'],loc=0)
    plt.title('Average relative effective sample size')  
    plt.savefig(path + 'ESSR|mobType')

    res = np.zeros([nSteps,3])
    for time in range(nSteps):
        for mobType in range(3):
            res[time,mobType] = np.mean(persMat[time,persMat[time,:,persPropDict['prefTyp'][0]]==mobType,persPropDict['ESSR'][0]])
    
    fig = plt.figure()
    plt.plot(res)
    #plt.title(enums['brands'][carLabel])
    if withoutBurnIn:
        plt.xlim([nBurnIn,nSteps])
    if years:
        years = (nSteps - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)    
    plt.legend(['Combution engine', 'Electic engine', 'other mobility types'],loc=0)
    plt.title('Average relative effective sample size')  
    plt.savefig(path + 'ESSR|prefType')
        
    
    
#%%
data = persMat[time,:,persPropDict['preferences']]

plt.figure()
plt.subplot(2,2,1)
plt.scatter(data[0,:],data[1,:],s=2)
plt.xlabel('conv')
plt.ylabel('eco')

plt.subplot(2,2,2)
plt.scatter(data[2,:],data[1,:],s=2)
plt.xlabel('eco')
plt.ylabel('money')

plt.subplot(2,2,3)
plt.scatter(data[0,:],data[2,:],s=2)
plt.xlabel('conv')
plt.ylabel('money')

plt.subplot(2,2,4)
plt.scatter(data[2,:],data[3,:],s=2)
plt.xlabel('eco')
plt.ylabel('inno')

#%%%
if peerBubbleSize:
    res = np.zeros([nSteps,3])
    #std = np.zeros([nSteps,3])
    for time in range(nSteps):
        for mobType in range(3):
            res[time,mobType] = np.mean(persMat[time,persMat[time,:,persPropDict['mobType'][0]]==mobType,persPropDict['peerBubbleHeterogeneity'][0]])
            #std[time,mobType] = np.std(persMat[time,persMat[time,:,persPropDict['mobType'][0]]==mobType,persPropDict['age'][0]])
    fig = plt.figure()
    
    plt.plot(res)
    #plt.title(enums['brands'][carLabel])
    if withoutBurnIn:
        plt.xlim([nBurnIn,nSteps])
    if years:
        years = (nSteps - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)    
    plt.legend(['Combution engine', 'Electic engine', 'other mobility types'],loc=0)
    plt.title('Average Bubble size')  
    plt.savefig(path + 'socialBubbleSize|mobType')

    res = np.zeros([nSteps,3])
    #std = np.zeros([nSteps,3])
    for time in range(nSteps):
        for mobType in range(3):
            res[time,mobType] = np.mean(persMat[time,persMat[time,:,persPropDict['prefTyp'][0]]==mobType,persPropDict['peerBubbleHeterogeneity'][0]])
            #std[time,mobType] = np.std(persMat[time,persMat[time,:,persPropDict['mobType'][0]]==mobType,persPropDict['age'][0]])
    fig = plt.figure()
    
    plt.plot(res)
    #plt.title(enums['brands'][carLabel])
    if withoutBurnIn:
        plt.xlim([nBurnIn,nSteps])
    if years:
        years = (nSteps - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)    
    plt.legend(['Combution engine', 'Electic engine', 'other mobility types'],loc=0)
    plt.title('Average Bubble size')  
    plt.savefig(path + 'socialBubbleSize|prefType')
    
#%%
if agePerMobType:
    res = np.zeros([nSteps,3])
    #std = np.zeros([nSteps,3])
    for time in range(nSteps):
        for mobType in range(3):
            res[time,mobType] = np.mean(persMat[time,persMat[time,:,persPropDict['mobType'][0]]==mobType,persPropDict['age'][0]])
            #std[time,mobType] = np.std(persMat[time,persMat[time,:,persPropDict['mobType'][0]]==mobType,persPropDict['age'][0]])
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
    plt.savefig(path + 'agePerMobType')



#%%
if womanSharePerMobType:
    res = np.zeros([nSteps,3])
    for time in range(nSteps):
        for mobType in range(3):
            res[time,mobType] = np.mean(persMat[time,persMat[time,:,persPropDict['mobType'][0]]==mobType,persPropDict['gender'][0]])-1
    
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
    plt.savefig(path + 'womanShareGreen')


if expectUtil  == 1:
    fig = plt.figure()
    data = np.asarray(persMat[:,:,persPropDict['commUtil']])
    
    plt.plot(np.mean(data,axis=1))
    if withoutBurnIn:
        plt.xlim([nBurnIn,nSteps])
    if years:
        years = (nSteps - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)    
    
    plt.title("Expectations for mobility types")    
    plt.legend(['Combution engine', 'Electic engine', 'other mobility types'],loc=0)
    plt.savefig(path + 'expectedUtility')
    #%%
if expectUtil  == 1:
    fig = plt.figure(figsize=(15,10))
    data = np.asarray(persMat[:,:,persPropDict['commUtil']])
    
    plt.plot(np.mean(data,axis=1),linewidth=3)
    legStr = list()
    for label in range(len(enums['brands'])):
        legStr.append(enums['brands'][label])  
    style = ['-','-', ':','--','-.']
    ledAdd = [' (all)', ' (convenience)', ' (ecology)', ' (money)', ' (immi)']
    newLegStr = []
    newLegStr += [ string + ledAdd[0] for string in  legStr]
    for prefType in range(4):
        plt.gca().set_prop_cycle(None)
        boolMask = np.asarray(persMat[0,:,persPropDict['prefTyp']]) == prefType
        plt.plot(np.mean(data[:,boolMask[0],0],axis=1),style[prefType+1])
        plt.plot(np.mean(data[:,boolMask[0],1],axis=1),style[prefType+1])
        plt.plot(np.mean(data[:,boolMask[0],2],axis=1),style[prefType+1])
        newLegStr += [ string + ledAdd[prefType+1] for string in  legStr]
    if withoutBurnIn:
        plt.xlim([nBurnIn,nSteps])
    if years:
        years = (nSteps - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)    
    
    plt.title("Expectations for mobility types ")    
    plt.legend(newLegStr,loc=0)
    plt.savefig(path + 'expectedUtility2')    
    
    
    #%%
    
    legStr = list()
    for label in range(len(enums['brands'])):
        legStr.append(enums['brands'][label])  
    fig = plt.figure(figsize=(15,10))
    res = np.zeros([nSteps,len(enums['brands']), 5])
    for time in range(nSteps):
        for carLabel in range(0,len(enums['brands'])):
            boolMask = persMat[time,:,persPropDict['mobType'][0]] == carLabel
            res[time, carLabel, 0] = np.mean(persMat[time,boolMask,persPropDict['util'][0]])    
            for prefType in range(4):
                boolMask2 = np.asarray(persMat[0,:,persPropDict['prefTyp'][0]]) == prefType
                res[time, carLabel, prefType+1] = np.mean(persMat[time,boolMask & boolMask2,persPropDict['util'][0]])    
    
    newLegStr= list()
  
#%%  plot car stock as bar plot
legStr = list()

carMat = np.zeros([nSteps,3])
for time in range(nSteps):
    carMat[time,:]= np.bincount(persMat[time,:,persPropDict['mobType'][0]].astype(int),minlength=3).astype(float)
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
    plt.legend(legStr,bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    plt.savefig(path + 'carStock')
#%% plot sales per mobility type

carSales = np.zeros([nSteps,3])
for time in range(nSteps):
    for brand in range(0,len(enums['brands'])):
        #idx = persMat[time,:,persPropDict['predMeth'][0]] == 1
        #carSales[time,:] = np.bincount(persMat[time,idx,persPropDict['type'][0]].astype(int),minlength=3).astype(float)
        boolMask = persMat[time,:,persPropDict['lastAction'][0]]== 0
        carSales[time,:] = np.bincount(persMat[time,boolMask,persPropDict['mobType'][0]].astype(int),minlength=3).astype(float)

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
    plt.savefig(path + 'salesPerMobType')
    
#%%
if salesProperties:
    plt.figure(figsize=[15,10])
    
    
    for brand in range(0,len(enums['brands'])):
        plt.subplot(2,2,brand+1)
        res = np.zeros([nSteps,len(enums['priorities'])])
        for time in range(nSteps):
            boolMask = persMat[time,:,persPropDict['lastAction'][0]]== 0
            boolMask2 = persMat[time,:,persPropDict['mobType'][0]]== brand
            res[time,:] = np.mean(persMat[np.ix_([time],boolMask & boolMask2,persPropDict['preferences'])],axis=1)
            
        plt.plot(res)    
        plt.legend(enums['priorities'].values())
        plt.title(enums['brands'][brand])
        if withoutBurnIn:
            plt.xlim([nBurnIn,nSteps])
        if years:
            years = (nSteps - nBurnIn) / 12
            plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)    
    plt.suptitle('preferences of current sales per time')
    plt.savefig(path + 'buyerPriorities')#
    
    #%%
if salesProperties:
    plt.figure(figsize=[15,10])
    
    propList = ['age', 'commUtil','lastAction', 'util']
    for i,prop in enumerate(propList):
        plt.subplot(2,2,i+1)
        res = np.zeros([nSteps,len(enums['brands'])])
    
        for brand in range(0,len(enums['brands'])):
            for time in range(nSteps):
                boolMask = persMat[time,:,persPropDict['lastAction'][0]]== 0
                boolMask2 = persMat[time,:,persPropDict['mobType'][0]]== brand
                if prop in ['lastAction']:
                    res[time,brand] = np.mean(persMat[np.ix_([np.max([0,time-1])],boolMask & boolMask2,persPropDict[prop]) ],axis=1)
                elif prop in ['commUtil']: 
                    res[time,brand] = np.mean(persMat[np.ix_([np.max([0,time-1])],boolMask & boolMask2,[persPropDict[prop][brand]]) ],axis=1)
                else:
                    res[time,brand] = np.mean(persMat[np.ix_([time],boolMask & boolMask2,persPropDict[prop]) ],axis=1)
                
        plt.plot(res)    
        plt.legend(enums['brands'].values(),loc=0)
        plt.title(prop)
        if withoutBurnIn:
            plt.xlim([nBurnIn,nSteps])
        if years:   
            years = (nSteps - nBurnIn) / 12
            plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)    
    plt.suptitle('preferences of current sales per time')
    plt.savefig(path + 'buyerProperties')
    
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
    plt.savefig(path + 'utilPerMobType')




#%%    
if utilPerLabel:
    legStr = list()
    for label in range(len(enums['brands'])):
        legStr.append(enums['brands'][label])  
    fig = plt.figure(figsize=(15,10))
    res = np.zeros([nSteps,len(enums['brands']), 5])
    for time in range(nSteps):
        for carLabel in range(0,len(enums['brands'])):
            boolMask = persMat[time,:,persPropDict['mobType'][0]] == carLabel
            res[time, carLabel, 0] = np.mean(persMat[time,boolMask,persPropDict['util'][0]])    
            for prefType in range(4):
                boolMask2 = np.asarray(persMat[0,:,persPropDict['prefTyp'][0]]) == prefType
                res[time, carLabel, prefType+1] = np.mean(persMat[time,boolMask & boolMask2,persPropDict['util'][0]])    
    
    newLegStr= list()
    
    
    style = ['-','-', ':','--','-.']
    plt.gca().set_prop_cycle(None)            
    plt.plot(res[:,:,0],style[0], linewidth =3)
    newLegStr += [ string + ' (all)' for string in  legStr]
    plt.gca().set_prop_cycle(None)            
    plt.plot(res[:,:,1],style[1])
    newLegStr += [ string + ' (convenience)' for string in  legStr]
    plt.gca().set_prop_cycle(None)            
    plt.plot(res[:,:,2],style[2])
    newLegStr += [ string + ' (ecology)' for string in  legStr]
    plt.gca().set_prop_cycle(None)            
    plt.plot(res[:,:,3],style[3])
    newLegStr += [ string + ' (money)' for string in  legStr]
    plt.gca().set_prop_cycle(None) 
    plt.plot(res[:,:,4],style[4])
    newLegStr += [ string + ' (immi)' for string in  legStr]
    plt.legend(newLegStr,loc=0, ncol=5)
    plt.title('Average utility by mobility type -=conv | ..=eco | --=mon ')
    if withoutBurnIn:
        plt.xlim([nBurnIn,nSteps])
    if years:
        years = (nSteps - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)  
    plt.ylim([np.nanpercentile(res,1), np.nanpercentile(res,99)])
    plt.tight_layout()
    plt.savefig(path + 'utilPerMobType2')
#%%  green cars per income




if greenPerIncome or incomePerLabel:


    hhglob2datIdx = dict()
    for idx in range(hhMat.shape[1]):
        hhglob2datIdx[hhMat[0,idx,hhPropDict['gID'][0]]] = idx


    greenHH = list()
    for time in range(nSteps):
        print time,
        gIDsofHH = persMat[time,persMat[time,:,persPropDict['mobType'][0]]==1,persPropDict['hhID'][0]].astype(int)
        hhIDs = [hhglob2datIdx[gID] for gID in gIDsofHH]
        greenHH.append(np.asarray(hhIDs))


if greenPerIncome:        
    res = np.zeros([nSteps,2])
    std = np.zeros([nSteps,2])
    plt.figure()
    for i,year in enumerate([2005, 2010, 2015, 2020, 2025, 2029]):
        
        plt.subplot(2,3,i+1)
        time = (nBurnIn + (year-2005) * 12)-1
        
        #for carLabel in range(len(enums['brands'])):
        if time < hhMat.shape[0]:
            #idx = hhMat[time,:,hhPropDict['type'][0]] == carLabel
            plt.hist(hhMat[time,:,hhPropDict['income'][0]],bins=np.linspace(0,11000,30), color='black')
            plt.title(str(year))
            if len(greenHH[time]) > 0:
                plt.hist(hhMat[time,:,hhPropDict['income'][0]][greenHH[time]],bins=np.linspace(0,11000,30), color='green')
                
            else:
                pass
        if i < 2:
            plt.xticks([])  
    plt.savefig(path + 'greenPerIncomeClass')
if incomePerLabel:
    
       
    res = np.zeros([nSteps,2])
    std = np.zeros([nSteps,2])
    for time in range(nSteps):
        for carLabel in range(len(enums['brands'])):
            #idx = hhMat[time,:,hhPropDict['type'][0]] == carLabel
            res[time, 0] = np.mean(hhMat[time,:,hhPropDict['income'][0]])
            std[time, 0] = np.std(hhMat[time,:,hhPropDict['income'][0]])
            idx = np.zeros(hhMat.shape[1])
            if len(greenHH[time]) > 0:
                res[time, 1] = np.mean(hhMat[time,:,hhPropDict['income'][0]][greenHH[time]])
                std[time, 1] = .5* np.std(hhMat[time,:,hhPropDict['income'][0]][greenHH[time]])
            else:
                res[time, 1] = np.nan
                std[time, 1] = np.nan
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
    
    plt.savefig(path + 'incomePerMobType')
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
    plt.savefig(path + 'meanPriorityPerMobType')
print 1

#%% mean consequences per mobility type
 
#enums['consequences'] = {0: 'convenience', 1: 'eco-friendliness', 2: 'remaining money', 3: 'similarity'}

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
    plt.savefig(path + 'meanConsequencesPerMobType')
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
    plt.savefig(path + 'meanEmmissionsPerMobType')
#plt.show()
#%% loading cell agent file


if printCellMaps:
    
    
    def cellDataAsMap(landLayer, posArray, cellData):

        array = landLayer*1.0
        #res[posArray[0],posArray[1]] = cellMat[step,:,cellPropDict['carsInCell'][iBrand]] / cellMat[step,:,cellPropDict['population']]
    
        array[posArray[0],posArray[1]] = cellData
                
        return array

    #%% loading cell agent file
    cellMatStep = getData(parameters,1,0)

    cellMat = np.zeros([parameters['nSteps'], cellMatStep.shape[0],cellMatStep.shape[1]])
    for step in range(parameters['nSteps']):
        cellMat[step,:,:] = getData(parameters,1,step)
    
    nSteps, nCells, nProp = persMat.shape
#    nSteps, nHhs,  nHhProp   = hhMat.shape   
#    cellMat      = np.load(path + 'agentFile_type1.npy')
    cellPropDict = loadObj(path + 'attributeList_type1')
#    enums        = loadObj(path + 'enumerations')
#    
#    nSteps, nCells, nProp = cellMat.shape
    posArray = cellMat[0,:,cellPropDict['pos']].astype(int)
    #posList = [tuple(posArray[:, i]) for i in range(posArray.shape[1])]
    #locIdxList = np.ravel_multi_index((posArray[0], posArray[1]),population.shape)
#%%    


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
    plt.savefig(path + 'conveniencePerCell')
    #%%
if cellMovie:
    from matplotlib.colors import ListedColormap
    my_cmap = ListedColormap(sns.color_palette('BuGn_d').as_hex())
    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage
    
    landLayer = np.zeros(np.max(cellMat[0,:,cellPropDict['pos']]+1,axis=1).astype(int).tolist())
    for iCell in range(cellMat.shape[1]):
        x = cellMat[0,iCell,cellPropDict['pos'][0]].astype(int)
        y = cellMat[0,iCell,cellPropDict['pos'][1]].astype(int)
        landLayer[x,y] = 1
    #plt.pcolormesh(landLayer)
    landLayer = landLayer.astype(bool)
    
    bounds = dict()
    plotDict = dict()
    tt = 0
    fig = plt.figure()
    plt.clf()
        
    res = landLayer*1.
    res[res == 0] = np.nan
    for iBrand in range(3):
        data = cellMat[-1,:,cellPropDict['carsInCell'][iBrand]] / cellMat[-1,:,cellPropDict['population'][0]] * 1000
        data[np.isinf(data)] = 0
        bounds[iBrand] = [np.nanmin(data), np.nanpercentile(data,95)]
        print bounds[iBrand]
        res[posArray[0],posArray[1]] = cellMat[tt,:,cellPropDict['carsInCell'][iBrand]] / cellMat[tt,:,cellPropDict['population'][0]] * 1000
        plt.subplot(2,2,iBrand+1)
        plotDict[iBrand] = plt.imshow(np.flipud(res), cmap=my_cmap) 
        plt.colorbar()
        plt.clim(bounds[iBrand])
    plt.tight_layout()
    
    def make_frame(t):
        #print t
        tt = int(t*15) + nBurnIn
        for iBrand in range(3):
            
            res = landLayer*1.
            #print(type(tt))
            #print tt
            #print cellMat[t,:,cellPropDict['carsInCell'][iBrand]]
            res[posArray[0],posArray[1]] = cellMat[tt,:,cellPropDict['carsInCell'][iBrand]] / cellMat[tt,:,cellPropDict['population'][0]] * 1000

            plotDict[iBrand].set_data(res)
            #plt.clim([0,1])
            #plt.colorbar()
            #plt.clim(bounds[iBrand])
            plt.title(enums['brands'][iBrand] + ' cars per cells')
            #print iBrand
        plt.tight_layout()
        plt.suptitle('TimeStep' + str(tt))
        return mplfig_to_npimage(fig)
    
    timeDur = (nSteps - nBurnIn)/15
    animation = VideoClip(make_frame, duration = timeDur)
    animation.write_gif(path + "svm.gif", fps=15)
    
    #dsfg
    

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
    step = nSteps-1
    test = landLayer*0
    for iBrand in range(3):
        res = landLayer*1.0
        res[posArray[0],posArray[1]] = cellMat[step,:,cellPropDict['carsInCell'][iBrand]] / cellMat[step,:,cellPropDict['population'][0]]
        #print np.max(res)
        test = test + res
        if iBrand == 1:
            arrayData = copy.copy(res)
        #res[landLayer==False] = np.nan
        plt.subplot(2,2,iBrand+1)
        plt.pcolormesh(np.flipud(res))
        #plt.clim([0,1])
        plt.colorbar()
        plt.title(enums['brands'][iBrand] + ' cars per cells')
        plt.savefig(path + 'carsPerCell')
    print 1
    if doFolium:  
        bounds = (0,1)
        sys.path.append('/media/sf_shared/python/modules')
        #sys.path.append('/media/sf_shared/python/database')
        import class_map
        import matplotlib
        import folium
        foMap = class_map.Map('toner', location = [53.9167-62*0.04166666, 13.9167], zoom_start = 7)
        geoFile = 'resources_NBH/regions.shp'
        import mod_geojson as gjs
        geoData = gjs.extractShapes(geoFile, [1518, 1520, 6321] , 'numID',None)
        foMap.map.choropleth(geo_str=geoData, fill_color=None,fill_opacity=0.00, line_color='green', line_weight=3)
        for year in [2005, 2010, 2015, 2020, 2025, 2030]:
            
            step = (nBurnIn + (year-2005) * 12)-1
            if step > nSteps:
                break
            # green cars per 1000 people
            data = cellMat[step,:,cellPropDict['carsInCell'][1]] / cellMat[step,:,cellPropDict['population']]*1000
            arrayData = cellDataAsMap(landLayer,posArray, data)
            # green cars per 1000 people
            #arrayData = cellMat[step,:,cellPropDict['carsInCell'][1]] / (cellMat[step,:,cellPropDict['carsInCell'][0]] + cellMat[step,:,cellPropDict['carsInCell'][1]])
            arrayData[np.isnan(arrayData)] = 0
            bounds = np.min([bounds[0], np.nanpercentile(arrayData,2)]) , np.max([bounds[1], np.nanpercentile(arrayData,98)])
        for year in  [2005, 2010, 2015, 2020, 2025, 2030]:
            step = (nBurnIn + (year-2005) * 12)-1
            if step > nSteps:
                break    
            # green cars per 1000 people
            data = cellMat[step,:,cellPropDict['carsInCell'][1]] / cellMat[step,:,cellPropDict['population']]*1000
            
            # green cars per 1000 people
            #data = cellMat[step,:,cellPropDict['carsInCell'][1]] / (cellMat[step,:,cellPropDict['carsInCell'][0]] + cellMat[step,:,cellPropDict['carsInCell'][1]])
            
            arrayData = cellDataAsMap(landLayer,posArray, data)
            arrayData[np.isnan(arrayData)] = 0
            cm = matplotlib.cm.get_cmap('YlGn')
            normed_data = (arrayData - bounds[0]) / (bounds[1]- bounds[0])
            #minmax = np.nanpercentile(arrayData,2), np.nanpercentile(arrayData,98)
            colored_data = cm(normed_data)
            foMap.addImage(colored_data, mercator=False, latMin=53.9167-62*0.04166666, latMax=53.9167,lonMin=6.625,lonMax=6.625+118*0.04166666,min_=0,max_=0, name = str(year))
            
        from branca.utilities import color_brewer
        cols = color_brewer('YlGn',6)
        cmap = folium.LinearColormap(cols,vmin=float(bounds[0]),vmax=float(bounds[1]), caption='Electric cars per 1000 people')
        
        
        foMap.map.add_child(cmap)
        #foMap.map.add_child(xx)
        foMap.save(path +  'carsPer1000.html')
        
        
        
        bounds = (0,1)
        sys.path.append('/media/sf_shared/python/modules')
        #sys.path.append('/media/sf_shared/python/database')
        import class_map
        import matplotlib
        import folium
        foMap = class_map.Map('toner',location = [53.9167-62*0.04166666, 13.9167], zoom_start = 7)
        geoFile = 'resources_NBH/regions.shp'
        import mod_geojson as gjs
        geoData = gjs.extractShapes(geoFile, [1518, 1520, 6321] , 'numID',None)
        foMap.map.choropleth(geo_str=geoData, fill_color=None,fill_opacity=0.00, line_color='green', line_weight=3)
        for year in [2005, 2010, 2015, 2020, 2025, 2030]:
            
            step = (nBurnIn + (year-2005) * 12)-1
            if step > nSteps:
                break
            # green cars per 1000 people
            data = cellMat[step,:,cellPropDict['carsInCell'][1]] / cellMat[step,:,cellPropDict['population']]*1000
            #arrayData = cellDataAsMap(landLayer,posArray, data)
            # green cars per 1000 people
            arrayData = cellMat[step,:,cellPropDict['carsInCell'][1]] / (cellMat[step,:,cellPropDict['carsInCell'][0]] + cellMat[step,:,cellPropDict['carsInCell'][1]])
            arrayData[np.isnan(arrayData)] = 0
            bounds = (0, 1)
        for year in [2005, 2010, 2015, 2020, 2025, 2030]:
            step = (nBurnIn + (year-2005) * 12)-1
            if step > nSteps:
                break    
            # green cars per 1000 people
            #data = cellMat[step,:,cellPropDict['carsInCell'][1]] / cellMat[step,:,cellPropDict['population']]*1000
            # green cars per 1000 people
            data = cellMat[step,:,cellPropDict['carsInCell'][1]] / (cellMat[step,:,cellPropDict['carsInCell'][0]] + cellMat[step,:,cellPropDict['carsInCell'][1]])
            arrayData = cellDataAsMap(landLayer,posArray, data)
            arrayData[np.isnan(arrayData)] = 0
            cm = matplotlib.cm.get_cmap('YlGn')
            normed_data = (arrayData - bounds[0]) / (bounds[1]- bounds[0])
            #minmax = np.nanpercentile(arrayData,2), np.nanpercentile(arrayData,98)
            colored_data = cm(normed_data)
            foMap.addImage(colored_data, mercator=False, latMin=53.9167-62*0.04166666, latMax=53.9167,lonMin=6.625,lonMax=6.625+118*0.04166666,min_=0,max_=0, name = str(year))
            
        from branca.utilities import color_brewer
        cols = color_brewer('YlGn',6)
        cmap = folium.LinearColormap(cols,vmin=float(bounds[0]),vmax=float(bounds[1]), caption='Electric car share')
        
        
        foMap.map.add_child(cmap)
        #foMap.map.add_child(xx)
        foMap.save(path +  'greenCarShare.html')
        
        
    #%%
    plt.figure()
    #plt.colormap('jet')
    plt.imshow(simParas['landLayer'])
    plt.colorbar()
    plt.figure()
    plt.clf()
    step = nSteps-1
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
        res[posArray[0],posArray[1]] = cellMat[step,:,cellPropDict['convenience'][iBrand]]
        test = test + res
        #res[landLayer==False] = np.nan
        plt.subplot(2,2,iBrand+1)
        plt.pcolormesh(np.flipud(res))
        #plt.clim([0,1])
        plt.colorbar()
        plt.title('convenience of ' + enums['brands'][iBrand] + ' cars per cells')
    plt.savefig(path + 'conveniencePerCell')
#%%
    plt.figure()
    res = landLayer*1.0
    
    
    res[posArray[0],posArray[1]] = cellMat[0,:,cellPropDict['population']][0]
    
    plt.imshow(np.flipud(res))
    plt.colorbar()
    #plt.clim([0,1])
    
    plt.title('population')
    plt.savefig(path + 'population')
plt.show()

print 'All done'
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
