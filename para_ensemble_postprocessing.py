#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:06:18 2017

@author: gcf

"""

import numpy as np
import tables as ta
from class_auxiliary import loadObj, getEnvironment
simulations = []
import os

#import matplotlib as mpl
#mpl.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.color_palette("Paired")
sns.set_color_codes("dark")
#%%

ensNo =7
#simulations= [222,223,224,225,226,227,228,229,230,231]

if ensNo == 1:
    realStart, realEnd = 2500, 2504
if ensNo == 2:
    realStart, realEnd = 295, 324
if ensNo == 3:
    realStart, realEnd = 325, 354
if ensNo == 6:
    realStart, realEnd = 528, 627 
if ensNo == 7:
    realStart, realEnd = 543, 744 

for i,simNo in enumerate(range(realStart,realEnd)):
    path = getEnvironment(None,getSimNo = False) +'sim' + str(simNo).zfill(4) + '/'
    print path
    if os.path.isfile(path+ '/globals.hdf5') and os.path.isfile(path+ '/simulation_parameters.pkl'):
        simulations.append(simNo)

print 'available simulations: ' +str(simulations) 


simDataNS = np.zeros([560,3,len(simulations)])
simDataBr = np.zeros([560,3,len(simulations)])
simDataHb = np.zeros([560,3,len(simulations)])

keys = []

#for key in paraDict.keys():
#    if type(paraDict[key]) in [int, float]:
#        keys.append(key)
#        
#paraDict = { key : paraDict[key] for key in keys}
        
#pd.DataFrame.from_dict(paraDict)

#df = pd.DataFrame
simNo = simulations[0]
path = getEnvironment(None,getSimNo = False) +'sim' + str(simNo).zfill(4) + '/'
paraDict = loadObj(path+ 'simulation_parameters')

for key in paraDict.keys():
    if type(paraDict[key]) in [int, float]:
        keys.append(key)
                
for i, simNo in enumerate(simulations):
    path = getEnvironment(None,getSimNo = False) +'sim' + str(simNo).zfill(4) + '/'
    h5File  = ta.File(path + '/globals.hdf5', 'r')
    
    paraDict = loadObj(path+ 'simulation_parameters')
    paraDict = { key : paraDict[key] for key in keys}
    parameters = pd.Series(paraDict,index=paraDict.keys())
    if i == 0:
        
        paraDf = parameters.to_frame()        
        
    else:
        paraDf[i] =  parameters.to_frame()    
    
    group = h5File.get_node('/glob/stockNiedersachsen')
    simDataNS[:,:,i] = group.read()
    group = h5File.get_node('/glob/stockBremen')
    simDataBr[:,:,i] = group.read()
    group = h5File.get_node('/glob/stockHamburg')
    simDataHb[:,:,i] = group.read()

np.save('output/E' + str(ensNo) + '_stock_nie.npy',simDataNS)
np.save('output/E' + str(ensNo) + '_stock_bre.npy',simDataBr)
np.save('output/E' + str(ensNo) + '_stock_ham.npy',simDataHb)
paraDf.to_json('output/E' + str(ensNo) + '_parameters.json')

group = h5File.get_node('/calData/stockNiedersachsen')
dataNS = group.read()
group = h5File.get_node('/calData/stockBremen')
dataBr = group.read()
group = h5File.get_node('/calData/stockHamburg')
dataHb = group.read()

np.save('output/E' + str(ensNo) + '_cal_stock_nie.npy',dataNS)
np.save('output/E' + str(ensNo) + '_cal_stock_bre.npy',dataBr)
np.save('output/E' + str(ensNo) + '_cal_stock_ham.npy',dataHb)

if False:
#%%s
    ensNo =7
    
    nSteps  = 560
    nBurnIn = 200
    factor = 5
    log    = 1
    lineWidth = .2

    years = (nSteps - nBurnIn) / 12 / factor
    plt.clf()
    plt.subplot(1,3,1) 
    plt.title('Niedersachsen')         
    data = np.load('poznan_out/E' + str(ensNo) + '_stock_nie.npy')    
    calData = np.load('poznan_out/E' + str(ensNo) + '_cal_stock_nie.npy')
    paraDf = pd.read_json('poznan_out/E' + str(ensNo) + '_parameters.json')
    
    plt.plot(data[:,0,:], 'b', linewidth=lineWidth)
    plt.plot(calData[:,0], calData[:,1],'bo')
    plt.plot(data[:,1,:], 'g', linewidth=lineWidth)
    plt.plot(calData[:,0], calData[:,2],'go')
    plt.plot(data[:,2,:], 'r', linewidth=lineWidth)
    
    
    errNs = np.nansum(np.abs(data[calData[:,0].astype(int),1,:] - np.tile(calData[:,2],(data.shape[2],1)).T,axis=0))
    print 'min err Niedersachsen: ' + str(np.argmin(errNs))
    plt.plot(data[:,1,np.argmin(errNs)], 'g', linewidth=lineWidth*5)
    
    plt.xlim([nBurnIn,nSteps])
    years = (nSteps - nBurnIn) / 12 / factor
    plt.xticks(np.linspace(nBurnIn,nSteps,years+1), [str(2005 + year*factor) for year in range(years+1)], rotation=30)    
        
    if log:
        plt.yscale('log')   
    
    plt.subplot(1,3,2) 
    plt.title('Bremen')      
    data = np.load('poznan_out/E' + str(ensNo) + '_stock_bre.npy')    
    calData = np.load('poznan_out/E' + str(ensNo) + '_cal_stock_bre.npy')
    plt.plot(data[:,0,:], 'b', linewidth=lineWidth)
    plt.plot(calData[:,0], calData[:,1],'bo')
    plt.plot(data[:,1,:], 'g', linewidth=lineWidth)
    plt.plot(calData[:,0], calData[:,2],'go')
    plt.plot(data[:,2,:], 'r', linewidth=lineWidth)
    
    errBr = np.nansum(np.abs(data[calData[:,0].astype(int),1,:] - np.tile(calData[:,2],(data.shape[2],1)).T,axis=0))
    print 'min err Bremen: ' + str(np.argmin(errBr))
    plt.plot(data[:,1,np.argmin(errBr)], 'g', linewidth=lineWidth*5)
    
    plt.xlim([nBurnIn,nSteps])
    years = (nSteps - nBurnIn) / 12 / factor
    plt.xticks(np.linspace(nBurnIn,nSteps,years+1), [str(2005 + year*factor) for year in range(years+1)], rotation=30)    
    if log:   
        plt.yscale('log')   
    
    plt.subplot(1,3,3) 
    plt.title('Hamburg')
    data = np.load('poznan_out/E' + str(ensNo) + '_stock_ham.npy')    
    calData = np.load('poznan_out/E' + str(ensNo) + '_cal_stock_ham.npy')
    plt.plot(data[:,0,:], 'b', linewidth=lineWidth)
    plt.plot(calData[:,0], calData[:,1],'bo')
    plt.plot(data[:,1,:], 'g', linewidth=lineWidth)
    plt.plot(calData[:,0], calData[:,2],'go')
    plt.plot(data[:,2,:], 'r', linewidth=lineWidth)
    
    errHb = np.nansum(np.abs(data[calData[:,0].astype(int),1,:] - np.tile(calData[:,2],(data.shape[2],1)).T,axis=0))
    print 'min err Hamburg: ' + str(np.argmin(errHb))
    plt.plot(data[:,1,np.argmin(errHb)], 'g', linewidth=lineWidth*5)
    
    plt.xlim([nBurnIn,nSteps])
    years = (nSteps - nBurnIn) / 12 / factor
    plt.xticks(np.linspace(nBurnIn,nSteps,years+1), [str(2005 + year*factor) for year in range(years+1)], rotation=30)    
        
    if log:
        plt.yscale('log')   
        
    h1 = plt.plot(data[:1,0,1], 'b', linewidth=2)
    h2 = plt.plot(data[:1,1,1], 'g', linewidth=2)
    h3 = plt.plot(data[:1,2,1], 'r', linewidth=2)
    plt.figlegend(h1 + h2 + h3,['Combustion engine', 'Electric engine', 'other mobility types'], loc = 'lower center', ncol=3, labelspacing=0. )
    plt.tight_layout()   
    plt.subplots_adjust(bottom=.15)
    if log:
        plt.savefig('poznan_out/E' + str(ensNo) + '_ensembleStockAllLog')
    else:
        plt.savefig('poznan_out/E' + str(ensNo) + '_ensembleStockAll')
    
    print 1
    
    #%%
    from biokit.viz import corrplot
    paraDf.ix['errNs'] = errNs.astype(float)
    paraDf.ix['errHb'] = errHb.astype(float)
    paraDf.ix['errBr'] = errBr.astype(float)
    paraDf.ix['errTot'] = errBr+errNs+errHb.astype(float)
    paraDf.ix['invErrTot'] = paraDf.ix['errTot'] / np.max(paraDf.ix['errTot'])
    catParaDf = paraDf.ix[['convC', 'initialGreen', 'initPriceBrown','urbanThreshold', 'urbanCritical','kappa','errNs','errHb','errBr', 'errTot']]
    c = corrplot.Corrplot(catParaDf.T)
    c.plot(method='circle' ,rotation=45)
    
    #catParaDf.plot()
    #paraDf.ix[['individualPrio','convC', 'initialGreen', 'initPriceBrown','mobNewPeriod', 'maxFriends']].T.plot.box()

#%%
    plt.clf()
    plt.figure()
    plt.scatter(paraDf.ix['urbanThreshold'], paraDf.ix['urbanCritical'])
    ax = plt.subplot(2,2,1)
    paraDf.T.plot.scatter(x='convC',y='initialGreen',c='errNs', s = paraDf.ix['errNs'])
    plt.subplot(2,2,2)
    paraDf.T.plot.scatter(x='convC',y='initPriceBrown',c='errNs', s = paraDf.ix['errNs']*.1)
    plt.subplot(2,2,3)
    paraDf.T.plot.scatter(x='urbanCritical',y='urbanThreshold',c='errNs', s = paraDf.ix['errNs']*.1)
    plt.subplot(2,2,4)
    paraDf.T.plot.scatter(x='kappa',y='initPriceBrown',c='errNs', s = paraDf.ix['errNs']*.1)

#%%
    
    testDf = paraDf.ix[['convC', 'initialGreen', 'initPriceBrown','urbanThreshold', 'urbanCritical','kappa','errNs','errHb','errBr', 'errTot']]
    from pandas.plotting import scatter_matrix 
    scatter_matrix(testDf.T, alpha=0.3, figsize=(8, 8), diagonal='kde')


