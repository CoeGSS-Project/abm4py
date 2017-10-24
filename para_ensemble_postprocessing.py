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
#simulations= [222,223,224,225,226,227,228,229,230,231]

for i,simNo in enumerate(range(295,324)):
    path = getEnvironment(None,getSimNo = False) +'sim' + str(simNo).zfill(4) + '/'
    print path
    if os.path.isfile(path+ '/globals.hdf5'):
        simulations.append(simNo)

print 'available simulations: ' +str(simulations) 

simDataNS = np.zeros([560,3,len(simulations)])
simDataBr = np.zeros([560,3,len(simulations)])
simDataHb = np.zeros([560,3,len(simulations)])


for i, simNo in enumerate(simulations):
    path = getEnvironment(None,getSimNo = False) +'sim' + str(simNo).zfill(4) + '/'
    h5File  = ta.File(path + '/globals.hdf5', 'r')
    
    
    
    group = h5File.get_node('/glob/stockNiedersachsen')
    simDataNS[:,:,i] = group.read()
    group = h5File.get_node('/glob/stockBremen')
    simDataBr[:,:,i] = group.read()
    group = h5File.get_node('/glob/stockHamburg')
    simDataHb[:,:,i] = group.read()

np.save('output/stock_nie.npy',simDataNS)
np.save('output/stock_bre.npy',simDataBr)
np.save('output/stock_ham.npy',simDataHb)


group = h5File.get_node('/calData/stockNiedersachsen')
dataNS = group.read()
group = h5File.get_node('/calData/stockBremen')
dataBr = group.read()
group = h5File.get_node('/calData/stockHamburg')
dataHb = group.read()

np.save('output/cal_stock_nie.npy',dataNS)
np.save('output/cal_stock_bre.npy',dataBr)
np.save('output/cal_stock_ham.npy',dataHb)

if False:
#%%
    nSteps  = 560
    nBurnIn = 200
    factor = 5
    log    = 1

    years = (nSteps - nBurnIn) / 12 / factor
    plt.clf()
    plt.subplot(1,3,1) 
    plt.title('Niedersachsen')         
    data = np.load('poznan_out/stock_nie.npy')    
    calData = np.load('poznan_out/cal_stock_nie.npy')
    plt.plot(data[:,0,:], 'b', linewidth=.5)
    plt.plot(calData[:,0], calData[:,1],'bo')
    plt.plot(data[:,1,:], 'g', linewidth=.5)
    plt.plot(calData[:,0], calData[:,2],'go')
    plt.plot(data[:,2,:], 'r', linewidth=.5)
    
    plt.xlim([nBurnIn,nSteps])
    years = (nSteps - nBurnIn) / 12 / factor
    plt.xticks(np.linspace(nBurnIn,nSteps,years+1), [str(2005 + year*factor) for year in range(years+1)], rotation=30)    
        
    if log:
        plt.yscale('log')   
    
    plt.subplot(1,3,2) 
    plt.title('Bremen')      
    data = np.load('poznan_out/stock_bre.npy')    
    calData = np.load('poznan_out/cal_stock_bre.npy')
    plt.plot(data[:,0,:], 'b', linewidth=.5)
    plt.plot(calData[:,0], calData[:,1],'bo')
    plt.plot(data[:,1,:], 'g', linewidth=.5)
    plt.plot(calData[:,0], calData[:,2],'go')
    plt.plot(data[:,2,:], 'r', linewidth=.5)
    
    plt.xlim([nBurnIn,nSteps])
    years = (nSteps - nBurnIn) / 12 / factor
    plt.xticks(np.linspace(nBurnIn,nSteps,years+1), [str(2005 + year*factor) for year in range(years+1)], rotation=30)    
    if log:   
        plt.yscale('log')   
    
    plt.subplot(1,3,3) 
    plt.title('Hamburg')
    data = np.load('poznan_out/stock_ham.npy')    
    calData = np.load('poznan_out/cal_stock_ham.npy')
    plt.plot(data[:,0,:], 'b', linewidth=.5)
    plt.plot(calData[:,0], calData[:,1],'bo')
    plt.plot(data[:,1,:], 'g', linewidth=.5)
    plt.plot(calData[:,0], calData[:,2],'go')
    plt.plot(data[:,2,:], 'r', linewidth=.5)
    
    plt.xlim([nBurnIn,nSteps])
    years = (nSteps - nBurnIn) / 12 / factor
    plt.xticks(np.linspace(nBurnIn,nSteps,years+1), [str(2005 + year*factor) for year in range(years+1)], rotation=30)    
        
    if log:
        plt.yscale('log')   
        
    h1 = plt.plot(data[:,0,:], 'b', linewidth=.5)
    h2 = plt.plot(data[:,1,:], 'g', linewidth=.5)
    h3 = plt.plot(data[:,2,:], 'r', linewidth=.5)
    plt.figlegend([h1, h2 ,h3],['Combustion engine', 'Electric engine', 'other mobility types'], loc = 'lower center', ncol=3, labelspacing=0. )
    plt.tight_layout()   
    plt.subplots_adjust(bottom=.15)
    if log:
        plt.savefig('poznan_out/ensembleStockAllLog')
    else:
        plt.savefig('poznan_out/ensembleStockAll')
    
