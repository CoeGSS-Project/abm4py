#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:59:47 2017

@author: gcf
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append('/media/sf_shared/python/modules/biokit')
sys.path.append('/home/geiges/database/modules/folium/')
sys.path.append('/home/geiges/database/')
import seaborn as sns; sns.set()
sns.set_color_codes("dark")

from os.path import expanduser
home = expanduser("~")

plotRecords       = 0
plotCarStockBar   = 1
plotCarSales      = 1
prefPerLabel      = 0
utilPerLabel      = 1
incomePerLabel    = 0
meanPrefPerLabel  = 1
meanConsequencePerLabel = 1
printCellMaps     = 1
emissionsPerLabel = 1

nBurnIn = 100
withoutBurnIn = False 
years         = True                        # only applicable in plots without burn-in

time = 0
plt.figure()
for calRun in range(0,1):
    plt.subplot(3,3,calRun+1)
    os.system('tar -xf ' + home + '/.openmole/gcf-VirtualBox/webui/projects/mobilityModel/out/output' + str(calRun) + '.tar.gz')

    path = 'mobilityABM/rootfs/mnt/ssd/geiges/python/agModel/output/sim9999/'
    #path = 'mobilityABM/rootfs/mnt/ssd/geiges/python/agModel/output/sim9999/'
    #%% init
        
    from class_auxiliary import loadObj
    

    
    
    #%% loading household agent file
    
    persMat      = np.load(path + 'agentFile_type3.npy')
    persPropDict = loadObj(path + 'attributeList_type3')
    hhMat        = np.load(path + 'agentFile_type2.npy')
    hhPropDict   = loadObj(path + 'attributeList_type2')
    parameters   = loadObj(path + 'simulation_parameters')
    enums        = loadObj(path + 'enumerations')
    print parameters['initialGreen']
    nSteps, nPers, nPersProp = persMat.shape
    nSteps, nHhs,  nHhProp   = hhMat.shape
    
    
    nPrior = len(enums['priorities'])
    print np.sum(persMat[time,:,persPropDict['mobType'][0]])
    
    #%%  plot car stock as bar plot
    legStr = list()
    
    carMat = np.zeros([nSteps,3])
    for time in range(nSteps):
        carMat[time,:]= np.bincount(persMat[time,:,persPropDict['mobType'][0]].astype(int),minlength=3).astype(float)
    if plotCarStockBar:
        
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
        plt.title('initial Green: ' + str(parameters['initialGreen']))