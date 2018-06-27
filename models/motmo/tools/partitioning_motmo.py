#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 11:47:23 2018

@author: ageiges
"""

import mpi4py
mpi4py.rc.threads = False
import sys, os

sys.path.append('../../../lib/')
sys.path.append('../')
                
import socket
import csv
#from bunch import Bunch
import numpy as np
from copy import copy
from os.path import expanduser
import pdb
import core as core
import lib_gcfabm as lib

import logging as lg
import matplotlib.pyplot as plt
#import class_auxiliary as aux
home = expanduser("~")


dirPath = os.path.dirname(os.path.realpath('.'))

fileName = sys.argv[1]
nParts   = int(sys.argv[2])
parameters = core.AttrDict()
for item in csv.DictReader(open('../' +fileName)):
    if item['name'][0] != '#':
        parameters[item['name']] = core.convertStr(item['value'])
for item in csv.DictReader(open('../parameters_all.csv')):
    if item['name'][0] != '#':
        parameters[item['name']] = core.convertStr(item['value'])        
lg.info('Setting loaded:')

import init_motmo as init 
import scenarios as sc

scenarioDict = dict()
scenarioDict[2] = sc.scenarioLueneburg
scenarioDict[3] = sc.scenarioNBH
scenarioDict[6] = sc.scenarioGer
#%%
parameters = scenarioDict[parameters.scenario] (parameters, dirPath)
parameters['connRadius'] = 1.5
simNo, outputPath = core.setupSimulationEnvironment(None, simNo=0)
earth = init.initEarth(999, outputPath, parameters, maxNodes=1000000, maxLinks=1000000, debug =True)
CELL, HH, PERS = init.initTypes(earth)
#init.initSpatialLayer(earth)
connRadius = 1.5
connList= core.computeConnectionList(parameters['connRadius'], ownWeight=1.5)

earth.spatial.initSpatialLayer(parameters['landLayer'],
                       connList, 
                       LocClassObject=init.Cell,
                       linkTypeID=1)    
    
parameters.population.clip(min=2)
for cell in earth.iterNodes(CELL):
    pos = cell.get('pos')
    cell.set('population', parameters.population[pos[0], pos[1]])
    
if parameters.scenario == 6:

    #earth.view('spatial_graph.png')

    earth.graph.addLink(1,1000995,1002057)
    earth.graph.addLink(1,1002057,1000995)
    earth.graph.addLink(1,1001310,1000810)
    earth.graph.addLink(1,1000810,1001310)
    #aux.writeAdjFile(earth.graph,'resources_ger/outGraph.txt')
    

core.writeAdjFile(earth, parameters['resourcePath'] + 'outGraph.txt', nodeTypeID=1)

metisPath = home + '/software/metis-5.1.0/build/Linux-x86_64/programs/gpmetis'

os.system(metisPath + ' ' + parameters['resourcePath'] + 'outGraph.txt ' + str(nParts) + ' -niter=500 -ncuts=500 -ufactor=2 -conti -no2hop -ctype=rm -objtype=vol')

clusterMap = parameters['population'] *0
fid = open(parameters['resourcePath'] +"outGraph.txt.part." + str(nParts),'r')
x = fid.readlines()
yy = np.asarray([int(xx) for xx in x])
nonNan = np.isnan(parameters['population'])==False
clusterMap[nonNan] = yy
plt.figure()
plt.imshow(clusterMap)
plt.savefig('clusterMap.png')

#%%
nAgents = np.zeros(nParts)
for iClust in range(nParts):
    #print str(iClust) + ': ' + str()
    nAgents[iClust] = np.sum(parameters['population'][clusterMap==iClust])
print('min: ' + str(nAgents.min()))
print('max: ' + str(nAgents.max()))
print('mean: ' + str(nAgents.mean()))
print('std: ' + str(nAgents.std()))
print('rel std: ' + str(nAgents.std() / nAgents.mean()))  
#%%
#clusterMap = clusterMap.astype(int)
#iMax = np.argmax(nAgents)
#xMax, yMax = np.where(clusterMap==iMax)
#idxMinMax =  np.argmin(parameters['population'][xMax,yMax])
#minValue = parameters['population'][xMax[idxMinMax],yMax[idxMinMax]]
#xNeig = []
#yNeig = []
#for x,y in zip(xMax,yMax):
#    for dx in [-1,1]:
#        for dy in [-1,1]:
#            xNeig.append(x+dx)
#            yNeig.append(y+dy)
#xNeig = np.asarray(xNeig)
#yNeig = np.asarray(yNeig)
#neigClustRank = np.unique(clusterMap[xNeig,yNeig])
#
#exchangeClust = neigClustRank[nAgents[neigClustRank] + minValue < nAgents[iMax]][0]
#clusterMap[xMax[idxMinMax],yMax[idxMinMax]] = exchangeClust

#%%

np.save(parameters['resourcePath'] +'partition_map_' + str(nParts) + '.npy',clusterMap) 
