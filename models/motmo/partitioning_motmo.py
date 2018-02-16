#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 11:47:23 2018

@author: ageiges
"""

import mpi4py
mpi4py.rc.threads = False
import sys, os

sys.path.append('../../lib/')
sys.path.append('../../modules/')
                
import socket
import csv
from bunch import Bunch
import numpy as np
from copy import copy
from os.path import expanduser
import pdb

import logging as lg
import matplotlib.pyplot as plt
import class_auxiliary as aux
home = expanduser("~")


dirPath = os.path.dirname(os.path.realpath(__file__))

fileName = sys.argv[1]
parameters = Bunch()
for item in csv.DictReader(open(fileName)):
    if item['name'][0] != '#':
        parameters[item['name']] = aux.convertStr(item['value'])
lg.info('Setting loaded:')

import init_motmo as init 

scenarioDict = dict()
scenarioDict[2] = init.scenarioLueneburg
scenarioDict[3] = init.scenarioNBH
scenarioDict[6] = init.scenarioGer
#%%
parameters = scenarioDict[parameters.scenario] (parameters, dirPath)
parameters['connRadius'] = 1.1

earth = init.initEarth(999, 'output/', parameters, maxNodes=1000000, debug =True)
CELL, HH, PERS = init.initTypes(earth)
init.initSpatialLayer(earth)
for cell in earth.iterEntRandom(CELL):
    cell.setValue('population', parameters.population[cell.getValue('pos')])
    
    
if parameters.scenario == 6:

    #earth.view('spatial_graph.png')

    earth.graph.add_edge(995,2057)
    earth.graph.add_edge(2057,995)
    earth.graph.add_edge(1310,810)
    earth.graph.add_edge(810,1310)
    #aux.writeAdjFile(earth.graph,'resources_ger/outGraph.txt')
    

aux.writeAdjFile(earth.graph, parameters['resourcePath'] + 'outGraph.txt')

metisPath = home + '/software/metis-5.1.0/build/Linux-x86_64/programs/gpmetis'
nParts = 10
os.system(metisPath + ' ' + parameters['resourcePath'] + 'outGraph.txt ' + str(nParts) + ' -niter=50 -ncuts=50 -ufactor=2 -conti -no2hop -ctype=rm -objtype=vol -seed=2')

clusterMap = parameters['population'] *0
fid = open(parameters['resourcePath'] +"outGraph.txt.part." + str(nParts),'r')
x = fid.readlines()
yy = np.asarray([int(xx) for xx in x])
nonNan = np.isnan(parameters['population'])==False
clusterMap[nonNan] = yy
plt.figure()
plt.imshow(clusterMap)
#%%
for iClust in range(nParts):
    print str(iClust) + ': ' + str(np.sum(parameters['population'][clusterMap==iClust]))

#%%

np.save(parameters['resourcePath'] +'rankMap_nClust' + str(nParts) + 'new.npy',clusterMap) 