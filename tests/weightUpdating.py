#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://wwww.globalclimateforum.org

CAR INNOVATION MARKET MODEL
-- TEST FILE --

This file is part of GCFABM.

GCFABM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GCFABM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GCFABM.  If not, see <http://earthw.gnu.org/licenses/>.
"""

from __future__ import division
import sys
from os.path import expanduser
#home = expanduser("~")
#sys.path.append(home + '/python/decorators/')
sys.path.append('../modules/')
sys.path.append('..')
#from deco_util import timing_function
import numpy as np
import time
#import mod_geotiff as gt
from class_agents import Household, Reporter, Cell
from class_world import Earth
import matplotlib
matplotlib.use('TKAgg')
import seaborn as sns; sns.set()
sns.set_color_codes("dark")
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
from bunch import Bunch

overallTime = time.time()
###### Enums ################
#connections
_tll = 1 # loc - loc
_tlh = 2 # loc - household
_thh = 3 # household, household
#nodes
_cell = 1
_hh   = 2

#%% model paramerter setup

para = Bunch()
para.simNo = None
para.nSteps         = 20
para.flgSpatial     = True
para.connRadius     = 1 
para.properties     = ['weig','range','consum','vol','speed', 'price']
para.memoryTime  = 10
para.addYourself = True
para.utilObsError = 10
para.writeOutput = True
para.writeNPY    = True
para.randomCarPropDeviationSTD = 0.01
landLayer   = np.asarray([[1]])
population = landLayer*2

#%% INITIALIZATION ##########################  
earth = Earth(para.nSteps, para.simNo, spatial=para.flgSpatial)
connList= earth.computeConnectionList(para.connRadius)
earth.initSpatialLayerNew(landLayer, connList, Cell)

# transfer all parameters to earth
earth.setParameters(Bunch.toDict(para)) 

# register enumerations
earth.enums = dict()
earth.enums['prefTypes'] = dict()
earth.enums['prefTypes'][0] = 'safety'
earth.enums['prefTypes'][1] = 'ecology'
earth.enums['prefTypes'][2] = 'convinience'
earth.enums['prefTypes'][3] = 'money'
earth.nPref = len(earth.enums['prefTypes'])
earth.nPrefTypes = [0]* earth.nPref

earth.enums['nodeTypes'] = dict()
earth.enums['nodeTypes'][1] = 'cell'
earth.enums['nodeTypes'][2] = 'household'


#init market
earth.initMarket(para.properties, para.randomCarPropDeviationSTD)
# init option class
earth.initOpinionGen(indiRatio = 0.33, ecoIncomeRange=(0,12000),convIncomeFraction=10000)
#init cell memory
earth.initMemory(para.properties + ['utility','label','hhID'], para.memoryTime)

#adding only one brand
earth.addBrand('medium',(2000, 600,  5.5,   5.5,  170,  180*12), 0)
earth.market.initCars()

print 1
# first basic test
# test if differences in preferences lead to differences in utility 
if False:
    plt.figure()
    for j in range(6):
        plt.subplot(2,3,j+1)
        properties = np.random.rand(6)
        prefReference =  np.random.rand(4)
        prefReference /= np.sum(prefReference)
        utilReference = earth.og.getUtililty(properties, prefReference)
        x,y = list(), list()
        for i in range(1000):
            preferences = np.random.rand(4)
            preferences /= np.sum(preferences)
            y.append(utilReference - earth.og.getUtililty(properties, preferences))
            x.append(np.sum(np.abs(prefReference - preferences)))
        plt.scatter(x,y)
        plt.xlabel('difference in preferences')
        plt.ylabel('difference in utility')
    
# generation of a two agents graph    
hhList = list()    
for h in range(3):
    hhList.append(Household(earth,'hh', 0, 0))
    hhList[-1].connectGeoNode(earth) 
    hhList[-1].setValue('income',np.random.randn(1)* 10000 + 18000)
    hhList[-1].setValue('income',18000)
    if h < 2:
        preferences = np.asarray([.5, .1, .2, .2])
    else:
        preferences = np.asarray([.2, .2, .1, .5])
    preferences = preferences**10    
    preferences = preferences / sum(preferences)
    print str(hhList[-1].nID) +'- ' +  str(preferences)
    hhList[-1].setValue('preferences', tuple(preferences))
    hhList[-1].prefTyp = np.argmax((preferences))
    hhList[-1].setValue('prefTyp',hhList[-1].prefTyp)
    hhList[-1].registerAgent(earth)
earth.dequeueEdges()

for agent in earth.iterNodes('hh'):
    print agent.nID
    for friend in earth.iterNodes('hh'):
        if True: #agent.nID != friend.nID:
            agent.addConnection(friend.nID, edgeType=_thh)
            edge = earth.graph.es[earth.graph.ecount()-1]
            print str(edge.index) + ' ' + str(edge.source) + ' -> ' + str(edge.target)
for edge in earth.iterEdges(_thh):
    edge['weig'] = .3333
earth.view("testGraph.png")   

#print  earth.graph.es['weig']
for agent in earth.randomIterNodes('hh'):
    for i in range(10):
        agent.buyCar(earth,0.0)
    agent.evalUtility(earth)  
    print str(agent.getValue('util')) + ': ',
    percProp = agent.getValue('preceivedProps')
    for value in percProp:
        print("%.2f" % value),
    agent.shareExperience(earth)
    print ' '
    
for agent in earth.iterNodes('hh'):
    agent.weightFriendExperience(earth)
#print earth.graph.es['weig']

for edge in earth.iterEdges(_thh):
    print str(edge.index) + ' ' + str(edge.source) + ' -> ' + str(edge.target) + ' weig:' + str(edge['weig'])
    
for agent in earth.iterNodes('hh'):
      
    agent.evalUtility(earth)  
    print str(agent.getValue('util')) + ': ',
    percProp = agent.getValue('preceivedProps')
    for value in percProp:
        print("%.2f" % value),
    agent.shareExperience(earth)
    agent.weightFriendExperience(earth)  
    print ' ' 
for edge in earth.iterEdges(_thh):
    print str(edge.index) + ' ' + str(edge.source) + ' -> ' + str(edge.target) + ' weig:' + str(edge['weig'])
        