#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://wwww.globalclimateforum.org

MOBILITY INNOVATION MARKET MODEL
-- INIT FILE --

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
import sys, os
from os.path import expanduser
home = expanduser("~")
#sys.path.append(home + '/python/decorators/')
sys.path.append(home + '/python/modules/')

#from deco_util import timing_function
import numpy as np
import time
import mod_geotiff as gt
from class_mobilityABM import Person, Household, Reporter, Cell,  Earth, Opinion

from class_auxiliary  import convertStr
import matplotlib
matplotlib.use('TKAgg')
import seaborn as sns; sns.set()
sns.set_color_codes("dark")
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
from bunch import Bunch
import csv

from init_mobilityABM import *

###### Enums ################
#connections
_cll = 1 # loc - loc
_clh = 2 # loc - household
_chh = 3 # household, household
_chp = 4 # household, person
_cpp = 5 # person, person

#nodes
_cell = 1
_hh   = 2
_pers = 3

#time spans
_month = 1
_year  = 2

################################################################

dirPath = os.path.dirname(os.path.realpath(__file__))

parameters = Bunch()
for item in csv.DictReader(open("test_parameters.csv")):
    parameters[item['name']] = convertStr(item['value'])

parameters.initialGreen =100
parameters.initialBrown =10000
parameters.initialOther =5000

parameters.burnIn = 10
parameters.showFigures = 1

parameters.resourcePath = dirPath + '/resources_nie/'
parameters = scenarioTestMedium(parameters)
            
earth = initEarth(parameters)

mobilitySetup(earth, parameters)


cellTest(earth, parameters)

householdSetup(earth, parameters)

generateNetwork(earth, parameters)

initMobilityTypes(earth, parameters)

initGlobalRecords(earth, parameters)
#%%
for household in tqdm.tqdm(earth.iterNodes(_hh)):
    
    household.takeAction(earth, household.adults, np.random.randint(0,earth.market.nMobTypes,len(household.adults)))

for cell in earth.iterNodes(_cell):
    cell.step(earth.market.kappa) 

for household in tqdm.tqdm(earth.iterNodes(_hh)):
    household.calculateConsequences(earth.market)
    household.util = household.evalUtility()
    household.shareExperience(earth)
    
colorPal =  sns.color_palette("Set3", n_colors=3, desat=1)    


test = 2

if test == 1:
    #%%    
    nAg = 1000
    plt.figure(1)
    plt.clf()
    legStr = list()
    x = np.zeros([nAg])
    y = np.zeros([4,nAg])
    z = np.zeros([nAg])
    for j, persID in enumerate(earth.nodeList[3][0:nAg]):
        pers = earth.entDict[persID]
        x[j]    = pers.node['util']
        y[:,j]  = pers.node['consequences']
        z[j]    = pers.node['mobType']
        
    for i, cons in enumerate(earth.enums['consequences'].keys()):
        plt.subplot(2,2,i+1)
        for mob in range(3):
            plt.scatter(y[i,z==mob],x[z==mob],color= colorPal[mob])
        plt.xlabel(earth.enums['consequences'][cons])
        plt.ylabel('utility')
    plt.tight_layout()
    
    #%%
    earth.step()
    
    #%%    
    nAg = 1000
    plt.figure(2)
    plt.clf()
    
    x = np.zeros([nAg])
    y = np.zeros([4,nAg])
    z = np.zeros([nAg])
    dist = np.zeros([2,nAg])
    distMeas = np.zeros([nAg])
    inno = np.zeros([nAg])
    
    for j, persID in enumerate(earth.nodeList[3][0:nAg]):
        pers = earth.entDict[persID]
        x[j]    = pers.node['util']
        y[:,j]  = pers.node['consequences']
        z[j]    = pers.node['mobType']
        mobProp = mobProps = pers.node['prop']
        dist[:,j]   = [(earth.market.mean[0]-mobProp[0])/earth.market.std[0], (mobProp[1]- earth.market.mean[1])/earth.market.std[1]]
        distMeas[j] = (earth.market.distanceFromMean(mobProps)-earth.market.meanDist)/earth.market.stdDist
        inno[j]     = pers.innovatorDegree
        
    for i, cons in enumerate(earth.enums['consequences'].keys()):
        plt.subplot(2,2,i+1)
        
        for mob in range(3):
            plt.scatter(y[i,z==mob],x[z==mob],color= colorPal[mob])
            legStr.append(earth.enums['mobilityTypes'][mob])
        plt.xlabel(earth.enums['consequences'][cons])
        plt.ylabel('utility')
    plt.tight_layout()
    plt.legend(legStr,loc=2)


    #%%
    plt.figure(3)
    plt.clf()
    plt.scatter(dist[0,:],dist[1,:],c= distMeas,cmap='jet')
    #plt.plot(earth.market.mean[0],earth.market.mean[1], 'd')
    plt.colorbar()
    legStr.append(earth.enums['mobilityTypes'][mob])
    plt.xlabel("distance emissions" )
    plt.ylabel("distance price")
    plt.tight_layout()
    #plt.legend(legStr,loc=2)
    
    #%%
    plt.figure(4)
    plt.clf()
    plt.scatter(inno,distMeas,c= y[3,:],cmap='jet')
    #plt.plot(earth.market.mean[0],earth.market.mean[1], 'd')
    plt.colorbar()
    legStr.append(earth.enums['mobilityTypes'][mob])
    plt.xlabel("innovationDegree")
    plt.ylabel("distance measure")
    plt.tight_layout()
#%%    

if test == 2:
    nSteps = 50
    nMobType = np.zeros([3, nSteps])
    for i in range(nSteps):
        print 'step ' + str(i)
        nAg = 1000
        
        
        x = np.zeros([nAg])
        y = np.zeros([4,nAg])
        z = np.zeros([nAg])
        dist = np.zeros([2,nAg])
        distMeas = np.zeros([nAg])
        inno = np.zeros([nAg])
        
        for j, persID in enumerate(earth.nodeList[3][0:nAg]):
            pers = earth.entDict[persID]
            x[j]    = pers.node['util']
            y[:,j]  = pers.node['consequences']
            z[j]    = pers.node['mobType']
            mobProp = mobProps = pers.node['prop']
            dist[:,j]   = [(earth.market.mean[0]-mobProp[0])/earth.market.std[0], (mobProp[1]- earth.market.mean[1])/earth.market.std[1]]
            distMeas[j] = earth.market.getDistanceFromMean(mobProp)
            
            inno[j]     = pers.innovatorDegree
            
    
        
        plt.clf()
        h = plt.subplot(1,2,1)
        plt.scatter(dist[0,z==0],dist[1,z==0],c= distMeas[z==0],cmap='jet', marker='d')
        plt.clim([np.min(distMeas), np.max(distMeas)])
        plt.scatter(dist[0,z==1],dist[1,z==1],c= distMeas[z==1],cmap='jet', marker='s')
        plt.clim([np.min(distMeas), np.max(distMeas)])
        plt.scatter(dist[0,z==2],dist[1,z==2],c= distMeas[z==2],cmap='jet', marker='x')
        plt.clim([np.min(distMeas), np.max(distMeas)])
        #plt.plot(earth.market.mean[0],earth.market.mean[1], 'd')
        plt.colorbar()
        legStr.append(earth.enums['mobilityTypes'][mob])
        plt.xlabel("distance emissions" )
        plt.ylabel("distance price")
        plt.tight_layout()
        h.set_aspect('equal', 'datalim')
        #plt.legend(legStr,loc=2)
    
        plt.subplot(1,2,2)
        plt.scatter(inno[z==0],distMeas[z==0],c= y[3,z==0],cmap='jet', marker='d')
        plt.scatter(inno[z==1],distMeas[z==1],c= y[3,z==1],cmap='jet', marker='s')
        plt.scatter(inno[z==2],distMeas[z==2],c= y[3,z==2],cmap='jet', marker='x')
        #plt.plot(earth.market.mean[0],earth.market.mean[1], 'd')
        plt.colorbar()
        legStr.append(earth.enums['mobilityTypes'][mob])
        plt.xlabel("innovationDegree")
        plt.ylabel("distance measure")
        plt.tight_layout()
        plt.legend(earth.enums['mobilityTypes'].values(),loc=2)
        plt.savefig('tests/' + str(earth.time) + '.png')
        
        
        earth.step()
        
        print "persons per mobType",
        for mob in range(3):
            nMobType[mob, i] = np.sum(z==mob)
            print (np.sum(z==mob)),
        print " "
    plt.figure(2)        
    for mob in range(3):
                    
        plt.plot(nMobType[mob, :])
    plt.legend(earth.enums['mobilityTypes'].values(),loc=2)
    
if test == 3:

    propMat = np.matrix(earth.graph.vs[earth.nodeList[3]]['preferences'])
    print np.mean(propMat,axis=0)
    