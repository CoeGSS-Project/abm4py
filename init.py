#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://www.globalclimateforum.org

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
along with GCFABM.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import division
import sys
from os.path import expanduser
home = expanduser("~")
sys.path.append(home + '/python/decorators/')
sys.path.append(home + '/python/modules')
#from deco_util import timing_function
import numpy as np
import time
import mod_geotiff as gt
from class_agents import Household, Reporter, Cell
from class_world import World
import seaborn as sns; sns.set()
sns.set_color_codes("dark")
import matplotlib.pyplot as plt
import tqdm
import pandas as pd

###### Enums ################
#connections
_tll = 1 # loc - loc
_tlh = 2 # loc - household
_thh = 3 # household, household
#nodes
_cell = 1
_hh   = 2
#%% INIT
flgSpatial = True
connRadius = 2.1  # radíus of cells that get an connection
tolerance  = 1.   # tolerance of friends when connecting to others (deviation in preferences)

scenario      = 1
randomAgents  = 1 # 0: prefrences dependent on agent properties - 1: random distribution
randPref      = 1 # 0: only exteme preferences (e.g. 0,0,1) - 1: random weighted preferences
radicality    = 3 # exponent of the preferences -> high values lead to extreme differences

nSteps     = 20 # number of simulation steps
initPhase = 20
properties = ['weig','range','consum','vol','speed', 'price']
randomCarPropDeviationSTD = 0
minFriends = 30  # number of desired friends
memoryTime  = 10  # length of the periode for which memories are stored

recAgent   = [5,10,15]   # reporter agents that return a diary

tt = time.time()

## Scenario definition ###################################################################
if scenario == 0: #small
    minFriends = 5
    nAgents    = 100
    landLayer   = np.asarray([[1, 1, 1,0,0,0],
                              [0, 0, 1,0,0,0],
                              [0, 0, 1,1,1,1]])
    population = landLayer*10
    
elif scenario == 1: # medium
    nAgents    = 1000
    landLayer   = np.asarray([[0, 0, 0, 0, 0, 1, 1, 1], 
                              [1, 1, 1, 0, 0, 1, 1, 1],
                              [1, 1, 1, 0, 0, 1, 0, 0],
                              [1, 1, 1, 1, 1, 1, 0, 0],
                              [1, 1, 1, 1, 1, 1, 0, 0]])    
    population = landLayer*100
    
elif scenario == 2: # Niedersachsen
    reductionFactor = 100
    landLayer= gt.load_array_from_tiff('resources_nie/land_layer_3432x8640.tiff')
    landLayer[np.isnan(landLayer)] = 0
    landLayer = landLayer.astype(int)
    plt.imshow(landLayer)
    population = gt.load_array_from_tiff('resources_nie/pop_counts_ww_2005_3432x8640.tiff') / reductionFactor
    plt.imshow(population)
    nAgents    = np.nansum(population)
    landLayer[landLayer == 1 & np.isnan(population)] =0
    assert np.sum(np.isnan(population[landLayer==1])) == 0
    print nAgents




#%% INITIALIZATION ##########################   
ww = World(nSteps, spatial=flgSpatial)
connList= ww.computeConnectionList(connRadius)
ww.initSpatialLayerNew(landLayer, connList, Cell)
#
ww.props = properties
ww.initMarket(properties, randomCarPropDeviationSTD)
#ww.initObsAtLoc(properties)
ww.initOpinionGen(indiRatio = 0.33, ecoIncomeRange=(10000,30000),convIncomeFraction=25000)
ww.scenario = scenario
ww.radicality = radicality
#init location memory


 
ww.initMemory(properties + ['utility','label','hhID'], memoryTime)


#                           weig range consum vol   speed price']
#ww.market.addBrand('green',      (1500,300,  3.0,   4,    130,  40000))    
#ww.addBrand('city', (1500, 600,  4.5,   4.7,  140,  35000))   #city car 
ww.addBrand('medium',(2000, 600,  5.5,   5.0,  140,  25000), 0)   #small car 
ww.addBrand('family',(2400, 800,  6.5,   8.0,  140,  30000), 0)   #family car

ww.addBrand('Diesel',(2500, 900,  7.5,   8.5,  150,  45000), 0)   #Diesels 
ww.addBrand('Sport', (1500, 800,  9.0,   5.0,  250,  35000), 0)   #sports 
ww.addBrand('small',(1800, 700,  5.0,   5.0,  160,  25000),50)   #small car 
ww.addBrand('Diesel+',(2600, 1000, 7.0,   8.5,  160,  45000),60) 
ww.addBrand('city', (1500, 600,  4.5,   4.7,  140,  35000),70)
ww.addBrand('SUV',   (3500, 500,  9.0,   9.0,  180,  60000),80)   #suv 
ww.addBrand('green',(1500, 450,  2.0,   4,    130,  40000),90)
    
ww.nPrefTypes = [0,0,0]
ww.record.loc[0] = 0
print 'Init finished after -- ' + str( time.time() - tt) + ' s'
tt = time.time()
df = pd.read_csv('resources_nie/hh_niedersachsen.csv')
hhMat = pd.read_csv('resources_nie/hh_niedersachsen.csv').values


#%% Init records

ww.registerRecord('avgUtil', 'Average utility',["overall", "prefSafety", "prefEcology", "prefConvinience"])
ww.registerRecord('carStock', 'Cars per label', ww.market.brandsToInit, style='stackedBar')
ww.registerRecord('sales', 'Sales per Preference',["salesSaf", "salesEco", "salesCon"])


#%% Init of Households
nAgents = 0
nHH     = 0
if randomAgents:
    idx = 0
    for x,y in tqdm.tqdm(ww.locDict.keys()):
        nAgentsCell = int(population[x,y])
        while True:
            
            if nHH in recAgent:
                hh = Reporter(ww,'hh', x, y)
            else:
                hh = Household(ww,'hh', x, y)
            hh.connectGeoNode(ww)
            hh.tolerance = tolerance
            hhSize, ageList, sexList, income,  nKids, prSaf, prEco, prCon = ww.generateHH()
            #hhSize = int(np.ceil(np.abs(np.random.randn(1)*2)))
            hh.setValue('hhSize', hhSize)
            hh.setValue('age',ageList)         
            hh.setValue('income',income)
            hh.setValue('nKids', nKids)
            
            
            # normal 
            #hh.setValue('preferences', (prSaf, prEco, prCon))
            #hh.prefTyp = np.argmax((prSaf, prEco, prCon))
            
            # test
            if randPref == 0:
                xx = np.zeros(3)
                xx[np.random.randint(0,3)] = 1
                #xx = xx / sum(xx)
                prSaf, prEco, prCon = xx
            elif randPref == 1:
                xx = np.random.random(3)** radicality
                xx = xx / sum(xx)
                prSaf, prEco, prCon = xx
                
            
            hh.setValue('preferences', (prSaf, prEco, prCon))
            hh.prefTyp = np.argmax((prSaf, prEco, prCon))
            
            hh.registerAgent(ww,_tlh)
            ww.nPrefTypes[hh.prefTyp] += 1
            nPers       = hhSize
            nAgentsCell -= nPers
            nAgents     += nPers
            idx         += nPers
            nHH         +=1
            if nAgentsCell < 0:
                break
                
else:
    idx = 0
    for x,y in tqdm.tqdm(ww.locDict.keys()):
        #print x,y
        nAgentsCell = int(population[x,y])
        while True:
            if nHH in recAgent:
                hh = Reporter(ww,'hh', x, y)
            else:
                hh = Household(ww,'hh', x, y)
            hh.connectGeoNode(ww)
            hh.tolerance = tolerance
            nPers = hhMat[idx,4]
            hh.setValue('hhSize',nPers)
            age= hhMat[idx:idx+nPers,12]
            hh.setValue('age',list(age))
            sex= hhMat[idx:idx+nPers,12]
            hh.setValue('sex',list(sex))
            
            nKids = np.sum(age<18)
            hh.setValue('nKids', nKids)
            income = hhMat[idx,16]
            hh.setValue('income',income)   
            if len(np.where(age>=18)[0]) > 0:
                idxAdult = np.random.choice(np.where(age>=18)[0])
            else:
                idxAdult = 0
            prSaf, prEco, prCon = ww.og.getPref(age[idxAdult],sex[idxAdult],nKids,income,radicality)
            # normal 
            hh.setValue('preferences', (prSaf, prEco, prCon))
            hh.prefTyp = np.argmax((prSaf, prEco, prCon))
            
            # test
            #hh.setValue('preferences', (0,0,1))
            #hh.prefTyp = 0

            hh.registerAgent(ww,_tlh)
            ww.nPrefTypes[hh.prefTyp] += 1
            nAgentsCell -= nPers
            nAgents     += nPers
            idx         += nPers
            nHH         +=1
            if nAgentsCell < 0:
                break
ww.dequeueEdges()
                
#ww.recAgent = ww.nodeList[_hh][recAgent]            
            
print 'Agents created in -- ' + str( time.time() - tt) + ' s'

ww.genFriendNetwork(minFriends)
ww.market.initCars()
if scenario == 0:
    ww.view()

tt = time.time()
   
import pandas as pd
#prefUtil = pd.DataFrame([],columns= ["prSaf", "prEco", "prCon"] )    
#prefUtil.loc[0] = 0
for agent in ww.iterNode(_hh):
    agent.buyCar(ww,np.random.choice(ww.market.brandProp.keys()))
    agent.car['age'] = np.random.randint(0,15)
    agent.util = agent.evalUtility(ww)
    agent.utilList.append(agent.util)
    agent.shareExperience(ww)

#ww.record.loc[ww.time, ww.rec["avgUtilPref"][1]] /= ww.nPrefTypes
ww.globRec['avgUtil'].div(ww.time, [ww.nAgents] + ww.nPrefTypes)

#ww.view()
util = np.asarray(ww.graph.vs[ww.nodeList[2]]['util'])
import matplotlib.pyplot as plt

if False:
    plt.figure()
    plt.subplot(2,5,1)
    plt.hist(util,30)
    plt.xlim([0,1])

brandDict = dict()
for key in ww.market.obsDict.keys():
    brandDict[key] = [np.mean(ww.market.obsDict[ww.time][key])]
#for es in ww.graph.es:
#    print (es.source, es.target)
#for x in xrange(3,21):
#    print ww.agDict[x].obs
    
    #choice = ww.agDict[x].optimalChoice(ww)  
    
ww.avgDegree = [np.mean(ww.graph.vs[ww.nodeList[2]].degree())]

#%% Simulation 
for step in xrange(1,nSteps):
    tt = time.time()
    ww.step()
#    for x in ww.nodeList[2]:
#        agent = ww.agDict[x]
        
        #agent.socialize(ww)
    
    print 'step ' + str(step) + ' done in ' +  str(time.time()-tt) + ' s'
    util = np.asarray(ww.graph.vs[ww.nodeList[2]]['util'])
    # plot utility per brand
    for key in ww.market.obsDict[ww.time].keys():
        if key in brandDict:
            brandDict[key].append(np.mean(ww.market.obsDict[ww.time][key]))
        else:
            brandDict[key] = [np.mean(ww.market.obsDict[ww.time][key])]
    
            
    if False:  
        plt.subplot(2,5,step)
        plt.hist(util,30)
        plt.xlim([0,1])   
    
        

for writer in ww.reporter:        
    writer.close()        
#%% post processing
legLabels = [ww.market.brandLabels[x] for x in ww.market.stockbyBrand.columns]
if False:
    #plot individual utilities
    plt.figure()
    for agent in ww.iterNode(_hh):
        plt.plot(agent.utilList)    

plt.figure()
for key in brandDict.keys():
    with sns.color_palette("Set3", n_colors=9, desat=.8):
        plt.plot(range(nSteps-len(brandDict[key]),nSteps), brandDict[key])
plt.legend(legLabels, loc=3)
plt.title('Utility per brand')
plt.savefig('utilityPerBrand.png')
#plt.figure()
#with sns.color_palette("Set3", n_colors=9, desat=.8):
#    plt.plot(ww.market.stockbyBrand)
#plt.legend(legLabels, loc=3)
#plt.title('Cars per brand')
#plt.savefig('carsPerBrand.png')


#%% Cars per brand
#plt.figure()
#legLabels = [ww.market.brandLabels[x] for x in ww.market.stockbyBrand.columns]
#nsteps = ww.market.stockbyBrand.shape[0]
#nCars = np.zeros(nsteps)
#colorPal =  sns.color_palette("Set3", n_colors=len(legLabels), desat=.8)
#for i, brand in enumerate(legLabels):
#   plt.bar(np.arange(nsteps),ww.market.stockbyBrand.loc[:,i],bottom=nCars, color =colorPal[i], width=1)
#   nCars += ww.market.stockbyBrand.loc[:,i]
#plt.legend(legLabels, loc=0)
#plt.savefig('carsPerBrand.png')

#%%
if False:
    df = pd.DataFrame([],columns=['prSaf','prEco','prCon'])
    for agID in ww.nodeList[2]:
        df.loc[agID] = ww.graph.vs[agID]['preferences']
    
    
    print 'Preferences -average'
    df.mean()
    print 'Preferences - standart deviation'
    print df.std()
    
    print 'Preferences - standart deviation within friends'
    avgStd= np.zeros([1,3])    
    for agent in ww.iterNode(_hh): 
        friendList = agent.getOutNeighNodes(_thh)
        if len(friendList)> 1:
            #print df.ix[friendList].std()
            avgStd += df.ix[friendList].std().values
    print avgStd / nAgents
    prfType = np.argmax(df.values,axis=1)
    #for i, agent in enumerate(ww.iterNode(_hh)):
    #    print agent.prefTyp, prfType[i]
    df['ref'] = prfType

#%%
#for key in ww.rec:
#    stat = ww.rec[key][1]
#    plt.figure()
#    plt.plot(ww.record[stat][initPhase:])
#    plt.legend(stat, loc=0)
#    plt.title(ww.rec[key][0])
#    plt.savefig(key + '.png')

for key in ww.globRec:    
    ww.globRec[key].saveCSV()

for key in ww.globRec:
    ww.globRec[key].plot()



#%% correlation testing of weights and deviation in preferences

preDiff = list()
weights = list()


for edge in ww.iterEdges(_thh):
    preDiff.append(np.sum(np.abs(df.ix[edge.target, :-1] - df.ix[edge.source,:-1])))
    weights.append(edge['weig'])
plt.figure()
plt.scatter(preDiff,weights)
plt.xlabel('difference in preferences')
plt.ylabel('connections weight')

plt.show()

print np.corrcoef(preDiff,weights)

