#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://wwww.globalclimateforum.org

CAR INNOVATION MARKET MODEL
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
import sys
from os.path import expanduser
#home = expanduser("~")
#sys.path.append(home + '/python/decorators/')
sys.path.append('modules/')
#from deco_util import timing_function
import numpy as np
import time
import mod_geotiff as gt
from class_agents import Household, Reporter, Cell
from class_world import Earth
import matplotlib
matplotlib.use('TKAgg')
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
randomAgents  = 0 # 0: prefrences dependent on agent properties - 1: random distribution
randPref      = 1 # 0: only exteme preferences (e.g. 0,0,1) - 1: random weighted preferences
radicality    = 3 # exponent of the preferences -> high values lead to extreme differences

nSteps     = 100 # number of simulation steps
initPhase = 20
properties = ['weig','range','consum','vol','speed', 'price']
randomCarPropDeviationSTD = 0.01
minFriends = 30  # number of desired friends
memoryTime  = 15  # length of the periode for which memories are stored
carNewPeriod = 5
recAgent   = [5,10,15]   # reporter agents that return a diary

tt = time.time()

## Scenario definition ###################################################################
if scenario == 0: #small
    minFriends = 5
    nAgents    = 100
    landLayer   = np.asarray([[1, 1, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 1, 1, 1, 1]])
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

if not randomAgents:
    df = pd.read_csv('resources_nie/hh_niedersachsen.csv')
    hhMat = pd.read_csv('resources_nie/hh_niedersachsen.csv').values


#%% INITIALIZATION ##########################   
earth = Earth(nSteps, spatial=flgSpatial)
connList= earth.computeConnectionList(connRadius)
earth.initSpatialLayerNew(landLayer, connList, Cell)
#
earth.props = properties
earth.initMarket(properties, randomCarPropDeviationSTD)
#earth.initObsAtLoc(properties)
earth.initOpinionGen(indiRatio = 0.33, ecoIncomeRange=(10000,30000),convIncomeFraction=25000)
earth.scenario = scenario
earth.radicality = radicality
earth.carNewPeriod = carNewPeriod
#init location memory

earth.enums = dict()
 
earth.initMemory(properties + ['utility','label','hhID'], memoryTime)


#                           weig range consum vol   speed price']
#earth.market.addBrand('green',      (1500,300,  3.0,   4,    130,  40000))    
#earth.addBrand('city', (1500, 600,  4.5,   4.7,  140,  35000))   #city car 
earth.addBrand('medium',(2000, 600,  5.5,   5.5,  170,  180*12), 0)   #small car 
earth.addBrand('family',(2400, 800,  6.5,   8.0,  140,  220*12), 0)   #family car

earth.addBrand('Diesel',(2500, 900,  7.5,   8.5,  150,  280*12), 0)   #Diesels 
earth.addBrand('Sport', (1500, 800,  9.0,   5.0,  250,  400*12), 0)   #sports 
earth.addBrand('small',(1800, 700,  5.0,   5.0,  160,  120*12),50)   #small car 
earth.addBrand('Diesel+',(2600, 1000, 7.0,   8.5,  160,  270*12),60) 
earth.addBrand('city', (1500, 600,  4.5,   4.7,  140,  160*12),70)
earth.addBrand('SUV',   (3500, 500,  9.0,   9.0,  180,  350*12),80)   #suv 
earth.addBrand('green',(1500, 450,  2.0,   4,    130,  250*12),90)
    

print 'Init finished after -- ' + str( time.time() - tt) + ' s'
tt = time.time()



#%% Init records

earth.registerRecord('avgUtil', 'Average utility',["overall", "prefSafety", "prefEcology", "prefConvinience","prefMoney"])
earth.registerRecord('carStock', 'Cars per label', earth.market.brandsToInit, style='stackedBar')
earth.registerRecord('sales', 'Sales per Preference',["salesSaf", "salesEco", "salesCon", "salesMoney"])
earth.registerRecord('maxSpeedStat', 'statistical distribution for max speed',["mean", "mean+STD", "mean-STD"])

#%% Init of Households
nAgents = 0
nHH     = 0

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

if randomAgents:
    idx = 0
    for x,y in tqdm.tqdm(earth.locDict.keys()):
        nAgentsCell = int(population[x,y])
        while True:
            
            if nHH in recAgent:
                hh = Reporter(earth,'hh', x, y)
            else:
                hh = Household(earth,'hh', x, y)
            hh.connectGeoNode(earth)
            hh.tolerance = tolerance
            hhSize, ageList, sexList, income,  nKids, prSaf, prEco, prCon ,prMon= earth.generateHH()
            #hhSize = int(np.ceil(np.abs(np.random.randn(1)*2)))
            #hh.setValue('hhSize', hhSize)
            #hh.setValue('age',ageList)         
            hh.setValue('income',income)
            #hh.setValue('nKids', nKids)
            
            
            # normal 
            #hh.setValue('preferences', (prSaf, prEco, prCon))
            #hh.prefTyp = np.argmax((prSaf, prEco, prCon))
            
            # test
            if randPref == 0:
                preferences = np.zeros(earth.nPref)
                preferences[np.random.randint(0,earth.nPref)] = 1
                #xx = xx / sum(xx)
                #prSaf, prEco, prCon = xx
            elif randPref == 1:
                preferences = np.random.random(earth.nPref)** radicality
                preferences = preferences / sum(preferences)
                #prSaf, prEco, prCon, prMon = xx
                
            
            hh.setValue('preferences', tuple(preferences))
            hh.prefTyp = np.argmax((preferences))
            hh.setValue('prefTyp',hh.prefTyp)
            
            hh.registerAgent(earth)
            earth.nPrefTypes[hh.prefTyp] += 1
            nPers       = hhSize
            nAgentsCell -= nPers
            nAgents     += nPers
            idx         += nPers
            nHH         +=1
            if nAgentsCell < 0:
                break
                
else:
    idx = 0
    for x,y in tqdm.tqdm(earth.locDict.keys()):
        #print x,y
        nAgentsCell = int(population[x,y])
        while True:
            if nHH in recAgent:
                hh = Reporter(earth,'hh', x, y)
            else:
                hh = Household(earth,'hh', x, y)
            hh.connectGeoNode(earth)
            hh.tolerance = tolerance
            nPers = hhMat[idx,4]
            hh.setValue('hhSize',nPers)
            age= hhMat[idx:idx+nPers,12]
            #hh.setValue('age',list(age))
            sex= hhMat[idx:idx+nPers,12]
            #hh.setValue('sex',list(sex))
            
            nKids = np.sum(age<18)
            hh.setValue('nKids', nKids)
            income = hhMat[idx,16]
            hh.setValue('income',income)   
            if len(np.where(age>=18)[0]) > 0:
                idxAdult = np.random.choice(np.where(age>=18)[0])
            else:
                idxAdult = 0
            prSaf, prEco, prCon, prMon = earth.og.getPref(age[idxAdult],sex[idxAdult],nKids,income,radicality)
            # normal 
            hh.setValue('preferences', (prSaf, prEco, prCon, prMon))
            
            hh.prefTyp = np.argmax((prSaf, prEco, prCon, prMon))
            hh.setValue('prefTyp',hh.prefTyp)
            # test
            #hh.setValue('preferences', (0,0,1))
            #hh.prefTyp = 0

            hh.registerAgent(earth)
            earth.nPrefTypes[hh.prefTyp] += 1
            nAgentsCell -= nPers
            nAgents     += nPers
            idx         += nPers
            nHH         +=1
            if nAgentsCell < 0:
                break
earth.dequeueEdges()
                
#earth.recAgent = earth.nodeList[_hh][recAgent]            
            
print 'Agents created in -- ' + str( time.time() - tt) + ' s'

earth.genFriendNetwork(minFriends)
earth.market.initCars()
if scenario == 0:
    earth.view()

tt = time.time()
   
import pandas as pd
#prefUtil = pd.DataFrame([],columns= ["prSaf", "prEco", "prCon"] )    
#prefUtil.loc[0] = 0
for household in earth.iterNode(_hh):
    household.buyCar(earth,np.random.choice(earth.market.brandProp.keys()))
    household.car['age'] = np.random.randint(0,15)
    household.util = household.evalUtility(earth)
    household.utilList.append(household.util)
    household.shareExperience(earth)
    
for cell in earth.iterNode(_cell):
    cell.step()
    
#earth.record.loc[earth.time, earth.rec["avgUtilPref"][1]] /= earth.nPrefTypes
earth.globalRec['avgUtil'].div(earth.time, [earth.nAgents] + earth.nPrefTypes)

#earth.view()
util = np.asarray(earth.graph.vs[earth.nodeList[2]]['util'])
import matplotlib.pyplot as plt

if False:
    plt.figure()
    plt.subplot(2,5,1)
    plt.hist(util,30)
    plt.xlim([0,1])

brandDict = dict()
for key in earth.market.obsDict.keys():
    brandDict[key] = [np.mean(earth.market.obsDict[earth.time][key])]
#for es in earth.graph.es:
#    print (es.source, es.target)
#for x in xrange(3,21):
#    print earth.agDict[x].obs
    
    #choice = earth.agDict[x].optimalChoice(earth)  
    
earth.avgDegree = [np.mean(earth.graph.vs[earth.nodeList[2]].degree())]

tt = time.time()
earth.initAgentFile(typ = _hh)
earth.initAgentFile(typ = _cell)
print 'Agent file initialized in ' + str( time.time() - tt) + ' s'
earth.writeAgentFile()
#%% Simulation 
for step in xrange(1,nSteps):
    
    earth.step() # looping over all cells
                 # and agents
    
#    for x in earth.nodeList[2]:
#        agent = earth.agDict[x]
        
        #agent.socialize(earth)
    
    print 'Step ' + str(step) + ' done in ' +  str(time.time()-tt) + ' s'
    tt = time.time()
    earth.writeAgentFile()
    print 'Agent file written in ' +  str(time.time()-tt) + ' s'
    util = np.asarray(earth.graph.vs[earth.nodeList[2]]['util'])
    # plot utility per brand
    for key in earth.market.obsDict[earth.time].keys():
        if key in brandDict:
            brandDict[key].append(np.mean(earth.market.obsDict[earth.time][key]))
        else:
            brandDict[key] = [np.mean(earth.market.obsDict[earth.time][key])]
    
            
    if False:  
        plt.subplot(2,5,step)
        plt.hist(util,30)
        plt.xlim([0,1])   
    
earth.finalize()        

       
#%% post processing
legLabels = [earth.market.brandLabels[x] for x in earth.market.stockbyBrand.columns]
if False:
    #plot individual utilities
    plt.figure()
    for agent in earth.iterNode(_hh):
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
#    plt.plot(earth.market.stockbyBrand)
#plt.legend(legLabels, loc=3)
#plt.title('Cars per brand')
#plt.savefig('carsPerBrand.png')


#%% Cars per brand
#plt.figure()
#legLabels = [earth.market.brandLabels[x] for x in earth.market.stockbyBrand.columns]
#nsteps = earth.market.stockbyBrand.shape[0]
#nCars = np.zeros(nsteps)
#colorPal =  sns.color_palette("Set3", n_colors=len(legLabels), desat=.8)
#for i, brand in enumerate(legLabels):
#   plt.bar(np.arange(nsteps),earth.market.stockbyBrand.loc[:,i],bottom=nCars, color =colorPal[i], width=1)
#   nCars += earth.market.stockbyBrand.loc[:,i]
#plt.legend(legLabels, loc=0)
#plt.savefig('carsPerBrand.png')

#%%
if False:
    df = pd.DataFrame([],columns=['prSaf','prEco','prCon'])
    for agID in earth.nodeList[2]:
        df.loc[agID] = earth.graph.vs[agID]['preferences']
    
    
    print 'Preferences -average'
    df.mean()
    print 'Preferences - standart deviation'
    print df.std()
    
    print 'Preferences - standart deviation within friends'
    avgStd= np.zeros([1,3])    
    for agent in earth.iterNode(_hh): 
        friendList = agent.getOutNeighNodes(_thh)
        if len(friendList)> 1:
            #print df.ix[friendList].std()
            avgStd += df.ix[friendList].std().values
    print avgStd / nAgents
    prfType = np.argmax(df.values,axis=1)
    #for i, agent in enumerate(earth.iterNode(_hh)):
    #    print agent.prefTyp, prfType[i]
    df['ref'] = prfType

#%%
#for key in earth.rec:
#    stat = earth.rec[key][1]
#    plt.figure()
#    plt.plot(earth.record[stat][initPhase:])
#    plt.legend(stat, loc=0)
#    plt.title(earth.rec[key][0])
#    plt.savefig(key + '.png')


#%% correlation testing of weights and deviation in preferences

preDiff = list()
weights = list()

pref = np.zeros([earth.graph.vcount(), 3])
pref[-earth.nAgents:,:] = np.array(earth.graph.vs[-earth.nAgents:]['preferences'])
for edge in earth.iterEdges(_thh):
    preDiff.append(np.sum(np.abs(pref[edge.target, :] - pref[edge.source,:])))
    weights.append(edge['weig'])
plt.figure()
plt.scatter(preDiff,weights)
plt.xlabel('difference in preferences')
plt.ylabel('connections weight')

plt.show()

print np.corrcoef(preDiff,weights)

