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

#%%
#TODO
# random iteration (even pairs of agents)
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
from class_mobilityABM import Household, Reporter, Cell,  Earth, Opinion
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


#%% INIT
para = Bunch()

#global parameter
para.scenario       = 0
para.nSteps         = 200 # number of simulation steps
para.flgSpatial     = True
para.connRadius     = 2.1  # radíus of cells that get an connection
para.tolerance      = 1.   # tolerance of friends when connecting to others (deviation in preferences)

para.initPhase      = 20
para.properties     = ['emmisions','price']
para.randomAgents   = 0 # 0: prefrences dependent on agent properties - 1: random distribution
para.randomCarPropDeviationSTD = 0.01


# agent parameter
para.randPref      = 1 # 0: only exteme preferences (e.g. 0,0,1) - 1: random weighted preferences
para.radicality    = 3 # exponent of the preferences -> high values lead to extreme differences
para.incomeShareForMobility = 0.5
para.minFriends = 50  # number of desired friends
para.memoryTime  = 10  # length of the periode for which memories are stored
para.addYourself = True
para.carNewPeriod = 6

para.utilObsError = 1
para.recAgent   = []   # reporter agents that return a diary

para.writeOutput = False
para.writeNPY    = True
para.writeCSV    = False

tt = time.time()

## Scenario definition ###################################################################
if para.scenario == 0: #small
    para.writeOutput = False
    para.minFriends = 10
    landLayer   = np.asarray([[1, 1, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 1, 1, 1, 1]])
    population = landLayer*20
    
elif para.scenario == 1: # medium
    landLayer   = np.asarray([[0, 0, 0, 0, 0, 1, 1, 1], 
                              [1, 1, 1, 0, 0, 1, 1, 1],
                              [1, 1, 1, 0, 0, 1, 0, 0],
                              [1, 1, 1, 1, 1, 1, 0, 0],
                              [1, 1, 1, 1, 1, 1, 0, 0]])    
    population = landLayer*100
    
elif para.scenario == 2: # Niedersachsen
    reductionFactor = 100
    landLayer= gt.load_array_from_tiff('resources_nie/land_layer_3432x8640.tiff')
    landLayer[np.isnan(landLayer)] = 0
    landLayer = landLayer.astype(int)
    plt.imshow(landLayer)
    population = gt.load_array_from_tiff('resources_nie/pop_counts_ww_2005_3432x8640.tiff') / reductionFactor
    plt.imshow(population)

    landLayer[landLayer == 1 & np.isnan(population)] =0

nAgents    = np.nansum(population)
assert np.sum(np.isnan(population[landLayer==1])) == 0
print 'Running with ' + str(nAgents) + ' agents'

if not para.randomAgents:
    para.synPopPath = 'resources_nie/hh_niedersachsen.csv'
    dfSynPop = pd.read_csv(para.synPopPath)
    hhMat = pd.read_csv('resources_nie/hh_niedersachsen.csv').values


#%% INITIALIZATION ##########################   
if para.writeOutput:
    # get simulation number
    fid = open("simNumber","r")
    para.simNo = int(fid.readline())
    fid = open("simNumber","w")
    fid.writelines(str(para.simNo+1))
    fid.close()
else:
    para.simNo = None

earth = Earth(para.nSteps, para.simNo, spatial=para.flgSpatial)
connList= earth.computeConnectionList(para.connRadius)
earth.initSpatialLayerNew(landLayer, connList, Cell)
ecoMin = np.percentile(dfSynPop['INCTOT']*para.incomeShareForMobility,20)
ecoMax = np.percentile(dfSynPop['INCTOT']*para.incomeShareForMobility,90)
opinion =  Opinion(indiRatio = 0.33, ecoIncomeRange=(ecoMin,ecoMax),convIncomeFraction=10000)

# transfer all parameters to earth
earth.setParameters(Bunch.toDict(para)) 

earth.initMarket(para.properties, para.randomCarPropDeviationSTD)

#init location memory
earth.enums = dict()
#earth.initMemory(para.properties + ['utility','label','hhID'], para.memoryTime)


#                       emmisions price']
earth.addBrand('green',(10,220*12), 0)   # green tech car
earth.addBrand('brown',(20, 100*12), 0)  # combustion car
earth.addBrand('none',(3, 150*12), 0)    # none or other

print 'Init finished after -- ' + str( time.time() - tt) + ' s'
tt = time.time()



#%% Init records

#%% Init of Households
nAgents = 0
nHH     = 0

earth.enums['prefTypes'] = dict()
earth.enums['prefTypes'][0] = 'ecology'
earth.enums['prefTypes'][1] = 'convinience'
earth.enums['prefTypes'][2] = 'money'
earth.nPref = len(earth.enums['prefTypes'])
earth.nPrefTypes = [0]* earth.nPref

earth.enums['nodeTypes'] = dict()
earth.enums['nodeTypes'][1] = 'cell'
earth.enums['nodeTypes'][2] = 'household'


idx = 0
for x,y in tqdm.tqdm(earth.locDict.keys()):
    #print x,y
    nAgentsCell = int(population[x,y])
    while True:
        if nHH in para.recAgent:
            hh = Reporter(earth,'hh', x, y)
        else:
            hh = Household(earth,'hh', x, y)
        hh.connectGeoNode(earth)
        hh.tolerance = para.tolerance
        nPers = hhMat[idx,4]
        hh.setValue('hhSize',nPers)
        age= hhMat[idx:idx+nPers,12]
        sex= hhMat[idx:idx+nPers,12]
        
        nKids = np.sum(age<18)
        hh.setValue('nKids', nKids)
        income = hhMat[idx,16]
        income *= para.incomeShareForMobility
        hh.setValue('income',income)   
        if len(np.where(age>=18)[0]) > 0:
            idxAdult = np.random.choice(np.where(age>=18)[0])
        else:
            idxAdult = 0
        prEco, prCon, prMon = opinion.getPref(age[idxAdult],sex[idxAdult],nKids,income,para.radicality)
        # normal 
        hh.setValue('preferences', (prEco, prCon, prMon))
        hh.prefTyp = np.argmax((prEco, prCon, prMon))
        hh.setValue('prefTyp',hh.prefTyp)
        hh.setValue('expUtil',0)
        hh.setValue('predMeth',0)
        hh.setValue('noisyUtil',0)

        hh.registerAgent(earth)
        earth.nPrefTypes[hh.prefTyp] += 1
        nAgentsCell -= nPers
        nAgents     += nPers
        idx         += nPers
        nHH         +=1
        if nAgentsCell < 0:
            break
earth.dequeueEdges()
print 'Agents created in -- ' + str( time.time() - tt) + ' s'

# %% Generate Network
tt = time.time()
earth.genFriendNetwork(para.minFriends)
earth.market.initCars()
if para.scenario == 0:
    earth.view('output/graph.png')
print 'Network initialized in -- ' + str( time.time() - tt) + ' s'

#%% Initial actions
tt = time.time()
for household in earth.iterNodes(_hh):
    household.buyCar(earth,np.random.choice(earth.market.brandProp.keys()))
    household.car['age'] = np.random.randint(0,15)
    household.util = household.evalUtility(earth)
    household.shareExperience(earth)
    
for cell in earth.iterNodes(_cell):
    cell.step()
print 'Initial actions randomized in -- ' + str( time.time() - tt) + ' s'


#%% Init of agent file
tt = time.time()
earth.initAgentFile(typ = _hh)
earth.initAgentFile(typ = _cell)
print 'Agent file initialized in ' + str( time.time() - tt) + ' s'


#%% Simulation 
print "Starting the simulation:"
for step in xrange(para.nSteps):
    tt = time.time()
    earth.step() # looping over all cells
    print 'Step ' + str(step) + ' done in ' +  str(time.time()-tt) + ' s',
    earth.writeAgentFile()
    print ' - agent file written in ' +  str(time.time()-tt) + ' s'
    

#%% Finishing the simulation    
print "Finishing thesimulation:"
if para.writeOutput:
    earth.finalizeAgentFile()
    earth.finalize()        

       
#%% post processing
legLabels = [earth.market.brandLabels[x] for x in earth.market.stockbyBrand.columns]
if False:
    #plot individual utilities
    plt.figure()
    for agent in earth.iterNodes(_hh):
        plt.plot(agent.utilList)    

plt.figure()
for key in brandDict.keys():
    with sns.color_palette("Set3", n_colors=9, desat=.8):
        plt.plot(range(para.nSteps-len(brandDict[key]),para.nSteps), brandDict[key])
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
if True:
    df = pd.DataFrame([],columns=['prSaf','prEco','prCon','prMon'])
    for agID in earth.nodeList[2]:
        df.loc[agID] = earth.graph.vs[agID]['preferences']
    
    
    print 'Preferences -average'
    df.mean()
    print 'Preferences - standart deviation'
    print df.std()
    
    print 'Preferences - standart deviation within friends'
    avgStd= np.zeros([1,4])    
    for agent in earth.iterNodes(_hh): 
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

pref = np.zeros([earth.graph.vcount(), 4])
pref[-earth.nAgents:,:] = np.array(earth.graph.vs[-earth.nAgents:]['preferences'])
idx = list()
for edge in earth.iterEdges(_thh):
    edge['prefDiff'] = np.sum(np.abs(pref[edge.target, :] - pref[edge.source,:]))
    idx.append(edge.index)
    
    
plt.figure()
plt.scatter(earth.graph.es['prefDiff'],earth.graph.es['weig'])
plt.xlabel('difference in preferences')
plt.ylabel('connections weight')

plt.show()
x = np.asarray(earth.graph.es['prefDiff'])[idx].astype(float)
y = np.asarray(earth.graph.es['weig'])[idx].astype(float)
print np.corrcoef(x,y)

print 'Simulation finished after -- ' + str( time.time() - overallTime) + ' s'