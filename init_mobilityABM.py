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
parameters = Bunch()

#global parameter
parameters.scenario       = 1
parameters.nSteps         = 50  # number of simulation steps
parameters.flgSpatial     = True
parameters.connRadius     = 1.5  # radíus of cells that get an connection
parameters.tolerance      = 1.   # tolerance of friends when connecting to others (deviation in preferences)
parameters.spatial        = True
parameters.util           = 'cobb'

parameters.burnIn         = 10
parameters.properties     = ['emmisions','price']
parameters.randomAgents   = 0    # 0: prefrences dependent on agent properties - 1: random distribution
parameters.randomCarPropDeviationSTD = 0.01


# agent parametersmeter
parameters.randPref      = 1 # 0: only exteme preferences (e.g. 0,0,1) - 1: random weighted preferences
parameters.radicality    = 3 # exponent of the preferences -> high values lead to extreme differences
parameters.incomeShareForMobility = 0.3
parameters.minFriends    = 30  # number of desired friends
parameters.memoryTime    = 10  # length of the periode for which memories are stored
parameters.addYourself   = True
parameters.carNewPeriod  = 60 # months

parameters.utilObsError  = 1
parameters.recAgent      = []   # reporter agents that return a diary

parameters.writeOutput   = 1
parameters.writeNPY      = 1
parameters.writeCSV      = 0


tt = time.time()

## Scenario definition ###################################################################
if parameters.scenario == 0: #small
    parameters.writeOutput = True
    parameters.minFriends = 10
    landLayer   = np.asarray([[1, 1, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 1, 1, 1, 1]])
    population = landLayer* np.random.randint(5,20,landLayer.shape)
    urbThreshold = 13
    
elif parameters.scenario == 1: # medium
    landLayer   = np.asarray([[0, 0, 0, 0, 1, 1, 1, 0, 0], 
                              [0, 1, 0, 0, 0, 1, 1, 1, 0],
                              [1, 1, 1, 0, 0, 1, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 1, 0, 0],
                              [1, 1, 1, 1, 0, 0, 0, 0, 0]])    
    
    convMat = np.asarray([[0,1,0],[1,0,1],[0,1,0]])
    from scipy import signal
    population = landLayer* signal.convolve2d(landLayer,convMat,boundary='symm',mode='same')
    population = 20*population+ landLayer* np.random.randint(1,4,landLayer.shape)
    urbThreshold = 64
    
    

    
    
elif parameters.scenario == 2: # Niedersachsen
    reductionFactor = 200
    landLayer= gt.load_array_from_tiff('resources_nie/land_layer_3432x8640.tiff')
    landLayer[np.isnan(landLayer)] = 0
    landLayer = landLayer.astype(int)
    plt.imshow(landLayer)
    population = gt.load_array_from_tiff('resources_nie/pop_counts_ww_2005_3432x8640.tiff') / reductionFactor
    plt.imshow(population,cmap='jet')
    plt.clim([0, np.nanpercentile(population,90)])
    plt.colorbar()
    landLayer[landLayer == 1 & np.isnan(population)] =0
    urbThreshold = np.nanpercentile(population,90)
nAgents    = np.nansum(population)

minPop = np.nanmin(population[population!=0])
maxPop = np.nanmax(population)
maxDeviation = np.nanmax([(minPop-urbThreshold)**2, (maxPop-urbThreshold)**2])
minCarConvenience = .6
parameters.paraB =  minCarConvenience / maxDeviation 
parameters.urbanPopulationThreshold = urbThreshold  


assert np.sum(np.isnan(population[landLayer==1])) == 0
print 'Running with ' + str(nAgents) + ' agents'

if not parameters.randomAgents:
    parameters.synPopPath = 'resources_nie/hh_niedersachsen.csv'
    dfSynPop = pd.read_csv(parameters.synPopPath)
    hhMat = pd.read_csv('resources_nie/hh_niedersachsen.csv').values


#%% INITIALIZATION ##########################   
if parameters.writeOutput:
    # get simulation number
    fid = open("simNumber","r")
    parameters.simNo = int(fid.readline())
    fid = open("simNumber","w")
    fid.writelines(str(parameters.simNo+1))
    fid.close()
else:
    parameters.simNo = None

earth = Earth(parameters)
earth.registerEdgeType('cell-cell')
earth.registerEdgeType('cell-hh')
earth.registerEdgeType('hh-hh')
connList= earth.computeConnectionList(parameters.connRadius)
earth.initSpatialLayerNew(landLayer, connList, Cell)
ecoMin = np.percentile(dfSynPop['INCTOT']*parameters.incomeShareForMobility,20)
ecoMax = np.percentile(dfSynPop['INCTOT']*parameters.incomeShareForMobility,90)
opinion =  Opinion(indiRatio = 0.33, ecoIncomeRange=(ecoMin,ecoMax),convIncomeFraction=10000)


for cell in earth.iterNodes(_cell):
    cell.selfTest()

earth.initMarket(parameters.properties, parameters.randomCarPropDeviationSTD, burnIn=parameters.burnIn)

#init location memory
earth.enums = dict()
earth.initMemory(parameters.properties + ['utility','label','hhID'], parameters.memoryTime)


#              label , properties, initTimestpe, allTimeProduced
earth.initBrand('brown',(440, 150), 0,5000)  # combustion car

earth.initBrand('green',(250,450), 0, 100)   # green tech car

earth.initBrand('other',(120, 80), 0, 500)    # none or other

print 'Init finished after -- ' + str( time.time() - tt) + ' s'
tt = time.time()



#%% Init records

#%% Init of Households
nAgents = 0
nHH     = 0

earth.enums['priorities'] = dict()

earth.enums['priorities'][0] = 'convinience'
earth.enums['priorities'][1] = 'ecology'
earth.enums['priorities'][2] = 'money'


earth.enums['nodeTypes'] = dict()
earth.enums['nodeTypes'][1] = 'cell'
earth.enums['nodeTypes'][2] = 'household'

earth.enums['consequences'] = dict()
earth.enums['consequences'][0] = 'comfort'
earth.enums['consequences'][1] = 'eco-friendlyness'
earth.enums['consequences'][2] = 'free money'

earth.enums['mobilityTypes'] = dict()
earth.enums['mobilityTypes'][1] = 'green'
earth.enums['mobilityTypes'][0] = 'brown'
earth.enums['mobilityTypes'][2] = 'other'


earth.nPref = len(earth.enums['priorities'])
earth.nPrefTypes = [0]* earth.nPref


idx = 0
for x,y in tqdm.tqdm(earth.locDict.keys()):
    #print x,y
    nAgentsCell = int(population[x,y])
    while True:
        
        
        nPers = hhMat[idx,4]
        ages= hhMat[idx:idx+nPers,12]
        
        for person in range(nPers):
            
            #creating persons as agents
            
            age= hhMat[idx+person,12]
            sex= hhMat[idx+person,13]
            
            nKids = np.sum(ages<18)
            income = hhMat[idx,16]
            income *= parameters.incomeShareForMobility
            
              
            if age< 18:
                continue
            
            if nHH in parameters.recAgent:
                hh = Reporter(earth,'hh', x, y)
            else:
                hh = Household(earth,'hh', x, y)
            prEco, prCon, prMon = opinion.getPref(age,sex,nKids, nPers, income,parameters.radicality)
            prefTyp = np.argmax((prCon ,prEco, prMon))
            
            # seting values of hh
            hh.tolerance = parameters.tolerance
            hh.setValue('hhSize',nPers)
            hh.setValue('nKids', nKids)
            hh.setValue('income',income) 
            hh.setValue('preferences', (prCon ,prEco , prMon))
            hh.setValue('prefTyp',prefTyp)
            hh.setValue('expUtil',0)
            hh.setValue('util',0)
            hh.setValue('predMeth',0)
            hh.setValue('noisyUtil',0)
            hh.setValue('consequences', [0,0,0])
            hh.registerAgent(earth)
            earth.nPrefTypes[prefTyp] += 1
            nAgentsCell -= 1
            nAgents     += 1
            hh.connectGeoNode(earth)
            
        idx         += nPers
        nHH         += 1
        
        if nAgentsCell <= 0:
                break

earth.dequeueEdges()
print 'Agents created in -- ' + str( time.time() - tt) + ' s'

# %% Generate Network
tt = time.time()
earth.genFriendNetwork(parameters.minFriends)
earth.market.initialCarInit()
if parameters.scenario == 0:
    earth.view('output/graph.png')
print 'Network initialized in -- ' + str( time.time() - tt) + ' s'

#%% Initial actions
tt = time.time()
for household in tqdm.tqdm(earth.iterNodes(_hh)):
    household.buyCar(earth,np.random.choice(earth.market.brandProp.keys()))
    earth.market.computeStatistics()
    household.setValue('carAge', np.random.randint(0,15))
    household.calculateConsequences(earth.market)
    household.util = household.evalUtility()
    household.shareExperience(earth)
    
for cell in earth.iterNodes(_cell):
    cell.step(earth.market.kappa)
print 'Initial actions randomized in -- ' + str( time.time() - tt) + ' s'


#%% Init of agent file
tt = time.time()
earth.initAgentFile(typ = _hh)
earth.initAgentFile(typ = _cell)
print 'Agent file initialized in ' + str( time.time() - tt) + ' s'


#%% Simulation 
earth.time = -1 # hot bugfix to have both models running #TODO Fix later
print "Starting the simulation:"
for step in xrange(parameters.nSteps):
    tt = time.time()
    earth.step() # looping over all cells
    print 'Step ' + str(step) + ' done in ' +  str(time.time()-tt) + ' s',
    earth.writeAgentFile()
    print ' - agent file written in ' +  str(time.time()-tt) + ' s'
    

#%% Finishing the simulation    
print "Finalizing the simulation:"
if parameters.writeOutput:
    earth.finalizeAgentFile()
    earth.finalize()        

       
#%% post processing
#legLabels = [earth.market.brandLabels[x] for x in earth.market.stockbyBrand.columns]
#if False:
#    #plot individual utilities
#    plt.figure()
#    for agent in earth.iterNodes(_hh):
#        plt.plot(agent.utilList)    
#
#plt.figure()
#for key in brandDict.keys():
#    with sns.color_palette("Set3", n_colors=9, desat=.8):
#        plt.plot(range(para.nSteps-len(brandDict[key]),para.nSteps), brandDict[key])
#plt.legend(legLabels, loc=3)
#plt.title('Utility per brand')
#plt.savefig('utilityPerBrand.png')
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
    df = pd.DataFrame([],columns=['prCon','prEco','prMon'])
    for agID in earth.nodeList[2]:
        df.loc[agID] = earth.graph.vs[agID]['preferences']
    
    
    print 'Preferences -average'
    print df.mean()
    print 'Preferences - standart deviation'
    print df.std()
    
    print 'Preferences - standart deviation within friends'
    avgStd= np.zeros([1,3])    
    for agent in earth.iterNodes(_hh): 
        friendList = agent.getConnNodeIDs(nodeType=_hh)
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

print 'Simulation ' + str(earth.simNo) + ' finished after -- ' + str( time.time() - overallTime) + ' s'
#%%
nPeople = np.nansum(population)
nCars = float(np.nansum(np.array(earth.graph.vs[earth.nodeList[_hh]]['mobilityType'])!=2))
nGreenCars = float(np.nansum(np.array(earth.graph.vs[earth.nodeList[_hh]]['mobilityType'])==0))
nBrownCars = float(np.nansum(np.array(earth.graph.vs[earth.nodeList[_hh]]['mobilityType'])==1))
print 'Number of agents: ' + str(nPeople)
print 'Number of agents: ' + str(nCars)
print 'cars per 1000 people: ' + str(nCars/nPeople*1000.)
print 'green cars per 1000 people: ' + str(nGreenCars/nPeople*1000.)
print 'brown cars per 1000 people: ' + str(nBrownCars/nPeople*1000.)
