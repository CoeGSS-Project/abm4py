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
import sys, os
from os.path import expanduser
import igraph as ig
#home = expanduser("~")
#sys.path.append(home + '/python/decorators/')
sys.path.append('/home/gcf/python/modules/')

#from deco_util import timing_function
import numpy as np
import time
import mod_geotiff as gt
from class_mobilityABM import Household, Reporter, Cell,  Earth, Opinion
from class_auxiliary  import convertStr
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

#time spans
_month = 1
_year  = 2

#%% Scenario definition without calibraton parameters

def scenarioTestSmall(parameters):
    setup = Bunch()

    #time
    setup.nSteps         = 340     # number of simulation steps
    setup.timeUint       = _month  # unit of time per step
    setup.startDate      = [01,2005]
    setup.burnIn         = 100
    
    #spatial
    setup.isSpatial     = True
    setup.connRadius    = 1.5      # radíus of cells that get an connection
    setup.landLayer   = np.asarray([[1, 1, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 1, 1, 1, 1]])
    setup.population = setup.landLayer* np.random.randint(5,20,setup.landLayer.shape)
    
    #social
    setup.tolerance     = 1.       # tolerance of friends when connecting to others (deviation in preferences)    
    setup.addYourself   = True     # have the agent herself as a friend (have own observation)
    setup.minFriends    = 30       # number of desired friends
    setup.memoryTime    = 20       # length of the periode for which memories are stored
    setup.utilObsError  = 1
    setup.recAgent      = []       # reporter agents that return a diary
    
    #output
    setup.writeOutput   = 0
    setup.writeNPY      = 0
    setup.writeCSV      = 0
    
    #cars and infrastructure
    setup.properties    = ['emmisions','TCO']
    setup.carNewPeriod  = 24 # months
    setup.randomCarPropDeviationSTD = 0.01
    setup.greenInfraMalus = -0.3                   # malus assumed for initial disadvantages of green cars
    
    #agents
    setup.util             = 'cobb'
    setup.randPref         = 1 # 0: only exteme preferences (e.g. 0,0,1) - 1: random weighted preferences
    setup.radicality       = 3 # exponent of the preferences -> high values lead to extreme differences
    setup.incomeShareForMobility = 0.2
    setup.randomAgents     = 0    # 0: prefrences dependent on agent properties - 1: random distribution
    setup.minFriends       = 10
    
    
    minPop = np.nanmin(setup.population[setup.population!=0])
    maxPop = np.nanmax(setup.population)
    maxDeviation = np.nanmax([(minPop-parameters.urbThreshold)**2, (maxPop-parameters.urbThreshold)**2])
    minCarConvenience = 1 + setup.greenInfraMalus
    parameters.paraB =  minCarConvenience / (maxDeviation*.5)
    parameters.urbanPopulationThreshold = parameters.urbThreshold  
    
    # redefinition of setup parameters by input
    setup.update(parameters.toDict())
    
    return setup


    
def scenarioTestMedium(parameters):
    from scipy import signal
    
    setup = Bunch()

    #time
    setup.nSteps         = 10     # number of simulation steps
    setup.timeUint       = _month  # unit of time per step
    setup.startDate      = [01,2005]
    setup.burnIn         = 100
    
    #spatial
    setup.isSpatial     = True
    setup.connRadius    = 1.5      # radíus of cells that get an connection
    setup.landLayer   = np.asarray([[0, 0, 0, 0, 1, 1, 1, 0, 0], 
                              [0, 1, 0, 0, 0, 1, 1, 1, 0],
                              [1, 1, 1, 0, 0, 1, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 1, 0, 0],
                              [1, 1, 1, 1, 0, 0, 0, 0, 0]])    
    convMat = np.asarray([[0,1,0],[1,0,1],[0,1,0]])
    setup.population = setup.landLayer* signal.convolve2d(setup.landLayer,convMat,boundary='symm',mode='same')
    setup.population = 20*setup.population+ setup.landLayer* np.random.randint(1,10,setup.landLayer.shape)
    
    #social
    setup.tolerance     = 1.       # tolerance of friends when connecting to others (deviation in preferences)    
    setup.addYourself   = True     # have the agent herself as a friend (have own observation)
    setup.minFriends    = 30       # number of desired friends
    setup.memoryTime    = 20       # length of the periode for which memories are stored
    setup.utilObsError  = 1
    setup.recAgent      = []       # reporter agents that return a diary
    
    #output
    setup.writeOutput   = 1
    setup.writeNPY      = 1
    setup.writeCSV      = 0
    
    #cars and infrastructure
    setup.properties    = ['emmisions','TCO']
    setup.carNewPeriod  = 24 # months
    setup.randomCarPropDeviationSTD = 0.01
    setup.greenInfraMalus = -0.3                   # malus assumed for initial disadvantages of green cars
    
    #agents
    setup.util             = 'cobb'
    setup.randPref         = 1 # 0: only exteme preferences (e.g. 0,0,1) - 1: random weighted preferences
    setup.radicality       = 3 # exponent of the preferences -> high values lead to extreme differences
    setup.incomeShareForMobility = 0.2
    setup.randomAgents     = 0    # 0: prefrences dependent on agent properties - 1: random distribution
    setup.minFriends       = 10
    

    minPop = np.nanmin(setup.population[setup.population!=0])
    maxPop = np.nanmax(setup.population)
    maxDeviation = np.nanmax([(minPop-parameters.urbThreshold)**2, (maxPop-parameters.urbThreshold)**2])
    minCarConvenience = 1 + setup.greenInfraMalus
    parameters.paraB =  minCarConvenience / (maxDeviation*.5)
    parameters.urbanPopulationThreshold = parameters.urbThreshold  

    # redefinition of setup parameters by input
    setup.update(parameters.toDict())
    
    return setup
    
    
def scenarioNiedersachsen(parameters):
    setup = Bunch()
    
    #time
    setup.nSteps         = 340     # number of simulation steps
    setup.timeUint       = _month  # unit of time per step
    setup.startDate      = [01,2005]
    setup.burnIn         = 100
    
    #spatial
    setup.isSpatial     = True
    setup.connRadius    = 1.5      # radíus of cells that get an connection
    reductionFactor = 200
    setup.landLayer= gt.load_array_from_tiff(parameters.resourcePath + 'land_layer_62x118.tiff')
    setup.landLayer[np.isnan(setup.landLayer)] = 0
    setup.landLayer = setup.landLayer.astype(int)
    try:
        plt.imshow(setup.landLayer)

        population = gt.load_array_from_tiff(parameters.resourcePath + 'pop_counts_ww_2005_62x118.tiff') / reductionFactor
        plt.imshow(population,cmap='jet')
        plt.clim([0, np.nanpercentile(population,90)])
        plt.colorbar()
    except:
        pass
    setup.landLayer[setup.landLayer == 1 & np.isnan(population)] =0
    urbThreshold = np.nanpercentile(population,90)
    nAgents    = np.nansum(population)
    
    #social
    setup.tolerance     = 1.       # tolerance of friends when connecting to others (deviation in preferences)    
    setup.addYourself   = True     # have the agent herself as a friend (have own observation)
    setup.minFriends    = 30       # number of desired friends
    setup.memoryTime    = 20       # length of the periode for which memories are stored
    setup.utilObsError  = 1
    setup.recAgent      = []       # reporter agents that return a diary
    
    #output
    setup.writeOutput   = 0
    setup.writeNPY      = 0
    setup.writeCSV      = 0
    
    #cars and infrastructure
    setup.properties    = ['emmisions','TCO']
    setup.carNewPeriod  = 24 # months
    setup.randomCarPropDeviationSTD = 0.01
    setup.greenInfraMalus = -0.3                   # malus assumed for initial disadvantages of green cars
    
    #agents
    setup.util             = 'cobb'
    setup.randPref         = 1 # 0: only exteme preferences (e.g. 0,0,1) - 1: random weighted preferences
    setup.radicality       = 3 # exponent of the preferences -> high values lead to extreme differences
    setup.incomeShareForMobility = 0.2
    setup.randomAgents     = 0    # 0: prefrences dependent on agent properties - 1: random distribution
    setup.minFriends       = 10
    
    
    minPop = np.nanmin(population[population!=0])
    maxPop = np.nanmax(population)
    maxDeviation = np.nanmax([(minPop-urbThreshold)**2, (maxPop-urbThreshold)**2])
    minCarConvenience = 1 + setup.greenInfraMalus
    parameters.paraB =  minCarConvenience / (maxDeviation*.1)
    parameters.urbanPopulationThreshold = urbThreshold  


    assert np.sum(np.isnan(population[setup.landLayer==1])) == 0
    print 'Running with ' + str(nAgents) + ' agents'
    
    # redefinition of setup parameters by input
    setup.update(parameters.toDict())
    
    return setup
    
def scenarioChina(parameters):
    pass
    
    
###############################################################################
###############################################################################


# Mobility setup setup
def mobilitySetup(earth, parameters):
    import math
    def convienienceBrown(popDensity, paraA, paraB, paraC ,paraD, cell):
         if popDensity<cell.urbanThreshold:
            conv = paraA
         else:
            conv = max(0.05,paraA - paraB*(popDensity - cell.urbanThreshold)**2)          
         return conv
            
    def convienienceGreen(popDensity, paraA, paraB, paraC ,paraD, cell):
        conv = max(0.05, paraA - paraB*(popDensity - cell.urbanThreshold)**2 + cell.kappa  )
        return conv
    
    def convienienceOther(popDensity, paraA, paraB, paraC ,paraD, cell):
        conv = paraC/(1+math.exp((-paraD)*(popDensity-cell.urbanThreshold)))
        return conv
    
                         #(emmisions, TCO)         
    earth.initBrand('brown',(440., 200.), convienienceBrown, 0, 20000)  # combustion car
    
    earth.initBrand('green',(250., 500.), convienienceGreen, 0, 400)   # green tech car
    
    earth.initBrand('other',(120., 80.), convienienceOther, 0, 2000)    # none or other
    
    return earth
    ##############################################################################

def householdSetup(earth, parameters):
    tt = time.time()
    idx = 0
    nAgents = 0
    nHH     = 0
    
    
    if not parameters.randomAgents:
        parameters.synPopPath = parameters['resourcePath'] + 'hh_niedersachsen.csv'
        dfSynPop = pd.read_csv(parameters.synPopPath)
        hhMat = pd.read_csv(parameters.synPopPath).values
        
    ecoMin = np.percentile(dfSynPop['INCTOT']*parameters.incomeShareForMobility,20)
    ecoMax = np.percentile(dfSynPop['INCTOT']*parameters.incomeShareForMobility,90)
    opinion =  Opinion(indiRatio = 0.33, ecoIncomeRange=(ecoMin,ecoMax),convIncomeFraction=10000)
    
    for x,y in tqdm.tqdm(earth.locDict.keys()):
        #print x,y
        nAgentsCell = int(parameters.population[x,y])
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
                prEco, prCon, prMon, prImi = opinion.getPref(age,sex,nKids, nPers, income,parameters.radicality)
                prefTyp = np.argmax((prCon, prEco, prMon, prImi))
                
                # seting values of hh
                hh.tolerance = parameters.tolerance
                hh.setValue('hhSize',nPers)
                hh.setValue('nKids', nKids)
                hh.setValue('income',income) 
                hh.setValue('preferences', (prCon ,prEco , prMon, prImi))
                hh.setValue('prefTyp',prefTyp)
                hh.setValue('expUtil',0)
                hh.setValue('util',0)
                hh.setValue('predMeth',0)
                hh.setValue('noisyUtil',0)
                hh.setValue('consequences', [0,0,0,0])
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
    return earth



#%% Run of the simulation model ###############################################   

def initEarth(parameters):
    tt = time.time()
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
    earth.initSpatialLayerNew(parameters.landLayer, connList, Cell)
    
    
    
    
    
    earth.initMarket(parameters.properties, 
                     parameters.randomCarPropDeviationSTD, 
                     burnIn=parameters.burnIn, 
                     greenInfraMalus=parameters.greenInfraMalus)
    earth.market.mean = np.array([400.,300.])
    earth.market.std = np.array([100.,50.])
    #init location memory
    earth.enums = dict()
    earth.initMemory(parameters.properties + ['utility','label','hhID'], parameters.memoryTime)

    
    earth.enums['priorities'] = dict()
    
    earth.enums['priorities'][0] = 'convinience'
    earth.enums['priorities'][1] = 'ecology'
    earth.enums['priorities'][2] = 'money'
    earth.enums['priorities'][3] = 'imitation'
    
    earth.enums['properties'] = dict()
    earth.enums['properties'][1] = 'emissions'
    earth.enums['properties'][2] = 'TCO'
    
    earth.enums['nodeTypes'] = dict()
    earth.enums['nodeTypes'][1] = 'cell'
    earth.enums['nodeTypes'][2] = 'household'
    
    earth.enums['consequences'] = dict()
    earth.enums['consequences'][0] = 'comfort'
    earth.enums['consequences'][1] = 'eco-friendliness'
    earth.enums['consequences'][2] = 'remaining money'
    earth.enums['consequences'][3] = 'similarity'
    
    earth.enums['mobilityTypes'] = dict()
    earth.enums['mobilityTypes'][1] = 'green'
    earth.enums['mobilityTypes'][0] = 'brown'
    earth.enums['mobilityTypes'][2] = 'other'
    
    
    earth.nPref = len(earth.enums['priorities'])
    earth.nPrefTypes = [0]* earth.nPref
    
    print 'Init finished after -- ' + str( time.time() - tt) + ' s'
    return earth                  

    
#%% cell convenience test
def cellTest(earth, parameters):    
    convArray = np.zeros([earth.market.getNTypes(),len(earth.nodeList[1])])
    popArray = np.zeros([len(earth.nodeList[1])])
    for i, cell in enumerate(earth.iterNodes(_cell)):
        convAll, population = cell.selfTest()
        convArray[:,i] = convAll
        popArray[i] = population
    plt.figure()
    
    for i in range(earth.market.getNTypes()):    
        plt.subplot(2,2,i+1)
        plt.scatter(popArray,convArray[i,:])    
        plt.title('convenience of ' + earth.enums['mobilityTypes'][i])
    plt.show()
    
    
def generateNetwork(earth, parameters):
    # %% Generate Network
    tt = time.time()
    earth.genFriendNetwork(parameters.minFriends)
    print 'Network initialized in -- ' + str( time.time() - tt) + ' s'
    
def initMobilityTypes(earth, parameters):
    earth.market.initialCarInit()
 
def initGlobalRecords(earth, parameters):
    earth.registerRecord('stock', 'total use per mobility type', earth.enums['mobilityTypes'].values(), style ='plot')
    
def runModel(earth, parameters):

    #%% Initial actions
    tt = time.time()
    for household in tqdm.tqdm(earth.iterNodes(_hh)):
        household.buyCar(earth,np.random.choice(range(earth.market.nMobTypes)))
        #earth.market.computeStatistics()
        household.setValue('carAge', np.random.randint(0,15))
        
    for household in tqdm.tqdm(earth.iterNodes(_hh)):    
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

       
import csv
if __name__ == '__main__':
    dirPath = os.path.dirname(os.path.realpath(__file__))
    # got csv file containing parameters
    if len(sys.argv) > 1:
        
        fileName = sys.argv[1]
        parameters = Bunch()
        for item in csv.DictReader(open(fileName)):
            parameters[item['name']] = convertStr(item['value'])
        
        
        
    # no csv file given
    else:
        parameters = Bunch()
        parameters.urbThreshold = 13
        parameters.convIncomeFraction = 1000
        

    parameters.scenario       = 1

    if parameters.scenario == 0:
        parameters.resourcePath = dirPath + '/resources_nie/'
        parameters = scenarioTestSmall(parameters)
   
    elif parameters.scenario == 1:
        parameters.resourcePath = dirPath + '/resources_nie/'
        parameters = scenarioTestMedium(parameters)
    
    elif parameters.scenario == 2:
        parameters.resourcePath = dirPath + '/resources_nie/'
        parameters = scenarioNiedersachsen(parameters)
        
    #%% Init 
    earth = initEarth(parameters)
    
    mobilitySetup(earth, parameters)
    
    householdSetup(earth, parameters)
    
    cellTest(earth, parameters)
    
    generateNetwork(earth, parameters)
    
    initMobilityTypes(earth, parameters)
    
    initGlobalRecords(earth, parameters)
    
    if parameters.scenario == 0:
        earth.view('output/graph.png')
  
    #%% run of the model ################################################
    runModel(earth, parameters)
    
    
    
    
    #%% postprocessing
    if True:
        df = pd.DataFrame([],columns=['prCon','prEco','prMon','prImi'])
    for agID in earth.nodeList[2]:
        df.loc[agID] = earth.graph.vs[agID]['preferences']
    
    
    print 'Preferences -average'
    print df.mean()
    print 'Preferences - standart deviation'
    print df.std()
    
    print 'Preferences - standart deviation within friends'
    avgStd= np.zeros([1,4])    
    for agent in earth.iterNodes(_hh): 
        friendList = agent.getConnNodeIDs(nodeType=_hh)
        if len(friendList)> 1:
            #print df.ix[friendList].std()
            avgStd += df.ix[friendList].std().values
    nAgents    = np.nansum(parameters.population)         
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
    
    print 'Simulation ' + str(earth.simNo) + ' finished after -- ' + str( time.time() - overallTime) + ' s'
    #%%
    nPeople = np.nansum(parameters.population)
    nCars = float(np.nansum(np.array(earth.graph.vs[earth.nodeList[_hh]]['mobilityType'])!=2))
    nGreenCars = float(np.nansum(np.array(earth.graph.vs[earth.nodeList[_hh]]['mobilityType'])==0))
    nBrownCars = float(np.nansum(np.array(earth.graph.vs[earth.nodeList[_hh]]['mobilityType'])==1))
    print 'Number of agents: ' + str(nPeople)
    print 'Number of agents: ' + str(nCars)
    print 'cars per 1000 people: ' + str(nCars/nPeople*1000.)
    print 'green cars per 1000 people: ' + str(nGreenCars/nPeople*1000.)
    print 'brown cars per 1000 people: ' + str(nBrownCars/nPeople*1000.)

    
 
