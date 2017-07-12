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
from copy import copy
import csv
from scipy import signal


overallTime = time.time()
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

#%% Scenario definition without calibraton parameters

def scenarioTestSmall(parameterInput, dirPath):
    setup = Bunch()
    
    #general 
    setup.resourcePath = dirPath + '/resources_nie/'
    setup.progressBar  = True
    
    #time
    setup.timeUint         = _month  # unit of time per step
    setup.startDate        = [01,2005]   

    #spatial
    setup.reductionFactor = 50000
    setup.isSpatial       = True
    setup.connRadius      = 2.1      # radíus of cells that get an connection
    setup.landLayer   = np.asarray([[1, 1, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 1, 1, 1, 1]])
    setup.regionIdRaster    = setup.landLayer*1518
    setup.regionIdRaster[0:,0:2] = 6321
    setup.population = setup.landLayer* np.random.randint(5,20,setup.landLayer.shape)
    
    #social
    setup.addYourself   = True     # have the agent herself as a friend (have own observation)
    setup.recAgent      = []       # reporter agents that return a diary
    
    #output
    setup.writeOutput   = 1
    setup.writeNPY      = 1
    setup.writeCSV      = 0
    
    #cars and infrastructure
    setup.properties    = ['emmisions','TCO']

    #agents
    setup.randomAgents     = False
    setup.omniscientAgents = False    

    # redefinition of setup parameters used for automatic calibration
    setup.update(parameterInput.toDict())
    
    # calculate dependent parameters
    maxDeviation = (setup.urbanCritical - setup.urbanThreshold)**2
    minCarConvenience = 1 + setup.kappa
    setup.convB =  minCarConvenience / (maxDeviation)
 
    # only for packing with care
    dummy   = gt.load_array_from_tiff(setup.resourcePath + 'land_layer_62x118.tiff')
    dummy   = gt.load_array_from_tiff(setup.resourcePath + 'pop_counts_ww_2005_62x118.tiff') / setup.reductionFactor
    dummy   = gt.load_array_from_tiff(setup.resourcePath + 'subRegionRaster_62x118.tiff')    
    del dummy
    
    return setup


    
def scenarioTestMedium(parameterInput, dirPath):

    
    setup = Bunch()


    #general 
    setup.resourcePath = dirPath + '/resources_nie/'
    setup.progressBar  = True
    
    #time
    setup.timeUint         = _month  # unit of time per step
    setup.startDate        = [01,2005]   

        
    #spatial
    setup.reductionFactor = 5000 # only and estimation in comparison to niedersachsen
    setup.isSpatial     = True
    setup.landLayer   = np.asarray([[0, 0, 0, 0, 1, 1, 1, 0, 0], 
                                    [0, 1, 0, 0, 0, 1, 1, 1, 0],
                                    [1, 1, 1, 0, 0, 1, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1, 1, 0, 0],
                                    [1, 1, 1, 1, 0, 0, 0, 0, 0]])    
    setup.regionIdRaster    = setup.landLayer*1518
    setup.regionIdRaster[3:,0:3] = 6321
    convMat = np.asarray([[0,1,0],[1,0,1],[0,1,0]])
    setup.population = setup.landLayer* signal.convolve2d(setup.landLayer,convMat,boundary='symm',mode='same')
    setup.population = 20*setup.population+ setup.landLayer* np.random.randint(1,10,setup.landLayer.shape)
    
    #social
    setup.addYourself   = True     # have the agent herself as a friend (have own observation)
    setup.recAgent      = []       # reporter agents that return a diary
    
    #output
    setup.writeOutput   = 1
    setup.writeNPY      = 1
    setup.writeCSV      = 0
    
    #cars and infrastructure
    setup.properties    = ['emmisions','TCO']

    #agents
    setup.randomAgents     = False
    setup.omniscientAgents = False    

    # redefinition of setup parameters used for automatic calibration
    setup.update(parameterInput.toDict())
    
    # calculate dependent parameters
    maxDeviation = (setup.urbanCritical - setup.urbanThreshold)**2
    minCarConvenience = 1 + setup.kappa
    setup.convB =  minCarConvenience / (maxDeviation)
    
    # only for packing with care
    dummy   = gt.load_array_from_tiff(setup.resourcePath + 'land_layer_62x118.tiff')
    dummy   = gt.load_array_from_tiff(setup.resourcePath + 'pop_counts_ww_2005_62x118.tiff') / setup.reductionFactor
    dummy   = gt.load_array_from_tiff(setup.resourcePath + 'subRegionRaster_62x118.tiff')    
    del dummy
    
    return setup
    
    
def scenarioNiedersachsen(parameterInput, dirPath):
    setup = Bunch()
    
    #general 
    setup.resourcePath = dirPath + '/resources_nie/'
    setup.progressBar  = True
    
    #time
    setup.nSteps           = 340     # number of simulation steps
    setup.timeUint         = _month  # unit of time per step
    setup.startDate        = [01,2005]
    setup.burnIn           = 100
    setup.omniscientBurnIn = 10       # no. of first steps of burn-in phase with omniscient agents, max. =burnIn
    
    #spatial
    setup.isSpatial     = True
    setup.connRadius    = 2.5      # radíus of cells that get an connection
    setup.reductionFactor = 200.
    
    if hasattr(parameterInput, "reductionFactor"):
        # overwrite the standart parameter
        setup.reductionFactor = parameterInput.reductionFactor
        
    setup.landLayer= gt.load_array_from_tiff(setup.resourcePath + 'land_layer_62x118.tiff')
    setup.landLayer[np.isnan(setup.landLayer)] = 0
    setup.landLayer = setup.landLayer.astype(int)
    
    setup.population        = gt.load_array_from_tiff(setup.resourcePath + 'pop_counts_ww_2005_62x118.tiff') / setup.reductionFactor
    setup.regionIdRaster    = gt.load_array_from_tiff(setup.resourcePath + 'subRegionRaster_62x118.tiff')
    
    
    if False:
        try:
            #plt.imshow(setup.landLayer)
            plt.imshow(setup.population,cmap='jet')
            plt.clim([0, np.nanpercentile(setup.population,90)])
            plt.colorbar()
        except:
            pass
    setup.landLayer[setup.landLayer == 1 & np.isnan(setup.population)] =0
    nAgents    = np.nansum(setup.population)
    
    #social
    setup.addYourself   = True     # have the agent herself as a friend (have own observation)
    setup.recAgent      = []       # reporter agents that return a diary
    
    #output
    setup.writeOutput   = 1
    setup.writeNPY      = 1
    setup.writeCSV      = 0
    
    #cars and infrastructure
    setup.properties    = ['emmisions','TCO']

    #agents
    setup.randomAgents     = False
    setup.omniscientAgents = False    

    # redefinition of setup parameters used for automatic calibration
    print "setting up the parameters from file"
    print parameterInput
    print "####################################"
    setup.update(parameterInput.toDict())
    
    # calculate dependent parameters
    maxDeviation = (setup.urbanCritical - setup.urbanThreshold)**2
    minCarConvenience = 1 + setup.kappa
    setup.convB =  minCarConvenience / (maxDeviation)
    
    
    assert np.sum(np.isnan(setup.population[setup.landLayer==1])) == 0
    print 'Running with ' + str(nAgents) + ' agents'
    
    return setup
    
def scenarioChina(calibatationInput):
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
            conv = max(0.2,paraA - paraB*(popDensity - cell.urbanThreshold)**2)          
         return conv
            
    def convienienceGreen(popDensity, paraA, paraB, paraC ,paraD, cell):
        conv = max(0.2, paraA - paraB*(popDensity - cell.urbanThreshold)**2 + cell.kappa  )
        return conv
    
    def convienienceOther(popDensity, paraA, paraB, paraC ,paraD, cell):
        conv = paraC/(1+math.exp((-paraD)*(popDensity-cell.urbanThreshold + cell.puplicTransBonus)))
        return conv
    
                         #(emmisions, TCO)         
    earth.initBrand('brown',(440., 200.), convienienceBrown, 0, earth.para['initialBrown']) # combustion car
    
    earth.initBrand('green',(350., 450.), convienienceGreen, 0, earth.para['initialGreen']) # green tech car

    earth.initBrand('other',(120., 100.), convienienceOther, 0, earth.para['initialOther'])  # none or other
            
    return earth
    ##############################################################################

def householdSetup(earth, parameters, calibration=False):
    tt = time.time()
    idx = 0
    nAgents = 0
    nHH     = 0
    
    # init additional properties of nodes 
    earth.registerNodeType('hh',   ['type',
                                   'hhSize',
                                   'nKids',
                                   'income',
                                   'expUtil',
                                   'util',
                                   'expenses'])
    earth.registerNodeType('pers', ['type',
                                   'hhID',
                                   'preferences',
                                   'prefTyp',
                                   'genders',
                                   'ages',
                                   'expUtil',
                                   'util',
                                   'mobType',
                                   'prop',
                                   'consequences'])
    
    if not parameters.randomAgents:
        parameters.synPopPath = parameters['resourcePath'] + 'hh_niedersachsen.csv'
        #dfSynPop = pd.read_csv(parameters.synPopPath)
        hhMat = pd.read_csv(parameters.synPopPath).values
        
    opinion =  Opinion(earth)
    
    for x,y in earth.locDict.keys():
        #print x,y
        nAgentsCell = int(parameters.population[x,y])
        #print nAgentsCell
        while True:
             
            #creating persons as agents
            nPers = hhMat[idx,4]    
            ages    = list(hhMat[idx:idx+nPers,12])
            genders = list(hhMat[idx:idx+nPers,13])
            income = hhMat[idx,16]
            income *= parameters.mobIncomeShare
            nKids = np.sum(ages<18)
            
            # creating houshold
            hh = Household(earth,'hh', x, y)
            hh.adults = list()
            hh.node['hhSize']   = nPers
            hh.node['nKids']    = nKids
            hh.node['income']   = income
            hh.node['expUtil']  = 0
            hh.node['util']     = 0
            hh.node['expenses'] = 0
            hh.register(earth)
            hh.connectGeoNode(earth)
            
            hh.loc.node['population'] += nPers
            
            for iPers in range(nPers):
                
                if ages[iPers]< 18:
                    continue    #skip kids
                
                pers = Person(earth,'pers')
                pers.hh = hh
                pers.register(earth)
                prefTuple = opinion.getPref(ages[iPers],genders[iPers],nKids,nPers,income,parameters.radicality)
                prefTyp = np.argmax(prefTuple)
                pers.node['preferences']    = prefTuple
                pers.node['hhID']           = hh.nID
                pers.node['prefTyp']        = prefTyp
                pers.node['genders']        = genders[iPers]
                pers.node['ages']           = ages[iPers]
                pers.node['expUtil']        = 0
                pers.node['util']           = 0
                pers.node['mobType']        = 0
                pers.node['prop']           = [0]*len(parameters.properties)
                pers.node['consequences']   = [0]*len(prefTuple)
                pers.node['lastAction']     = 0
                pers.innovatorDegree = np.random.randn()
                pers.queueConnection(hh.nID,edgeType=_chp)
                pers.registerAtGeoNode(earth, hh.loc.nID)
                
                # adding reference to the person to household
                hh.adults.append(pers)
            
                earth.nPrefTypes[prefTyp] += 1
                nAgentsCell -= 1
                nAgents     += 1
    
            idx         += nPers
            nHH         += 1
            
            if nAgentsCell <= 0:
                    break
    
    if not(calibration):
        earth.dequeueEdges(_clh)
        earth.dequeueEdges(_chp)

    print str(nAgents) + ' Agents and ' + str(nHH) + ' Housholds created in -- ' + str( time.time() - tt) + ' s'
    return earth




def initEarth(parameters):
    tt = time.time()
    if parameters.writeOutput: 
        if not parameters['calibration']:
            
           
                
            # get simulation number from file
            try:
                fid = open("simNumber","r")
                parameters.simNo = int(fid.readline())
                fid = open("simNumber","w")
                fid.writelines(str(parameters.simNo+1))
                fid.close()
            except:
                parameters.simNo = 0
        
    else:
        parameters.simNo = None
    
    print 'simulation number is: ' + str(parameters.simNo)

    


    earth = Earth(parameters)
    earth.registerEdgeType('cell-cell', ['type','weig'])
    earth.registerEdgeType('cell-hh')
    earth.registerEdgeType('hh-hh')
    earth.registerEdgeType('hh-pers')
    earth.registerEdgeType('pers-pers', ['type','weig'])
    connList= earth.computeConnectionList(parameters.connRadius, ownWeight=1)
    earth.initSpatialLayerNew(parameters.landLayer, connList, Cell)
    
    if hasattr(parameters,'regionIdRaster'):
        
        for cell in earth.iterNodes(_cell):
            cell.node['regionId'] = parameters.regionIdRaster[cell.x,cell.y]
    
    
    earth.initMarket(earth,
                     parameters.properties, 
                     parameters.randomCarPropDeviationSTD, 
                     burnIn=parameters.burnIn, 
                     greenInfraMalus=parameters.kappa)
    
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
    earth.enums['consequences'][0] = 'convenience'
    earth.enums['consequences'][1] = 'eco-friendliness'
    earth.enums['consequences'][2] = 'remaining money'
    earth.enums['consequences'][3] = 'innovation'
    
    earth.enums['mobilityTypes'] = dict()
    earth.enums['mobilityTypes'][1] = 'green'
    earth.enums['mobilityTypes'][0] = 'brown'
    earth.enums['mobilityTypes'][2] = 'other'
    
    
    earth.nPref = len(earth.enums['priorities'])
    earth.nPrefTypes = [0]* earth.nPref
    
    print 'Init finished after -- ' + str( time.time() - tt) + ' s'
    return earth                  

    

def cellTest(earth, parameters):   
    #%% cell convenience test
    convArray = np.zeros([earth.market.getNTypes(),len(earth.nodeDict[1])])
    popArray = np.zeros([len(earth.nodeDict[1])])
    for i, cell in enumerate(earth.iterNodes(_cell)):
        convAll, population = cell.selfTest()
        convArray[:,i] = convAll
        popArray[i] = population
    
    
    if earth.para['showFigures']:
        plt.figure()
        for i in range(earth.market.getNTypes()):    
            plt.subplot(2,2,i+1)
            plt.scatter(popArray,convArray[i,:])    
            plt.title('convenience of ' + earth.enums['mobilityTypes'][i])
        plt.show()
    
    
def generateNetwork(earth, parameters):
    # %% Generate Network
    tt = time.time()
    earth.genFriendNetwork(_pers,_cpp)
    print 'Network initialized in -- ' + str( time.time() - tt) + ' s'
    
def initMobilityTypes(earth, parameters):
    earth.market.initialCarInit()
    earth.market.setInitialStatistics([1000.,2.,500.])
 
def initGlobalRecords(earth, parameters):
    earth.registerRecord('stockBremen', 'total use per mobility type - Bremen', earth.enums['mobilityTypes'].values(), style ='plot')
    
    calDataDfCV = pd.read_csv(parameters.resourcePath + 'calDataCV.csv',index_col=0, header=1)
    calDataDfEV = pd.read_csv(parameters.resourcePath + 'calDataEV.csv',index_col=0, header=1)
    timeIdxs = list()
    values   = list()
    for column in calDataDfCV.columns[1:]:
        value = [np.nan]*3
        year = int(column)
        timeIdx = 12* (year - parameters['startDate'][1]) + earth.para['burnIn']
        value[0] = (calDataDfCV[column]['re_1518'] ) / parameters['reductionFactor']
        if column in calDataDfEV.columns[1:]:
            value[1] = (calDataDfEV[column]['re_1518'] ) / parameters['reductionFactor']
        
    
        timeIdxs.append(timeIdx)
        values.append(value)
          
    earth.globalData['stockBremen'].addCalibrationData(timeIdxs,values)
 
    earth.registerRecord('stockNiedersachsen', 'total use per mobility type - Niedersachsen', earth.enums['mobilityTypes'].values(), style ='plot')
    
    timeIdxs = list()
    values   = list()
    for column in calDataDfCV.columns[1:]:
        value = [np.nan]*3
        year = int(column)
        timeIdx = 12* (year - parameters['startDate'][1]) + earth.para['burnIn']
        value[0] = ( calDataDfCV[column]['re_6321']) / parameters['reductionFactor']
        if column in calDataDfEV.columns[1:]:
            value[1] = ( calDataDfEV[column]['re_6321']) / parameters['reductionFactor']
        
    
        timeIdxs.append(timeIdx)
        values.append(value)
         
    earth.globalData['stockNiedersachsen'].addCalibrationData(timeIdxs,values)
    
def initAgentOutput(earth):
    #%% Init of agent file

    tt = time.time()
    earth.initAgentFile(typ = _hh)
    earth.initAgentFile(typ = _pers)
    earth.initAgentFile(typ = _cell)
    print 'Agent file initialized in ' + str( time.time() - tt) + ' s'
    

def calGreenNeigbourhoodShareDist(earth):
    if parameters.showFigures:
        #%%
        import matplotlib.pylab as pl
        
        relarsPerNeigborhood = np.zeros([len(earth.nodeDict[_pers]),3])
        for i, persId in enumerate(earth.nodeDict[_pers]):
            person = earth.entDict[persId]
            x,__ = person.getConnNodeValues('mobType',_pers)
            for mobType in range(3):
                relarsPerNeigborhood[i,mobType] = float(np.sum(np.asarray(x)==mobType))/len(x)
            
        n, bins, patches = pl.hist(relarsPerNeigborhood, 30, normed=0, histtype='bar',
                                label=['brown', 'green', 'other'])
        pl.legend()
        
def plotIncomePerNetwork(earth):        
    #%%
        import matplotlib.pylab as pl
        
        incomeList = np.zeros([len(earth.nodeDict[_pers]),1])
        for i, persId in enumerate(earth.nodeDict[_pers]):
            person = earth.entDict[persId]
            x, friends = person.getConnNodeValues('mobType',_pers)
            incomes = [earth.entDict[friend].hh.node['income'] for friend in friends]
            incomeList[i,0] = np.mean(incomes)
            
        n, bins, patches = pl.hist(incomeList, 20, normed=0, histtype='bar',
                                label=['average imcome '])
        pl.legend()
    #%%
def runModel(earth, parameters):

    #%% Initial actions
    tt = time.time()
    for household in earth.iterNodes(_hh):
        
        household.takeAction(earth, household.adults, np.random.randint(0,earth.market.nMobTypes,len(household.adults)))
        #for adult in household.adults:
            #adult.setValue('lastAction', 0)
        
    for cell in earth.iterNodes(_cell):
        cell.step(earth.market.kappa)
        
    for household in earth.iterNodes(_hh):
        household.calculateConsequences(earth.market)
        household.util = household.evalUtility()
        household.shareExperience(earth)
    print 'Initial actions randomized in -- ' + str( time.time() - tt) + ' s'
    
    #plotIncomePerNetwork(earth)
    
    
    #%% Simulation 
    earth.time = -1 # hot bugfix to have both models running #TODO Fix later
    print "Starting the simulation:"
    for step in xrange(parameters.nSteps):
        tt = time.time()
        earth.step() # looping over all cells
        print 'Step ' + str(step) + ' done in ' +  str(time.time()-tt) + ' s',
        #plt.figure()
        #calGreenNeigbourhoodShareDist(earth)
        #plt.show()
        tt = time.time()
        earth.writeAgentFile()
        
        print ' - agent file written in ' +  str(time.time()-tt) + ' s'
        
    
    #%% Finishing the simulation    
    print "Finalizing the simulation (No." + str(earth.simNo) +"):"
    if parameters.writeOutput:
        earth.finalizeAgentFile()
    earth.finalize()        

def writeSummary(earth, calRunId, paraDf):
    errBremen = earth.globalData['stockBremen'].evaluateRelativeError()
    errNiedersachsen = earth.globalData['stockNiedersachsen'].evaluateRelativeError()
    fid = open('summary.out','w')
    fid.writelines('Calibration run id: ' + str(calRunId) + '\n')
    fid.writelines('Parameters:')
    paraDf.to_csv(fid)
    fid.writelines('Error:')
    fid.writelines('Bremen: ' + str(errBremen))
    fid.writelines('Niedersachsen: ' + str(errNiedersachsen))
    fid.writelines('Gesammt:')
    fid.writelines(str(errBremen + errNiedersachsen))
    fid.close()
    print 'Calibration Run: ' + str(calRunId)
    print paraDf
    print 'The simulation error is: ' + str(errBremen + errNiedersachsen) 

def onlinePostProcessing(earth):
    # calculate the mean and standart deviation of priorities
    if True:
        df = pd.DataFrame([],columns=['prCon','prEco','prMon','prImi'])
        for agID in earth.nodeDict[3]:
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

    # calculate the correlation between weights and differences in priorities        
    if True:
        pref = np.zeros([earth.graph.vcount(), 4])
        pref[earth.nodeDict[_pers],:] = np.array(earth.graph.vs[earth.nodeDict[_pers]]['preferences'])
        idx = list()
        for edge in earth.iterEdges(_cpp):
            edge['prefDiff'] = np.sum(np.abs(pref[edge.target, :] - pref[edge.source,:]))
            idx.append(edge.index)
            
            
        plt.figure()
        plt.scatter(np.asarray(earth.graph.es['prefDiff'])[idx],np.asarray(earth.graph.es['weig'])[idx])
        plt.xlabel('difference in preferences')
        plt.ylabel('connections weight')
        
        plt.show()
        x = np.asarray(earth.graph.es['prefDiff'])[idx].astype(float)
        y = np.asarray(earth.graph.es['weig'])[idx].astype(float)
        print np.corrcoef(x,y)


#%%############### Tests ############################################################ 


def prioritiesCalibrationTest():
             
    householdSetup(earth, parameters, calibration=True)
    df = pd.DataFrame([],columns=['prCon','prEco','prMon','prImi'])
    for agID in earth.nodeDict[3]:
        df.loc[agID] = earth.graph.vs[agID]['preferences']

#    df = pd.DataFrame([],columns=['prCon','prEco','prMon','prImi'])
#    for agID in earth.nodeDict[3]:
#        df.loc[agID] = earth.graph.vs[agID]['preferences']

    propMat = np.array(np.matrix(earth.graph.vs[earth.nodeDict[3]]['preferences']))

    return earth 


def setupHouseholdsWithOptimalChoice():

    householdSetup(earth, parameters)            
    initMobilityTypes(earth, parameters)    
    #earth.market.setInitialStatistics([500.0,10.0,200.0])
    for household in earth.iterNodes(_hh):    
        household.takeAction(earth, household.adults, np.random.randint(0,earth.market.nMobTypes,len(household.adults)))

    for cell in earth.iterNodes(_cell):
        cell.step(earth.market.kappa) 
    
    earth.market.setInitialStatistics([1000.,5.,300.])

    for household in earth.iterNodes(_hh):
        household.calculateConsequences(earth.market)
        household.util = household.evalUtility()
        
    for hh in iter(earth.nodeDict[_hh]):
        oldEarth = copy(earth)
        earth.entDict[hh].bestMobilityChoice(oldEarth,forcedTryAll = True)    
    return earth    
     


#%%###################################################################################
########## 
   
######################################################################################

if __name__ == '__main__':
#from pycallgraph import PyCallGraph
#from pycallgraph.output import GraphvizOutput
#
#with PyCallGraph(output=GraphvizOutput()):
        
    
    dirPath = os.path.dirname(os.path.realpath(__file__))
    
    
    # loading of standart parameters
    fileName = sys.argv[1]
    parameters = Bunch()
    for item in csv.DictReader(open(fileName)):
        parameters[item['name']] = convertStr(item['value'])
    print 'Setting loaded:'
    print parameters.toDict()
    
    # loading of changing calibration parameters
    if len(sys.argv) > 3:
        # got calibration id
        paraFileName = sys.argv[2]
        colID = int(sys.argv[3])
        parameters['calibration'] = True
        
        if parameters.mpi:
            from mpi4py import MPI
    
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            colID = rank
            parameters['simNo']       = rank
        else:
            
            parameters['simNo']       = 9999
        calParaDf = pd.read_csv(paraFileName, index_col= 0, header = 0, skiprows = range(1,colID+1), nrows = 1)
        calRunID = calParaDf.index[0]
        for colName in calParaDf.columns:
            print 'Setting "' + colName + '" to value: ' + str(calParaDf[colName][calRunID]) 
            parameters[colName] = convertStr(str(calParaDf[colName][calRunID]))
        

    else:
        parameters['calibration'] = False
        calRunID = -999
        calParaDf = pd.DataFrame()
        # no csv file given
        print "no input of parameters"
        
    showFigures    = 0
    

    if parameters.scenario == 0:
        
        parameters = scenarioTestSmall(parameters, dirPath)

    elif parameters.scenario == 1:

        parameters = scenarioTestMedium(parameters, dirPath)
        
    elif parameters.scenario == 2:
        
        parameters = scenarioNiedersachsen(parameters, dirPath)
        
        
    if parameters.scenario == 4:
        # test scenario 
        
        parameters.resourcePath = dirPath + '/resources_nie/'
        parameters = scenarioTestMedium(parameters, dirPath)
        parameters.showFigures = showFigures
        #parameters = scenarioNiedersachsen(parameters)
        earth = initEarth(parameters)
        mobilitySetup(earth, parameters)
        #earth = prioritiesCalibrationTest()
        earth = setupHouseholdsWithOptimalChoice()
    else:
        #%% Init 
        parameters.showFigures = showFigures
        
        earth = initEarth(parameters)
        
        mobilitySetup(earth, parameters)
        
        householdSetup(earth, parameters)
        
        cellTest(earth, parameters)

        generateNetwork(earth, parameters)
        
        initMobilityTypes(earth, parameters)
        
        initGlobalRecords(earth, parameters)
        
        initAgentOutput(earth)
        
        if parameters.scenario == 0:
            earth.view('output/graph.png')
      
        #%% run of the model ################################################
        print '####### Running model with paramertes: #########################'
        import pprint 
        pprint.pprint(parameters.toDict())
        print '################################################################'
        
        runModel(earth, parameters)
        
        if earth.para['showFigures']:
            onlinePostProcessing(earth)
        
        writeSummary(earth, calRunID, calParaDf)
    
        print 'Simulation ' + str(earth.simNo) + ' finished after -- ' + str( time.time() - overallTime) + ' s'
        #%%
        nPeople = np.nansum(parameters.population)
        nCars = float(np.nansum(np.array(earth.graph.vs[earth.nodeDict[_pers]]['mobType'])!=2))
        nGreenCars = float(np.nansum(np.array(earth.graph.vs[earth.nodeDict[_pers]]['mobType'])==0))
        nBrownCars = float(np.nansum(np.array(earth.graph.vs[earth.nodeDict[_pers]]['mobType'])==1))

        print 'Number of agents: ' + str(nPeople)
        print 'Number of agents: ' + str(nCars)
        print 'cars per 1000 people: ' + str(nCars/nPeople*1000.)
        print 'green cars per 1000 people: ' + str(nGreenCars/nPeople*1000.)
        print 'brown cars per 1000 people: ' + str(nBrownCars/nPeople*1000.)


        cellList = earth.graph.vs[earth.nodeDict[_cell]]
        cellListBremen = cellList.select(regionId_eq=1518)
        cellListNieder = cellList.select(regionId_eq=6321)

        carsInBremen = np.asarray(cellListBremen['carsInCell'])
        carsInNieder = np.asarray(cellListNieder['carsInCell'])

        nPeopleBremen = np.nansum(parameters.population[parameters.regionIdRaster==1518])
        nPeopleNieder = np.nansum(parameters.population[parameters.regionIdRaster==6321])

        print 'Bremem - green cars per 1000 people: ' + str(np.sum(carsInBremen[:,1])/np.sum(carsInBremen)*1000)
        print 'Bremem - brown cars per 1000 people: ' + str(np.sum(carsInBremen[:,0])/np.sum(carsInBremen)*1000)

        print 'Niedersachsen - green cars per 1000 people: ' + str(np.sum(carsInNieder[:,1])/np.sum(carsInNieder)*1000)
        print 'Niedersachsen - brown cars per 1000 people: ' + str(np.sum(carsInNieder[:,0])/np.sum(carsInNieder)*1000)




        
 
