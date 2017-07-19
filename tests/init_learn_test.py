#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:12:09 2017
model for testing purposes
@author: gcf
"""

from __future__ import division
import sys, os
from os.path import expanduser
import igraph as ig
home = expanduser("~")
#sys.path.append(home + '/python/decorators/')
sys.path.append(home + '/python/modules/')
sys.path.append(home + '/python/synEarth/agModel/')
#from deco_util import timing_function
import numpy as np
import time
import mod_geotiff as gt
from class_mobilityABM import Cell, Earth, Opinion
from lib_gcfabm import Agent
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

###### Enums ################
#connections
_cll = 1
_cal = 2
_caa = 3
#nodes
_loc    = 1
_agent = 2
#time
_month = 1

class Person(Agent):
    
    def __init__(self, world, nodeType = 'ag', xPos = np.nan, yPos = np.nan):
        Agent.__init__(self, world, nodeType,  xPos, yPos)
        self.obs  = dict()
        #self.innovatorDegree = 0.    

    def getExpectedUtility(self,world):
        """ 
        return all possible actions with their expected consequences
        """

        if len(self.obs) == 0:
            return np.array([-1]), np.array(self.node['util'])

        obsMat, timeWeights    = self.getObservationsMat(world,['hhID', 'utility','label'])
       
        if obsMat.shape[0] == 0:
            return np.array([-1]), np.array(self.node['util'])
        
        observedActions       = np.hstack([np.array([-1]), np.unique(obsMat[:,-1])]) # action = -1 -> keep old car
        observedUtil          = observedActions*0
        observedUtil[0]       = self.node['util']              # utility of keeping old car = present utility
        
        tmpWeights = np.zeros([len(observedActions)-1,obsMat.shape[0]])
        weighted = True
        
        if weighted:
                
            weights, edges = self.getEdgeValuesFast('weig', edgeType=_caa) 
            
            target = [edge.target for edge in edges]
            srcDict =  dict(zip(target,weights))
            for i, id_ in enumerate(observedActions[1:]):
                
                tmpWeights[i,obsMat[:,-1] == id_] = map(srcDict.__getitem__,obsMat[obsMat[:,-1] == id_,0].tolist())
                #tmpWeights[i,obsMat[:,-1] == id_] = tmpWeights[i,obsMat[:,-1] == id_] * timeWeights[obsMat[:,-1] == id_]
        else:
            for i, id_ in enumerate(observedActions[1:]):
                tmpWeights[i,obsMat[:,-1] == id_] = 1
            
        avgUtil = np.dot(obsMat[:,1],tmpWeights.T) / np.sum(tmpWeights,axis=1)
        #maxid = np.argmax(avgUtil)
        observedUtil[1:] = avgUtil
        return observedActions.tolist(), observedUtil.tolist()
        
    
    def takeAction(self, earth, actionId):
        properties = earth.market.mobilityProp[actionId]
        self.node['choice']   = int(actionId)
        self.node['prop']      = [properties]
        self.node['obsID']     = None
        
    def shareExperience(self, world):
        
        # adding noise to the observations
        noisyUtil = self.getValue('util') + np.random.randn()* world.para['utilObsError']/10
        if noisyUtil<0:
           noisyUtil =0
        self.setValue('noisyUtil',noisyUtil)
        mobility = self.getValue('choice')
        # save util based on label
        world.market.obsDict[world.time][mobility].append(noisyUtil)
        obsID = self.loc.registerObs(self.nID, self.getValue('prop'), noisyUtil, mobility)
        self.setValue('obsID', obsID)
        if hasattr(world, 'globalRec'):
            world.globalRec['avgUtil'].addIdx(world.time, noisyUtil ,[0, self.prefTyp+1]) 

    def getObservationsMat(self, world, labelList):
        if len(self.obs) == 0:
            return None
        mat = np.zeros([0,len(labelList)])
        fullTimeList= list()
        for key in self.obs.keys():
            idxList, timeList = self.obs[key]
            idxList = [x for x,y in zip(idxList, timeList) if world.time - y < world.memoryTime  ]
            timeList = [x for x in timeList if world.time - x < world.memoryTime  ]
            self.obs[key] = idxList, timeList
            
            fullTimeList.extend(timeList)    
            mat = np.vstack(( mat, world.entDict[key].obsMemory.getMeme(idxList,labelList)))
        
        fullTimeList = world.time - np.asarray(fullTimeList)    
        weights = np.exp(-(fullTimeList**2) / (world.para['memoryTime']))
        return mat, weights


    def tell(self, locID, obsID, time):
        if locID in self.obs:
            self.obs[locID][0].append(obsID)
            self.obs[locID][1].append(time)
        else:
            self.obs[locID] = [obsID], [time]
    
    
    def weightFriendExperience(self, world):
        friendUtil, friendIDs = self.getConnNodeValues( 'noisyUtil' ,nodeType= _agent)
        carLabels, _        = self.getConnNodeValues( 'choice' ,nodeType= _agent)
        #friendID            = self.getOutNeighNodes(edgeType=_chh)
        ownLabel = self.getValue('choice')
        ownUtil  = self.getValue('util')
        
        edges = self.getEdges(_caa)
        
        idxT = list()
        for i, label in enumerate(carLabels):
            if label == ownLabel:
                idxT.append(i)
        #indexedEdges = [ edges[x].index for x in idxT]
        
        if len(idxT) < 5:
            return

        diff = np.asarray(friendUtil)[idxT] - ownUtil
        prop = np.exp(-(diff**2) / (2* world.para['utilObsError']**2))
        prop = prop / np.sum(prop)
        #TODO  try of an bayesian update - check for right math
        
        onlyEqualCars = True
        if onlyEqualCars:
        #### only weighting agents with same cars
            prior = np.asarray(edges[idxT]['weig'])
            post = prior * prop 

            sumPrior = np.sum(prior)
            post = post / np.sum(post) * sumPrior
            if not(np.any(np.isnan(post)) or np.any(np.isinf(post))):
                if np.sum(post) > 0:
                    edges[idxT]['weig'] = post
                else:
                    print 'updating failed, sum of weights are zero'
                    
        # tell agents that are friends with you - not your friends ("IN")
        for neig in self.getConnNodeIDs( _agent, 'IN'):
            agent = world.entDict[neig]
            agent.tell(self.loc.nID,self.getValue('obsID'), world.time)
            
    def getUtil(self,earth):
        self.node['util'] = np.exp(- (self.node['preference']- earth.market.mobilityProp[self.node['choice']])**2)
    

    
    def generateFriendNetwork(self, world, nFriends):
        """
        Method to generate a preliminary friend network that accounts for 
        proximity in space, priorities and income
        """
        nFriends = np.random.randint(10,nFriends)
        friendList = list()
        connList   = list()
        ownPref    = self.node['preference']

        contactIds     = list()
        propDiffList   = list()
        #incoDiffList   = list()
        spatWeigList   = list()
        


        #get spatial weights to all connected cells
        cellConnWeights, edgeIds, cellIds = self.loc.getConnCellsPlus()                    
        
        
        for cellWeight, cellIdx in zip(cellConnWeights, cellIds):
            
            personIds = world.entDict[cellIdx].getPersons()
            
            
            for personIdx in personIds:
                person = world.entDict[personIdx]
                propDiffList.append((person.node['preference']- ownPref)**2 ) # TODO look if get faster
                
            contactIds.extend(personIds)
            spatWeigList.extend([cellWeight]*len(personIds))
        
        propWeigList = np.array(propDiffList)    
        nullIds  = propWeigList == 0
        propWeigList = 1 / propWeigList
        propWeigList[nullIds] = 0 
        propWeigList = propWeigList / np.sum(propWeigList)
        

        
        weights = propWeigList * spatWeigList
        weights = weights / np.sum(weights)

        if len(weights)-1 < nFriends:
            print "reducting the number of friends"
        nFriends = min(len(weights)-1,nFriends)
        ids = np.random.choice(len(weights), nFriends, replace=False, p=weights)
        friendList = [ contactIds[idx] for idx in ids ]
        connList   = [(self.nID, contactIds[idx]) for idx in ids]
        
        if world.para['addYourself']:
            #add yourself as a friend
            friendList.append(self.nID)
            connList.append((self.nID,self.nID))
        
        weigList   = [1./len(connList)]*len(connList)    
        return friendList, connList, weigList

    def registerAtGeoNode(self, world, cellID):
        self.loc = world.entDict[cellID]        
        self.loc.peList.append(self.nID)
        
    def connectGeoNode(self, world):
        geoNodeID = int(self.graph.IdArray[int(self.x),int(self.y)])
        
        self.queueConnection(geoNodeID,_cal)         
        self.loc = world.entDict[geoNodeID]        
        self.loc.peList.append(self.nID)
        
def scenarioTestSmall(parameterInput):
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
    setup.properties    = ['property']

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


# Mobility setup setup
def mobilitySetup(earth, parameters):
    import math

    def foo():
        return 0
                         #(emmisions, TCO)         
    earth.initBrand('0',(0.), foo,0, earth.para['initialBrown']) # combustion car
    
    earth.initBrand('1',(1.), foo, 0, earth.para['initialGreen']) # green tech car

    earth.initBrand('2',(2.),  foo,0, earth.para['initialOther'])  # none or other

    earth.initBrand('3',(3.),  foo,0, earth.para['initialBrown']) # combustion car
    
    earth.initBrand('4',(4.),  foo,0, earth.para['initialGreen']) # green tech car

def householdSetup(earth, parameters, calibration=False):
    tt = time.time()
    idx = 0
    nAgents = 0
    nHH     = 0
    parameters.synPopPath = parameters['resourcePath'] + 'hh_niedersachsen.csv'
    # init additional properties of nodes 
    earth.registerNodeType('agent',   ['type', 'perference', 'choice'])
    hhMat = pd.read_csv(parameters.synPopPath).values
    
    
    for x,y in earth.locDict.keys():
        #print x,y
        nAgentsCell = int(parameters.population[x,y])
        nPers = hhMat[idx,4]
        
        while True:
             
            for iPers in range(nPers):
                #creating agents
                pers = Person(earth,'agent', x, y)
                pers.register(earth)
                pers.connectGeoNode(earth)
                pers.node['preference']    = np.random.rand()*4
                pers.takeAction(earth, np.random.randint(5))

                nAgentsCell -= 1
            if nAgentsCell <= 0:
                    break
                
    if not(calibration):
        earth.dequeueEdges(_cal)

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



    earth = Earth(parameters)
    earth.registerEdgeType('cell-cell', ['type','weig'])
    earth.registerEdgeType('cell-hh')
    earth.registerEdgeType('ag-ag', ['type','weig'])
    
    connList= earth.computeConnectionList(parameters.connRadius)
    earth.initSpatialLayerNew(parameters.landLayer, connList, Cell)            
    earth.enums = dict()
    earth.initMarket(earth,
                     parameters.properties, 
                     parameters.randomCarPropDeviationSTD, 
                     burnIn=parameters.burnIn, 
                     greenInfraMalus=parameters.kappa)
    
    return earth 

def correlation(earth):
    pref = np.zeros([earth.graph.vcount()])
    pref[earth.nodeList[_agent]] = np.array(earth.graph.vs[earth.nodeList[_agent]]['preference'])
    idx = list()
    for edge in earth.iterEdges(_caa):
        edge['prefDiff'] = np.sum(np.abs(pref[edge.target] - pref[edge.source]))
        idx.append(edge.index)
        
        
    plt.scatter(np.asarray(earth.graph.es['prefDiff'])[idx],np.asarray(earth.graph.es['weig'])[idx], 1)
    plt.xlabel('difference in preferences')
    plt.ylabel('connections weight')
    
    x = np.asarray(earth.graph.es['prefDiff'])[idx].astype(float)
    y = np.asarray(earth.graph.es['weig'])[idx].astype(float)
    print np.corrcoef(x,y)

def generateNetwork(earth, parameters, nodeType, edgeType):
    # %% Generate Network
    tt = time.time()
    earth.genFriendNetwork(parameters.minFriends, nodeType, edgeType)
    print 'Network initialized in -- ' + str( time.time() - tt) + ' s'    
    
    
if __name__ == '__main__':

    dirPath = os.path.dirname(os.path.realpath(__file__))
    dirPath = '/'.join(dirPath.split('/')[:-1])
    
    fileName = sys.argv[1]
    parameters = Bunch()
    for item in csv.DictReader(open(fileName)):
        parameters[item['name']] = convertStr(item['value'])
    print 'Setting loaded:'
    print parameters.toDict()
    
    parameters = scenarioTestSmall(parameters)
    
    parameters.calibration = False
    
    
    for flgOpt in ['opt','prop','rnd']:
        earth = initEarth(parameters)
            
        mobilitySetup(earth, parameters)
        earth.initMemory(parameters.properties + ['utility','label','hhID'], parameters.memoryTime)
        earth.market.initialCarInit()
        
        householdSetup(earth, parameters)
        
        generateNetwork(earth,parameters,_agent,_caa)
        
        
        for agent in earth.iterNodes(_agent):
            agent.getUtil(earth)
            agent.shareExperience(earth)
        for agent in earth.iterNodes(_agent):    
            agent.weightFriendExperience(earth)
        print np.mean(np.asarray(earth.graph.vs['util'])[earth.nodeList[_agent]])    
        
        
        plt.figure()
        for i in range(120): 
            for agent in earth.iterNodes(_agent):
                            
                if flgOpt == 'opt':
                    #optimal action
                    option, eUtil  = agent.getExpectedUtility(earth)
                    idx = option.index(-1)
                    del option[idx]
                    del eUtil[idx]
                    agent.takeAction(earth, option[np.argmax(eUtil)])
                elif flgOpt == 'rnd':
                    #random action                
                    agent.takeAction(earth, np.random.randint(5))
                elif flgOpt == 'prop':
                    option, eUtil  = agent.getExpectedUtility(earth)
                    idx = option.index(-1)
                    del option[idx]
                    del eUtil[idx]
                    if len(eUtil)> 1:
                        weig = np.asarray(eUtil) - np.min(np.asarray(eUtil))
                        weig =weig / np.sum(weig)
                        idx = np.random.choice(option, 1, replace=False, p=weig)
                        agent.takeAction(earth, option[np.argmax(eUtil)])
                agent.getUtil(earth)
                agent.shareExperience(earth)
                
            for agent in earth.iterNodes(_agent):    
                agent.weightFriendExperience(earth)
            
            if np.mod(i,10) ==0:
                plt.subplot(3,4,i/10+1)
                correlation(earth)
                print np.mean(np.asarray(earth.graph.vs['util'])[earth.nodeList[_agent]])
#        for agent in earth.iterNodes(_agent):
#            print str(agent.node['choice']) + ',' + str(agent.node['preference']) 
            
        print np.mean(np.asarray(earth.graph.vs['util'])[earth.nodeList[_agent]])
        for agent in earth.iterNodes(_agent):
            option, eUtil  = agent.getExpectedUtility(earth)
            idx = option.index(-1)
            del option[idx]
            del eUtil[idx]
            agent.takeAction(earth, option[np.argmax(eUtil)])
            agent.getUtil(earth)
        
        print 'final: ' +str(np.mean(np.asarray(earth.graph.vs['util'])[earth.nodeList[_agent]]))
        
        if flgOpt == 'opt':
            plt.suptitle('Optimized')
        elif flgOpt == 'rnd':
            plt.suptitle('Random')
        elif flgOpt == 'prop':
            plt.suptitle('probability')