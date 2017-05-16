#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://www.globalclimateforum.org

MOBILITY INNOVATION MARKET MODEL
-- CLASS FILE --

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

from lib_gcfabm import World, Agent, Location
from class_auxiliary import Record, Memory, Writer
import igraph as ig
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import time
import os
import math
import copy
from bunch import Bunch
#%% --- ENUMERATIONS ---
#connections
_tll = 1 # loc - loc
_tlh = 2 # loc - household
_thh = 3 # household, household
#nodes
_cell = 1
_hh   = 2

#%% --- Global classes ---
class Earth(World):
    
    def __init__(self, parameters):
        
        World.__init__(self, parameters.spatial)
        self.simNo      = parameters.simNo
        self.agentRec   = dict()   
        self.time       = 0
        self.nSteps     = parameters.nSteps
        self.reporter   = list()
        self.nAgents    = 0
        self.brandDict  = dict()
        self.brands     = list()
        
        # transfer all parameters to earth
        self.setParameters(Bunch.toDict(parameters)) 
        
        if not os.path.isdir('output'):
            os.mkdir('output')
        
        if not self.simNo is None:
            self.para['outPath']    = 'output/sim' + str(self.simNo).zfill(4)
            if not os.path.isdir(self.para['outPath']):
                os.mkdir(self.para['outPath'])
            if not os.path.isdir(self.para['outPath'] + '/rec'):
                os.mkdir(self.para['outPath'] + '/rec')
                
    def registerRecord(self, name, title, colLables, style ='plot'):
        if not hasattr(self, 'globalRec'):
            self.globalRec  = dict()
        self.globalRec[name] = Record(name, colLables, self.nSteps, title, style)
        
    # init car market    
    def initMarket(self, properties, propRelDev=0.01, time = 0):
        self.market = Market(properties, propRelDev=propRelDev, time=time)
    
    def initMemory(self, memeLabels, memoryTime):
        self.memoryTime = memoryTime
        for location in tqdm.tqdm(self.iterNodes(_cell)):
            location.initCellMemory(memoryTime, memeLabels)
#    
#    def initObsAtLoc(self,properties):
#        for loc in self.nodeList[1]:
#            #self.agDict[loc].obsMat = np.zeros([0,len(properties)+1])
#            columns = properties + ['utility','label']
#            self.agDict[loc].obsDf = pd.DataFrame(columns = columns)
    
    def addBrand(self, label, propertyTuple, initTimeStep):
        brandID = self.market.addBrand(label, propertyTuple, initTimeStep)
        
        for cell in self.iterNodes(_cell):
            cell.traffic[brandID] = 0
        if 'brands' not in self.enums.keys():
            self.enums['brands'] = dict()
        self.enums['brands'][brandID] = label
        
    # init opinion class        
#    def initOpinionGen(self,indiRatio = 0.33, ecoIncomeRange=(1000,4000),convIncomeFraction=7000):
#        self.og     = OpinionGenerator(indiRatio, ecoIncomeRange, convIncomeFraction)
#        # read raster and init surface
        
        # init location nodes
        
        # populate required properties
        
        # populate global variables and lists
             
    def plotTraffic(self,label):
        #import matplotlib.pyplot as plt
        import numpy.ma as ma
        traffic = self.graph.IdArray *0
        if label in self.market.brandLabels.itervalues():
            brandID = self.market.brandLabels.keys()[self.market.brandLabels.values().index(label)]
            
            for cell in self.iterNodes(_cell):
                traffic[cell.x,cell.y] += cell.traffic[brandID]
        #Zm = ma.masked_invalid(traffic)
        plt.clf()
        plt.imshow(traffic, cmap='jet',interpolation=None)
        plt.colorbar()
        plt.tight_layout()
        #plt.pcolormesh(Zm.T)
        #plt.pcolormesh(traffic)
        plt.title('traffic of ' + label)
        plt.savefig('output/traffic' + label + str(self.time).zfill(3) + '.png')
        
    def generateHH(self):
        hhSize  = int(np.ceil(np.abs(np.random.randn(1)*2)))
        while True:
            ageList = np.random.randint(1,60,hhSize) 
            if np.sum(ageList>17) > 0: #ensure one adult
                break 
        ageList = ageList.tolist()
        sexList = np.random.randint(1,3,hhSize).tolist()  
        income  = int(np.random.randint(5000))
        nKids   = np.sum(ageList<19)
        #print ageList
        idxAdult = [ n for n,i in enumerate(ageList) if i>17 ]
        idx = np.random.choice(idxAdult)
        prSaf, prEco, prCon, prMon = self.og.getPref(ageList[idx],sexList[idx],nKids,income, self.radicality)
        
        return hhSize, ageList, sexList, income,  nKids, prSaf, prEco, prCon, prMon
    
    
    def genFriendNetwork(self, nFriendsPerPerson):
        """ 
        Function for the generation of a simple network that regards the 
        distance between the agents and their similarity in their preferences
        """
        tt = time.time()
        edgeList = list()
        weigList  = list()
        for agent, x in tqdm.tqdm(self.iterNodeAndID(_hh)):
            
            frList, edges, weights = agent.generateFriends(self,nFriendsPerPerson)
            edgeList += edges
            weigList += weights
        eStart = self.graph.ecount()
        self.graph.add_edges(edgeList)
        self.graph.es[eStart:]['type'] = _thh
        self.graph.es[eStart:]['weig'] = weigList
        print 'Network created in -- ' + str( time.time() - tt) + ' s'
        
        tt = time.time()
        for node in tqdm.tqdm(self.entList):
            node.updateEdges()
        print 'Edges updated in -- ' + str( time.time() - tt) + ' s'
        tt = time.time()
        

    def step(self):
        """ 
        Method to proceed the next time step
        """
        self.time += 1       
        
        # proceed market in time
        self.market.step() # Statistics are computed here
        # The utility of a single agent depends on the whole market
        # if you want a fast car, you are not as happy if everyone
        # else has a fast car same for ecology, etc.
        
        #loop over cells
        for cell in self.iterNodes(_cell):
            cell.step(self.market)
        # Update observations (remove old ones)
        # Compute the number of cars in each cell

        #self.avgDegree.append(np.mean(self.graph.vs[self.nodeList[2]].degree()))
        self.market.stockbyBrand.loc[len(self.market.stockbyBrand)] = self.market.stockbyBrand.loc[len(self.market.stockbyBrand)-1]

        # get number of new cars
        
        # find which agents will by a new car
        
        #loop over buyers
#        markerlist = ['s', 'o', 'd', 'x', '*', '^','p','d','8','_']
#        colors = sns.color_palette("Set1", n_colors=9, desat=.5)
        # Iterate over households with a progress bar
        for agent in tqdm.tqdm(self.randomIterNodes(_hh)):
            #agent = self.agDict[agID]
            agent.step(self)

        # proceed step
        self.writeAgentFile()
        
                
        
    def finalize(self):
        
        from class_auxiliary import saveObj
        
        # finishing reporter files
        for writer in self.reporter:        
            writer.close() 
        
        # writing global records to file
        if hasattr(self, 'globalRec'):
            for key in self.globalRec:    
                self.globalRec[key].saveCSV(self.para['outPath'] + '/rec')


        # saving enumerations            
        saveObj(self.enums, self.para['outPath'] + '/enumerations')
        
        try:
            # plotting and saving figures
            for key in self.globalRec:
                self.globalRec[key].plot(self.para['outPath'] + '/rec')
        except:
            pass

        
class Market():

    def __init__(self, properties, propRelDev=0.01, time = 0):

        self.time          = time
        self.properties    = properties
        self.brandProp     = dict()                     # brandID -> [properties]
        self.nProp         = len(properties)
        self.stock         = np.zeros([0,self.nProp+1]) # first column gives the brandID  (rest: properties, sorted by car ID)
        self.owners        = list()                     # List of (ownerID, brandID) index: carID
        self.propRelDev    = propRelDev                 # relative deviation of the actual car propeties
        self.obsDict       = dict()                     # time -> other dictionary (see next line)
        self.obsDict[self.time] = dict()                # (used by agents, observations (=utilities) also saved in locations)
        self.freeSlots     = list()
        self.stockbyBrand  = pd.DataFrame([])           # stock by brands 
        self.nBrands       = 0
        self.brandLabels   = dict()                     # brandID -> label
        #self.prctValues = [50,80]        
        self.brandInitDict = dict()
        self.brandsToInit  = list()                     # list of brand labels
        self.carsPerLabel  = list()
        self.brandGrowthRates = list()                   # list of growth rates of brand
        self.techProgress  = list()                     # list of productivities of brand

       
    def initCars(self):
        # actually puts the car on the market
        for label, propertyTuple, _, brandID in  self.brandInitDict[0]:
             self.initBrand(label, propertyTuple, brandID)

    
    def computeStatistics(self):  
        # prct = self.percentiles.keys()
        # for item in self.percentiles.keys():
        #    self.percentiles[item] = np.percentile
        #self.percentiles = np.percentile(self.stock[:,1:],self.prctValues,axis=0)
        #print self.percentiles
        self.mean = np.mean(self.stock[:,1:],axis=0)                           # list of means of properties
        self.std  = np.std(self.stock[:,1:],axis=0)                            # same for std
        
    def step(self):
        self.time +=1 
        self.obsDict[self.time] = dict()
        #re-init key for next dict in new timestep
        for key in self.obsDict[self.time-1]:
            self.obsDict[self.time][key] = list()
        
        # calculate growth rates per brand:
        oldCarsPerLabel = copy.copy(self.carsPerLabel)
        self.carsPerLabel = np.bincount(self.stock[:,0].astype(int)).astype(float)           
        for i in range(self.nBrands):
            if not oldCarsPerLabel[i] == 0.:
                newGrowthRate = (self.carsPerLabel[i]-oldCarsPerLabel[i])/oldCarsPerLabel[i]
            else: 
                newGrowthRate = 0
            self.brandGrowthRates[i] = newGrowthRate          
        
        # technological progress:
        oldEtas = copy.copy(self.techProgress)
        for brandID in range(self.nBrands):
            self.techProgress[brandID] = oldEtas[brandID] * (1+max(0,self.brandGrowthRates[brandID]))       
        
        if self.time in self.brandInitDict.keys():
            
            for label, propertyTuple, _, brandID in  self.brandInitDict[self.time]:
                self.initBrand(label, propertyTuple, brandID)

        
    def addBrand(self, label, propertyTuple, initTimeStep):
        
        brandID = self.nBrands
        self.nBrands +=1 
        self.brandGrowthRates.append(0.)
        self.techProgress.append(1.)
        self.carsPerLabel = np.zeros(self.nBrands)
        self.brandsToInit.append(label)
        if initTimeStep not in self.brandInitDict.keys():
            self.brandInitDict[initTimeStep] = [[label, propertyTuple, initTimeStep , brandID]]
        else:
            self.brandInitDict[initTimeStep].append([label, propertyTuple, initTimeStep, brandID])
       
        return brandID
    
    def initBrand(self, label, propertyTuple, brandID):
        
        #add brand to the market
           
        self.stockbyBrand[brandID] = 0
        if self.stockbyBrand.shape[0] == 0:
            # init first time step row
            self.stockbyBrand.loc[len(self.stockbyBrand)] = 0    
        self.brandProp[brandID]   = propertyTuple
        self.brandLabels[brandID] = label
        #self.buyCar(brandID,0)
        #self.stockbyBrand.loc[self.stockbyBrand.index[-1],brandID] -= 1
        self.obsDict[self.time][brandID] = list()
        
        
    def remBrand(self,label):
        #remove brand from the market
        del self.brandProp[label]
    
    def buyCar(self, brandID, eID):
        # draw the actual car property with a random component
        prop =[float(x * y) for x,y in zip( self.brandProp[brandID], (1 + np.random.randn(self.nProp)*self.propRelDev))]
        
        if len(self.freeSlots) > 0:
            carID = self.freeSlots.pop()
            self.stock[carID] = [brandID] + prop
            self.owners[carID] = (eID, brandID)
        else:
            self.stock = np.vstack(( self.stock, [brandID] + prop))
            self.owners.append((eID, brandID))
            carID = len(self.owners)-1
        #self.stockbyBrand.loc[self.stockbyBrand.index[-1],brandID] += 1
        self.computeStatistics()
        
        return carID, prop
    
    def sellCar(self, carID):
        self.stock[carID] = np.Inf
        self.freeSlots.append(carID)
        label = self.owners[carID][1]
        self.owners[carID] = None
        self.stockbyBrand.loc[self.stockbyBrand.index[-1],label] -= 1
    

# %% --- entity classes ---

class Household(Agent):

    def __init__(self, world, nodeType = 'ag', xPos = np.nan, yPos = np.nan):
        Agent.__init__(self, world, nodeType,  xPos, yPos)
        self.obs  = dict()
        self.car  = dict()
        self.util = 0
        if world.para['util'] == 'cobb':
            self.utilFunc = self.cobbDouglasUtil
        elif world.para['util'] == 'ces':
            self.utilFunc = self.CESUtil

    def cobbDouglasUtil(self, x,alpha):
        utility = 1.
        factor = 100        
        # standard:
        for i in range(len(x)):
            utility *= (factor*x[i])**alpha[i]
        if  np.isnan(utility) or np.isinf(utility):
            print sdfsdf      
        return utility

    
    def CESUtil(self, x,alpha):
        uti = 0
        s = 2.    # elasticity of substitution        
        # standard:
        for i in range(len(x)):
            uti += alpha[i]**(1/s)*x[i]**((s-1)/s)        
     

        utility = uti**(s/(s-1))
        if  np.isnan(utility) or np.isinf(utility):
            print 1
        return utility
        

    def registerAgent(self, world):

        self.register(world)    
        world.nAgents += 1

    
    
    def weightFriendExperience(self, world):
        friendUtil, friendIDs = self.getConnNodeValues( 'noisyUtil' ,nodeType= _hh)
        carLabels, _        = self.getConnNodeValues( 'mobilityType' ,nodeType= _hh)
        #friendID            = self.getOutNeighNodes(edgeType=_thh)
        ownLabel = self.getValue('mobilityType')
        ownUtil  = self.getValue('util')
        
        edges = self.getEdges(_thh)
        
        idxT = list()
        for i, label in enumerate(carLabels):
            if label == ownLabel:
                idxT.append(i)
        #indexedEdges = [ edges[x].index for x in idxT]
        
        if len(idxT) < 2:
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
            #neig = [self.graph.es[x].target for x in indexedEdges]
            #diff = [np.sum(np.abs(np.asarray(self.graph.vs[x]['preferences']) - np.asarray(self.graph.vs[self.nID]['preferences']))) for x in neig]
#            plt.scatter(diff,prop)
#            for y,z in zip(diff,prop):
#                if y > .5 and z > .8:
#                    print 1
#        else:
        #### reducting also weight of owners of other cars -> factor .99
#            idxF = np.where(carLabels!=ownLabel)[0]
#            #otherEdges = [ edgeIDs[x] for x in idxF]
#       
#            prior = np.asarray(world.getEdgeValues(edgeIDs,'weig'))
#            post = prior
#            sumPrior = np.sum(prior)
#            post[idxT] = prior[idxT] * prop 
#            post[idxF] = prior[idxF] * .999
#            post = post / np.sum(post) * sumPrior
#            if not(np.any(np.isnan(post)) or np.any(np.isinf(post))):
#                if np.sum(post) > 0:
#                    world.setEdgeValues(edgeIDs,'weig',post)
#                else:
#                    print 'updating failed, sum of weights are zero'
                    
    def connectGeoNode(self, world):
        geoNodeID = int(self.graph.IdArray[int(self.x),int(self.y)])
        self.queueConnection(geoNodeID,_tlh)         
        self.loc = world.entDict[geoNodeID]        
        self.loc.agList.append(self.nID)
        
    
    def generateFriends(self,world, nFriends):
        """
        Method to make/delete or alter the agents social connections
        details to be defined
        """
        friendList = list()
        connList   = list()
        ownPref = self.getValue('preferences') 
        weights, eList, cellList = self.loc.getConnCellsPlus()
        cumWeights = np.cumsum(weights)
        
        iFriend = 0
        i = 0
        if world.para['addYourself']:
            friendList.append(self.nID)
            connList.append((self.nID,self.nID))
        while iFriend < nFriends:
            
            idx = np.argmax(cumWeights > np.random.random(1)) 
            cellID = cellList[idx]
            agList = world.entDict[cellID].getAgents()
            if len(agList) > 0:

                newFrID = np.random.choice(agList)
                diff = 0
                for x,y in zip (world.graph.vs[newFrID]['preferences'], ownPref ): #TODO change to lib-cal
                    diff += abs(x-y)
                    
                if not newFrID == self.nID and newFrID not in friendList and  diff < self.tolerance:   
                    friendList.append(newFrID)
                    connList.append((self.nID,newFrID))
                    iFriend +=1
                i +=1
                if i > 1000:
                    break
        
            if len(connList) > 0:
                weigList = [1./len(connList)]*len(connList)
            else:
                weigList = []
        return friendList, connList, weigList
    
        
    def evalUtility(self,):
        util = self.utilFunc(self.getValue('x'), self.getValue('preferences'))
        assert not( np.isnan(util) or np.isinf(util))
        self.setValue('util',util)
        return util
                        
    def buyCar(self,world, label):
        """
        Method to execute the optimal choice, observe its real consequences and
        tell others in your network about it.
        """
        # add sale to record
        #world.record.loc[world.time, world.rec["sales"][1][self.prefTyp]] += 1
        if hasattr(world, 'globalRec'):
            world.globalRec['sales'].addIdx(world.time, 1 ,self.prefTyp) 
        carID, properties = world.market.buyCar(label, self.nID)
        self.loc.addToTraffic(label)
        self.setValue('mobilityType',label)        
        self.setValue('carID',carID)
        self.setValue('prop', properties)
        self.setValue('obsID', None)
        self.setValue('carAge', 0)
        
    def shareExperience(self, world):
        
        
        # adding noise to the observations
        noisyUtil = self.getValue('util') + np.random.randn(1)* world.para['utilObsError']
        self.setValue('noisyUtil',noisyUtil[0])
        mobility = self.getValue('mobilityType')
        # save util based on label
        world.market.obsDict[world.time][mobility].append(noisyUtil)
        obsID = self.loc.registerObs(self.nID, self.getValue('prop'), noisyUtil, mobility)
        self.setValue('obsID', obsID)
        if hasattr(world, 'globalRec'):
            world.globalRec['avgUtil'].addIdx(world.time, noisyUtil ,[0, self.prefTyp+1]) 

        
        # tell agents that are friends with you - not your friends ("IN")
        for neig in self.getConnNodeIDs( _hh, 'in'):
            agent = world.entDict[neig]
            agent.tell(self.loc.nID,self.getValue('obsID'), world.time)
    
    def sellCar(self,world, carID):
        world.market.sellCar(carID)
        self.loc.remFromTraffic(self.getValue('mobilityType'))
    
    def tell(self, locID, obsID, time):
        if locID in self.obs:
            self.obs[locID][0].append(obsID)
            self.obs[locID][1].append(time)
        else:
            self.obs[locID] = [obsID], [time]

    def step(self, world):
        self.car['age']  +=1
        carBought = False
        self.setValue('predMeth',0)
        self.setValue('expUtil',0)
        # If the car is older than a constant, we have a 50% of searching
        # for a new car.
        if self.car['age'] > world.para['carNewPeriod'] and np.random.rand(1)>.5: 
            # Check what cars are owned by my friends, and what are their utilities,
            # and make a choice based on that.
            choice = self.optimalChoice(world)  
            
            if choice is not None:
                # If the utility of the new choice is higher than
                # the current utility times 1.2, we perform a transaction

                if choice[1] > self.util *1.05:
                    self.setValue('predMeth',1) # predition method
                    self.setValue('expUtil',choice[1]) # expected utility
                    self.sellCar(world, self.getValue('carID'))
                    self.buyCar(world, choice[0])
                    carBought = True
                # Otherwise, we have a 25% chance of looking at properties
                # of cars owned by friends, and perform linear sensitivity
                # analysis based on the utility of your friends.
                
                    
                
        self.evalIndividualConsequences(world)
        self.util = self.evalUtility()               
        if carBought:
            self.weightFriendExperience(world)
        self.shareExperience(world)

    def evalIndividualConsequences(self,world):
        
       
        x = self.loc.getX(self.getValue('mobilityType'))
        x[-1] = max(0,1 - x[-1] / self.getValue('income'))
        self.setValue('x', x)
        
        
    def getObservationsMat(self, world, labelList):
        if len(self.obs) == 0:
            return None
        mat = np.zeros([0,len(labelList)])
        for key in self.obs.keys():
            idxList, timeList = self.obs[key]
            idxList = [x for x,y in zip(idxList,timeList) if world.time - y < world.memoryTime  ]
            timeList = [x for x in timeList if world.time - x < world.memoryTime  ]
            self.obs[key] = idxList, timeList
            mat = np.vstack(( mat, world.entDict[key].obsMemory.getMeme(idxList,labelList)))
        return mat
    
    def optimalChoice(self,world):
        """ 
        Method for searching the optimal choice that lead to the highest
        expected utility
        return a_opt = arg_max (E(u(a)))
        """
        from operator import itemgetter
        if len(self.obs) == 0:
            return None

        obsMat    = self.getObservationsMat(world,['hhID', 'utility','label'])
       
        if obsMat.shape[0] == 0:
            return None
        
        carIDs       = np.unique(obsMat[:,-1])
        
        tmp = np.zeros([len(carIDs),obsMat.shape[0]])
        weighted = True
        
        if weighted:
                
            weights, edges = self.getEdgeValuesFast('weig', edgeType=_thh) 
            target = [edge.target for edge in edges]
            srcDict =  dict(zip(target,weights))
            for i, id_ in enumerate(carIDs):
                
                tmp[i,obsMat[:,-1] == id_] = map(srcDict.__getitem__,obsMat[obsMat[:,-1] == id_,0].tolist())
        else:
            for i, id_ in enumerate(carIDs):
                tmp[i,obsMat[:,-1] == id_] = 1
            
        avgUtil = np.dot(obsMat[:,1],tmp.T) / np.sum(tmp,axis=1)
        maxid = np.argmax(avgUtil)
        return carIDs[maxid], avgUtil[maxid]
    
class Reporter(Household):
    
    def __init__(self, world, nodeType = 'ag', xPos = np.nan, yPos = np.nan):
        Household.__init__(self, world, nodeType , xPos, yPos)
        self.writer = Writer(world, str(self.nID) + '_diary')
        raise('do not use - or update')
    
class Cell(Location):
    
    def __init__(self, earth,  xPos, yPos):
        Location.__init__(self, earth,  xPos, yPos)
        self.agList = list()
        self.carsToBuy = 0
        self.deleteQueue =1
        self.currID = 0
        self.traffic = dict()
        self.sigmaEps = 1.
        self.muEps = 1.               
        self.cellSize = 1.
        

    def initCellMemory(self, memoryLen, memeLabels):
        from collections import deque
        self.deleteQueue = deque([list()]*(memoryLen+1))
        self.currDelList = list()
        self.obsMemory   = Memory(memeLabels)

        
    def getConnCellsPlus(self):
        self.weights, self.eIDs = self.getEdgeValues('weig',edgeType=_tll, mode='out')
        self.connNodeList = [self.graph.es[x].target for x in self.eIDs ]
        return self.weights, self.eIDs, self.connNodeList
    
    
    def getAgents(self):
        return self.agList
    
    def getConnLoc(self,edgeType=1):
        return self.getAgentOfCell(edgeType=1)
    
    
    def addToTraffic(self,label):
        self.traffic[label] += 1
        
    def remFromTraffic(self,label):
        self.traffic[label] -= 1
        
    def ecology(self, emissions, market):
        ecology = 1/(1+math.exp((emissions-market.mean[0])/market.std[0]))
        return ecology        
    
    def getX(self, choice):
        return copy.copy(self.xCell[choice,:])
   
    def updateX(self, market):
        # convenience parameters:        
        a, b, c, d = 1., 0.005, 1., 0.1
        kappa = 0.
        popT = 13   # population threshold for urban area
        popDensity = float(len(self.agList))/self.cellSize
        
        # calculate conveniences
        convenienceG = a - b*(popDensity - popT)**2 + kappa
        if popDensity<popT:
            convenienceB = a
        else:
            convenienceB = a - b*(popDensity - popT)**2
        convenienceN = c/(1+math.exp((-d)*(popDensity-popT)))
        conveniences = [convenienceG, convenienceB, convenienceN]
        
        emissions = list()
        prices = list()
        ecologies = list()
        
        for brandID in range(market.nBrands):
            properties = market.brandProp[brandID]
            em = properties[0] / market.techProgress[brandID]
            emissions.append(em)
            ecologies.append(self.ecology(em, market))
            prices.append(properties[1]/market.techProgress[brandID])
        ecologies[-1] = 0.99
        
        self.xCell = np.asarray([conveniences, ecologies, prices]).T
        
        
    def step(self, market):
        """
        Manages the deletion og observation after a while
        """
        
        self.deleteQueue.append(self.currDelList) # add current list to the queue for later
        delList = self.deleteQueue.popleft()      # takes the list to delete
        for obsID in delList:
            self.obsMemory.remMeme(obsID)         # removes the obs from memory
        self.currDelList = list()                 # restarts the list for the next step
        
        #write cell traffic to graph
        if len(self.traffic.values()) > 1:
            self.setValue('carsInCell', tuple(self.traffic.values()))
        else:
            self.setValue('carsInCell', self.traffic.values()[0])
            
        self.updateX(market)
    
    def registerObs(self, hhID, prop, util, label):
        """
        Adds a car to the cell pool of observations
        """
        #prop.append(util)
        #obsID = self.currID
        meme = prop + [util, label, hhID]
        assert not any(np.isnan(meme))
        obsID = self.obsMemory.addMeme(meme)
        self.currDelList.append(obsID)
        
        #self.currID +=1
        return obsID
        
    def removeObs(self, label):
        """
        Removes a car for the pool of observations
        - not used right now -
        """
        self.traffic[label] -= 1
        
class Opinion():
    import numpy as np
    """ 
    Creates preferences for households, given their properties
    """
    def __init__(self, indiRatio = 0.33, ecoIncomeRange=(1000,4000),convIncomeFraction=7000):
        self.indiRatio = indiRatio
        self.ecoIncomeRange = ecoIncomeRange
        self.convIncomeFraction = convIncomeFraction
        
    def getPref(self,age,sex,nKids,income, radicality):
        
        #safety
        cs = 0
        if nKids < 0:
            if sex == 2:
                cs += 4
            else:
                cs += 2
        cs += int(float(age)/10)
        if sex == 2:
            cs += 1
        cs = float(cs)**2
        
        #ecology
        ce = 3
        if sex == 2:
            ce +=2
        if income>self.ecoIncomeRange[0] and income<self.ecoIncomeRange[1]:
            rn = np.random.rand(1)
            if rn > 0.9:
                ce += 5
            elif rn > 0.5:
                ce += 3
            else:
                ce +=1
        ce = float(ce)**2
        
        #convinience
        cc = 1
        cc += nKids
        cc += income/self.convIncomeFraction
        if sex == 1:
            cc +=1
        if age > 60:
            cc += 3
        elif age > 50:
            cc += 2
        elif age > 40:
            cc += 1
        cc = float(cc)**2

        
        sumC = cc + cs + ce
        cc /= sumC
        ce /= sumC
        cs /= sumC

        #individual preferences
        cci, cei, csi = np.random.rand(3)
        sumC = cci + cei + csi
        cci /= sumC
        cei /= sumC
        csi /= sumC
        
        csAll = cs* (1-self.indiRatio) + csi*self.indiRatio
        ceAll = ce* (1-self.indiRatio) + cei*self.indiRatio
        ccAll = cc* (1-self.indiRatio) + cci*self.indiRatio
        cmAll = np.random.rand(1) # only random component
        
        pref = np.asarray([ ceAll, ccAll, cmAll])
        pref = pref ** radicality
        pref = pref / np.sum(pref)
        return tuple(pref)
     
# %% --- main ---
if __name__ == "__main__":
    market = Market(["wei","ran"])
    
    market.addBrand('g',(2,1))    
    market.addBrand('b',(1,2))    
    market.buyCar('g',1)    
    market.buyCar('b',2) 
    market.buyCar('g',1)
    market.buyCar('b',1)
    
    print market.getPropPercentiles((2,2))

    og = OpinionGenerator(indiRatio= 0.2)
    print 'male young' + str(og.getPref(20,1,0,3001))
    print 'female young' + str(og.getPref(20,2,0,5001))
    print 'male family' + str(og.getPref(25,1,2,5001))
    print 'female family' + str(og.getPref(25,1,4,3501))
    print 'female singe' + str(og.getPref(25,2,0,3501))
    print 'male old' + str(og.getPref(64,1,3,2501))
    print 'female old' + str(og.getPref(64,2,3,2501))
