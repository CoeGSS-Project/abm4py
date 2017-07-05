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
from class_auxiliary import Record, Memory, Writer, cartesian
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
_cll = 1 # loc - loc
_clh = 2 # loc - household
_chh = 3 # household, household
_chp = 4 # household, person
_cpp = 5 # household, person
#nodes
_cell = 1
_hh   = 2
_pers = 3

#%% --- Global classes ---
class Earth(World):

    def __init__(self, parameters):
        
        World.__init__(self, parameters.isSpatial)
        self.simNo      = parameters.simNo
        self.agentRec   = dict()   
        self.time       = 0
        self.date       = list(parameters.startDate)
        self.timeUnit   = parameters.timeUint
        self.nSteps     = parameters.nSteps
        self.reporter   = list()
        self.nAgents    = 0
        self.brandDict  = dict()
        self.brands     = list()
        
        self.globalData  = dict() # storage of global data
        
        # transfer all parameters to earth
        self.setParameters(Bunch.toDict(parameters))
        
        if self.para['omniscientBurnIn']>self.para['burnIn']:
            self.para['omniscientBurnIn']=self.para['burnIn']
        
        try: 
            import git
            repo = git.Repo(search_parent_directories=True)
            self.para["gitVersionSHA"] = repo.head.object.hexsha
        except:
            print "Warning git version of the code not documented"
            print "Please install gitpython using: pip install gitpython"
        
        if not os.path.isdir('output'):
            os.mkdir('output')
        
        if not self.simNo is None:
            self.para['outPath']    = 'output/sim' + str(self.simNo).zfill(4)
            if not os.path.isdir(self.para['outPath']):
                os.mkdir(self.para['outPath'])
            if not os.path.isdir(self.para['outPath'] + '/rec'):
                os.mkdir(self.para['outPath'] + '/rec')
                
    def registerRecord(self, name, title, colLables, style ='plot'):
        self.globalData[name] = Record(name, colLables, self.nSteps, title, style)
        
    # init car market    
    def initMarket(self, earth, properties, propRelDev=0.01, time = 0, burnIn = 0, greenInfraMalus=0):
        self.market = Market(earth, properties, propRelDev=propRelDev, time=time, burnIn=burnIn, greenInfraMalus=greenInfraMalus)
    
    def initMemory(self, memeLabels, memoryTime):
        self.memoryTime = memoryTime
        for location in self.iterNodes(_cell):
            location.initCellMemory(memoryTime, memeLabels)
#    
#    def initObsAtLoc(self,properties):
#        for loc in self.nodeList[1]:
#            #self.agDict[loc].obsMat = np.zeros([0,len(properties)+1])
#            columns = properties + ['utility','label']
#            self.agDict[loc].obsDf = pd.DataFrame(columns = columns)
    
    def initBrand(self, label, propertyTuple, convFunction, initTimeStep, allTimeProduced):
        brandID = self.market.initBrand(label, propertyTuple, initTimeStep, allTimeProduced)
        
        for cell in self.iterNodes(_cell):
            cell.traffic[brandID] = 0
            cell.convFunctions.append(convFunction)
            
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
        if label in self.market.mobilityLables.itervalues():
            brandID = self.market.mobilityLables.keys()[self.market.mobilityLables.values().index(label)]
            
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
        prSaf, prEco, prCon, prMon, prImi = self.og.getPref(ageList[idx],sexList[idx],nKids,income, self.radicality)
        
        return hhSize, ageList, sexList, income,  nKids, prSaf, prEco, prCon, prMon, prImi
    
    
    def genFriendNetwork(self, nodeType, edgeType):
        """ 
        Function for the generation of a simple network that regards the 
        distance between the agents and their similarity in their preferences
        """
        tt = time.time()
        edgeList = list()
        weigList  = list()
        for agent, x in self.iterNodeAndID(nodeType):
            
            frList, edges, weights = agent.generateFriendNetwork(self)
            edgeList += edges
            weigList += weights
        eStart = self.graph.ecount()
        self.graph.add_edges(edgeList)
        self.graph.es[eStart:]['type'] = edgeType
        self.graph.es[eStart:]['weig'] = weigList
        print 'Network created in -- ' + str( time.time() - tt) + ' s'
        
        tt = time.time()
        for node in self.entList:
            node.updateEdges()
        print 'Edges updated in -- ' + str( time.time() - tt) + ' s'
        tt = time.time()
        

    def step(self):
        """ 
        Method to proceed the next time step
        """
        self.time += 1
        
        # progressing time
        if self.timeUnit == 1: #months
            self.date[0] += 1  
            if self.date[0] == 13:
                self.date[0] = 1
                self.date[1] += 1
        elif self.timeUnit == 1: # years
            self.date[1] +=1
            
        
        # proceed market in time
        self.market.step() # Statistics are computed here
        
        
        #loop over cells
        for cell in self.iterNodes(_cell):
            cell.step(self.market.kappa)


        #update global data
        for cell in self.iterNodes(_cell):
            if cell.node['regionId'] == 6321:
                self.globalData['stockNiedersachsen'].add(self.time,np.asarray(cell.node['carsInCell']))
            elif cell.node['regionId'] == 1518:
                self.globalData['stockBremen'].add(self.time,np.asarray(cell.node['carsInCell']))

                
        # Iterate over households with a progress bar
        if self.para['omniscientAgents'] or (self.time < self.para['omniscientBurnIn']):       
            for household in self.iterNodes(_hh):
                #agent = self.agDict[agID]
                household.stepOmniscient(self)        
        else:
            for household in self.iterNodes(_hh):
                #agent = self.agDict[agID]
                household.step(self)
            
        for adult in self.iterNodes(_pers):
            adult.weightFriendExperience(self)   
        
        # proceed step
        #self.writeAgentFile()

        
    def finalize(self):
        
        from class_auxiliary import saveObj
        
        # finishing reporter files
        for writer in self.reporter:        
            writer.close() 
        
        # writing global records to file
        for key in self.globalData:    
            self.globalData[key].saveCSV(self.para['outPath'] + '/rec')


        # saving enumerations            
        saveObj(self.enums, self.para['outPath'] + '/enumerations')
        
        
        # saving enumerations            
        saveObj(self.para, self.para['outPath'] + '/simulation_parameters')
        
        if self.para['showFigures']:
            # plotting and saving figures
            for key in self.globalData:
                self.globalData[key].plot(self.para['outPath'] + '/rec')
            #except:
        #    pass
        if self.para['mpi']:
            os.system("tar -zcvf mobilityABM/rootfs/mnt/ssd/geiges/python/agModel/output.tar.gz mobilityABM/rootfs/mnt/ssd/geiges/python/agModel/" + self.para['outPath'])

        
class Market():

    def __init__(self, earth, properties, propRelDev=0.01, time = 0, burnIn=0, greenInfraMalus=0.):

        self.time               = time
        self.properties         = properties                 # (currently: emissions, price)
        self.mobilityProp       = dict()                     # mobType -> [properties]
        self.nProp              = len(properties)
        self.stock              = np.zeros([0,self.nProp+1]) # first column gives the brandID  (rest: properties, sorted by car ID)
        self.owners             = list()                     # List of (ownerID, brandID) index: mobID
        self.propRelDev         = propRelDev                 # relative deviation of the actual car propeties
        self.obsDict            = dict()                     # time -> other dictionary (see next line)
        self.obsDict[self.time] = dict()                # (used by agents, observations (=utilities) also saved in locations)
        self.freeSlots          = list()
        self.stockByMobType     = list()                     # stock by brands 
        self.nMobTypes          = 0
        self.mobilityLables     = dict()                     # brandID -> label
        self.mobilityInitDict   = dict()
        self.mobilityTypesToInit  = list()                     # list of brand labels
        self.carsPerLabel       = list()
        self.mobilityGrowthRates = list()                  # list of growth rates of brand
        self.techProgress       = list()                     # list of productivities of brand
        self.burnIn             = burnIn
        self.greenInfraMalus    = greenInfraMalus
        self.kappa              = self.greenInfraMalus
        self.sales              = list()
        self.meanDist           = 0.
        self.stdDist            = 1.
        self.innovationWeig     = [1- earth.para['innoWeigPrice'], earth.para['innoWeigPrice']]
        self.allTimeProduced    = list()                   # list of previous produced numbers -> for technical progress
        self.mobNewPeriod       = earth.para['mobNewPeriod']
        self.innoDevRange       = earth.para['innoDevRange']
        
    def getNTypes(self):
        return self.nMobTypes
    
    def getTypes(self):
        return self.mobilityLables
    
    def initialCarInit(self):
        # actually puts the car on the market
        for label, propertyTuple, _, brandID, allTimeProduced in  self.mobilityInitDict[0]:
             self.addBrand2Market(label, propertyTuple, brandID)
    
    def computeStatistics(self):
        distances=list()
        # prct = self.percentiles.keys()
        # for item in self.percentiles.keys():
        #    self.percentiles[item] = np.percentile
        #self.percentiles = np.percentile(self.stock[:,1:],self.prctValues,axis=0)
        #print self.percentiles
        
        
        #if self.time < self.burnIn:
            #self.setInitialStatistics([1000.,5.,300.])
         #   self.mean = np.mean(self.stock[:,1:],axis=0)                           # list of means of properties
         #   self.std  = np.std(self.stock[:,1:],axis=0)                            # same for std
        #else:
        self.mean = np.mean(self.stock[:,1:],axis=0)                           # list of means of properties
        self.std  = np.std(self.stock[:,1:],axis=0)                            # same for std
        
        distances = list()         
        for mobID in range(len(self.stock)):
            properties = self.stock[mobID,1:]
            distance = self.distanceFromMean(properties)
            distances.append(distance)
        self.meanDist = np.mean(distances)
        self.stdDist = np.std(distances)

    def distanceFromMean(self, properties):        
        distance = self.innovationWeig[0]*(self.mean[0]-properties[0])/self.std[0]+self.innovationWeig[1]*(properties[1]-self.mean[1])/self.std[1]            
        return distance
    
    def getDistanceFromMean(self, properties): 
        distance = (self.innovationWeig[0]*(self.mean[0]-properties[0])/self.std[0]+self.innovationWeig[1]*(properties[1]-self.mean[1])/self.std[1]  - self.meanDist)/self.stdDist          
        return distance
        
    def setInitialStatistics(self, typeQuantities):
        total = sum(typeQuantities[mobIdx] for mobIdx in range(self.nMobTypes))
        shares = np.zeros(self.nMobTypes)
        self.mean = np.zeros(self.nProp)
        self.std = np.zeros(self.nProp)
        
        for mobIdx in range(self.nMobTypes):
            shares[mobIdx] = typeQuantities[mobIdx]/total
        
        for propIdx in range(self.nProp):
            propMean = sum(shares[mobIdx]*self.mobilityProp[mobIdx][propIdx] for mobIdx in range(self.nMobTypes))
            propVar = sum(shares[mobIdx]*(self.mobilityProp[mobIdx][propIdx])**2 for mobIdx in range(self.nMobTypes))-propMean**2
            propStd = math.sqrt(propVar)            
            self.mean[propIdx] = propMean
            self.std[propIdx] = propStd

                        
    def ecology(self, emissions):
        if self.std[0] == 0:
            ecology = 1/(1+math.exp((emissions-self.mean[0])/1))
        else:
            ecology = 1/(1+math.exp((emissions-self.mean[0])/self.std[0]))
        return ecology        

        
    def step(self):
        self.time +=1 
        self.obsDict[self.time] = dict()
        #re-init key for next dict in new timestep
        for key in self.obsDict[self.time-1]:
            self.obsDict[self.time][key] = list()
        
        # check if a new car is entering the market
        if self.time in self.mobilityInitDict.keys():
                
            for label, propertyTuple, _, brandID in  self.mobilityInitDict[self.time]:
                self.addBrand2Market(label, propertyTuple, brandID)
        
        # only do technical change after the burn in phase
        if self.time > self.burnIn:
            self.computeTechnicalProgress()
        
        # add sales to allTimeProduced
        self.allTimeProduced = [x+y for x,y in zip(self.allTimeProduced, self.sales)]
        # reset sales
        #print self.sales
        self.sales = [0]*len(self.sales)
        
        #compute new statistics        
        self.computeStatistics()
        
        
            
    def computeTechnicalProgress(self):
            # calculate growth rates per brand:
            # oldCarsPerLabel = copy.copy(self.carsPerLabel)
            # self.carsPerLabel = np.bincount(self.stock[:,0].astype(int), minlength=self.nMobTypes).astype(float)           
            for i in range(self.nMobTypes):
                if not self.allTimeProduced[i] == 0.:
                    newGrowthRate = (self.sales[i])/float(self.allTimeProduced[i])
                else: 
                    newGrowthRate = 0
                self.mobilityGrowthRates[i] = newGrowthRate          
            
            # technological progress:
            oldEtas = copy.copy(self.techProgress)
            for brandID in range(self.nMobTypes):
                self.techProgress[brandID] = oldEtas[brandID] * (1+ max(0,self.mobilityGrowthRates[brandID]))   
                
            # technical process of infrastructure -> given to cells                
            self.kappa = self.greenInfraMalus/(np.sqrt(self.techProgress[0]))
            

        
    def initBrand(self, label, propertyTuple, initTimeStep, allTimeProduced):
        
        mobType = self.nMobTypes
        self.nMobTypes +=1 
        self.mobilityGrowthRates.append(0.)
        self.techProgress.append(1.)
        self.sales.append(0)
        self.stockByMobType.append(0)
        self.allTimeProduced.append(allTimeProduced)
        self.carsPerLabel = np.zeros(self.nMobTypes)
        self.mobilityTypesToInit.append(label)
        if initTimeStep not in self.mobilityInitDict.keys():
            self.mobilityInitDict[initTimeStep] = [[label, propertyTuple, initTimeStep , mobType, allTimeProduced]]
        else:
            self.mobilityInitDict[initTimeStep].append([label, propertyTuple, initTimeStep, mobType, allTimeProduced])
       
        return mobType
    
    def addBrand2Market(self, label, propertyTuple, mobType):
        
        #add brand to the market
           
        self.stockByMobType[mobType] = 0
        self.mobilityProp[mobType]   = propertyTuple
        self.mobilityLables[mobType] = label
        #self.buyCar(brandID,0)
        #self.stockByMobType.loc[self.stockByMobType.index[-1],brandID] -= 1
        self.obsDict[self.time][mobType] = list()
        
        
    def remBrand(self,label):
        #remove brand from the market
        del self.mobilityProp[label]
    
    def currentCarProperties(self, mobTypeIdx):
        eta = self.techProgress[int(mobTypeIdx)]
        # draw the actual car property with a random component
        prop =[float(x/eta * y) for x,y in zip( self.mobilityProp[mobTypeIdx], (1 + np.random.randn(self.nProp)*self.propRelDev))]
        return prop
    
    def buyCar(self, mobTypeIdx, eID):
        prop = self.currentCarProperties(mobTypeIdx)
        if self.time > self.burnIn:
            self.sales[int(mobTypeIdx)] += 1
            
        if len(self.freeSlots) > 0:
            mobID = self.freeSlots.pop()
            self.stock[mobID] = [mobTypeIdx] + prop
            self.owners[mobID] = (eID, mobTypeIdx)
        else:
            self.stock = np.vstack(( self.stock, [mobTypeIdx] + prop))
            self.owners.append((eID, mobTypeIdx))
            mobID = len(self.owners)-1
        #self.stockByMobType.loc[self.stockByMobType.index[-1],brandID] += 1
        self.stockByMobType[int(mobTypeIdx)] += 1
        #self.computeStatistics()
        
        return mobID, prop
    
    def sellCar(self, mobID):
        self.stock[mobID] = np.Inf
        self.freeSlots.append(mobID)
        mobType = self.owners[mobID][1]
        self.owners[mobID] = None
        #self.stockByMobType.loc[self.stockByMobType.index[-1],label] -= 1
        self.stockByMobType[int(mobType)] -= 1

# %% --- entity classes ---

class Person(Agent):
    def __init__(self, world, nodeType = 'ag', xPos = np.nan, yPos = np.nan):
        Agent.__init__(self, world, nodeType,  xPos, yPos)
        self.obs  = dict()
        #self.innovatorDegree = 0.
        
    def registerAtGeoNode(self, world, cellID):
        self.loc = world.entDict[cellID]        
        self.loc.peList.append(self.nID)
        

    def shareExperience(self, world):
        
        # adding noise to the observations
        noisyUtil = self.getValue('util') + np.random.randn(1)* world.para['utilObsError']/10
        self.setValue('noisyUtil',noisyUtil[0])
        mobility = self.getValue('mobType')
        # save util based on label
        world.market.obsDict[world.time][mobility].append(noisyUtil)
        obsID = self.loc.registerObs(self.nID, self.getValue('prop'), noisyUtil, mobility)
        self.setValue('obsID', obsID)
        if hasattr(world, 'globalRec'):
            world.globalRec['avgUtil'].addIdx(world.time, noisyUtil ,[0, self.prefTyp+1]) 

        
        # tell agents that are friends with you - not your friends ("IN")
        for neig in self.getConnNodeIDs( _pers, 'IN'):
            agent = world.entDict[neig]
            agent.tell(self.loc.nID,self.getValue('obsID'), world.time)
       
#    def getObservationsMat(self, world, labelList):
#        if len(self.obs) == 0:
#            return None
#        mat = np.zeros([0,len(labelList)])
#        for key in self.obs.keys():
#            idxList, timeList = self.obs[key]
#            idxList = [x for x,y in zip(idxList, timeList) if world.time - y < world.memoryTime  ]
#            timeList = [x for x in timeList if world.time - x < world.memoryTime  ]
#            self.obs[key] = idxList, timeList
#            mat = np.vstack(( mat, world.entDict[key].obsMemory.getMeme(idxList,labelList)))
#        return mat
    
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
        friendUtil, friendIDs = self.getConnNodeValues( 'noisyUtil' ,nodeType= _pers)
        carLabels, _        = self.getConnNodeValues( 'mobType' ,nodeType= _pers)
        #friendID            = self.getOutNeighNodes(edgeType=_chh)
        ownLabel = self.getValue('mobType')
        ownUtil  = self.getValue('util')
        
        edges = self.getEdges(_cpp)
        
        idxT = list()
        for i, label in enumerate(carLabels):
            if label == ownLabel:
                idxT.append(i)
        #indexedEdges = [ edges[x].index for x in idxT]
        
        if len(idxT) < 10:
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
                    
#        plt.subplot(3,1,1)
#        plt.scatter(diff,prop)            
#        plt.subplot(3,1,2)        
#        plt.scatter(diff,post/prop)
#        plt.subplot(3,1,3)        
#        plt.scatter(diff,post)
#        plt.show

                    
    def generateFriendNetwork(self, world):
        """
        Method to generate a preliminary friend network that accounts for 
        proximity in space, priorities and income
        """
        
        nFriends = np.random.randint(world.para['minFriends'],world.para['maxFriends'])
        
        friendList = list()
        connList   = list()
        ownPref    = self.node['preferences']
        ownIncome  = self.hh.node['income']

        contactIds     = list()
        propDiffList   = list()
        incoDiffList   = list()
        spatWeigList   = list()
        


        #get spatial weights to all connected cells
        cellConnWeights, edgeIds, cellIds = self.loc.getConnCellsPlus()                    
        
        
        for cellWeight, cellIdx in zip(cellConnWeights, cellIds):
            
            personIds = world.entDict[cellIdx].getPersons()
            
            
            for personIdx in personIds:
                person = world.entDict[personIdx]
                propDiffList.append(np.sum([(x-y)**2 for x,y in zip (person.node['preferences'], ownPref ) ])) # TODO look if get faster
                incoDiffList.append(np.abs(person.hh.node['income'] - ownIncome))
                
            contactIds.extend(personIds)
            spatWeigList.extend([cellWeight]*len(personIds))
        
        propWeigList = np.array(propDiffList)    
        nullIds  = propWeigList == 0
        propWeigList = 1 / propWeigList
        propWeigList[nullIds] = 0 
        propWeigList = propWeigList / np.sum(propWeigList)
        
        incoWeigList = np.array(incoDiffList)    
        nullIds  = incoWeigList == 0
        incoWeigList = 1 / incoWeigList
        incoWeigList[nullIds] = 0 
        incoWeigList = incoWeigList / np.sum(incoWeigList)
        
        spatWeigList = spatWeigList / np.sum(spatWeigList)
        spatWeigList = spatWeigList / np.sum(spatWeigList)
        
        weights = propWeigList * spatWeigList * incoWeigList
        weights = weights / np.sum(weights)

        if len(weights)-1 < nFriends:
            print "reducting the number of friends"
        nFriends = min(np.sum(weights>0)-1,nFriends)
        try:
            ids = np.random.choice(len(weights), nFriends, replace=False, p=weights)
        except:
            import pdb
            pdb.set_trace()
        friendList = [ contactIds[idx] for idx in ids ]
        connList   = [(self.nID, contactIds[idx]) for idx in ids]
        
        
        if world.para['addYourself']:
            #add yourself as a friend
            friendList.append(self.nID)
            connList.append((self.nID,self.nID))
        
        weigList   = [1./len(connList)]*len(connList)    
        return friendList, connList, weigList


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
                
            weights, edges = self.getEdgeValuesFast('weig', edgeType=_cpp) 
            
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
          

class Household(Agent):

    def __init__(self, world, nodeType = 'ag', xPos = np.nan, yPos = np.nan):
        Agent.__init__(self, world, nodeType,  xPos, yPos)
        
 #       self.car  = dict()
        self.util = 0
        if world.para['util'] == 'cobb':
            self.utilFunc = self.cobbDouglasUtil
        elif world.para['util'] == 'ces':
            self.utilFunc = self.CESUtil
 #       self.consequences = list()

    def cobbDouglasUtil(self, x, alpha):
        utility = 1.
        factor = 100        
        for i in range(len(x)):
            utility *= (factor*x[i])**alpha[i]
        if np.isnan(utility) or np.isinf(utility):
            import pdb
            pdb.set_trace()
        return utility

    
    def CESUtil(self, x, alpha):
        uti = 0.
        s = 2.    # elasticity of substitution, has to be float!        
        for i in range(len(x)):
            uti += (alpha[i]*x[i]**(s-1))**(1/s)
            #print uti 
        utility = uti**(s/(s-1))
        if  np.isnan(utility) or np.isinf(utility):
            import pdb
            pdb.set_trace()
        return utility
 
    
    
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
        
        self.queueConnection(geoNodeID,_clh)         
        self.loc = world.entDict[geoNodeID]        
        self.loc.hhList.append(self.nID)
        
    
    
    def shareExperience(self, world):
        for adult in self.adults:
            adult.shareExperience(world)
        
    def evalUtility(self):
        """
        Method to evaluate the utilty of all persons and the overall household
        """
        hhUtility = 0
        for adult in self.adults:
            
            utility = self.utilFunc(adult.node['consequences'], adult.node['preferences'])
            assert not( np.isnan(utility) or np.isinf(utility)), utility
            
            adult.node['util'] = utility
            hhUtility += utility
        
        self.node['util'] = hhUtility
        return hhUtility
             

    def takeAction(self, earth, persons, actionIds):
        """
        Method to execute the optimal actions for selected persons of the household
        """
        for person, actionIdx in zip(persons, actionIds):
            
            mobID, properties = earth.market.buyCar(actionIdx, self.nID)
            self.loc.addToTraffic(actionIdx)
            
            person.node['mobType']   = int(actionIdx)
            person.node['mobID']     = int(mobID)
            person.node['prop']      = properties
            person.node['obsID']     = None
            if earth.time <  earth.para['burnIn']:
                person.node['lastAction'] = np.random.randint(0, 2*earth.para['mobNewPeriod'])
            else:
                person.node['lastAction'] = 0
            # add cost of mobility to the expenses
            self.node['expenses'] += properties[1]
            
 
    def undoActions(self,world, persons):
        """
        Method to undo actions
        """
        for adult in persons:
            mobID = adult.node['mobID']
            world.market.sellCar(mobID)
            self.loc.remFromTraffic(adult.node['mobType'])
            
            # remove cost of mobility to the expenses
            self.node['expenses'] -= adult.node['prop'][1]



    def calculateConsequences(self, market):
        
        carInHh = False
        # at least one car in househould?
        if any([adult.node['mobType'] !=2 for adult in self.adults]):       
            carInHh = True
            
        # calculate money consequence
        money = max(0, 1 - self.node['expenses'] /self.node['income'])
                
        for adult in self.adults:
            hhCarBonus = 0.
            #get action of the person
            
            actionIdx = adult.node['mobType']
            mobProps = adult.node['prop']
                
            # calculate convenience:
            if (adult.node['lastAction'] > 2*market.mobNewPeriod) and (actionIdx != 2):
                decay = math.exp(-(adult.node['lastAction'] - 2*market.mobNewPeriod)**2)
            else:
                decay = 1.
                
            if (actionIdx == 2) and carInHh:
                hhCarBonus = 0.2
                
            convenience = decay * self.loc.getValue('convenience')[actionIdx] + hhCarBonus               
            
            # calculate ecology:
            emissions = mobProps[0]
            ecology = market.ecology(emissions)
            
            # get innovation consequence
            #imitation = self.loc.brandShares[action]
            distance   = (market.distanceFromMean(mobProps)-market.meanDist)/market.stdDist
            innovation = math.exp(-((adult.innovatorDegree - distance)**2)/ market.innoDevRange)

            
            adult.node['consequences'] = [convenience, ecology, money, innovation]


    def bestMobilityChoice(self, earth, forcedTryAll = False):          # for test setupHouseholdsWithOptimalCars   (It's the best choice. It's true.)        
        market = earth.market
        actionTaken = True
        if len(self.adults) > 0 :
            combinedActions = self.possibleActions(earth, forcedTryAll)            
            utilities = list()
            
            # save current values
            oldMobType = list()
            oldProp = list()
            oldLastAction = list()
            for adult in self.adults:
                oldMobType.append(copy.copy(adult.node['mobType']))
                oldProp.append(copy.copy(adult.node['prop']))
                oldLastAction.append(copy.copy(adult.node['lastAction']))             
            oldExpenses = copy.copy(self.node['expenses'])
            oldUtil = copy.copy(self.util)
            
            # try all mobility combinations
            for combinationIdx in range(len(combinedActions)):
                self.node['expenses'] = 0            
                for adultIdx, adult in enumerate(self.adults):
                    if combinedActions[combinationIdx][adultIdx] == -1:     # no action taken
                        adult.node['mobType'] = oldMobType[adultIdx]
                        adult.node['prop'] = oldProp[adultIdx]
                        adult.node['lastAction'] = oldLastAction[adultIdx]
                    else:
                        adult.node['mobType'] = combinedActions[combinationIdx][adultIdx]
                        adult.node['prop'] = market.currentCarProperties(adult.node['mobType'])
                        adult.node['lastAction'] = 0                    
                    self.node['expenses'] += adult.node['prop'][1]
                
                self.calculateConsequences(market)
                utility = self.evalUtility()
                utilities.append(utility)
            
            # reset old node values
            for adultIdx, adult in enumerate(self.adults):
                adult.node['mobType'] = oldMobType[adultIdx] 
                adult.node['prop'] = oldProp[adultIdx]
                adult.node['lastAction'] = oldLastAction[adultIdx]
            self.node['expenses'] = oldExpenses
                       
            # get best combination
            bestUtilIdx = np.argmax(utilities)
            bestCombination = combinedActions[bestUtilIdx]
            #print bestCombination
                
            # set best combination 
            if utilities[bestUtilIdx] > oldUtil:
                persons = np.array(self.adults)
                actionIds = np.array(bestCombination)
                actors = persons[ actionIds != -1 ] # remove persons that don't take action
                if len(actors) == 0:
                    actionTaken = False
                actions = actionIds[ actionIds != -1]
                self.undoActions(earth, actors)                  
                self.takeAction(earth, actors, actions)     # remove not-actions (i.e. -1 in list)     
                self.calculateConsequences(market)
                self.util = self.evalUtility()
             
        else:
            actionTaken = False
        
        return actionTaken


    def possibleActions(self, earth, forcedTryAll = False):               
        actionsList = list()
        nMobTypes = earth.market.nMobTypes
        
        for adultIdx, adult in enumerate(self.adults):
            if forcedTryAll or (adult.node['lastAction'] > earth.para['mobNewPeriod']) or (earth.time < earth.para['burnIn']):
                actionsList.append([-1]+range(nMobTypes))
            else:
                actionsList.append([-1])
        if len(actionsList) > 6:                            # to avoid the problem of too many possibilities (if more than 7 adults)
            minNoAction = len(actionsList) - 6              # minum number of adults not to take action    
            while len(filter(lambda x: x == [-1], actionsList)) < minNoAction:
                randIdx = np.random.randint(len(actionsList))
                actionsList[randIdx] = [-1]
            #print 'large Household'
                                          
        possibilities = cartesian(actionsList)
        return possibilities    
            
            
    def evaluateExpectedUtility(self, earth):
    
        actionIdsList   = list()
        eUtilsList      = list()

        for adult in self.adults:
            
            if adult.node['lastAction'] > earth.para['mobNewPeriod'] or (earth.time < earth.para['burnIn']):
                actionIds, eUtils = adult.getExpectedUtility(earth)
            else:
                actionIds, eUtils = [-1], [adult.node['util']]
            
            actionIdsList.append(actionIds)
            eUtilsList.append(eUtils)
            
            if eUtils is None:
                print 1
        
        if len(actionIdsList) == 0:
            return None, None
        
        elif len(actionIdsList) > 6:                            # to avoid the problem of too many possibilities (if more than 7 adults)
            minNoAction = len(actionIdsList) - 6                # minum number of adults not to take action    
            #import pdb
            #pdb.set_trace()
            while len(filter(lambda x: x == [-1], actionIdsList)) < minNoAction:
                randIdx = np.random.randint(len(actionIdsList))
                actionIdsList[randIdx] = [-1]
                eUtilsList[randIdx] =  [adult.node['util']]#[ eUtilsList[randIdx][0] ]
            #print 'large Household'
        
        combActions = cartesian(actionIdsList)
        overallUtil = np.sum(cartesian(eUtilsList),axis=1)

        
        
        
        return combActions, overallUtil
    
    def maxUtilChoice(self, combActions, overallUtil):
        #best action 
        bestActionIdx = np.argmax(overallUtil)
        actions = combActions[bestActionIdx]
        # return persons that buy a new car (action is not -1)
        actors = np.array(self.adults)[ actions != -1]
        actions = actions[ actions != -1]     
        if overallUtil[bestActionIdx] is None:
            print 1
        return actors, actions, overallUtil[bestActionIdx]
    
    def propUtilChoice(self, combActions, overallUtil):
        weig = np.asarray(overallUtil) - np.min(np.asarray(overallUtil))
        weig =weig / np.sum(weig)
        propActionIdx = np.random.choice(range(len(weig)), p=weig)
        actions = combActions[propActionIdx]
        # return persons that buy a new car (action is not -1)
        actors = np.array(self.adults)[ actions != -1]
        actions = actions[ actions != -1]     
        if overallUtil[propActionIdx] is None:
            print 1
        return actors, actions, overallUtil[propActionIdx]
    
        
    def step(self, earth):
        for adult in self.adults:
            adult.addValue('lastAction', 1)
            #adult.node['lastAction'] += 1
        actionTaken = False
        doCheckMobAlternatives = False

        if earth.time < earth.para['burnIn']:
            doCheckMobAlternatives = True
        elif any( [adult.node['lastAction']> earth.para['mobNewPeriod'] for adult in self.adults]):
            doCheckMobAlternatives = True
                
            
        if doCheckMobAlternatives:
            
            # return persons that are potentially performing an action, the action and the expected overall utility
            
            
            combActions, overallUtil = self.evaluateExpectedUtility(earth)
            
            personsToTakeAction, actions, expectedUtil = self.maxUtilChoice(combActions, overallUtil)
            #personsToTakeAction, actions, expectedUtil = self.propUtilChoice(combActions, overallUtil)
            
            if (personsToTakeAction is not None) and len(personsToTakeAction) > 0:
            
                # the propbabilty of taking action is equal to the expected raise of the expected utility
                if self.node['util'] == 0:
                    actionTaken = True                   
                elif (expectedUtil / self.node['util'] ) - 1 > np.random.rand() or (earth.time < earth.para['burnIn']):
                    actionTaken = True                   
                           
            # the action is only performed if flag is True
          
            if actionTaken:
                self.undoActions(earth, personsToTakeAction)
                self.takeAction(earth, personsToTakeAction, actions)

            self.calculateConsequences(earth.market)
            self.util = self.evalUtility()
            
            if actionTaken:                
                self.shareExperience(earth)



    def stepOmniscient(self, earth):
        for adult in self.adults:
            adult.addValue('lastAction', 1)
            #adult.node['lastAction'] += 1
        actionTaken = False
        doCheckMobAlternatives = False

        if earth.time < earth.para['burnIn'] or (any([adult.node['lastAction']> earth.para['mobNewPeriod'] for adult in self.adults])) :
            doCheckMobAlternatives = True
                            
        if doCheckMobAlternatives:            
            actionTaken = self.bestMobilityChoice(earth)
            self.calculateConsequences(earth.market)
            self.util = self.evalUtility()
            
            if actionTaken:                
                for adult in self.adults:
                    adult.weightFriendExperience(earth)
                self.shareExperience(earth)

                                

#    def step(self, world):
#        self.addValue('carAge', 1)
#        carBought = False
#        self.setValue('predMeth',0)
#        self.setValue('expUtil',0)
#        # If the car is older than a constant, we have a 50% of searching
#        # for a new car.
#        if (self.getValue('lastAction') > world.para['newPeriod'] and np.random.rand(1)>.5) or world.time < world.para['burnIn']: 
#            # Check what cars are owned by my friends, and what are their utilities,
#            # and make a choice based on that.
#            brandID, expUtil = self.optimalChoice(world)  
#            
#            if brandID is not None:
#                # If the utility of the new choice is higher than
#                # the current utility times 1.2, we perform a transaction
#
#                buySellBecauseOld = max(0.,self.getValue('lastAction') - 2*world.para['newPeriod'])/world.para['newPeriod'] > np.random.rand(1)
#                
#                # three reasons to buy a car:
#                    # new car has a higher utility 
#                    # current car is very old
#                    # in the burn in phase, every time step a new car is chosen
#                if expUtil > self.util *1.05 or buySellBecauseOld or world.time < world.para['burnIn']:
#                    self.setValue('predMeth',1) # predition method
#                    self.setValue('expUtil',expUtil) # expected utility
#                    self.sellCar(world, self.getValue('mobID'))
#                    self.buyCar(world, brandID)
#                    carBought = True
#                # Otherwise, we have a 25% chance of looking at properties
#                # of cars owned by friends, and perform linear sensitivity
#                # analysis based on the utility of your friends.
#                
#        self.calculateConsequences(world.market)
#        self.util = self.evalUtility()               
#        if carBought:
#            self.weightFriendExperience(world)
#        self.shareExperience(world)


#    def evalIndividualConsequences(self,world):
#        x = self.loc.getX(self.getValue('mobilityType'))
#        x[-1] = max(0,1 - x[-1] / self.getValue('income'))
#        self.setValue('x', x)
 
             
    
#    def optimalChoice(self,world):
#        """ 
#        Method for searching the optimal choice that lead to the highest
#        expected utility
#        return a_opt = arg_max (E(u(a)))
#        """
#        if len(self.obs) == 0:
#            return None, None
#
#        obsMat    = self.getObservationsMat(world,['hhID', 'utility','label'])
#       
#        if obsMat.shape[0] == 0:
#            return None
#        
#        mobIDs       = np.unique(obsMat[:,-1])
#        
#        tmpWeights = np.zeros([len(mobIDs),obsMat.shape[0]])
#        weighted = True
#        
#        if weighted:
#                
#            weights, edges = self.getEdgeValuesFast('weig', edgeType=_chh) 
#            target = [edge.target for edge in edges]
#            srcDict =  dict(zip(target,weights))
#            for i, id_ in enumerate(mobIDs):
#                
#                tmpWeights[i,obsMat[:,-1] == id_] = map(srcDict.__getitem__,obsMat[obsMat[:,-1] == id_,0].tolist())
#        else:
#            for i, id_ in enumerate(mobIDs):
#                tmpWeights[i,obsMat[:,-1] == id_] = 1
#            
#        avgUtil = np.dot(obsMat[:,1],tmpWeights.T) / np.sum(tmpWeights,axis=1)
#        maxid = np.argmax(avgUtil)
#        return mobIDs[maxid], avgUtil[maxid]
    
class Reporter(Household):
    
    def __init__(self, world, nodeType = 'ag', xPos = np.nan, yPos = np.nan):
        Household.__init__(self, world, nodeType , xPos, yPos)
        self.writer = Writer(world, str(self.nID) + '_diary')
        raise('do not use - or update')
        
        #self.writer.write(
        
    
class Cell(Location):
    
    def __init__(self, earth,  xPos, yPos):
        Location.__init__(self, earth,  xPos, yPos)
        self.hhList = list()
        self.peList = list()
        self.carsToBuy = 0
        self.deleteQueue =1
        self.currID = 0
        self.traffic = dict()
        self.sigmaEps = 1.
        self.muEps = 1.               
        self.cellSize = 1.
        self.setValue('population', 0)
        self.setValue('convenience', [0,0,0])
        self.setValue('carsInCell', [0,0,0])
        self.urbanThreshold = earth.para['urbanThreshold']
        self.puplicTransBonus = earth.para['puplicTransBonus']
        
        self.conveniences = list()
        
        self.convFunctions = list()
        self.brandShares = [1.,1.,1.,1.]
        
        self.paraA = earth.para['convA']
        self.paraB = earth.para['convB']
        self.paraC = earth.para['convC']
        self.paraD = earth.para['convD']
        self.kappa = earth.para['kappa']
        
    def initCellMemory(self, memoryLen, memeLabels):
        from collections import deque
        self.deleteQueue = deque([list()]*(memoryLen+1))
        self.currDelList = list()
        self.obsMemory   = Memory(memeLabels)


    def getConnCellsPlus(self):
         self.weights, edges = self.getEdgeValuesFast('weig',edgeType=_cll)
         self.connnodeDict = [edge.target for edge in edges ]
         return self.weights, edges.indices, self.connnodeDict    
 
    def _getConnCellsPlusOld(self):
         self.weights, self.eIDs = self.getEdgeValues('weig',edgeType=_cll, mode='out')
         self.connnodeDict = [self.graph.es[x].target for x in self.eIDs ]
         return self.weights, self.eIDs, self.connnodeDict

    
    def getHHs(self):
        return self.hhList
    
    def getPersons(self):
        return self.peList
    
    def getConnLoc(self,edgeType=1):
        return self.getAgentOfCell(edgeType=1)
    
    
    def addToTraffic(self,brandID):
        self.traffic[brandID] += 1      
        
    def remFromTraffic(self,label):
        self.traffic[label] -= 1
        
    def trafficMixture(self):
        total = sum(self.traffic.values()) 
        shares = list()
        for key in self.traffic.keys():         
            shares.append(float(self.traffic[key])/total)       
        self.brandShares = shares
            
    def getX(self, choice):
        return copy.copy(self.xCell[choice,:])

    def selfTest(self):
        convAll = self.calculateConveniences()
   
        for x in convAll:
            if np.isinf(x) or np.isnan(x) or x == 0:
                import pdb
                pdb.set_trace()
        return convAll, self.getValue('population')
    
    def calculateConveniences(self):
        
        convAll = list()
        # convenience parameters:    

        #paraA, paraC, paraD = 1., .2, 0.07
        popDensity = float(self.getValue('population'))/self.cellSize        
        for funcCall in self.convFunctions:            
            convAll.append(min(1., max(0.05,funcCall(popDensity, self.paraA, self.paraB, self.paraC, self.paraD, self))))            
        return convAll

        
    def step(self, kappa):
        """
        Manages the deletion of observation after a while
        """
        self.kappa = kappa
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
            
        convAll = self.calculateConveniences()
        self.setValue('convenience', convAll)
        self.trafficMixture()

    
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
    def __init__(self, world):
        self.innovationPriority = world.para['innoPriority']
        self.charAge            = world.para['charAge']
        self.indiRatio          = world.para['individualPrio']
        self.minIncomeEco       =world.para['minIncomeEco']
        self.convIncomeFraction =world.para['charIncome']
        
    def getPref(self,age,sex,nKids, nPers,income, radicality):
        
#        # priority of safety
#        cs = 0
#        if nKids < 0:
#            if sex == 2:
#                cs += 4
#            else:
#                cs += 2
#        cs += int(float(age)/10) #
#        if sex == 2:
#            cs += 1
#        cs = float(cs)**2
        
        # priority of ecology
        ce = 2
        if sex == 2:
            ce +=2
        if income>self.minIncomeEco:
            rn = np.random.rand(1)
            if rn > 0.9:
                ce += 3
            elif rn > 0.6:
                ce += 2
            else:
                ce +=1
        ce = float(ce)**2
        
        # priority of convinience
        cc = 0
        cc += nKids
        cc += income/self.convIncomeFraction/2
        if sex == 1:
            cc +=1
        
        cc += int(float(age)/self.charAge)  
        cc = float(cc)**2
        
        # priority of money
        cm = 0
        cm += nKids
        cm += self.convIncomeFraction/income
        cm += nPers
        cm = float(cm)**2
        
        
        sumC = cc + ce + cm
        cc /= sumC
        ce /= sumC
        #cs /= sumC
        cm /= sumC

        # priority of innovation
        ci = self.innovationPriority  
        
        # normalization
        sumC = cc +  ce + cm +ci
        cc /= sumC
        ce /= sumC
        #cs /= sumC
        cm /= sumC
        ci /= sumC

        #individual preferences
        cci, cei,  cmi, cii = np.random.rand(4)
        sumC = cci + cei + + cmi + cii
        cci /= sumC
        cei /= sumC
        #csi /= sumC
        cmi /= sumC
        cii /= sumC
        
        #csAll = cs* (1-self.indiRatio) + csi*self.indiRatio
        ceAll = ce* (1-self.indiRatio) + cei*self.indiRatio
        ccAll = cc* (1-self.indiRatio) + cci*self.indiRatio
        cmAll = cm* (1-self.indiRatio) + cmi*self.indiRatio
        ciAll = ci* (1-self.indiRatio) + cii*self.indiRatio
        
        pref = np.asarray([ ccAll, ceAll, cmAll, ciAll])
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
