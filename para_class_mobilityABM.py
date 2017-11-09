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

from para_lib_gcfabm import World, Agent, GhostAgent, Location, GhostLocation, aux, h5py, MPI

#import class_auxiliary as aux # Record, Memory, Writer, cartesian

import igraph as ig
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import h5py
import time
import os
import math
import copy
import logging as lg

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

    def __init__(self,
                 simNo,
                 outPath,
                 parameters,
                 maxNodes,
                 debug, 
                 mpiComm=None,
                 caching=True,
                 queuing=True):
        
        nSteps     = parameters.nSteps
        
        self.computeTime = np.zeros(parameters.nSteps)
        self.syncTime    = np.zeros(parameters.nSteps)
        self.waitTime    = np.zeros(parameters.nSteps)
        self.ioTime      = np.zeros(parameters.nSteps)
        
        World.__init__(self,
                       simNo,
                       outPath,
                       parameters.isSpatial, 
                       nSteps, 
                       maxNodes = maxNodes, 
                       debug=debug, 
                       mpiComm=mpiComm, 
                       caching=caching)
        
        self.agentRec   = dict()   
        self.time       = 0
        self.date       = list(parameters.startDate)
        self.timeUnit   = parameters.timeUnit
        
        self.reporter   = list()
        self.nAgents    = 0
        self.brandDict  = dict()
        self.brands     = list()
        
        self.globalRecord  = dict() # storage of global data
        
        # transfer all parameters to earth
        parameters.simNo = simNo
        self.setParameters(Bunch.toDict(parameters))
        
        if self.para['omniscientBurnIn']>self.para['burnIn']:
            self.para['omniscientBurnIn']=self.para['burnIn']
        
        
#        try:
#            import socket
#            if (mpiComm is None or mpiComm.rank ==0) and socket.gethostname() != 'gcf-VirtualBox':
#                #os.system('git commit -a -m "automatic commit"')
#            
#                import git
#                repo = git.Repo(search_parent_directories=True)
#                self.para["gitVersionSHA"] = repo.head.object.hexsha
#        except:
#            print "Warning git version of the code not documented"
#            print "Please install gitpython using: pip install gitpython"

            
        
        if not os.path.isdir('output'):
            os.mkdir('output')
        
        

                
    def registerRecord(self, name, title, colLables, style ='plot', mpiReduce=None):

        self.globalRecord[name] = aux.Record(name, colLables, self.nSteps, title, style)
        #print self.globalRecord[name]
        if mpiReduce is not None:
            self.glob.registerValue(name , np.asarray([0]*len(colLables)),mpiReduce)
            self.globalRecord[name].glob = self.glob
            
    # init car market    
    def initMarket(self, earth, properties, propRelDev=0.01, time = 0, burnIn = 0):
        self.market = Market(earth, properties, propRelDev=propRelDev, time=time, burnIn=burnIn)
    
    def initMemory(self, memeLabels, memoryTime):
        self.memoryTime = memoryTime
        for location in self.iterEntRandom(_cell):
            location.initCellMemory(memoryTime, memeLabels)

    def initBrand(self, label, propertyTuple, convFunction, initTimeStep, allTimeProduced):
        brandID = self.market.initBrand(label, propertyTuple, initTimeStep, allTimeProduced)
        
        for cell in self.iterEntRandom(_cell):
            cell.traffic[brandID] = 0
            cell.convFunctions.append(convFunction)
            
        if 'brands' not in self.enums.keys():
            self.enums['brands'] = dict()
        self.enums['brands'][brandID] = label

             
    def plotTraffic(self,label):
        #import matplotlib.pyplot as plt
        #import numpy.ma as ma
        traffic = self.graph.IdArray *0
        if label in self.market.mobilityLables.itervalues():
            brandID = self.market.mobilityLables.keys()[self.market.mobilityLables.values().index(label)]
            
            for cell in self.iterEntRandom(_cell):
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
        while 1:
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
    
    
    def generateSocialNetwork(self, nodeType, edgeType):
        """ 
        Function for the generation of a simple network that regards the 
        distance between the agents and their similarity in their preferences
        """
        tt = time.time()
        edgeList = list()
        weigList  = list()
        for agent, x in self.iterEntAndIDRandom(nodeType):
            
            frList, edges, weights = agent.generateContactNetwork(self)
            edgeList += edges
            weigList += weights
        self.addEdges(edgeList, type=edgeType, weig=weigList)

        if self.queuing:                
            self.queue.dequeueEdges(self)     
            
        lg.info( 'Network created in -- ' + str( time.time() - tt) + ' s')
        

    def step(self):
        """ 
        Method to proceed the next time step
        """
        tt = time.time()
        self.time += 1
        self.timeStep = self.time
        
        
        ttComp = time.time()
        # time management
        if self.timeStep == 0:
            lg.info( 'setting up time warp during burnin by factor of ' + str(self.para['burnInTimeFactor']))
            self.para['mobNewPeriod'] = int(self.para['mobNewPeriod'] / self.para['burnInTimeFactor'])
            newValue = np.rint(self.getNodeValues('lastAction',_pers) / self.para['burnInTimeFactor']).astype(int)
            self.setNodeValues('lastAction',_pers, newValue)
            
        if self.timeStep+5 == self.para['burnIn']:
            lg.info( 'reducting time speed to normal')
            self.para['mobNewPeriod'] = int(self.para['mobNewPeriod'] * self.para['burnInTimeFactor'])
            oldValue = self.getNodeValues('lastAction',_pers) * self.para['burnInTimeFactor']
            newValue = oldValue.astype(int)
            stochastricRoundValue = newValue + (np.random.random(len(oldValue)) < oldValue-newValue).astype(int)
            
            self.setNodeValues('lastAction',_pers, stochastricRoundValue)
        
        # progressing time
        if self.timeUnit == 1: #months
            self.date[0] += 1  
            if self.date[0] == 13:
                self.date[0] = 1
                self.date[1] += 1
        elif self.timeUnit == 1: # years
            self.date[1] +=1
            
        #update global data
#        for cell in self.iterEntRandom(_cell):
#            if cell.node['regionId'] == 6321:
#                self.globalRecord['stockNiedersachsen'].add(self.time,np.asarray(cell.node['carsInCell']))
#            elif cell.node['regionId'] == 1518:
#                self.globalRecord['stockBremen'].add(self.time,np.asarray(cell.node['carsInCell']))
#            elif cell.node['regionId'] == 1520:
#                self.globalRecord['stockHamburg'].add(self.time,np.asarray(cell.node['carsInCell']))
        
        for cell in self.iterEntRandom(_cell):
            self.globalRecord['stock_' + str(int(cell.node['regionId']))].add(self.time,np.asarray(cell.node['carsInCell']))

        # move values to global data class
        for re in self.para['regionIDList']:
            self.globalRecord['stock_' + str(re)].updateValues(self.time) 
        
#        self.globalRecord['stockNiedersachsen'].updateValues(self.time) 
#        self.globalRecord['stockBremen'].updateValues(self.time) 
#        self.globalRecord['stockHamburg'].updateValues(self.time) 
        
        self.computeTime[self.time] = time.time()-ttComp
        
        ttSync = time.time()
        self.glob.updateStatValues('meanEmm', np.asarray(self.graph.vs[self.nodeDict[_pers]]['prop'])[:,0])
        self.glob.updateStatValues('stdEmm', np.asarray(self.graph.vs[self.nodeDict[_pers]]['prop'])[:,0])
        self.glob.updateStatValues('meanPrc', np.asarray(self.graph.vs[self.nodeDict[_pers]]['prop'])[:,1])
        self.glob.updateStatValues('stdPrc', np.asarray(self.graph.vs[self.nodeDict[_pers]]['prop'])[:,1])
        self.glob.sync()
        self.syncTime[self.time] = time.time()-ttSync
        lg.debug('globals synced in ' +str(time.time()- ttSync) + ' seconds')
        
        ttComp = time.time()
        #gather data back to the records
        
        for re in self.para['regionIDList']:
            self.globalRecord['stock_' + str(re)].gatherSyncDataToRec(self.time) 
            
#        self.globalRecord['stockNiedersachsen'].gatherSyncDataToRec(self.time) 
#        self.globalRecord['stockBremen'].gatherSyncDataToRec(self.time) 
#        self.globalRecord['stockHamburg'].gatherSyncDataToRec(self.time) 
        
        
        # proceed market in time
        #print 'sales after sync: ',self.glob['sales']
        self.market.step(self) # Statistics are computed here
        
        ttCell = time.time()   
        #loop over cells
        for cell in self.iterEntRandom(_cell):
            cell.step(self.market.currKappa)
        lg.debug('Cell step required ' + str(time.time()- ttCell) + ' seconds')        


        tthh = time.time()        
        # Iterate over households with a progress bar
        if self.para['omniscientAgents'] or (self.time < self.para['omniscientBurnIn']):
            for household in self.iterEntRandom(_hh):
                #agent = self.agDict[agID]
                household.stepOmniscient(self)        
        else:
            for household in self.iterEntRandom(_hh):
                
                if np.random.rand()<1e-4:      
                    household.stepOmniscient(self)
                else:
                    household.step(self)
        lg.debug('Household step required ' + str(time.time()- tthh) + ' seconds')        

        if self.queuing:                
            self.queue.dequeueEdgeDeleteList(self)        
            

        self.computeTime[self.time] += time.time()-ttComp
        
        ttWait = time.time()
        self.mpi.comm.Barrier()
        
        self.waitTime[self.time] = time.time()-ttWait
        
        ttSync = time.time()
        self.mpi.updateGhostNodes([_pers],['commUtil'])
        self.syncTime[self.time] += time.time()-ttSync
        
        lg.debug('Ghosts synced in ' + str(time.time()- ttSync) + ' seconds')

        ttComp = time.time()
        for person in self.iterEntRandom(_pers):
            person.step(self)
        lg.debug('Person step required ' + str(time.time()- ttComp) + ' seconds')
        
        if self.queuing:                
            self.queue.dequeueEdges(self)        
            
        self.computeTime[self.time] += time.time()-ttComp
        # proceed step
        #self.writeAgentFile()
        
        ttWait = time.time()
        self.mpi.comm.Barrier()
        
        self.waitTime[self.time] += time.time()-ttWait

        ttIO = time.time()
        #earth.writeAgentFile()
        self.io.gatherNodeData(self.time)
        self.io.writeDataToFile()
        self.ioTime[self.time] = time.time()-ttIO
        
        
        lg.info(('Times: tComp: '+ '{:10.5f}'.format(self.computeTime[self.time])+ 
              ' - tSync: '+ '{:10.5f}'.format(self.syncTime[self.time])+ 
              ' - tWait: '+ '{:10.5f}'.format(self.waitTime[self.time])+ 
              ' - tIO: '+ '{:10.5f}'.format(self.ioTime[self.time]) ))
        
        if self.para['omniscientAgents']:
            lg.info( 'Omincent step ' + str(self.time) + ' done in ' +  str(time.time()-tt) + ' s')
        else:
            lg.info( 'Step ' + str(self.time) + ' done in ' +  str(time.time()-tt) + ' s')
        
        if self.isRoot:
            print 'Step ' + str(self.time) + ' done in ' +  str(time.time()-tt) + ' s'
        
    def finalize(self):
        
        from class_auxiliary import saveObj
        
        # finishing reporter files
        for writer in self.reporter:        
            writer.close() 
        
        if self.isRoot:
            # writing global records to file
            h5File = h5py.File(self.para['outPath'] + '/globals.hdf5', 'w')
            for key in self.globalRecord:    
                self.globalRecord[key].saveCSV(self.para['outPath'])
                self.globalRecord[key].save2Hdf5(h5File)
                
            h5File.close()
            # saving enumerations            
            saveObj(self.enums, self.para['outPath'] + '/enumerations')
            
            
            # saving enumerations            
            saveObj(self.para, self.para['outPath'] + '/simulation_parameters')
            
            if self.para['showFigures']:
                # plotting and saving figures
                for key in self.globalRecord:
                    self.globalRecord[key].plot(self.para['outPath'])


class Market():

    def __init__(self, earth, properties, propRelDev=0.01, time = 0, burnIn=0):
        #import global variables
        self.globalRecord       = earth.globalRecord
        self.comm               = earth.mpi.comm
        self.glob               = earth.glob
        self.glob.registerValue('sales' , np.asarray([0]),'sum')
        self.glob.registerStat('meanEmm' , np.asarray([0]*len(properties)),'mean')
        self.glob.registerStat('stdEmm' , np.asarray([0]*len(properties)),'std')
        self.glob.registerStat('meanPrc' , np.asarray([0]*len(properties)),'mean')
        self.glob.registerStat('stdPrc' , np.asarray([0]*len(properties)),'std')        
        self.time                = time
        self.graph               = earth.graph
        self.nodeDict            = earth.nodeDict
        self.properties          = properties                 # (currently: emissions, price)
        self.mobilityProp        = dict()                     # mobType -> [properties]
        self.nProp               = len(properties)
        #self.stock               = np.zeros([0,self.nProp+1]) # first column gives the mobTypeID  (rest: properties, sorted by car ID)
        #self.owners              = list()                     # List of (ownerID, mobTypeID) index: mobID
        self.propRelDev          = propRelDev                 # relative deviation of the actual car propeties
        self.obsDict             = dict()                     # time -> other dictionary (see next line)
        self.obsDict[self.time]  = dict()                     # (used by agents, observations (=utilities) also saved in locations)
        self.stockByMobType      = list()                     # list of total numbers per mobility type 
        self.nMobTypes           = 0                          # number of different mobility types
        self.mobilityLables      = dict()                     # mobTypeID -> label
        self.mobilityInitDict    = dict()                     # list of (list of) initial values for each mobility type
        self.mobilityTypesToInit = list()                     # list of (mobility type) labels
        self.mobilityGrowthRates = list()                     # list of growth rates of brand
        self.techProgress        = list()                     # list of productivities of brand
        self.burnIn              = burnIn
        #self.greenInfraMalus     = greenInfraMalus
        #self.kappa               = self.greenInfraMalus
        self.sales               = list()
        self.meanDist            = 0.
        self.stdDist             = 1.
        self.innovationWeig      = [1-earth.para['innoWeigPrice'], earth.para['innoWeigPrice']]   # weights for calculating innovation distance
        self.allTimeProduced     = list()                     # list of previous produced numbers -> for technical progress
        self.para                = earth.para
        self.currKappa           = self.para['kappa']

        
    def getNTypes(self):
        return self.nMobTypes
    
    def getTypes(self):
        return self.mobilityLables
    
    def initialCarInit(self):
        # actually puts the car on the market
        for label, propertyTuple, _, brandID, allTimeProduced in self.mobilityInitDict['start']:
             self.addBrand2Market(label, propertyTuple, brandID)

    
    def computeStatistics(self):
        distances=list()
        
        stock = np.asarray(self.graph.vs[self.nodeDict[_pers]]['prop'])
        #self.glob['sumProp'] = np.sum(stock, axis=0)
        #self.mean = np.mean(stock,axis=0)                           # list of means of properties
        #self.std  = np.std(stock,axis=0)                            # same for std
        
        self.mean = [self.glob['meanEmm'], self.glob['meanPrc']]
        self.std  = [self.glob['stdEmm'],  self.glob['stdPrc']]
        
        lg.debug('Mean properties- mean: ' + str(self.mean) + ' std: ' + str(self.std))
        distances = list()         
        for idx in range(len(stock)):
            properties = stock[idx,:]
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
            ecology = 1 / (1+math.exp((emissions-self.mean[0])/1))
        else:
            ecology = 1 / (1+math.exp((emissions-self.mean[0])/self.std[0]))
        
        return ecology        

        
    def step(self, world):
        
        self.obsDict[self.time] = dict()
        #re-init key for next dict in new timestep
        for key in self.obsDict[self.time]:
            self.obsDict[self.time][key] = list()
        
        # check if a new car is entering the market
        if self.time in self.mobilityInitDict.keys():
            
            for mobTuple in self.mobilityInitDict[self.time]:
                for label, propertyTuple, _, mobTypeID in  mobTuple:
                    self.addBrand2Market(label, propertyTuple, mobTypeID)
        
        # only do technical change after the burn in phase
        if self.time > self.burnIn:
            self.computeTechnicalProgress()
        self.globalRecord['infraKappa'].set(self.time, self.currKappa)
        
        # add sales to allTimeProduced
        self.allTimeProduced = [x+y for x,y in zip(self.allTimeProduced, self.glob['sales'])]
        
        lg.debug('new value of allTimeProduced: ' + str(self.allTimeProduced))
        # reset sales
        self.glob['sales'] = self.glob['sales']*0
        
        
        #compute new statistics        
        self.computeStatistics()
        
        self.time +=1 
                    
    def computeTechnicalProgress(self):
        # calculate growth rates per brand:
        # oldCarsPerLabel = copy.copy(self.carsPerLabel)
        # self.carsPerLabel = np.bincount(self.stock[:,0].astype(int), minlength=self.nMobTypes).astype(float)           
        lg.info( 'sales in market: ' +str(self.glob['sales']))
        for i in range(self.nMobTypes):
            if not self.allTimeProduced[i] == 0.:
                
                newGrowthRate = (self.glob['sales'][i]) / float(self.allTimeProduced[i])
            else: 
                newGrowthRate = 0
            self.mobilityGrowthRates[i] = newGrowthRate          
        
        lg.debug('growth rate: ' + str(newGrowthRate))
        
        # technological progress:
        oldEtas = copy.copy(self.techProgress)
        for brandID in range(self.nMobTypes):
            self.techProgress[brandID] = oldEtas[brandID] * (1+ max(0,self.mobilityGrowthRates[brandID]))   
        
        if self.comm.rank == 0:
            self.globalRecord['growthRate'].set(self.time, self.mobilityGrowthRates)
            self.globalRecord['allTimeProduced'].set(self.time, self.allTimeProduced)
        
        # technical process of infrastructure -> given to cells                
        self.currKappa = self.para['kappa'] / self.techProgress[1]
        
        
        
        lg.debug('techProgress: ' + str(self.techProgress))
        
    def initBrand(self, label, propertyTuple, initTimeStep, allTimeProduced):        
        mobType = self.nMobTypes
        self.nMobTypes +=1 
        self.mobilityGrowthRates.append(0.)
        self.techProgress.append(1.)
        self.glob['sales'] = np.asarray([0]*self.nMobTypes)

        self.stockByMobType.append(0)
        self.allTimeProduced.append(allTimeProduced)
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
    
    def buyCar(self, mobTypeIdx):
        prop = self.currentCarProperties(mobTypeIdx)
        if self.time > self.burnIn:
            self.glob['sales'][int(mobTypeIdx)] += 1
    
        self.stockByMobType[int(mobTypeIdx)] += 1        
        return prop    #return mobID, prop
    

# %% --- entity classes ---


class Person(Agent):
    
    def __init__(self, world, **kwProperties):
        Agent.__init__(self, world, **kwProperties)


    def isAware(self,mobNewPeriod):
        # method that returns if the Persion is aktively searching for information
        return (self.node['lastAction'] - mobNewPeriod/10.) / (mobNewPeriod)  > np.random.rand()
    
        
    def register(self, world, parentEntity=None, edgeType=None):
        
        Agent.register(self, world, parentEntity, edgeType)
        self.loc = parentEntity.loc 
        self.loc.peList.append(self.nID)
        self.hh = parentEntity


    def weightFriendExperience(self, world):
        friendUtil = np.asarray(self.getPeerValues('commUtil',_cpp)[0])[:,self.node['mobType']]
        ownUtil  = self.getValue('util')
        edges = self.getEdges(_cpp)
        diff = friendUtil - ownUtil +  np.random.randn(len(friendUtil))*world.para['utilObsError']
        prop = np.exp(-(diff**2) / (2* world.para['utilObsError']**2))
        prop = prop / np.sum(prop)        
        
        prior = np.asarray(edges['weig'])
        prior = prior / np.sum(prior)      
        #if any([value is None for value in prop]) or any([value is None for value in prior]):
        #    import pdb
        #    pdb.set_trace()
        
        #print prop
        #print prior
        
        try:
            post = prior * prop 
        except:
            import pdb
            pdb.set_trace()

        #sumPrior = np.sum(prior)
        #post = post / np.sum(post) * sumPrior
        
        post = post / np.sum(post) 
        
        if not(np.any(np.isnan(post)) or np.any(np.isinf(post))):
            if np.sum(post) > 0:
                edges['weig'] = post    
        
                self.node['ESSR'] =  (1 / np.sum(post**2)) / float(len(post))
                #assert self.edges[_cpp][self.ownEdgeIdx[0]].target == self.nID
                #assert self.edges[_cpp][self.ownEdgeIdx[0]].source == self.nID
                if np.sum(self.getEdgeValues('weig', edgeType=_cpp)[0]) < 0.99:
                    import pdb
                    pdb.set_trace()
            return post, self.node['ESSR']
        else:
            import pdb
            pdb.set_trace()
    
    def socialize(self, world):
        
        tt = time.time()
#        drop 10% of old connections
#        print 'ID:', self.nID, 
#        print "prior", 
#        weights, edges = self.getEdgeValues('weig', edgeType=_cpp) 
#        print 'sum of weights', np.sum(weights)
#        for weig, idx in zip(weights, edges.indices):
#            print '(',weig, idx,')',
#        print ' ' 
        weights, edges = self.getEdgeValues('weig', edgeType=_cpp) 
        nContacts = len(weights)
        nDrops    = int(nContacts/10)
        dropIds = np.asarray(edges.indices)[np.argsort(weights)[:nDrops].tolist()].tolist()
        
        if world.queuing:
            world.queue.edgeDeleteList.extend(dropIds)
        else:
            world.graph.delete_edges(dropIds)
        
        # add t new connections
        currContacts = self.getPeerIDs(_cpp)
        
        frList, edgeList           = self.getRandomNewContacts(world, nDrops, currContacts)
        
        if len(edgeList) > 0:
        # update edges
            lg.debug('adding contact edges')
            world.addEdges(edgeList, type=_cpp, weig=1.0/nContacts)

        if world.caching:    
            self.cache.resetEdgeCache(edgeType=_cpp) 
            self.cache.resetPeerCache(edgeType=_cpp) 

    def getRandomNewContacts(self, world, nContacts, currentContacts):
        cellConnWeights, edgeIds, cellIds = self.loc.getConnCellsPlus()
        
        cell = world.entDict[np.random.choice(cellIds)]
        personIds = cell.getPersons()
        
        if len(personIds) > nContacts:
            contactIds = np.random.choice(personIds, size= nContacts, replace=False)
        else:
            return [], []
            
        contactIds = [person for person in contactIds if person not in currentContacts]    
                    
        contactList = contactIds
        connList    = [(self.nID, idx) for idx in contactIds]
        
        return contactList, connList
        
    def generateContactNetwork(self, world, nContacts = None,  currentContacts = None, addYourself = True):
        """
        Method to generate a preliminary friend network that accounts for 
        proximity in space, priorities and income
        """
        
    
        if currentContacts is None:
            isInit=True
        else:
            isInit=False
        if currentContacts is None:
            currentContacts = [self.nID]
        else:
            currentContacts.append(self.nID)
        
        if nContacts is None:
            nContacts = np.random.randint(world.para['minFriends'],world.para['maxFriends'])
        
        contactList = list()
        connList   = list()
        ownPref    = self.node['preferences']
        ownIncome  = self.hh.node['income']

        contactIds     = list()
        propDiffList   = list()
        incoDiffList   = list()
        spatWeigList   = list()
        


        #get spatial weights to all connected cells
        cellConnWeights, edgeIds, cellIds = self.loc.getConnCellsPlus()                    
        #print cellConnWeights
        #print [world.graph.vs[i]['gID']  for i in cellIds]
        
        for cellWeight, cellIdx in zip(cellConnWeights, cellIds):
            
            personIds = world.entDict[cellIdx].getPersons()
            

            if currentContacts is not None:
                #remove current contact from potential connections
                personIds = [person for person in personIds if person not in currentContacts]
            
            for personIdx in personIds:
                person = world.entDict[personIdx]
                propDiffList.append(np.sum([(x-y)**2 for x,y in zip (person.node['preferences'], ownPref ) ])) # TODO look if get faster
                incoDiffList.append(np.abs(person.hh.node['income'] - ownIncome))
                
            contactIds.extend(personIds)
            spatWeigList.extend([cellWeight]*len(personIds))


        # return nothing if too few candidates
        if not isInit and not len(contactIds) > nContacts:
            lg.info('ID: ' + str(self.nID) + ' failed to generate friend')
            
        else:
            np.seterr(divide='ignore')
            propWeigList = np.array(propDiffList)    
            nullIds  = propWeigList == 0
            propWeigList = 1 / propWeigList
            propWeigList[nullIds] = 0 
            propWeigList = propWeigList / np.sum(propWeigList)
            
    
            incoWeigList = np.array(incoDiffList)    
            nullIds  = incoWeigList == 0
            incoWeigList = 1 / incoWeigList
            incoWeigList[nullIds] = 0 
            np.seterr(divide='warn')
            incoWeigList = incoWeigList / np.sum(incoWeigList)
            
            
            spatWeigList = spatWeigList / np.sum(spatWeigList)
            spatWeigList = spatWeigList / np.sum(spatWeigList)
            
            weights = propWeigList * spatWeigList * incoWeigList
            weights = weights / np.sum(weights)
    
            if np.sum(weights>0) < nContacts:
                lg.info( "nID: " + str(self.nID) + ": Reducting the number of friends at " + str(self.loc.node['pos']))
                lg.info( "population = " + str(self.loc.node['population']) + " surrounding population: " +str(np.sum(self.loc.getPeerValues('population',_cll)[0])))
                
                nContacts = min(np.sum(weights>0)-1,nContacts)
    
            if nContacts < 1:
                lg.info('ID: ' + str(self.nID) + ' failed to generate friend')
            else:
                # adding contacts
                ids = np.random.choice(len(weights), nContacts, replace=False, p=weights)
                contactList = [ contactIds[idx] for idx in ids ]
                connList   = [(self.nID, contactIds[idx]) for idx in ids]
                
            
        if world.para['addYourself'] and addYourself:
            #add yourself as a friend
            contactList.append(self.nID)
            connList.append((self.nID,self.nID))
        
        weigList   = [1./len(connList)]*len(connList)    
        return contactList, connList, weigList

    def computeExpUtil(self,world):
        #get weights from friends
        weights, edges = self.getEdgeValues('weig', edgeType=_cpp) 
        weights = np.asarray(weights)
        
        # compute weighted mean of all friends
        communityUtil = np.dot(weights,np.asarray(self.getPeerValues('commUtil',_cpp)[0]))
        
        selfUtil = self.node['selfUtil'][:]
        mobType   = self.node['mobType']
        
        # weighting by 3
        selfUtil[mobType] *= world.para['selfTrust']
       
        if len(selfUtil) != 3 or len(communityUtil) != 3:
            print 'error: ' 
            print 'communityUtil: ' + str(communityUtil)
            print 'selfUtil: ' + str(selfUtil)
            return 
            
        self.node['commUtil'] = np.nanmean(np.asarray([communityUtil,selfUtil]),axis=0) 
        
        # adjust mean since double of weigth - very bad code - sorry        
        self.node['commUtil'][mobType] /= (world.para['selfTrust']+1)/2


    def step(self, world):
        
        
        # weight friends
        weights, ESSR = self.weightFriendExperience(world)
        
        
        # compute similarity
        weights, edges = self.getEdgeValues('weig', edgeType=_cpp) 
        weights = np.asarray(weights)
        preferences = np.asarray(self.getPeerValues('preferences',_cpp)[0]) 
        
        average = np.average(preferences, axis= 0, weights=weights)
        self.node['peerBubbleHeterogeneity'] = np.sum(np.sqrt(np.average((preferences-average)**2, axis=0, weights=weights)))
    
        # socialize
#        if ESSR < 0.1 and np.random.rand() >0.99:
#            self.socialize(world)
    
    
 
class GhostPerson(GhostAgent):
    
    def __init__(self, world, mpiOwner, nID=None, **kwProperties):
        GhostAgent.__init__(self, world, mpiOwner, nID, **kwProperties)
    
    def register(self, world, parentEntity=None, edgeType=None):
        
        GhostAgent.register(self, world, parentEntity, edgeType)
        
        
        self.loc = parentEntity.loc 
        self.loc.peList.append(self.nID)
        self.hh = parentEntity
    
class GhostHousehold(GhostAgent):       
    
    def __init__(self, world, mpiOwner, nID=None, **kwProperties):
        GhostAgent.__init__(self, world, mpiOwner, nID, **kwProperties)

    def register(self, world, parentEntity=None, edgeType=None):
        
        GhostAgent.register(self, world, parentEntity, edgeType)
        
        #self.queueConnection(locID,_clh)         
        self.loc = parentEntity
        self.loc.hhList.append(self.nID)        
        
class Household(Agent):

    def __init__(self, world, **kwProperties):
        Agent.__init__(self, world, **kwProperties)
        
        
        if world.para['util'] == 'cobb':
            self.utilFunc = self.cobbDouglasUtil
        elif world.para['util'] == 'ces':
            self.utilFunc = self.CESUtil
        self.computeTime = 0

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
 
                        
    #def registerAtLocation(self, world,x,y, nodeType, edgeType):
    def register(self, world, parentEntity=None, edgeType=None):
        
        
        Agent.register(self, world, parentEntity, edgeType)
        
        
        #self.queueConnection(locID,_clh)         
        self.loc = parentEntity
        self.loc.hhList.append(self.nID)
        
  
        
    def evalUtility(self, world, actionTaken=False):
        """
        Method to evaluate the utilty of all persons and the overall household
        """
        hhUtility = 0
        for adult in self.adults:
            
            utility = self.utilFunc(adult.node['consequences'], adult.node['preferences'])
            assert not( np.isnan(utility) or np.isinf(utility)), utility
            
            #adult.node['expUtilNew'][adult.node['mobType']] = utility + np.random.randn()* world.para['utilObsError']/10
            adult.node['util'] = utility
            
            
            if actionTaken:
                # self-util is only saved if an action is taken
                adult.node['selfUtil'][adult.node['mobType']] = utility
                
            hhUtility += utility
        
        self.node['util'] = hhUtility
        
        return hhUtility

    def evalExpectedUtility(self, earth, getInfoList):
    
        actionIdsList   = list()
        eUtilsList      = list()
        
        for i,adult in enumerate(self.adults):
            
            if getInfoList[i]: #or (earth.time < earth.para['burnIn']):
            #actionIds, eUtils = adult.getExpectedUtility(earth)
                adult.computeExpUtil(earth)
                actionIds = [-1, 0, 1, 2]
                eUtils = [adult.node['util']] + adult.node['commUtil'].tolist()
            #nonNanIdx = np.isnan(eUtils) == False
            #eUtils = eUtils[nonNanIdx]
            #actions = actionIds[nonNanIdx].tolist()
            else:
                actionIds, eUtils = [-1], [adult.node['util']]
#               
            actionIdsList.append(actionIds)
            eUtilsList.append(eUtils)
            

        
        if len(actionIdsList) == 0:
            return None, None
        
        elif len(actionIdsList) > 6:                            # to avoid the problem of too many possibilities (if more than 7 adults)
            minNoAction = len(actionIdsList) - 6                # minum number of adults not to take action    
            #import pdb
            #pdb.set_trace()
            #print 'Action List: ',actionIdsList
            while len(filter(lambda x: x == [-1], actionIdsList)) < minNoAction:
                randIdx = np.random.randint(len(actionIdsList))
                actionIdsList[randIdx] = [-1]
                eUtilsList[randIdx] =  [adult.node['util']]#[ eUtilsList[randIdx][0] ]
            #print 'large Household'
        
        combActions = aux.cartesian(actionIdsList)
        overallUtil = np.sum(aux.cartesian(eUtilsList),axis=1,keepdims=True)

        if len(combActions) != len(overallUtil):
            import pdb
            pdb.set_trace()
        
        return combActions, overallUtil
             

    def takeAction(self, earth, persons, actionIds):
        """
        Method to execute the optimal actions for selected persons of the household
        """
        for person, actionIdx in zip(persons, actionIds):
            #print 1
            properties = earth.market.buyCar(actionIdx)
            self.loc.addToTraffic(actionIdx)
            
            person.node['mobType']   = int(actionIdx)
            
            person.node['prop']      = properties
            person.node['obsID']     = None
            if earth.time <  earth.para['omniscientBurnIn']:
                person.node['lastAction'] = np.random.randint(0, int(1.5*earth.para['mobNewPeriod']))
            else:
                person.node['lastAction'] = 0
            # add cost of mobility to the expenses
            self.node['expenses'] += properties[1]
            
 
    def undoActions(self,world, persons):
        """
        Method to undo actions
        """
        for adult in persons:
            #mobID = adult.node['mobID']
            #world.market.sellCar(mobID)
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
            #if (adult.node['lastAction'] > 2*market.mobNewPeriod) and (actionIdx != 2):
                #decay = math.exp(-(adult.node['lastAction'] - 2*market.mobNewPeriod)**2)
                
            #else:
            #    decay = 1.
            if (actionIdx != 2):
                decay = 1- (1/(1+math.exp(-0.1*(adult.node['lastAction']-market.para['mobNewPeriod']))))                
            else:
                decay = 1.
            if (actionIdx == 2) and carInHh:
                hhCarBonus = 0.2
                
            convenience = decay * self.loc.getValue('convenience')[actionIdx] + hhCarBonus               
            
            # calculate ecology:
            emissions = mobProps[0]
            ecology = market.ecology(emissions)
            

            innovation = 1 - ( (market.allTimeProduced[adult.node['mobType']] 
                             / np.sum(market.allTimeProduced))**.5 )
            
            adult.node['consequences'] = [convenience, ecology, money, innovation]


    def bestMobilityChoice(self, earth, persGetInfoList , forcedTryAll = False):          # for test setupHouseholdsWithOptimalCars   (It's the best choice. It's true.)        
        market = earth.market
        actionTaken = True
        if len(self.adults) > 0 :
            combinedActions = self.possibleActions(earth, persGetInfoList, forcedTryAll)            
            utilities = list()
            
            # save current values
            oldMobType = list()
            oldProp = list()
            oldLastAction = list()
            for adult in self.adults:
                oldMobType.append(adult.node['mobType'])
                oldProp.append(adult.node['prop'])
                oldLastAction.append(adult.node['lastAction'])            
            oldExpenses = self.node['expenses']
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
                        if earth.time <  earth.para['burnIn']:
                            adult.node['lastAction'] = np.random.randint(0, earth.para['mobNewPeriod'])                    
                        else:
                            adult.node['lastAction'] = 0
                    self.node['expenses'] += adult.node['prop'][1]
                
                self.calculateConsequences(market)
                utility = self.evalUtility(earth)
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
            if self.decisionFunction(oldUtil, utilities[bestUtilIdx]):
                persons = np.array(self.adults)
                actionIds = np.array(bestCombination)
                actors = persons[ actionIds != -1 ] # remove persons that don't take action
                if len(actors) == 0:
                    actionTaken = False
                actions = actionIds[ actionIds != -1]
                self.undoActions(earth, actors)                  
                self.takeAction(earth, actors, actions)     # remove not-actions (i.e. -1 in list)     
                self.calculateConsequences(market)
                self.util = self.evalUtility(earth)
             
        else:
            actionTaken = False
        
        return actionTaken


    def decisionFunction(self, oldUtil, expNewUtil):
        if oldUtil == 0:
            return True
        else:
            return  expNewUtil / oldUtil > 1.05 and (expNewUtil / oldUtil ) - 1 > np.random.rand() 

    def possibleActions(self, earth, persGetInfoList , forcedTryAll = False):               
        actionsList = list()
        nMobTypes = earth.market.nMobTypes
        
        for adultIdx, adult in enumerate(self.adults):
            if forcedTryAll or (earth.time < earth.para['burnIn']) or persGetInfoList[adultIdx]:
                actionsList.append([-1]+range(nMobTypes))
            else:
                actionsList.append([-1])
        if len(actionsList) > 6:                            # to avoid the problem of too many possibilities (if more than 7 adults)
            minNoAction = len(actionsList) - 6              # minum number of adults not to take action    
            while len(filter(lambda x: x == [-1], actionsList)) < minNoAction:
                randIdx = np.random.randint(len(actionsList))
                actionsList[randIdx] = [-1]
            #print 'large Household'
                                          
        possibilities = aux.cartesian(actionsList)
        return possibilities    
            
            
    
    def maxUtilChoice(self, combActions, overallUtil):
        #best action 
        bestActionIdx = np.argmax(overallUtil)

        if bestActionIdx > len(combActions):
            import pdb
            pdb.set_trace()        
        actions = combActions[bestActionIdx]
        # return persons that buy a new car (action is not -1)

        actors = np.array(self.adults)[ actions != -1]
        actions = actions[ actions != -1]     

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
        tt = time.time()
        for adult in self.adults:
            adult.addValue('lastAction', 1)
            #adult.node['lastAction'] += 1
        actionTaken = False
        doCheckMobAlternatives = False

        if False: #earth.time < earth.para['burnIn']:
            doCheckMobAlternatives = True
            persGetInfoList = [True] * len(self.adults) # list of persons that gather information about new mobility options
            
        else:
            persGetInfoList = [adult.isAware(earth.para['mobNewPeriod'])  for adult in self.adults]
            #print persGetInfoList
            if any(persGetInfoList):
                doCheckMobAlternatives = True
                
            
        if doCheckMobAlternatives:
            
            # return persons that are potentially performing an action, the action and the expected overall utility
            
            
            combActions, overallUtil = self.evalExpectedUtility(earth,persGetInfoList)
            
            
            if (combActions is not None):
                personsToTakeAction, actions, expectedUtil = self.maxUtilChoice(combActions, overallUtil)
                #personsToTakeAction, actions, expectedUtil = self.propUtilChoice(combActions, overallUtil)
                
                if (personsToTakeAction is not None) and len(personsToTakeAction) > 0:
                
                    # the propbabilty of taking action is equal to the expected raise of the expected utility
                    if self.node['util'] == 0:
                        actionTaken = True                   
                    elif self.decisionFunction(self.node['util'], expectedUtil): #or (earth.time < earth.para['burnIn']):
                        actionTaken = True                   
                           
            # the action is only performed if flag is True
            
            if actionTaken:
                self.undoActions(earth, personsToTakeAction)
                self.takeAction(earth, personsToTakeAction, actions)

            self.calculateConsequences(earth.market)
            self.util = self.evalUtility(earth, actionTaken)
            
#            if actionTaken:                
#                self.shareExperience(earth)
        self.computeTime += time.time() - tt


    def stepOmniscient(self, earth):
        tt = time.time()
        
        for adult in self.adults:
            adult.addValue('lastAction', 1)
            #adult.node['lastAction'] += 1
        actionTaken = False
        doCheckMobAlternatives = False

        
        if earth.time < earth.para['burnIn']:
            doCheckMobAlternatives = True
            persGetInfoList = [True] * len(self.adults) # list of persons that gather information about new mobility options
        else:
            persGetInfoList = [adult.isAware(earth.para['mobNewPeriod'])  for adult in self.adults]
            if any (persGetInfoList):
                doCheckMobAlternatives = True
                            
        if doCheckMobAlternatives:            
            actionTaken = self.bestMobilityChoice(earth, persGetInfoList)
            self.calculateConsequences(earth.market)
            self.util = self.evalUtility(earth, actionTaken)
            self.evalExpectedUtility(earth, [True] * len(self.adults))
            
        self.computeTime += time.time() - tt
            
class Reporter(Household):
    
    def __init__(self, world, nodeType = 'ag', xPos = np.nan, yPos = np.nan):
        Household.__init__(self, world, nodeType , xPos, yPos)
        self.writer = aux.Writer(world, str(self.nID) + '_diary')
        raise('do not use - or update')
        
        #self.writer.write(


    
class Cell(Location):
    
    def __init__(self, earth,  **kwProperties):
        kwProperties.update({'population': 0, 'convenience': [0,0,0], 'carsInCell':[0,0,0], 'regionId':0})
        Location.__init__(self, earth, **kwProperties)
        self.hhList = list()
        self.peList = list()
        self.carsToBuy = 0
        self.deleteQueue =1
        self.currID = 0
        self.traffic = dict()
        self.sigmaEps = 1.
        self.muEps = 1.               
        self.cellSize = 1.
        
        self.urbanThreshold = earth.para['urbanThreshold']
        self.puplicTransBonus = earth.para['puplicTransBonus']     
       
        self.convFunctions = list()
        
        self.paraA = earth.para['convA']
        self.paraB = earth.para['convB']
        self.paraC = earth.para['convC']
        self.paraD = earth.para['convD']
        self.kappa = earth.para['kappa']
        
    def initCellMemory(self, memoryLen, memeLabels):
        from collections import deque
        self.deleteQueue = deque([list()]*(memoryLen+1))
        self.currDelList = list()
        self.obsMemory   = aux.Memory(memeLabels)


    def getConnCellsPlus(self):
         self.weights, edges = self.getEdgeValues('weig',edgeType=_cll)
         self.connNodeDict = [edge.target for edge in edges ]
         return self.weights, edges.indices, self.connNodeDict    
 
    def _getConnCellsPlusOld(self):
         self.weights, self.eIDs = self.getEdgeValues('weig',edgeType=_cll)
         self.connnodeDict = [self.graph.es[x].target for x in self.eIDs ]
         return self.weights, self.eIDs, self.connnodeDict

    
    def getHHs(self):
        return self.hhList
    
    def getPersons(self):
        return self.peList
    
    def getConnLoc(self,edgeType=1):
        return self.getAgentOfCell(edgeType=1)
    
    
    def addToTraffic(self,mobTypeID):
        self.addValue('carsInCell', 1, idx=int(mobTypeID))

        
    def remFromTraffic(self,mobTypeID):
        self.addValue('carsInCell', -1, idx=int(mobTypeID))



            
    def getX(self, choice):
        return copy.copy(self.xCell[choice,:])

    def selfTest(self, world):
        self.node['population'] = world.para['population'][self.node['pos']]
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
            convAll.append(min(1., max(0.0,funcCall(popDensity, self.paraA, self.paraB, self.paraC, self.paraD, self))))            
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
        
        
        convAll = self.calculateConveniences()
        self.setValue('convenience', convAll)
        #self.trafficMixture()

    
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
#        


class GhostCell(GhostLocation, Cell):
    getPersons = Cell.__dict__['getPersons']    
        
    def __init__(self, earth, **kwProperties):
        GhostLocation.__init__(self, earth, **kwProperties)
        self.hhList = list()
        self.peList = list()

    def updateHHList(self, graph):
        nodeType = graph.class2NodeType[Household]
        hhIDList = self.getPeerIDs(nodeType)
        self.hhList = graph.vs[hhIDList]
        
    def updatePeList(self, graph):
        nodeType = graph.class2NodeType[Person]
        hhIDList = self.getPeerIDs(nodeType)
        self.hhList = graph.vs[hhIDList]
        
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
        
        
        # priority of ecology
        ce = 0
        if sex == 2:
            ce +=1.5
        if income>self.minIncomeEco:
            rn = np.random.rand(1)
            if rn > 0.9:
                ce += 2
            elif rn > 0.6:
                ce += 1
        elif income>2*self.minIncomeEco:
            if np.random.rand(1) > 0.8:
                ce+=1.5
            

        ce = float(ce)**2
        
        # priority of convinience
        cc = 2.5
        cc += nKids
        cc += income/self.convIncomeFraction/2
        if sex == 1:
            cc +=1
        
        cc += float(age)/self.charAge
        cc = float(cc)**2
        
        # priority of money
        cm = 0
        cm += self.convIncomeFraction/income
        cm += nPers
        cm = float(cm)**2
        
        
        sumC = cc + ce + cm
        #cc /= sumC
        #ce /= sumC
        #cs /= sumC
        #cm /= sumC

        # priority of innovation
        if sex == 1:
            if income>self.minIncomeEco:
                ci = np.random.rand()*5
            else:
                ci = np.random.rand()*2
        else:
            if income>self.minIncomeEco:
                ci = np.random.rand()*3
            else:
                ci = np.random.rand()*1
        ci = float(ci)**2
        
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
