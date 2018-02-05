#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://www.globalclimateforum.org

---- MoTMo ----
MOBILITY TRANSFORMATION MODEL
-- Class definition FILE --

This file is based on GCFABM.

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

import class_auxiliary as aux # Record, Memory, Writer, cartesian


from lib_gcfabm import World, Agent, GhostAgent, Location, GhostLocation, h5py, MPI
#import pdb
#import igraph as ig
import numpy as np
#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
import time
import os
import math
import copy
import logging as lg
from bunch import Bunch
#%% --- ENUMERATIONS / CONSTANTS---
#connections
CON_LL = 1 # loc - loc
CON_LH = 2 # loc - household
CON_HH = 3 # household, household
CON_HP = 4 # household, person
CON_PP = 5 # household, person

#nodes
CELL = 1
HH   = 2
PERS = 3

#consequences
CONV = 0
ECO  = 1
MON  = 2
INNO = 3

#mobility types
BROWN  = 0
GREEN  = 1
PUBLIC = 2
SHARED  = 3
NONE   = 4


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
                       maxNodes=maxNodes,
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

        self.globalRecord = dict() # storage of global data


        # transfer all parameters to earth
        parameters.simNo = simNo
        self.setParameters(Bunch.toDict(parameters))

        if self.para['omniscientBurnIn']>self.para['burnIn']:
            self.para['omniscientBurnIn']=self.para['burnIn']

        if not os.path.isdir('output'):
            os.mkdir('output')


    def nPriorities(self):
        return len(self.enums['priorities'])


    def registerRecord(self, name, title, colLables, style ='plot', mpiReduce=None):
        """
        Creation of of a new record instance. 
        If mpiReduce is given, the record is connected with a global variable with the
        same name
        """
        self.globalRecord[name] = aux.Record(name, colLables, self.nSteps, title, style)

        if mpiReduce is not None:
            self.graph.glob.registerValue(name , np.asarray([np.nan]*len(colLables)),mpiReduce)
            self.globalRecord[name].glob = self.graph.glob

    # init car market
    def initMarket(self, earth, properties, propRelDev=0.01, time = 0, burnIn = 0):
        """
        Init of the market instance
        """
        self.market = Market(earth, properties, propRelDev=propRelDev, time=time, burnIn=burnIn)

    def initChargInfrastructure(self):
        self.chargingInfra = Infrastructure(self, self.para['roadKmPerCell'], 2.0, 4.0, 5.0)
    
    def registerBrand(self, label, propertyTuple, convFunction, initTimeStep, slope, initialProgress, allTimeProduced):
        """
        Method to register a new Brand in the Earth and therefore in the market.
        It currently adds the related convenience function in the cells.
        """
        brandID = self.market.initBrand(label, propertyTuple, initTimeStep, slope, initialProgress, allTimeProduced)

        # TODO: move convenience Function from a instance variable to a class variable            
        for cell in self.iterEntRandom(CELL):
            cell.traffic[brandID] = 0
            cell.convFunctions.append(convFunction)

        if 'brands' not in self.enums.keys():
            self.enums['brands'] = dict()
        self.enums['brands'][brandID] = label

        # adding a record about the properties of each goood
        self.registerRecord('prop_' + label, 'properties of ' + label,
             self.enums['properties'].values(), style='plot')


    def generateSocialNetwork(self, nodeType, edgeType):
        """
        Function for the generation of a simple network that regards the
        distance between the agents and their similarity in their preferences
        """
        tt = time.time()
        edgeList = list()
        weigList  = list()
        populationList = list()
        for agent, x in self.iterEntAndIDRandom(nodeType):

            nContacts = np.random.randint(self.para['minFriends'],self.para['maxFriends'])
            frList, edges, weights = agent.generateContactNetwork(self, nContacts)

            edgeList += edges
            weigList += weights
            populationList.append(agent.loc._node['population'])

        self.addEdges(edgeList, type=edgeType, weig=weigList)
        lg.info( 'Connections queued in -- ' + str( time.time() - tt) + ' s')

        if self.queuing:
            self.queue.dequeueEdges(self)

        lg.info( 'Social network created in -- ' + str( time.time() - tt) + ' s')
        lg.info( 'Average population: ' + str(np.mean(np.asarray(populationList))) + ' - Ecount: ' + str(self.graph.ecount()))

        fid = open(self.para['outPath']+ '/initTimes.out', 'a')
        fid.writelines('r' + str(self.mpi.comm.rank) + ', ' +
                       str( time.time() - tt) + ',' +
                       str(np.mean(np.asarray(populationList))) + ',' +
                       str(self.graph.ecount()) + '\n')

    def updateRecords(self):
        """
        Encapsulating method for the update of records
        """
        for re in self.para['regionIDList']:
            self.globalRecord['stock_' + str(re)].set(self.time,0)

        for cell in self.iterEntRandom(CELL):
            self.globalRecord['stock_' + str(int(cell.getValue('regionId')))].add(self.time,np.asarray(cell.getValue(('carsInCell')))* self.para['reductionFactor'])

        # move values to global data class
        for re in self.para['regionIDList']:
            self.globalRecord['stock_' + str(re)].updateValues(self.time)

    def syncGlobals(self):
        """
        Encapsulating method for the sync of global variables
        """
        ttSync = time.time()
        self.graph.glob.updateLocalValues('meanEmm', np.asarray(self.graph.vs[self.nodeDict[PERS]]['prop'])[:,1])
        self.graph.glob.updateLocalValues('stdEmm', np.asarray(self.graph.vs[self.nodeDict[PERS]]['prop'])[:,1])
        self.graph.glob.updateLocalValues('meanPrc', np.asarray(self.graph.vs[self.nodeDict[PERS]]['prop'])[:,0])
        self.graph.glob.updateLocalValues('stdPrc', np.asarray(self.graph.vs[self.nodeDict[PERS]]['prop'])[:,0])
        
        # local values are used to update the new global values
        self.graph.glob.sync()
        
        #gather data back to the records
        for re in self.para['regionIDList']:
            self.globalRecord['stock_' + str(re)].gatherSyncDataToRec(self.time)

        tmp = [self.graph.glob['meanEmm'], self.graph.glob['stdEmm'], self.graph.glob['meanPrc'], self.graph.glob['stdPrc']]
        self.globalRecord['mobProp'].set(self.time,np.asarray(tmp))


        self.syncTime[self.time] = time.time()-ttSync        
        lg.debug('globals synced in ' +str(time.time()- ttSync) + ' seconds')

    def syncGhosts(self):
        """
        Encapsulating method for the syncronization of selected ghost agents
        """
        ttSync = time.time()
        self.mpi.updateGhostNodes([PERS],['commUtil'])
        self.mpi.updateGhostNodes([CELL],['chargStat', 'carsInCell'])

        self.syncTime[self.time] += time.time()-ttSync
        lg.debug('Ghosts synced in ' + str(time.time()- ttSync) + ' seconds')

    
    def updateGraph(self):
        """
        Encapsulating method for the update of the graph structure 
        """
        ttUpd = time.time()
        if self.queuing:
            self.queue.dequeueEdges(self)
            self.queue.dequeueEdgeDeleteList(self)
        lg.debug('Graph updated in ' + str(time.time()- ttUpd) + ' seconds')

        
        
    def progressTime(self):
        """ 
        Progressing time and date
        """
        ttComp = time.time()
        self.time += 1
        self.timeStep = self.time

        # time management
        if self.timeStep == 0:
            lg.info( 'setting up time warp during burnin by factor of ' + str(self.para['burnInTimeFactor']))
            self.para['mobNewPeriod'] = int(self.para['mobNewPeriod'] / self.para['burnInTimeFactor'])
            newValue = np.rint(self.getNodeValues('lastAction',PERS) / self.para['burnInTimeFactor']).astype(int)
            self.setNodeValues('lastAction', newValue, PERS)

        elif self.timeStep+5 == self.para['burnIn']:
            lg.info( 'reducting time speed to normal')
            self.para['mobNewPeriod'] = int(self.para['mobNewPeriod'] * self.para['burnInTimeFactor'])
            oldValue = self.getNodeValues('lastAction',PERS) * self.para['burnInTimeFactor']
            newValue = oldValue.astype(int)
            stochastricRoundValue = newValue + (np.random.random(len(oldValue)) < oldValue-newValue).astype(int)

            self.setNodeValues('lastAction', stochastricRoundValue, PERS)
            
        else:
            lastActions = self.getNodeValues('lastAction',PERS)
            self.setNodeValues('lastAction',lastActions+1, PERS)
        
        # progressing time
        if self.timeUnit == 1: #months
            self.date[0] += 1
            if self.date[0] == 13:
                self.date[0] = 1
                self.date[1] += 1
        elif self.timeUnit == 1: # years
            self.date[1] +=1
        self.computeTime[self.time] += time.time()-ttComp
    
    
    def step(self):
        """
        Method to proceed the next time step
        """
        tt = time.time()

        self.progressTime()
        
        self.updateRecords()
    
        self.syncGlobals()
        
        self.syncGhosts()
        
        ttComp = time.time()

        # proceed infrastructure in time
        if self.time > self.para['burnIn']:
            self.chargingInfra.step(self)

        # proceed market in time
        self.market.step(self) # Statistics are computed here

        
        ###### Cell loop ######
        ttCell = time.time()
        for cell in self.iterEntRandom(CELL):
            cell.step(self.para, self.market.getCurrentMaturity())
        lg.debug('Cell step required ' + str(time.time()- ttCell) + ' seconds')



        ###### Person loop ######
        ttComp = time.time()
        for person in self.iterEntRandom(PERS):
            person.step(self)
        lg.debug('Person step required ' + str(time.time()- ttComp) + ' seconds')
   


        ###### Household loop ######
        tthh = time.time()
        if self.para['omniscientAgents'] or (self.time < self.para['omniscientBurnIn']):
            for household in self.iterEntRandom(HH):
                household.stepOmniscient(self)
        else:
            for household in self.iterEntRandom(HH):

                if np.random.rand()<1e-4:
                    household.stepOmniscient(self)
                else:
                    household.evolutionaryStep(self)
        lg.debug('Household step required ' + str(time.time()- tthh) + ' seconds')



        self.updateGraph()        

        self.market.updateSales()
        
        self.computeTime[self.time] += time.time()-ttComp

        # waiting for other processes
        ttWait = time.time()
        self.mpi.comm.Barrier()
        self.waitTime[self.time] += time.time()-ttWait

        # I/O
        ttIO = time.time()
        self.io.gatherNodeData(self.time)
        self.io.writeDataToFile(self.time)
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
        """
        Method to finalize records, I/O and reporter
        """
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

# ToDo
class Good():
    """
    Definition of an economic good in an sence. The class is used to compute
    technical progress in production and related unit costs.

    Technical progress is models according to the puplication of nagy+al2010
    https://doi.org/10.1371/journal.pone.0052669
    """
    
    #class variables (common for all instances, - only updated by class mehtods)
    
    lastGlobalSales  = list()

    @classmethod
    def updateGlobalSales(cls, sales):
        """
        Method to update the class variable sales
        """
        cls.lastGlobalSales = sales
        
    def __init__(self, label, progressType, initialProgress, slope, propDict, experience):
        #print self.lastGlobalSales
        self.goodID             = len(self.lastGlobalSales)
        self.label              = label
        self.currGrowthRate     = 1
        self.oldStock           = 0
        self.currStock          = 0
        self.replacementRate    = 0.01
        self.currLocalSales     = 0
        
        self.updateGlobalSales(self.lastGlobalSales + [0])
        
        
        
        if progressType == 'wright':

            self.experience        = experience # cummulative production up to now
            self.technicalProgress = initialProgress
            self.initialProperties = propDict.copy()
            self.properties        = propDict
            for key in self.properties.keys():
                self.initialProperties[key] = (self.initialProperties[key][0] - self.initialProperties[key][1], self.initialProperties[key][1])
                self.properties[key] = (self.initialProperties[key][0] / initialProgress) + self.initialProperties[key][1]


            self.slope        = slope
            self.maturity     = 1 - (1 / self.technicalProgress)

        else:
            print 'not implemented'
            # TODO add Moore and SKC + ...



        
    def updateTechnicalProgress(self, production=None):
        """
        Computes the technical progress
        If the production is not given, internally the stock and the replacement
        rate is used calcualte sales
        """ 
        
        # if production is not given, the internal sales is used
        if production is None:
            self.currLocalSales = self.salesModel()
        else:
            self.currLocalSales = production
            
        self.currGrowthRate = 1 + (self.lastGlobalSales[self.goodID]) / float(self.experience)
        self.technicalProgress = self.technicalProgress * (self.currGrowthRate)**self.slope
        for prop in self.properties.keys():
            self.properties[prop] = (self.initialProperties[prop][0] / self.technicalProgress) + self.initialProperties[prop][1]

        self.maturity       =    1 - (1 / self.technicalProgress)

        #update experience
        self.experience += self.lastGlobalSales[self.goodID]

        

    def step(self, doTechProgress=True):
        """ replaces the old stock by the current Stock and computes the 
        technical progress
        """
        if doTechProgress:
            self.updateTechnicalProgress()
        
        self.oldStock = self.currStock
        
        return self.properties, self.technicalProgress, self.maturity
        

    def buy(self,quantity=1):
        """
        Update of the internal stock
        """
        self.currStock +=1
        return self.properties.values()

    def sell(self, quantity=1):
        """ 
        Update of the internal stock
        """
        self.currStock -=1
    
    
    def salesModel(self):
        """
        Surrogate model for sales that can be used if agents sales are not 
        computed explicitely.
        """
        sales = np.max([0,self.currStock - self.oldStock])
        sales = sales + self.oldStock * self.replacementRate
        return sales
    
    
    def getProperties(self):
        return self.properties.values()

    def getProgress(self):
        return self.technicalProgress

    def getMaturity(self):
        return self.maturity

    def getExperience(self):
        return self.experience

    def getGrowthRate(self):
        return self.currGrowthRate

class Market():
    """
    Market class that mangages goood, technical progress and stocks.
    """
    def __init__(self, earth, properties, propRelDev=0.01, time = 0, burnIn=0):
        #import global variables
        self.globalRecord        = earth.returnGlobalRecord()
        self.comm                = earth.returnMpiComm()
        self.glob                = earth.returnGlobals()
        self._graph              = earth.returnGraph()
        self.para                = earth.getParameter()

        self.time                = time
        self.nodeDict            = earth.nodeDict
        self.properties          = properties                 # (currently: emission, costs)
        self.mobilityProp        = dict()                     # mobType -> [properties]
        self.nProp               = len(properties)
        self.propRelDev          = propRelDev                 # relative deviation of the actual car propeties
        self.obsDict             = dict()                     # time -> other dictionary (see next line)
        self.obsDict[self.time]  = dict()                     # (used by agents, observations (=utilities) also saved in locations)
        self.stockByMobType      = list()                     # list of total numbers per mobility type
        self.__nMobTypes__       = 0                          # number of different mobility types
        self.mobilityLables      = dict()                     # mobTypeID -> label
        self.mobilityInitDict    = dict()                     # list of (list of) initial values for each mobility type
        self.mobilityTypesToInit = list()                     # list of (mobility type) labels
        self.burnIn              = burnIn
        self.goods               = dict()
        self.sales               = list()

        #adding market globals
        self.glob.registerValue('sales' , np.asarray([0]),'sum')
        self.glob.registerStat('meanEmm' , np.asarray([0]*len(properties)),'mean')
        self.glob.registerStat('stdEmm' , np.asarray([0]*len(properties)),'std')
        self.glob.registerStat('meanPrc' , np.asarray([0]*len(properties)),'mean')
        self.glob.registerStat('stdPrc' , np.asarray([0]*len(properties)),'std')

    def updateSales(self):
        
        # push current Sales to globals for sync
        #ToDo: check this implementation of class variables
        sales = np.asarray([good.currLocalSales for good in self.goods.itervalues()])
        self.glob.updateLocalValues('sales', sales)
            
        # pull sales from last time step to oldSales for technical chance
        self.goods[0].updateGlobalSales(self.glob['sales'])
            
    def getNMobTypes(self):
        return self.__nMobTypes__

    def getTypes(self):
        return self.mobilityLables

    def initialCarInit(self):
        # actually puts the car on the market
        for label, propertyTuple, _, brandID, allTimeProduced in self.mobilityInitDict['start']:
            self.addBrand2Market(label, propertyTuple, brandID)

    def getCurrentMaturity(self):
        return [self.goods[iGood].getMaturity() for iGood in self.goods.keys()]

    def getCurrentExperience(self):
        return [self.goods[iGood].getExperience() for iGood in self.goods.keys()]


    def computeStatistics(self):
        self.mean['emissions'] = self.glob['meanEmm']
        self.mean['costs']     = self.glob['meanPrc']

        self.std['emissions']  = self.glob['stdEmm']
        self.std['costs']  = self.glob['stdPrc']


        lg.debug('Mean properties- mean: ' + str(self.mean) + ' std: ' + str(self.std))

    def setInitialStatistics(self, typeQuantities):
        total = sum(typeQuantities[mobIdx] for mobIdx in range(self.__nMobTypes__))
        shares = np.zeros(self.__nMobTypes__)

        self.mean = dict()
        self.std  = dict()

        for mobIdx in range(self.__nMobTypes__):
            shares[mobIdx] = typeQuantities[mobIdx]/total

        for prop in self.properties:
            propMean = sum(shares[mobIdx]*self.mobilityProp[mobIdx][prop] for mobIdx in range(self.__nMobTypes__))
            propVar = sum(shares[mobIdx]*(self.mobilityProp[mobIdx][prop])**2 for mobIdx in range(self.__nMobTypes__))-propMean**2
            propStd = math.sqrt(propVar)
            self.mean[prop] = propMean
            self.std[prop] = propStd


    def ecology(self, emissions):

        if self.std['emissions'] == 0:
            ecology = 1 / (1+np.exp((emissions-self.mean['emissions'])/1))
        else:
            ecology = 1 / (1+np.exp((emissions-self.mean['emissions'])/self.std['emissions']))

        return ecology


    def step(self, world):

        # check if a new car is entering the market
        if self.time in self.mobilityInitDict.keys():

            for mobTuple in self.mobilityInitDict[self.time]:
                for label, propertyDict, _, mobTypeID in  mobTuple:
                    self.addBrand2Market(label, propertyDict, mobTypeID)

        # only do technical change after the burn in phase
        doTechProgress = self.time > self.burnIn
        if doTechProgress:
            lg.info( 'sales in market: ' + str(self.glob['sales']))
            
        for iGood in self.goods.keys():
            self.goods[iGood].step(doTechProgress)

        if doTechProgress:
            lg.debug('techProgress: ' + str([self.glob['sales'][iGood] for iGood in self.goods.keys()]))
             
            
        if self.comm.rank == 0:
            self.globalRecord['growthRate'].set(self.time, [self.goods[iGood].getGrowthRate() for iGood in self.goods.keys()])
            for iGood in self.goods.keys():
                self.globalRecord['prop_' + self.goods[iGood].label].set(self.time, self.goods[iGood].getProperties())
            self.globalRecord['allTimeProduced'].set(self.time, self.getCurrentExperience())
            self.globalRecord['kappas'].set(self.time, self.getCurrentMaturity())

        lg.debug('new value of allTimeProduced: ' + str(self.getCurrentExperience()))
        # reset sales
        #self.glob['sales'] = self.glob['sales']*0

        self.minPrice = np.min([good.getProperties()[0] for good in self.goods.itervalues()])

        #compute new statistics
        self.computeStatistics()

        self.time +=1


    def initBrand(self, label, propertyDict, initTimeStep, slope, initialProgress, allTimeProduced):
        mobType = self.__nMobTypes__
        self.__nMobTypes__ +=1
        self.glob['sales'] = np.asarray([0]*self.__nMobTypes__)
        self.glob.updateLocalValues('sales', np.asarray([0]*self.__nMobTypes__))
        self.stockByMobType.append(0)
        self.mobilityTypesToInit.append(label)
        self.goods[mobType] = Good(label, 'wright',initialProgress, slope, propertyDict, experience=allTimeProduced)

        if initTimeStep not in self.mobilityInitDict.keys():
            self.mobilityInitDict[initTimeStep] = [[label, propertyDict, initTimeStep , mobType, allTimeProduced]]
        else:
            self.mobilityInitDict[initTimeStep].append([label, propertyDict, initTimeStep, mobType, allTimeProduced])

        return mobType

    def addBrand2Market(self, label, propertyDict, mobType):
        #add brand to the market
        self.stockByMobType[mobType]     = 0
        self.mobilityProp[mobType]       = propertyDict
        self.mobilityLables[mobType]     = label
        self.obsDict[self.time][mobType] = list()

    def remBrand(self,label):
        #remove brand from the market
        del self.mobilityProp[label]


    def getMobProps(self):
        return np.asarray([self.goods[iGood].getProperties() * 
                           (1 + np.random.randn(self.nProp) * self.propRelDev) 
                           for iGood in range(self.__nMobTypes__)])
    
    
    
    def buyCar(self, mobTypeIdx):
        # get current properties form good class
        propDict = self.goods[mobTypeIdx].buy() * (1 + np.random.randn(self.nProp)*self.propRelDev)

        return propDict
    
    
    def sellCar(self, mobTypeIdx):
        self.goods[mobTypeIdx].sell()
        


# %% --- entity classes ---

class Infrastructure():
    """
    Model for the development  of the charging infrastructure.
    """
    
    def __init__(self, earth, potMap, potFactor, immiFactor, dampFactor):
        self.potentialMap = potMap[earth.cellMapIds] ** potFactor# basic proxi variable for drawing new charging stations
        self.potentialMap = self.potentialMap / np.sum(self.potentialMap)
        
        # share of new stations that are build in the are of this process
        self.shareStationsOfProcess = np.sum(potMap[earth.cellMapIds]) / np.nansum(potMap)
        lg.debug('Share of new station for this process: ' + str(self.shareStationsOfProcess))
        self.immitationFactor = immiFactor
        self.dampenFactor     = dampFactor
        
        self.carsPerStation = 15. # number taken from assumptions of the German government
        
        self.sigPara = 2.56141377e+02, 3.39506037e-2 # calibarted parameters

    @staticmethod
    def sigmoid(x, x0, k):
        y = (1. / (1. + np.exp(-k*(x-x0)))) * 1e6
        return y
    
    def step(self, earth, nNewStations = None):
        
        if nNewStations is None:
            timeStep = earth.timeStep - earth.para['burnIn']
            nNewStations = self.sigmoid(np.asarray([timeStep-1, timeStep]), *self.sigPara) 
            nNewStations = int(np.diff(nNewStations) * self.shareStationsOfProcess)
        
        lg.debug('Adding ' + str(nNewStations) + ' new stations')##OPTPRODUCTION
        
        #get the current number of charging stations
        currNumStations  = earth.getNodeValues('chargStat', nodeType=CELL)
        greenCarsPerCell = earth.getNodeValues('carsInCell',CELL)[:,GREEN]+1.
        
        #immition factor (related to hotelings law that new competitiors tent to open at the same location)
        
        if np.sum(currNumStations) == 0:
            propability = self.potentialMap
        else:
            
            propImmi = (currNumStations)**self.immitationFactor
            propImmi = propImmi / np.nansum(propImmi)
            
            
            # dampening factor that applies for infrastructure that is not used and
            # reduces the potential increase of charging stations
            overSupplyFactor = 3
            demand  = greenCarsPerCell * earth.para['reductionFactor'] / self.carsPerStation * overSupplyFactor
            supply  = currNumStations
            dampFac  = (demand / supply) ** self.dampenFactor
            dampFac[np.isnan(dampFac)] = 1
            dampFac[dampFac > 1] = 1
            
            lg.debug('Dampening growth rate for ' + str(np.sum(dampFac < 1)) + ' cells with')##OPTPRODUCTION
            lg.debug(str(currNumStations[dampFac < 1]))##OPTPRODUCTION
            lg.debug('charging stations per cell - by factor of:')##OPTPRODUCTION
            lg.debug(str(dampFac[dampFac < 1]))##OPTPRODUCTION
            
            propability = (propImmi + self.potentialMap) * dampFac #* (usageMap[nonNanIdx] / currMap[nonNanIdx]*14.)**2
            propability = propability / np.sum(propability)
            
      
        
        
        randIdx = np.random.choice(range(len(currNumStations)), int(nNewStations), p=propability)
        
        uniqueRandIdx, count = np.unique(randIdx,return_counts=True)
        
        currNumStations[uniqueRandIdx] += count   
        earth.setNodeValues('chargStat', currNumStations, nodeType=CELL)

class Person(Agent):
    __slots__ = ['gID', 'nID']
    def __init__(self, world, **kwProperties):
        Agent.__init__(self, world, **kwProperties)
        

    def isAware(self, mobNewPeriod):
        # method that returns if the Persion is aktively searching for information
        return (self._node['lastAction'] - mobNewPeriod/10.) / (mobNewPeriod)  > np.random.rand()


    def register(self, world, parentEntity=None, edgeType=None):

        Agent.register(self, world, parentEntity, edgeType)
        self.loc = parentEntity.loc
        self.loc.peList.append(self.nID)
        self.hh = parentEntity
        self.hh.addAdult(self)


    def weightFriendExperience(self, world, commUtilPeers, edges, weights):
        friendUtil = commUtilPeers[:,self._node['mobType']]
        ownUtil  = self.getValue('util')
        
        diff = friendUtil - ownUtil +  np.random.randn(len(friendUtil))*world.para['utilObsError']
        prop = np.exp(-(diff**2) / (2* world.para['utilObsError']**2))
        prop = prop / np.sum(prop)

        prior = weights
        prior = prior / np.sum(prior)
        assert not any(np.isnan(prior)) ##OPTPRODUCTION


        # TODO - re-think how to avoide
        try:
            post = prior * prop
        except:
            import pdb
            pdb.set_trace()


        post = post / np.sum(post)
        
        if not(np.any(np.isnan(post)) or np.any(np.isinf(post))):
            if np.sum(post) > 0:
                edges['weig'] = post

                self._node['ESSR'] =  (1 / np.sum(post**2)) / float(len(post))
                #assert self.edges[CON_PP][self.ownEdgeIdx[0]].target == self.nID ##OPTPRODUCTION
                #assert self.edges[CON_PP][self.ownEdgeIdx[0]].source == self.nID ##OPTPRODUCTION
                if np.sum(self.getEdgeValues('weig', edgeType=CON_PP)[0]) < 0.99: ##OPTPRODUCTION
                    import pdb      ##OPTPRODUCTION
                    pdb.set_trace() ##OPTPRODUCTION
            return post, self._node['ESSR']

        else:
            lg.debug( 'post values:')
            lg.debug([ value for value in post])
            lg.debug('diff values:')
            lg.debug([value for value in diff])
            lg.debug('friendUtil values:')
            lg.debug([value for value in friendUtil])

            return weights, self._node['ESSR']


    def socialize(self, world):

#        drop 10% of old connections
#        print 'ID:', self.nID,
#        print "prior",
#        weights, edges = self.getEdgeValues('weig', edgeType=CON_PP)
#        print 'sum of weights', np.sum(weights)
#        for weig, idx in zip(weights, edges.indices):
#            print '(',weig, idx,')',
#        print ' '
        weights, edges = self.getEdgeValues('weig', edgeType=CON_PP)
        nContacts = len(weights)
        nDrops    = int(nContacts/10)
        dropIds = np.asarray(edges.indices)[np.argsort(weights)[:nDrops].tolist()].tolist()

        if world.queuing:
            world.queue.edgeDeleteList.extend(dropIds)
        else:
            world.graph.delete_edges(dropIds)

        # add t new connections
        currContacts = self.getPeerIDs(CON_PP)

        frList, edgeList           = self.getRandomNewContacts(world, nDrops, currContacts)

        if len(edgeList) > 0:
        # update edges
            lg.debug('adding contact edges')
            world.addEdges(edgeList, type=CON_PP, weig=1.0/nContacts)

        if world.caching:
            self.resetEdgeCache(edgeType=CON_PP)
            self.resetPeerCache(edgeType=CON_PP)

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

    def generateContactNetwork(self, world, nContacts,  currentContacts = None, addYourself = True):
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

        contactList = list()
        connList   = list()
        ownPref    = self._node['preferences']
        ownIncome  = self.hh._node['income']


        #get spatial weights to all connected cells
        cellConnWeights, edgeIds, cellIds = self.loc.getConnCellsPlus()
        personIdsAll = list()
        nPers = list()
        cellWeigList = list()
        for cellWeight, cellIdx in zip(cellConnWeights, cellIds):

            cellWeigList.append(cellWeight)
            personIds = world.getEntity(cellIdx).getPersons()

            personIdsAll.extend(personIds)
            nPers.append(len(personIds))

        # return nothing if too few candidates
        if not isInit and nPers > nContacts:
            lg.info('ID: ' + str(self.nID) + ' failed to generate friend')
            return [],[],[]

        #setup of indices of columes:
        idxColSp = 0
        idxColIn = 1
        idxColPr = range(idxColIn+1,world.nPriorities()+idxColIn+1)
        weightData = np.zeros([np.sum(nPers),len(idxColPr) +2])

        idx = 0
        for nP, we in zip(nPers, cellWeigList):
            weightData[idx:idx+nP,idxColSp] = we
            idx = idx+ nP
        del idx

        hhIDs = [world.glob2loc(x) for x in world.getNodeValues('hhID', idxList=personIdsAll)]
        weightData[:,idxColIn] = np.abs(world.getNodeValues('income', idxList=hhIDs) - ownIncome)
        weightData[:,idxColPr] = world.getNodeValues('preferences',   idxList=personIdsAll)


        for i in idxColPr:
            weightData[:,i]  = (weightData[:,i] - ownPref[i-2])**2

        weightData[:,idxColPr[0]] = np.sum(weightData[:,idxColPr], axis=1)

        nullIds  = weightData== 0

        #weight = inverse of distance
        np.seterr(divide='ignore')
        weightData = 1/weightData
        np.seterr(divide='warn')

        weightData[nullIds] = 0

        # normalization per row
        weightData[:,:3] = weightData[:,:3] / np.sum(weightData[:,:3],axis=0)
        #np.sum(weightData[:,:3] / np.sum(weightData[:,:3],axis=0),axis=0)

        #combining weights to the first row
        weightData[:,0] = np.prod(weightData[:,:3],axis=1)

        #normalizing final row
        weightData[:,0] = weightData[:,0] / np.sum(weightData[:,0],axis=0)

        if np.sum(weightData[:,0]>0) < nContacts:
            lg.info( "nID: " + str(self.nID) + ": Reducting the number of friends at " + str(self.loc.getValue('pos')))
            lg.info( "population = " + str(self.loc.getValue('population')) + " surrounding population: " +str(np.sum(self.loc.getPeerValues('population',CON_LL)[0])))

            nContacts = min(np.sum(weightData[:,0]>0)-1,nContacts)

        if nContacts < 1:                                                       ##OPTPRODUCTION
            lg.info('ID: ' + str(self.nID) + ' failed to generate friend')      ##OPTPRODUCTION

        else:
            # adding contacts
            ids = np.random.choice(weightData.shape[0], nContacts, replace=False, p=weightData[:,0])
            contactList = [ personIdsAll[idx] for idx in ids ]
            connList   = [(self.nID, personIdsAll[idx]) for idx in ids]

        if isInit and world.getParameter('addYourself') and addYourself:
            #add yourself as a friend
            contactList.append(self.nID)
            connList.append((self.nID,self.nID))

        weigList   = [1./len(connList)]*len(connList)
        return contactList, connList, weigList


    def computeCommunityUtility(self,earth, weights, edges, commUtilPeers):
        #get weights from friends
        #weights, edges = self.getEdgeValues('weig', edgeType=CON_PP)
        commUtil = self._node['commUtil'] # old value
        
        # compute weighted mean of all friends
        if earth.para['weightConnections']:
            commUtil += np.dot(weights, commUtilPeers)
        else:
            commUtil += np.mean(commUtilPeers,axis=0)

        mobType  = self.getValue('mobType')
        
        
        # adding weighted selfUtil selftrust
        commUtil[mobType] += self.getValue('selfUtil')[mobType] * earth.para['selfTrust']

        commUtil /= 2.
        commUtil[mobType] /= (earth.para['selfTrust']+2)/2.


        if len(self.getValue('selfUtil')) != earth.para['nMobTypes'] or len(commUtil.shape) == 0 or commUtil.shape[0] != earth.para['nMobTypes']:       ##OPTPRODUCTION
            print 'nID: ' + str(self.nID)                                       ##OPTPRODUCTION
            print 'error: '                                                     ##OPTPRODUCTION
            print 'communityUtil: ' + str(commUtil)                             ##OPTPRODUCTION
            print 'selfUtil: ' + str(self.getValue('selfUtil'))                 ##OPTPRODUCTION
            print 'nEdges: ' + str(len(edges))                                  ##OPTPRODUCTION

            return                                                              ##OPTPRODUCTION
        
        self.setValue('commUtil', commUtil.tolist())

        
        

    def imitate(self, utilPeers, weights, mobTypePeers):
        #pdb.set_trace()
        if np.random.rand() > .99:
            self.imitation = [np.random.choice(len(self.getValue('commUtil')))]
        else:
            #peerUtil     = np.asarray(self.getPeerValues('commUtil',CON_PP)[0])
            #peerMobType  = np.asarray(self.getPeerValues('mobType',CON_PP)[0])
            #weights      = np.asarray(self.getEdgeValues('weig', edgeType=CON_PP)[0])
            
            # weight of the fitness (quality) of the memes
            w_fitness = utilPeers / np.sum(utilPeers)
            
            # weight of reliability of the information (evolving over time)
            w_reliability = weights

            # combination of weights for random drawing
            w_full = w_fitness * w_reliability 
            w_full = w_full / np.sum(w_full)
        
            self.imitation =  np.random.choice(mobTypePeers, 2, p=w_full)
        

    def step(self, earth):
        
        #load data
        commUtilPeers  = np.asarray(self.getPeerValues('commUtil',CON_PP)[0])
        utilPeers      = np.asarray(self.getPeerValues('util',CON_PP)[0])
        weights, edges = self.getEdgeValues('weig', edgeType=CON_PP)
        weights        = np.asarray(weights)        
        mobTypePeers   = np.asarray(self.getPeerValues('mobType',CON_PP)[0])
        
        if earth.para['weightConnections'] and np.random.rand() > self.getValue('util'): 
            # weight friends
            weights, ESSR = self.weightFriendExperience(earth, commUtilPeers, edges, weights)

            # compute similarity
            #weights = np.asarray(self.getEdgeValues('weig', edgeType=CON_PP)[0])
            preferences = np.asarray(self.getPeerValues('preferences',CON_PP)[0])

            average = np.average(preferences, axis= 0, weights=weights)
            self._node['peerBubbleHeterogeneity'] = np.sum(np.sqrt(np.average((preferences-average)**2, axis=0, weights=weights)))
        
        self.computeCommunityUtility(earth, weights, edges, commUtilPeers) 
        if self.isAware(earth.para['mobNewPeriod']):
            self.imitate(utilPeers, weights, mobTypePeers)
        else:
            self.imitation = [-1]
        
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

        #self.queueConnection(locID,CON_LH)
        self.loc = parentEntity
        self.loc.hhList.append(self.nID)

class Household(Agent):
    __slots__ = ['gID', 'nID']
    

        
    def __init__(self, world, **kwProperties):
        Agent.__init__(self, world, **kwProperties)


        if world.para['util'] == 'cobb':
            self.utilFunc = self.cobbDouglasUtil
        elif world.getParamerter('util') == 'ces':
            self.utilFunc = self.CESUtil
        self.computeTime = 0


    @staticmethod
    def cobbDouglasUtil(x, alpha):
        utility = 1.
        
        for i in xrange(len(x)):
            utility *= (100.*x[i])**alpha[i] 
        #if np.isnan(utility) or np.isinf(utility):  ##DEEP_DEBUG
        #    import pdb                              ##DEEP_DEBUG
        #    pdb.set_trace()                         ##DEEP_DEBUG

        # assert the limit of utility
        #assert utility > 0 and utility <= factor     ##DEEP_DEBUG

        return utility / 100.
    
    @staticmethod
    def cobbDouglasUtilArray(x, alpha):
        utility = 1.
        
        
        utility = np.sum((100. * x) ** alpha) / 100.
        
        # assert the limit of utility
        #assert utility > 0 and utility <= factor      ##DEEP_DEBUG

        return utility
    

    @staticmethod
    def CESUtil(x, alpha):
        uti = 0.
        s = 2.    # elasticity of substitution, has to be float!
        factor = 100.
        for i in xrange(len(x)):
            uti += (alpha[i]*(100. * x[i])**(s-1))**(1/s)
            #print uti
        utility = uti**(s/(s-1))
        #if  np.isnan(utility) or np.isinf(utility): ##DEEP_DEBUG
        #    import pdb                              ##DEEP_DEBUG
        #    pdb.set_trace()                         ##DEEP_DEBUG

        # assert the limit of utility
        #assert utility > 0 and utility <= factor ##DEEP_DEBUG

        return utility / 100.


    #def registerAtLocation(self, world,x,y, nodeType, edgeType):
    def register(self, world, parentEntity=None, edgeType=None):


        Agent.register(self, world, parentEntity, edgeType)


        #self.queueConnection(locID,CON_LH)
        self.loc = parentEntity
        self.loc.hhList.append(self.nID)

    def addAdult(self, personInstance):
        """adding reference to the person to household"""
        self.adults.append(personInstance)
        
    def setAdultNodeList(self, world):
        adultIdList = [adult.nID for adult in self.adults]
        self.adultNodeList = world.graph.vs[adultIdList]

    def evalUtility(self, world, actionTaken=False):
        """
        Method to evaluate the utilty of all persons and the overall household
        """
        hhUtility = 0
        for adult in self.adults:

            utility = self.utilFunc(adult.getValue('consequences'), adult.getValue('preferences'))
            #assert not( np.isnan(utility) or np.isinf(utility)), utility ##OPTPRODUCTION

            #adult.node['expUtilNew'][adult.node['mobType']] = utility + np.random.randn()* world.para['utilObsError']/10
            adult.setValue('util', utility)


            if actionTaken:
                # self-util is only saved if an action is taken
                adult._node['selfUtil'][adult.getValue('mobType')] = utility

            hhUtility += utility

        self.setValue('util', hhUtility)

        return hhUtility

    def evalExpectedUtility(self, earth, getInfoList):

        actionIdsList   = list()
        eUtilsList      = list()

        for i,adult in enumerate(self.adults):

            if getInfoList[i]: #or (earth.time < earth.para['burnIn']):
                adult.computeCommunityUtility(earth)
                actionIds = [-1] + range(earth.para['nMobTypes'])

                eUtils = [adult.getValue('util')] + adult.getValue('commUtil')

            else:
                actionIds, eUtils = [-1], [adult.getValue('util')]
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
                eUtilsList[randIdx] =  [adult.getValue('util')]#[ eUtilsList[randIdx][0] ]
            #print 'large Household'

        combActions = aux.cartesian(actionIdsList)
        overallUtil = np.sum(aux.cartesian(eUtilsList),axis=1,keepdims=True)

        if len(combActions) != len(overallUtil):        ##OPTPRODUCTION
            import pdb                                  ##OPTPRODUCTION
            pdb.set_trace()                             ##OPTPRODUCTION

        return combActions, overallUtil


    def takeActions(self, earth, persons, actionIds):
        """
        Method to execute the optimal actions for selected persons of the household
        """
        for person, actionIdx in zip(persons, actionIds):

            properties = earth.market.buyCar(actionIdx)
            self.loc.addToTraffic(actionIdx)

            person.setValue('mobType', int(actionIdx))

            person.setValue('prop', properties)
            if earth.time <  earth.para['omniscientBurnIn']:
                person.setValue('lastAction', np.random.randint(0, int(1.5*earth.para['mobNewPeriod'])))
            else:
                person.setValue('lastAction', 0)
            # add cost of mobility to the expenses
            self.addValue('expenses', properties[0])


    def undoActions(self, world, persons):
        """
        Method to undo actions
        """
        for adult in persons:
            mobType = adult.getValue('mobType')
            self.loc.remFromTraffic(mobType)

            # remove cost of mobility to the expenses
            self.addValue('expenses', -1 * adult.getValue('prop')[0])
            world.market.sellCar(mobType)

    def testConsequences(self, earth, actionIds):
        
        consMat = np.zeros([actionIds.shape[0], actionIds.shape[1], earth.nPriorities()])
        

        
        hhCarBonus = 0.2
        mobProperties = earth.market.getMobProps()
        experience    = np.asarray(earth.market.getCurrentExperience())
        sumExperience = sum(experience)
        convCell      = np.asarray(self.loc.getValue('convenience'))
        income        = self.getValue('income')
        
        for ix, actions in enumerate(actionIds):
            
            #convenience
            consMat[ix,:,CONV] = convCell[actions]
            if any(actions < 2):
                consMat[ix,actions==2,CONV] += hhCarBonus
                consMat[ix,actions==4,CONV] += hhCarBonus
                
            #ecology
            consMat[ix,:,ECO] = earth.market.ecology(mobProperties[actions,1])
            
            consMat[ix,:,MON] = max(1e-5, 1 - np.sum(mobProperties[actions,0]) / income)
            
            #immitation
            consMat[ix,:,INNO] = 1 - ( (experience[actions] / sumExperience) **.5)
            
        return consMat


    def testConsequences2(self, earth, actionIds):
        
        consMat = np.zeros([actionIds.shape[0], actionIds.shape[1], earth.nPriorities()])
        

        
        hhCarBonus = 0.2
        mobProperties = earth.market.getMobProps()
        experience    = earth.market.getCurrentExperience()
        sumExperience = sum(experience)
        convCell      = self.loc.getValue('convenience')
        income        = self.getValue('income')
        
        for ix, actions in enumerate(actionIds):
            
            #convenience
            consMat[ix,:,CONV] = [convCell[action] for action in actions]
            if any(actions < 2):
                consMat[ix,actions==2,CONV] += hhCarBonus
                consMat[ix,actions==4,CONV] += hhCarBonus
                
            #ecology
            consMat[ix,:,ECO] = [earth.market.ecology(mobProperties[action,1]) for action in actions]
            
            consMat[ix,:,MON] = max(1e-5, 1 - sum([mobProperties[action,0] for action in actions] / income)) 
            
            #immitation
            consMat[ix,:,INNO] = [1 - ( (experience[action] / sumExperience) **.5) for action in actions]
            
        return consMat
        
        

    def calculateConsequences(self, market):

        carInHh = False
        # at least one car in househould?
        if any([adult.getValue('mobType') !=2 for adult in self.adults]):
            carInHh = True

        # calculate money consequence
        money = min(1., max(1e-5, 1 - self.getValue('expenses') / self.getValue('income')))


        for adult in self.adults:
            hhCarBonus = 0.
            #get action of the person

            actionIdx = adult.getValue('mobType')
            mobProps  = adult.getValue('prop')

#            if (actionIdx != 2):
#                decay = 1- (1/(1+math.exp(-0.1*(adult.getValue('lastAction')-market.para['mobNewPeriod']))))
#            else:
#                decay = 1.
            if (actionIdx > 2) and carInHh:
                hhCarBonus = 0.2

            convenience = self.loc.getValue('convenience')[actionIdx] + hhCarBonus

            # calculate ecology:
            emissions = mobProps[1]
            ecology   = market.ecology(emissions)

            experience = market.getCurrentExperience()

            innovation = 1 - ( (experience[adult.getValue('mobType')] / np.sum(experience))**.5 )
            
            # assuring that consequences are within 0 and 1
            for consequence in [convenience, ecology, money, innovation]:     ##OPTPRODUCTION
                if not((consequence <= 1) and (consequence >= 0)):            ##DEEPDEBUG
                    print consequence                                         ##DEEPDEBUG      
                    import pdb                                                ##DEEPDEBUG
                    pdb.set_trace()                                           ##DEEPDEBUG  
                assert (consequence <= 1) and (consequence >= 0)              ##OPTPRODUCTION

            adult.setValue('consequences', [convenience, ecology, money, innovation])


    def bestMobilityChoice(self, earth, persGetInfoList , forcedTryAll = False):
        """
        (It's the best choice. It's true. It's huge)
        """
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
                oldMobType.append(adult.getValue('mobType'))
                oldProp.append(adult.getValue('prop'))
                oldLastAction.append(adult.getValue('lastAction'))
            oldExpenses = self.getValue('expenses')
            oldUtil = copy.copy(self.getValue('util'))


            # try all mobility combinations
            for combinationIdx in range(len(combinedActions)):
                self._node['expenses'] = 0
                for adultIdx, adult in enumerate(self.adults):
                    if combinedActions[combinationIdx][adultIdx] == -1:     # no action taken
                        adult.setValue('mobType', oldMobType[adultIdx])
                        adult.setValue('prop', oldProp[adultIdx])
                        adult.setValue('lastAction', oldLastAction[adultIdx])
                    else:
                        adult.setValue('mobType', combinedActions[combinationIdx][adultIdx])
                        adult.setValue('prop', market.goods[adult.getValue('mobType')].getProperties())
                        if earth.time <  earth.para['burnIn']:
                            adult.setValue('lastAction', np.random.randint(0, int(1.5* earth.para['mobNewPeriod'])))
                        else:
                            adult.setValue('lastAction', 0)
                    self.setValue('expenses', adult.getValue('prop')[0])

                self.calculateConsequences(market)
                utility = self.evalUtility(earth)

#                    import pprint as pp
#                    print combinedActions[combinationIdx,:]
#                    [pp.pprint(adult.node['prop'][0]) + ', ' for adult in self.adults]
#                    [pp.pprint(adult.node['consequences']) for adult in self.adults]
#                    #[pp.pprint(adult.node['preferences']) for adult in self.adults]
#                    [pp.pprint(adult.node['util']) for adult in self.adults]



                utilities.append(utility)

            # reset old node values
            for adultIdx, adult in enumerate(self.adults):
                adult.setValue('mobType', oldMobType[adultIdx])
                adult.setValue('prop', oldProp[adultIdx])
                adult.setValue('lastAction', oldLastAction[adultIdx])
            self.setValue('expenses', oldExpenses)

            # get best combination
            bestUtilIdx = np.argmax(utilities)
            bestCombination = combinedActions[bestUtilIdx]


            # set best combination
            if self.decisionFunction(oldUtil, utilities[bestUtilIdx]):
                persons = np.array(self.adults)
                actionIds = np.array(bestCombination)
                actors = persons[ actionIds != -1 ] # remove persons that don't take action
                if len(actors) == 0:
                    actionTaken = False
                actions = actionIds[ actionIds != -1]
                self.undoActions(earth, actors)
                self.takeActions(earth, actors, actions)     # remove not-actions (i.e. -1 in list)
                self.calculateConsequences(market)
                util = self.evalUtility(earth)

                if util < 1:                                                                                    ##OPTPRODUCTION
                    lg.debug('####' + str(self.nID) + '#####')                                                  ##OPTPRODUCTION
                    lg.debug('New Util: ' +str(util) + ' old util: '                                            ##OPTPRODUCTION
                             + str(oldUtil) + ' exp. Util: ' + str(utilities[bestUtilIdx]))                     ##OPTPRODUCTION
                    lg.debug('possible utilitties: ' + str(utilities))                                          ##OPTPRODUCTION                
                    lg.debug(self._node)                                                                        ##OPTPRODUCTION
                    lg.debug('properties: ' + str([adult.getValue('prop') for adult in self.adults]))           ##OPTPRODUCTION
                    lg.debug('consequence: '+ str([adult.getValue('consequences') for adult in self.adults]))   ##OPTPRODUCTION
                    lg.debug('preferences: '+ str([adult.getValue('preferences') for adult in self.adults]))    ##OPTPRODUCTION
                    lg.debug('utility: ' +    str([adult.getValue('util')  for adult in self.adults]))          ##OPTPRODUCTION
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
        nMobTypes = earth.market.getNMobTypes()

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

        if bestActionIdx > len(combActions):##OPTPRODUCTION
            import pdb                      ##OPTPRODUCTION
            pdb.set_trace()                 ##OPTPRODUCTION
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
        #if overallUtil[propActionIdx] is None: ##OPTPRODUCTION
        #    print 1                            ##OPTPRODUCTION
        return actors, actions, overallUtil[propActionIdx]


    def householdOptimization(self, earth, actionOptions):
 
        #removing possible none-actions (encoded as -1)
        try:
            actorIds = [idx for idx, option in enumerate(actionOptions) if option[0] != -1 ]
            actionOptions = [actionOptions[idx] for idx in actorIds]
        except:
            import pdb
            pdb.set_trace()

        if len(actionOptions) == 0:
            return None, None, None
         
        if len(actorIds) > 6:
            actorIds = np.random.choice(actorIds,6,replace=False)
            actionOptions = [actionOptions[idx] for idx in actorIds]
#        else:
#            actorIds = None
        combinedActionsOptions = aux.cartesian(actionOptions)
        
        #tt2 = time.time()
        consMat = self.testConsequences(earth, aux.cartesian(actionOptions)) #shape [nOptions x nPersons x nConsequences]
        #print 'testConsequences : ' +str(time.time() -tt2)
        #tt2 = time.time()
        #consMat = self.testConsequences2(earth, aux.cartesian(actionOptions)) #shape [nOptions x nPersons x nConsequences]
        #print 'testConsequences2: ' +str(time.time() -tt2)
        utilities = np.zeros(consMat.shape[0])
        #print 'done'
        prioMat = self.adultNodeList['preferences'] # [nPers x nPriorities]
        
        if actorIds is None:
            for iAction in range(consMat.shape[0]):
                for iPers in range(len(self.adults)):
                    utilities[iAction] += self.utilFunc(consMat[iAction,iPers,:], prioMat[iPers])
        else:
            for iAction in range(consMat.shape[0]):
                for ii, iPers in enumerate(actorIds):
                    utilities[iAction] += self.utilFunc(consMat[iAction,ii,:], prioMat[iPers])
                
        bestOpt = combinedActionsOptions[np.argmax(utilities)]
        return bestOpt, np.max(utilities), actorIds

    def evolutionaryStep(self, earth):
        """
        Evolutionary time step that uses components of genetic algorithms
        on social networks to mimic the evolutions of norms / social learning.
        The friends of an agents are replacing the population in a genetic
        algorithm from which no genes evolve by mutation, crossover and random.
        """
        tt = time.time()
        
        bestIndividualActionsIds = [adult.imitation for adult in self.adults]
        #print 'imiate: ' +str(time.time() -tt)
        #tt2 = time.time()
        bestOpt, bestUtil, actorIds = self.householdOptimization(earth, bestIndividualActionsIds)
        #print 'opt: ' +str(time.time() -tt2)
        #tt3 = time.time()
        
        if actorIds is not None:
#            self.undoActions(earth, self.adults)
#            self.takeActions(earth, self.adults, bestOpt)
#        else:
            self.undoActions(earth, [self.adults[idx] for idx in actorIds])
            self.takeActions(earth, [self.adults[idx] for idx in actorIds], bestOpt)
        
        self.calculateConsequences(earth.market)
        
        self.evalUtility(earth, actionTaken=True)
        
        #print 'eval: ' +str(time.time() -tt3)
        #print str(bestUtil) + ' -> ' + str(util)


            
        self.computeTime += time.time() - tt

    def step(self, earth):
        tt = time.time()

        actionTaken = False
        doCheckMobAlternatives = False

        if earth.time < earth.para['burnIn']:
            doCheckMobAlternatives = True
            persGetInfoList = [True] * len(self.adults) # list of persons that gather information about new mobility options

        else:
            if len(self.adults)*earth.market.minPrice < self._node['income']: #TODO

                #self._node['nPers']
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
                    if self._node['util'] == 0:
                        actionTaken = True
                    elif self.decisionFunction(self._node['util'], expectedUtil): #or (earth.time < earth.para['burnIn']):
                        actionTaken = True

            # the action is only performed if flag is True

            if actionTaken:
                self.undoActions(earth, personsToTakeAction)
                self.takeActions(earth, personsToTakeAction, actions)

            self.calculateConsequences(earth.market)
            self.evalUtility(earth, actionTaken)

#            if actionTaken:
#                self.shareExperience(earth)
        self.computeTime += time.time() - tt


    def stepOmniscient(self, earth):
        tt = time.time()

        actionTaken = False
        doCheckMobAlternatives = False

        if len(self.adults)*earth.market.minPrice < self._node['income']:

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
            util = self.evalUtility(earth, actionTaken)
            #self.evalExpectedUtility(earth, [True] * len(self.adults))

        self.computeTime += time.time() - tt

#TODO Check if reporter class is still useful
class Reporter(Household):

    def __init__(self, world, nodeType = 'ag', xPos = np.nan, yPos = np.nan):
        Household.__init__(self, world, nodeType , xPos, yPos)
        self.writer = aux.Writer(world, str(self.nID) + '_diary')
        raise('do not use - or update')

        #self.writer.write(



class Cell(Location):

    def __init__(self, earth,  **kwProperties):
        kwProperties.update({'population': 0, 'convenience': [0,0,0,0,0], 'carsInCell':[0,0,0,0,0], 'regionId':0})
        Location.__init__(self, earth, **kwProperties)
        self.hhList = list()
        self.peList = list()
        self.carsToBuy = 0
        self.deleteQueue =1
        self.traffic = dict()
        self.cellSize = 1.
        self.convFunctions = list()


    def initCellMemory(self, memoryLen, memeLabels):
        """
        deprectated
        """
        from collections import deque
        self.deleteQueue = deque([list()]*(memoryLen+1))
        self.currDelList = list()
        self.obsMemory   = aux.Memory(memeLabels)


    def getConnCellsPlus(self):
        """ 
        ToDo: check if not deprecated 
        """
        self.weights, edges = self.getEdgeValues('weig',edgeType=CON_LL)
        self.connNodeDict = [edge.target for edge in edges ]
        return self.weights, edges.indices, self.connNodeDict

    def _getConnCellsPlusOld(self):
        """
        depreciated
        """
        self.weights, self.eIDs = self.getEdgeValues('weig',edgeType=CON_LL)
        self.connnodeDict = [self._graph.es[x].target for x in self.eIDs ]
        return self.weights, self.eIDs, self.connnodeDict


    def getHHs(self):
        return self.hhList

    def getPersons(self):
        return self.peList

    def getConnLoc(self,edgeType=1):
        return self.getAgentOfCell(edgeType=1)


    def addToTraffic(self,mobTypeID):
        """
        
        """
        self.addValue('carsInCell', 1, idx=int(mobTypeID))


    def remFromTraffic(self,mobTypeID):
        self.addValue('carsInCell', -1, idx=int(mobTypeID))




    def getX(self, choice):
        return copy.copy(self.xCell[choice,:])

    def selfTest(self, world):
        """
        Not used in the simulations, but for testing purposes.
        
        """
        #self._node['population'] = population #/ float(world.getParameter('reductionFactor'))

        convAll = self.calculateConveniences(world.getParameter(), world.market.getCurrentMaturity())

#        for x in convAll:                             ##OPTPRODUCTION
#            if np.isinf(x) or np.isnan(x) or x == 0:  ##OPTPRODUCTION
#                import pdb                            ##OPTPRODUCTION
#                pdb.set_trace()                       ##OPTPRODUCTION
        #self._node['population'] = 0
        return convAll, self._node['popDensity']

    def calculateConveniences(self, parameters, currentMaturity):
        """
        Calculation of convenience for all goods + the electric infrastructure.
        
        returns list of conveniences Range: [0,1]
        
        ToDo:
            - integration in goods?
            - seperation of infrastructure and convenience of goods
        
        """
        convAll = list()

        popDensity = np.float(self.getValue('popDensity'))
        for i, funcCall in enumerate(self.convFunctions):
            convAll.append(funcCall(popDensity, parameters, currentMaturity[i], self))
            
        #convenience of electric mobility is additionally dependen on the infrastructure
        convAll[GREEN] *= self.electricInfrastructure()
        return convAll


    def electricInfrastructure(self, greenMeanCars = None):
        """ 
        Method for a more detailed estimation of the convenience of 
        electric intrastructure.
        Two components are considered: 
            - minimal infrastructure
            - capacity use
        Mapping between [0,1]
        """
        
        weights       = self.getEdgeValues('weig',CON_LL)[0]
        
        if greenMeanCars is None:
            
            carsInCells   = np.asarray(self.getPeerValues('carsInCell',CON_LL)[0])
            greenMeanCars =  sum([x*y for  x,y in zip(carsInCells[:,GREEN],weights)])    
        
        nStation      = self.getPeerValues('chargStat',CON_LL)[0]
        #print nStation
        if np.sum(nStation) == 0:
            return 0.
        
        
        # part of the convenience that is related to the capacity that is used
        avgStatPerCell = sum([x*y for  x,y in zip(nStation, weights)])
        
        capacityUse = greenMeanCars / (avgStatPerCell * 200.)
        
        if capacityUse > 100:
            useConv = 0
        else:
            useConv = 1 /  math.exp(capacityUse)
        
        # part of the convenience that is related to a minimal value of charging
        # stations
        minRequirement = 1
        
        if avgStatPerCell < minRequirement:
            statMinRequ = 1 / math.exp((1.-avgStatPerCell)**2 / .2)
        else:
            statMinRequ = 1.
        
        # overall convenience is product og both 
        eConv = statMinRequ * useConv
#        if self.getValue('chargStat') == 26:
#            import pdb
#            pdb.set_trace()
        assert (eConv >= 0) and (eConv <= 1) ##OPTPRODUCTION
        return eConv

        #%%
#        xx = range(1,1000)
#        cap = 200. # cars per load station
#        nStat = 2.
#        y = [ 1 /  np.exp(x / (nStat*cap)) for x in xx]
#        plt.clf()
#        plt.plot(xx,y)
#        #%%
#        x = np.linspace(0.01,1.0,50)
#        y = 1 / np.exp((1.-x)**2 / .1)
#        plt.clf()
#        plt.plot(x,y)
         #%%
        
    def step(self, parameters, currentMaturity):
        """
        Step method for cells
        """

        convAll = self.calculateConveniences(parameters,currentMaturity)
        self.setValue('convenience', convAll)


    def registerObs(self, hhID, prop, util, label):
        """
        Adds a car to the cell pool of observations
        """
        meme = prop + [util, label, hhID]
        assert not any(np.isnan(meme)) ##OPTPRODUCTION
        obsID = self.obsMemory.addMeme(meme)
        self.currDelList.append(obsID)

        return obsID



class GhostCell(GhostLocation, Cell):
    getPersons = Cell.__dict__['getPersons']

    def __init__(self, earth, **kwProperties):
        GhostLocation.__init__(self, earth, **kwProperties)
        self.hhList = list()
        self.peList = list()

    def updateHHList(self, graph):
        """
        updated method for the household list, which is required since
        ghost cells are not active on their own
        """
        nodeType = graph.class2NodeType[Household]
        hhIDList = self.getPeerIDs(nodeType)
        self.hhList = graph.vs[hhIDList]

    def updatePeList(self, graph):
        """
        updated method for the people list, which is required since
        ghost cells are not active on their own
        """
        nodeType = graph.class2NodeType[Person]
        hhIDList = self.getPeerIDs(nodeType)
        self.hhList = graph.vs[hhIDList]

class Opinion():
    """
    Creates preferences for households, given their properties
    ToDO:
        - Update the method + more sophisticate method maybe using the DLR Data
    """
    def __init__(self, world):
        self.charAge            = world.getParameter('charAge')
        self.indiRatio          = world.getParameter('individualPrio')
        self.minIncomeEco       = world.getParameter('minIncomeEco')
        self.convIncomeFraction = world.getParameter('charIncome')

    def getPref(self, age, sex, nKids, nPers, income, radicality):


        # priority of ecology
        ce = 0.5
        if sex == 2:
            ce +=1.5
        if income > self.minIncomeEco:
            rn = np.random.rand(1)
            if rn > 0.9:
                ce += 2
            elif rn > 0.6:
                ce += 1
        elif income > 2 * self.minIncomeEco:
            if np.random.rand(1) > 0.8:
                ce+=2.5


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

        assert all (pref > 0) and all (pref < 1) ##OPTPRODUCTION

        return tuple(pref)

# %% --- main ---
if __name__ == "__main__":
    pass