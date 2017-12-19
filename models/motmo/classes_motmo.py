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

from lib_gcfabm import World, Agent, GhostAgent, Location, GhostLocation, aux, h5py, MPI
import pdb
import igraph as ig
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

        self.globalRecord[name] = aux.Record(name, colLables, self.nSteps, title, style)

        if mpiReduce is not None:
            self.graph.glob.registerValue(name , np.asarray([np.nan]*len(colLables)),mpiReduce)
            self.globalRecord[name].glob = self.graph.glob

    # init car market
    def initMarket(self, earth, properties, propRelDev=0.01, time = 0, burnIn = 0):
        self.market = Market(earth, properties, propRelDev=propRelDev, time=time, burnIn=burnIn)


    def initBrand(self, label, propertyTuple, convFunction, initTimeStep, slope, initialProgress, allTimeProduced):
        brandID = self.market.initBrand(label, propertyTuple, initTimeStep, slope, initialProgress, allTimeProduced)

        for cell in self.iterEntRandom(_cell):
            cell.traffic[brandID] = 0
            cell.convFunctions.append(convFunction)

        if 'brands' not in self.enums.keys():
            self.enums['brands'] = dict()
        self.enums['brands'][brandID] = label


#    def plotTraffic(self,label):
#        traffic = self.graph.IdArray *0
#        if label in self.market.mobilityLables.itervalues():
#            brandID = self.market.mobilityLables.keys()[self.market.mobilityLables.values().index(label)]
#
#            for cell in self.iterEntRandom(_cell):
#                traffic[cell.x,cell.y] += cell.traffic[brandID]
#
#        plt.clf()
#        plt.imshow(traffic, cmap='jet',interpolation=None)
#        plt.colorbar()
#        plt.tight_layout()
#        plt.title('traffic of ' + label)
#        plt.savefig('output/traffic' + label + str(self.time).zfill(3) + '.png')
#
#    def generateHH(self):
#        hhSize  = int(np.ceil(np.abs(np.random.randn(1)*2)))
#        while 1:
#            ageList = np.random.randint(1,60,hhSize)
#            if np.sum(ageList>17) > 0: #ensure one adult
#                break
#        ageList = ageList.tolist()
#        sexList = np.random.randint(1,3,hhSize).tolist()
#        income  = int(np.random.randint(5000))
#        nKids   = np.sum(ageList<19)
#        #print ageList
#        idxAdult = [ n for n,i in enumerate(ageList) if i>17 ]
#        idx = np.random.choice(idxAdult)
#        prSaf, prEco, prCon, prMon, prImi = self.og.getPref(ageList[idx],sexList[idx],nKids,income, self.radicality)
#
#        return hhSize, ageList, sexList, income,  nKids, prSaf, prEco, prCon, prMon, prImi


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

    def step(self):
        """
        Method to proceed the next time step
        """
        tt = time.time()
        self.time += 1
        self.timeStep = self.time
#        self.minPrice = 1

        ttComp = time.time()
        # time management
        if self.timeStep == 0:
            lg.info( 'setting up time warp during burnin by factor of ' + str(self.para['burnInTimeFactor']))
            self.para['mobNewPeriod'] = int(self.para['mobNewPeriod'] / self.para['burnInTimeFactor'])
            newValue = np.rint(self.getNodeValues('lastAction',_pers) / self.para['burnInTimeFactor']).astype(int)
            self.setNodeValues('lastAction', newValue, _pers)

        elif self.timeStep+5 == self.para['burnIn']:
            lg.info( 'reducting time speed to normal')
            self.para['mobNewPeriod'] = int(self.para['mobNewPeriod'] * self.para['burnInTimeFactor'])
            oldValue = self.getNodeValues('lastAction',_pers) * self.para['burnInTimeFactor']
            newValue = oldValue.astype(int)
            stochastricRoundValue = newValue + (np.random.random(len(oldValue)) < oldValue-newValue).astype(int)

            self.setNodeValues('lastAction', stochastricRoundValue, _pers)
            
        else:
            lastActions = self.getNodeValues('lastAction',_pers)
            self.setNodeValues('lastAction',lastActions+1, _pers)
            

        # progressing time
        if self.timeUnit == 1: #months
            self.date[0] += 1
            if self.date[0] == 13:
                self.date[0] = 1
                self.date[1] += 1
        elif self.timeUnit == 1: # years
            self.date[1] +=1

        for re in self.para['regionIDList']:
            self.globalRecord['stock_' + str(re)].set(self.time,0)

        for cell in self.iterEntRandom(_cell):
            self.globalRecord['stock_' + str(int(cell.getValue('regionId')))].add(self.time,np.asarray(cell.getValue(('carsInCell'))))

        # move values to global data class
        for re in self.para['regionIDList']:
            self.globalRecord['stock_' + str(re)].updateValues(self.time)

        self.market.updateSales()


        self.computeTime[self.time] = time.time()-ttComp

        ttSync = time.time()
        self.graph.glob.updateStatValues('meanEmm', np.asarray(self.graph.vs[self.nodeDict[_pers]]['prop'])[:,1])
        self.graph.glob.updateStatValues('stdEmm', np.asarray(self.graph.vs[self.nodeDict[_pers]]['prop'])[:,1])
        self.graph.glob.updateStatValues('meanPrc', np.asarray(self.graph.vs[self.nodeDict[_pers]]['prop'])[:,0])
        self.graph.glob.updateStatValues('stdPrc', np.asarray(self.graph.vs[self.nodeDict[_pers]]['prop'])[:,0])
        self.graph.glob.sync()
        self.syncTime[self.time] = time.time()-ttSync
        lg.debug('globals synced in ' +str(time.time()- ttSync) + ' seconds')

        ttComp = time.time()
        #gather data back to the records

        for re in self.para['regionIDList']:
            self.globalRecord['stock_' + str(re)].gatherSyncDataToRec(self.time)

        # proceed market in time
        self.market.step(self) # Statistics are computed here

        ttCell = time.time()
        #loop over cells
        for cell in self.iterEntRandom(_cell):
            cell.step(self.para, self.market.getCurrentMaturity())
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
                    household.evolutionaryStep(self)
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

    def __init__(self, label, progressType, initialProgress, slope, propDict, experience):

        self.label = label
        self.currGrowthRate     = 1
        self.oldStock           = 0
        self.currStock          = 0
        self.replacementRate    = 0.01
        
        if progressType == 'wright':

            self.experience = experience # cummulative production up to now
            self.technicalProgress = initialProgress
            self.initialProperties = propDict.copy()
            self.properties   = propDict

            # setting initial price to match the initial technical progress
            for propKey in self.initialProperties.keys():
                self.initialProperties[propKey] *= initialProgress
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
            production = self.salesModel()

        self.currGrowthRate = 1 + (production) / float(self.experience)
        self.technicalProgress = self.technicalProgress * (self.currGrowthRate)**self.slope
        for prop in self.properties.keys():
            self.properties[prop] = self.initialProperties[prop] / self.technicalProgress

        self.maturity       =    1 - (1 / self.technicalProgress)

        #update experience
        self.experience += production

        

    def step(self, doTechProgress=True):
        """ replaces the old stock by the current Stock and computes the 
        technical progress
        """
        if doTechProgress:
            self.updateTechnicalProgress()
        
        self.oldStock = self.currStock
        
        return self.properties, self.technicalProgress, self.maturity
        

    def buy(self,quantity=1):
        self.currStock +=1
        return self.properties.values()

    def sell(self, quantity=1):
        self.currStock -=1
    
    
    def salesModel(self):
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
        for iGood in self.goods.keys():
            self.glob['sales'][int(iGood)] = self.goods[iGood].salesModel()
            
            
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
        self.std['emissions']  = self.glob['stdPrc']


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

        if self.std['emission'] == 0:
            ecology = 1 / (1+math.exp((emissions-self.mean['emission'])/1))
        else:
            ecology = 1 / (1+math.exp((emissions-self.mean['emission'])/self.std['emission']))

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
            self.globalRecord['allTimeProduced'].set(self.time, self.getCurrentExperience())
            self.globalRecord['kappas'].set(self.time, self.getCurrentMaturity())

        lg.debug('new value of allTimeProduced: ' + str(self.getCurrentExperience()))
        # reset sales
        self.glob['sales'] = self.glob['sales']*0

        self.minPrice = np.min([good.getProperties()[0] for good in self.goods.itervalues()])

        #compute new statistics
        self.computeStatistics()

        self.time +=1


    def initBrand(self, label, propertyDict, initTimeStep, slope, initialProgress, allTimeProduced):
        mobType = self.__nMobTypes__
        self.__nMobTypes__ +=1
        self.glob['sales'] = np.asarray([0]*self.__nMobTypes__)
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


    def buyCar(self, mobTypeIdx):
        # get current properties form good class
        propDict = self.goods[mobTypeIdx].buy() * (1 + np.random.randn(self.nProp)*self.propRelDev)

        return propDict
    
    
    def sellCar(self, mobTypeIdx):
        self.goods[mobTypeIdx].sell()
        


# %% --- entity classes ---


class Person(Agent):
    __slots__ = ['gID', 'nID']
    def __init__(self, world, **kwProperties):
        Agent.__init__(self, world, **kwProperties)


    def isAware(self,mobNewPeriod):
        # method that returns if the Persion is aktively searching for information
        return (self._node['lastAction'] - mobNewPeriod/10.) / (mobNewPeriod)  > np.random.rand()


    def register(self, world, parentEntity=None, edgeType=None):

        Agent.register(self, world, parentEntity, edgeType)
        self.loc = parentEntity.loc
        self.loc.peList.append(self.nID)
        self.hh = parentEntity


    def weightFriendExperience(self, world):
        friendUtil = np.asarray(self.getPeerValues('commUtil',_cpp)[0])[:,self._node['mobType']]
        ownUtil  = self.getValue('util')
        edges = self.getEdges(_cpp)
        diff = friendUtil - ownUtil +  np.random.randn(len(friendUtil))*world.para['utilObsError']
        prop = np.exp(-(diff**2) / (2* world.para['utilObsError']**2))
        prop = prop / np.sum(prop)

        prior = np.asarray(edges['weig'])
        prior = prior / np.sum(prior)
        assert not any(np.isnan(prior))


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
                #assert self.edges[_cpp][self.ownEdgeIdx[0]].target == self.nID
                #assert self.edges[_cpp][self.ownEdgeIdx[0]].source == self.nID
                if np.sum(self.getEdgeValues('weig', edgeType=_cpp)[0]) < 0.99:
                    import pdb
                    pdb.set_trace()
            return post, self._node['ESSR']

        else:
            lg.debug( 'post values:')
            lg.debug([ value for value in post])
            lg.debug('diff values:')
            lg.debug([value for value in diff])
            lg.debug('friendUtil values:')
            lg.debug([value for value in friendUtil])

            return prior, self._node['ESSR']


    def socialize(self, world):

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
            self.resetEdgeCache(edgeType=_cpp)
            self.resetPeerCache(edgeType=_cpp)

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
            lg.info( "population = " + str(self.loc.getValue('population')) + " surrounding population: " +str(np.sum(self.loc.getPeerValues('population',_cll)[0])))

            nContacts = min(np.sum(weightData[:,0]>0)-1,nContacts)

        if nContacts < 1:
            lg.info('ID: ' + str(self.nID) + ' failed to generate friend')

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



    def computeExpUtil(self,earth):
        #get weights from friends
        weights, edges = self.getEdgeValues('weig', edgeType=_cpp)
        weights = np.asarray(weights)


        # compute weighted mean of all friends
        if earth.para['weightConnections']:
            communityUtil = np.dot(weights,np.asarray(self.getPeerValues('commUtil',_cpp)[0]))
        else:
            communityUtil = np.mean(np.asarray(self.getPeerValues('commUtil',_cpp)[0]),axis=0)

        selfUtil = self._node['selfUtil'][:]
        mobType  = self._node['mobType']

        # weighting by selftrust
        selfUtil[mobType] *= earth.para['selfTrust']

        if len(selfUtil) != earth.para['nMobTypes'] or len(communityUtil.shape) == 0 or communityUtil.shape[0] != earth.para['nMobTypes']:
            print 'nID: ' + str(self.nID)
            print 'error: '
            print 'communityUtil: ' + str(communityUtil)
            print 'selfUtil: ' + str(selfUtil)
            print 'nEdges: ' + str(len(edges))

            return
        tmp = np.nanmean(np.asarray([communityUtil,selfUtil]),axis=0)
        tmp[mobType] /= (earth.para['selfTrust']+1)/2.
        
        self._node['commUtil'] = tmp.tolist()

        # adjust mean since double of weigth - very bad code - sorry
        

    def imitate(self):
        #pdb.set_trace()
        if np.random.rand() > .99:
            return np.random.choice(len(self.getValue('commUtil')))
        else:
            peerUtil     = np.asarray(self.getPeerValues('util',_cpp)[0])
            peerMobType  = np.asarray(self.getPeerValues('mobType',_cpp)[0])
            weights      = np.asarray(self.getEdgeValues('weig', edgeType=_cpp)[0])
            
            fitness = peerUtil * weights
            fitness = fitness / np.sum(fitness)
        
            return np.random.choice(peerMobType,p=fitness)
        

    def step(self, earth):


        # weight friends
        if earth.para['weightConnections'] and self.isAware(earth.para['mobNewPeriod']):
            weights, ESSR = self.weightFriendExperience(earth)

            # compute similarity
            weights = np.asarray(self.getEdgeValues('weig', edgeType=_cpp)[0])
            preferences = np.asarray(self.getPeerValues('preferences',_cpp)[0])

            average = np.average(preferences, axis= 0, weights=weights)
            self._node['peerBubbleHeterogeneity'] = np.sum(np.sqrt(np.average((preferences-average)**2, axis=0, weights=weights)))

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
        factor = 100
        for i in range(len(x)):
            utility *= (factor*x[i])**alpha[i]
        if np.isnan(utility) or np.isinf(utility):
            import pdb
            pdb.set_trace()

        # assert the limit of utility
        assert utility > 0 and utility <= factor

        return utility

    @staticmethod
    def CESUtil(x, alpha):
        uti = 0.
        s = 2.    # elasticity of substitution, has to be float!
        factor = 100
        for i in range(len(x)):
            uti += (alpha[i]*(factor * x[i])**(s-1))**(1/s)
            #print uti
        utility = uti**(s/(s-1))
        if  np.isnan(utility) or np.isinf(utility):
            import pdb
            pdb.set_trace()

        # assert the limit of utility
        assert utility > 0 and utility <= factor

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

            utility = self.utilFunc(adult.getValue('consequences'), adult.getValue('preferences'))
            assert not( np.isnan(utility) or np.isinf(utility)), utility

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
                adult.computeExpUtil(earth)
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

        if len(combActions) != len(overallUtil):
            import pdb
            pdb.set_trace()

        return combActions, overallUtil


    def takeAction(self, earth, persons, actionIds):
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


    def calculateConsequences(self, market):

        carInHh = False
        # at least one car in househould?
        if any([adult.getValue('mobType') !=2 for adult in self.adults]):
            carInHh = True

        # calculate money consequence
        money = max(1e-5, 1 - self.getValue('expenses') / self.getValue('income'))


        for adult in self.adults:
            hhCarBonus = 0.
            #get action of the person

            actionIdx = adult.getValue('mobType')
            mobProps = adult.getValue('prop')

            if (actionIdx != 2):
                decay = 1- (1/(1+math.exp(-0.1*(adult.getValue('lastAction')-market.para['mobNewPeriod']))))
            else:
                decay = 1.
            if (actionIdx == 2) and carInHh:
                hhCarBonus = 0.2

            convenience = decay * self.loc.getValue('convenience')[actionIdx] + hhCarBonus

            # calculate ecology:
            emissions = mobProps[1]
            ecology = market.ecology(emissions)

            experience = market.getCurrentExperience()

            innovation = 1 - ( (experience[adult.getValue('mobType')] / np.sum(experience))**.5 )

            # assuring that consequences are within 0 and 1
            for consequence in [convenience, ecology, money, innovation]:
                assert consequence <= 1 and consequence > 0

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
                self.takeAction(earth, actors, actions)     # remove not-actions (i.e. -1 in list)
                self.calculateConsequences(market)
                util = self.evalUtility(earth)

                if util < 1:
                    lg.debug('####' + str(self.nID) + '#####')
                    lg.debug('New Util: ' +str(util) + ' old util: ' + str(oldUtil) + ' exp. Util: ' + str(utilities[bestUtilIdx]))
                    lg.debug('possible utilitties: ' + str(utilities))
                    lg.debug(self._node)
                    lg.debug('properties: ' + str([adult.getValue('prop') for adult in self.adults]))
                    lg.debug('consequence: '+ str([adult.getValue('consequences') for adult in self.adults]))
                    lg.debug('preferences: '+ str([adult.getValue('preferences') for adult in self.adults]))
                    lg.debug('utility: ' +    str([adult.getValue('util')  for adult in self.adults]))
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


    def evolutionaryStep(self, earth):
        """
        Evolutionary time step that uses components of genetic algorithms
        on social networks to mimic the evolutions of norms / social learning.
        The friends of an agents are replacing the population in a genetic
        algorithm from which no genes evolve by mutation, crossover and random.
        """
        tt = time.time()
        
        actionIds = [adult.imitate() for adult in self.adults]
        
        
        self.undoActions(earth, self.adults)
        self.takeAction(earth, self.adults, actionIds)
            
        self.calculateConsequences(earth.market)
        self.evalUtility(earth, actionTaken=True)

        for adult in self.adults:
            adult.computeExpUtil(earth)
            
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
                self.takeAction(earth, personsToTakeAction, actions)

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
            self.evalExpectedUtility(earth, [True] * len(self.adults))

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
        self.currID = 0
        self.traffic = dict()
        self.sigmaEps = 1.
        self.muEps = 1.
        self.cellSize = 1.
        self.convFunctions = list()


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
        self.connnodeDict = [self._graph.es[x].target for x in self.eIDs ]
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
        population = world.getParameter('population')[self._node['pos']]
        self._node['population'] = population

        convAll = self.calculateConveniences(world.getParameter(), world.market.getCurrentMaturity())

        for x in convAll:
            if np.isinf(x) or np.isnan(x) or x == 0:
                import pdb
                pdb.set_trace()
        self._node['population'] = 0
        return convAll, population

    def calculateConveniences(self, parameters, currentMaturity):

        convAll = list()

        popDensity = np.float(self.getValue('population'))/self.cellSize
        for i, funcCall in enumerate(self.convFunctions):
            convAll.append(funcCall(popDensity, parameters, currentMaturity[i], self))

        return convAll



    def step(self, parameters, currentMaturity):
        """
        Manages the deletion of observation after a while
        """

        convAll = self.calculateConveniences(parameters,currentMaturity)
        self.setValue('convenience', convAll)


    def registerObs(self, hhID, prop, util, label):
        """
        Adds a car to the cell pool of observations
        """
        meme = prop + [util, label, hhID]
        assert not any(np.isnan(meme))
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
        nodeType = graph.class2NodeType[Household]
        hhIDList = self.getPeerIDs(nodeType)
        self.hhList = graph.vs[hhIDList]

    def updatePeList(self, graph):
        nodeType = graph.class2NodeType[Person]
        hhIDList = self.getPeerIDs(nodeType)
        self.hhList = graph.vs[hhIDList]

class Opinion():
    """
    Creates preferences for households, given their properties
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

        assert all (pref > 0) and all (pref < 1)

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
