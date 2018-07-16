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

from lib import core

from lib import World, Agent, GhostAgent, Location, GhostLocation
from lib.traits import Parallel

#import pdb
#import igraph as ig
import numpy as np
import time
import os
import math
import copy
import random
import logging as lg

#%% --- ENUMERATIONS / CONSTANTS---
#connections
CON_LL = 1 # loc - loc
CON_LH = 2 # loc - household
CON_HH = 3 # household, household
CON_HP = 4 # household, person
CON_PP = 5 # household, person

#properties
EMISSIONS      = 0
FIXEDCOSTS     = 1
OPERATINGCOSTS = 2

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

MEAN_KM_PER_TRIP = [.25, 3., 7.5, 30., 75. ]

#%% --- Global classes ---
from numba import njit

@njit("f8 (f8[:], f8[:])",cache=True)
def cobbDouglasUtilNumba(x, alpha):
    utility = 1.
    
    for i in range(len(x)):
        utility = utility * (100.*x[i])**alpha[i] 
    #if np.isnan(utility) or np.isinf(utility):  ##DEEP_DEBUG
    #    import pdb                              ##DEEP_DEBUG
    #    pdb.set_trace()                         ##DEEP_DEBUG

    # assert the limit of utility
    #assert utility > 0 and utility <=  100.     ##DEEP_DEBUG
    
    
    return utility / 100.

@njit("f8[:] (f8[:])",cache=True)
def normalize(array):
    return  array / np.sum(array)

@njit(cache=True)
def sum1D(array):
    return np.sum(array)

@njit(cache=True)
def sumSquared1D(array):
    return np.sum(array**2)

@njit(cache=True)
def prod1D(array1, array2):
    return np.multiply(array1,array2)

#@njit(cache=True)
def normalizedGaussian(array, center, errStd):
    diff = (array - center) +  np.random.randn(array.shape[0])*errStd
    normDiff = np.exp(-(diff**2.) / (2.* errStd**2.))  
    return normDiff / np.sum(normDiff)


class Earth(World):

    def __init__(self,
                 simNo,
                 outPath,
                 parameters,
                 maxNodes,
                 maxLinks,
                 debug,
                 mpiComm=None,
                 agentOutput=True):

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
                       maxLinks=maxLinks,
                       debug=debug,
                       mpiComm=mpiComm,
                       agentOutput=agentOutput)

        self.agentRec   = dict()
        self.time       = 0
        self.date       = list(parameters.startDate)
        self.timeUnit   = parameters.timeUnit

        self.reporter   = list()
        self.brandDict  = dict()
        self.brands     = list()

        


        # transfer all parameters to earth
        parameters.simNo = simNo
        self.setParameters(parameters)

        if self.para['omniscientBurnIn']>self.para['burnIn']:
            self.para['omniscientBurnIn']=self.para['burnIn']

    def nPriorities(self):
        return len(self.getEnums()['priorities'])


    # init car market
    def initMarket(self, earth, properties, propRelDev=0.01, time = 0, burnIn = 0):
        """
        Init of the market instance
        """
        self.market = Market(earth, properties, propRelDev=propRelDev, time=time, burnIn=burnIn)

    def initChargInfrastructure(self):
        self.chargingInfra = Infrastructure(self, self.para['roadKmPerCell'], 2.0, 4.0, 5.0)
    
    def registerGood(self, label, propDict, convFunction, initTimeStep, **kwProperties):
        """
        Method to register a new good (i.e. mobility type) in the Earth and therefore in the market.
        It currently adds the related convenience function in the cells.
        """
        goodID = self.market.initGood(initTimeStep, label, propDict, **kwProperties)

        # TODO: move convenience Function from a instance variable to a class variable            
        for cell in self.random.shuffleAgentsOfType(CELL):
            cell.convFunctions.append(convFunction)

        if 'brands' not in list(self.getEnums().keys()):
            self.getEnums()['brands'] = {}
        self.getEnums()['brands'][goodID] = label

        # adding a record about the properties of each goood
        self.registerRecord('prop_' + label,
                            'properties of ' + label,
                            list(self.getEnums()['properties'].values()), 
                            style='plot')


    def generateSocialNetwork(self, agTypeID, liTypeID):
        """
        Function for the generation of a simple network that regards the
        distance between the agents and their similarity in their preferences
        """
        tt = time.time()
        sourceList = list()
        targetList = list()
        weigList  = list()
        populationList = list()
        for agent in self.random.shuffleAgentsOfType(agTypeID):

            nContacts = random.randint(self.para['minFriends'],self.para['maxFriends'])
            ttx = time.time()
            frList, links, weights = agent.generateContactNetwork(self, nContacts)
            
            sourceList += [agent.nID] * len(frList)
            targetList += frList
            weigList += weights
            populationList.append(agent.loc.attr['population'])
        
        timePerAgent =  (time.time() - tt ) / self.nAgents(agTypeID)
        print('Average generation time per agent: {:3.4f}'.format(timePerAgent))
        
        ttx = time.time()
        self.addLinks(eTypeID=liTypeID, sources=sourceList, targets=targetList, weig=weigList)
        lg.info( 'Connections added in -- ' + str( time.time() - ttx) + ' s')

        lg.info( 'Social network created in -- ' + str( time.time() - tt) + ' s')
        lg.info( 'Average population: ' + str(np.mean(populationList)) + ' - Ecount: ' + str(self.graph.eCount()))

        fid = open(self.para['outPath']+ '/initTimes.out', 'a')
        fid.writelines('r' + str(core.mpiRank) + ', ' +
                       str( time.time() - tt) + ',' +
                       str(np.mean(populationList)) + ',' +
                       str(self.graph.eCount()) + '\n')

    def updateRecords(self):
        """
        Encapsulating method for the update of records
        """
        #emissions = np.zeros(len(self.enums['mobilityTypes'])+1)
        for re in self.para['regionIDList']:
            self.globalRecord['stock_' + str(re)].set(self.time,0)
            self.globalRecord['elDemand_' + str(re)].set(self.time,0.)
            self.globalRecord['emissions_' + str(re)].set(self.time,0.)
            self.globalRecord['nChargStations_' + str(re)].set(self.time,0)
            
            
        for cell in self.getAgents.byType(CELL):
            regionID = str(int(cell.get('regionId')))
            self.globalRecord['stock_' + regionID].add(self.time,np.asarray(cell['carsInCell']) * self.para['reductionFactor'])
            self.globalRecord['elDemand_' + regionID].add(self.time,np.asarray(cell['electricConsumption']))
            self.globalRecord['emissions_' + regionID].add(self.time,np.asarray(cell['emissions']))
            self.globalRecord['nChargStations_' + regionID].add(self.time,np.asarray(cell['chargStat']))
                
        # move values to global data class
        for re in self.para['regionIDList']:
            self.globalRecord['stock_' + str(re)].updateLocalValues(self.time)
            self.globalRecord['elDemand_' + str(re)].updateLocalValues(self.time)
            self.globalRecord['emissions_' + str(re)].updateLocalValues(self.time)
            self.globalRecord['nChargStations_' + str(re)].updateLocalValues(self.time)
            
            
    def syncGlobals(self):
        """
        Encapsulating method for the sync of global variables
        """
        ttSync = time.time()
        self.graph.glob.updateLocalValues('meanEmm', self.getAttrOfAgentType('prop', agTypeID=PERS)[:,EMISSIONS])
        self.graph.glob.updateLocalValues('stdEmm', self.getAttrOfAgentType('prop', agTypeID=PERS)[:,EMISSIONS])
        self.graph.glob.updateLocalValues('meanFiC', self.getAttrOfAgentType('prop', agTypeID=PERS)[:,FIXEDCOSTS])
        self.graph.glob.updateLocalValues('stdFiC', self.getAttrOfAgentType('prop', agTypeID=PERS)[:,FIXEDCOSTS])
        self.graph.glob.updateLocalValues('meanOpC', self.getAttrOfAgentType('prop', agTypeID=PERS)[:,OPERATINGCOSTS])
        self.graph.glob.updateLocalValues('stdOpC', self.getAttrOfAgentType('prop', agTypeID=PERS)[:,OPERATINGCOSTS])
     
        # local values are used to update the new global values
        self.graph.glob.sync()
        
        #gather data back to the records
        globalStock = np.zeros(self.para['nMobTypes'])
        for re in self.para['regionIDList']:
            reStock = self.globalRecord['stock_' + str(re)].gatherGlobalDataToRec(self.time)
            globalStock += reStock
            self.globalRecord['elDemand_' + str(re)].gatherGlobalDataToRec(self.time)
            self.globalRecord['emissions_' + str(re)].gatherGlobalDataToRec(self.time)
            self.globalRecord['nChargStations_' + str(re)].gatherGlobalDataToRec(self.time)
            

        tmp = [self.graph.glob.globalValue['meanEmm'], 
               self.graph.glob.globalValue['stdEmm'], 
               self.graph.glob.globalValue['meanFiC'], 
               self.graph.glob.globalValue['stdFiC'],
               self.graph.glob.globalValue['meanOpC'], 
               self.graph.glob.globalValue['stdOpC']]

        self.globalRecord['globEmmAndPrice'].set(self.time,np.asarray(tmp))


        self.syncTime[self.time] = time.time()-ttSync        
        lg.debug('globals synced in ' +str(time.time()- ttSync) + ' seconds') ##OPTPRODUCTION

    def syncGhosts(self):
        """
        Encapsulating method for the syncronization of selected ghost agents
        """
        ttSync = time.time()
        if core.comm is not None:
            self.papi.updateGhostAgents([PERS],['commUtil','util', 'mobType'])
            self.papi.updateGhostAgents([CELL],['chargStat', 'carsInCell'])

        self.syncTime[self.time] += time.time()-ttSync
        lg.debug('Ghosts synced in ' + str(time.time()- ttSync) + ' seconds')##OPTPRODUCTION
        
        
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
            newValue = np.rint(self.getAttrOfAgentType('lastAction', agTypeID=PERS) / self.para['burnInTimeFactor']).astype(int)
            self.setAttrOfAgentType('lastAction', newValue, agTypeID=PERS)

        elif self.timeStep+5 == self.para['burnIn']:
            lg.info( 'reducting time speed to normal')
            self.para['mobNewPeriod'] = int(self.para['mobNewPeriod'] * self.para['burnInTimeFactor'])
            oldValue = self.getAttrOfAgentType('lastAction', agTypeID=PERS) * self.para['burnInTimeFactor']
            newValue = oldValue.astype(int)
            stochastricRoundValue = newValue + (np.random.random(len(oldValue)) < oldValue-newValue).astype(int)

            self.setAttrOfAgentType('lastAction', stochastricRoundValue,  agTypeID=PERS)
            
        else:
            lastActions = self.getAttrOfAgentType('lastAction',agTypeID=PERS)
            self.setAttrOfAgentType('lastAction',lastActions+1, agTypeID=PERS)
        
        # progressing time
        if self.timeStep > self.para['burnIn']:
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
        if self.isParallel:
            self.syncGhosts()
        
        ttComp = time.time()

        # proceed infrastructure in time
        if self.time > self.para['burnIn']:
            self.chargingInfra.step(self)

        # proceed market in time
        # stf: sollte das nicht eher am Ende des steps aufgerufen
        # werden? z.b. passiert bei updateTechnologicalProgress nichts
        # da die sales=0 sind. Hab mir aber nicht alle Details
        # angeschaut, welche im Market step passieren
        self.market.step(self) # Statistics are computed here

        
        ###### Cell loop ######
        ttCell = time.time()
        for cell in self.random.shuffleAgentsOfType(CELL):
            cell.step(self.para, self.market.getCurrentMaturity())
        lg.debug('Cell step required ' + str(time.time()- ttCell) + ' seconds')##OPTPRODUCTION



        ###### Person loop ######

        for person in self.random.shuffleAgentsOfType(PERS):
            person.step(self)
        lg.debug('Person step required ' + str(time.time()- ttComp) + ' seconds')##OPTPRODUCTION
   


        ###### Household loop ######
        tthh = time.time()
        if self.para['omniscientAgents'] or (self.time < self.para['omniscientBurnIn']):
            for household in self.random.shuffleAgentsOfType(HH):
                household.stepOmniscient(self)
        else:
            for household in self.random.shuffleAgentsOfType(HH):

#                if random.random()<1e-4:
#                    household.stepOmniscient(self)
#                else:
                household.evolutionaryStep(self)
        lg.debug('Household step required ' + str(time.time()- tthh) + ' seconds')##OPTPRODUCTION

        for cell in self.getAgents.byType(CELL):
            cell.aggregateEmission(self)

               

        self.market.updateSales()
        
        self.computeTime[self.time] += time.time()-ttComp

        # waiting for other processes
        ttWait = time.time()
        core.mpiBarrier()
        self.waitTime[self.time] += time.time()-ttWait

        # I/O
        ioStep = self.getParameters()["ioSteps"]
        if self.getParameters()['writeAgentFile']:
            if ioStep !=0 and (self.timeStep%ioStep == 0 or self.timeStep == self.getParameters()['nSteps']):
                ttIO = time.time()
                self.io.writeDataToFile(self.time, [CELL, HH, PERS])
                self.ioTime[self.time] = time.time()-ttIO
            else: 
                self.ioTime[self.time] = 0

        lg.info(('Times: tComp: '+ '{:10.5f}'.format(self.computeTime[self.time])+
              ' - tSync: '+ '{:10.5f}'.format(self.syncTime[self.time])+
              ' - tWait: '+ '{:10.5f}'.format(self.waitTime[self.time])+
              ' - tIO: '+ '{:10.5f}'.format(self.ioTime[self.time]) ))

        if self.getParameters()['omniscientAgents']:
            lg.info( 'Omincent step ' + str(self.time) + ' done in ' + "{:2.4f}".format((time.time()-tt)) + ' s')
        else:
            lg.info( 'Step ' + str(self.time) + ' done in ' + "{:2.4f}".format((time.time()-tt)) + ' s')

        if self.isRoot:
            print('Step ' + str(self.time) + ' done in ' + "{:2.4f}".format((time.time()-tt)) + ' s --', end='')

    def year2step(self, year):
        return 12 * (year - self.para['startDate'][1]) + self.para['burnIn']


# ToDo
class Good():
    """
    Definition of an economic good in a sence. The class is used to compute
    technical progress in production and related unit costs.

    Technical progress is models according to the publication of nagy+al2010
    https://doi.org/10.1371/journal.pone.0052669
    """
    
    #class variables (common for all instances, - only updated by class mehtods)

    # stf: ich weiss nicht genau, wie das bei Python aussieht, aber Du
    # missbrauchst hier eine Liste als ein Array mit einer fixen
    # Länge. Das hat aber zur Laufzeit normalerweise echte Nachteile,
    # da Listenelemente nicht am Stück im Speicher stehen (sprich mehr
    # Cache-misses), und vor allem nur das vorrige und nächste Element
    # kennen, sprich, wenn Du auf goodid Nummer 4 zugreifen willst,
    # muss erst von drei Elemente nextElement() aufgerufen werden.
    lastGlobalSales  = list()
    overallExperience = 0.
    globalStock      = list()

    @classmethod
    def updateGlobalSales(cls, sales):
        """
        Method to update the class variable lastGlobalSales
        """
        cls.lastGlobalSales = sales

    @classmethod
    def updateGlobalStock(cls, newStock, mpiRank=1):
        """
        Method to update the class variable lastGlobalSales
        """
        if mpiRank == 0:
            print((' Current global stock: ' + str(newStock)))
        cls.globalStock = newStock

    @classmethod
    def addToOverallExperience(cls, exp):
        """
        Method to update the class variable overallExperience
        """
        cls.overallExperience += exp
        
    def __init__(self, label, propDict, initExperience, **parameters):
        # stf: das finde ich einen recht schrägen Hack um die id zu setzen, siehe auch
        # mein Kommentar zur Liste selber
        self.goodID           = len(self.lastGlobalSales)
        self.label            = label
        self.replacementRate  = 0.01
        self.properties       = propDict
        self.paras            = parameters
        self.currGrowthRate  = 1.
        self.currStock       = 0
        self.experience      = 1.
        self.initExperience  = initExperience
        self.addToOverallExperience(initExperience)
        self.progress        = 1.
        self.maturity        = 0.0001         
        
        self.oldStock        = 0
        self.currLocalSales  = 0
        
        self.updateGlobalSales(self.lastGlobalSales + [0.])
        self.updateGlobalStock(self.globalStock + [1.])
        self.initEmissionFunction()
             

    def initMaturity(self):
        if (self.label == 'shared' or self.label == 'none'):
            self.maturity = self.paras['initMaturity']
            self.progress = 1./(1.- self.paras['initMaturity'])
                  
        
    def initEmissionFunction(self):

        if self.label == 'brown':
            def emissionFn(self, market):    
                correctionFactor = .3  # for maturity
                weight = self.paras['weight']
                
                yearIdx = max(0, market.time-market.burnIn) #int((market.time - market.burnIn)/12.)
                if market.germany:
                    exp = market.experienceBrownExo[yearIdx] + self.experience
                else:
                    exp = market.experienceBrownExo[yearIdx]
                expIn10Mio = exp/10000000.
                emissionsPerKg = self.paras['emFactor'] * expIn10Mio**(self.paras['emRed']) + self.paras['emLimit']
                maturity = self.paras['emLimit'] / emissionsPerKg + correctionFactor
                emissions = emissionsPerKg * weight
                return  emissions, maturity
                
        elif self.label == 'green':
            def emissionFn(self, market):                  
                weight = self.paras['weight']
                electrProdFactor = 1.                   # CO2 per KWh compared to 2007, 2Do?
                yearIdx = max(0, market.time-market.burnIn) #int((market.time - market.burnIn)/12.)                
                if market.germany:
                    exp = market.experienceGreenExo[yearIdx] + self.experience
                else:
                    exp = market.experienceGreenExo[yearIdx]                
                emissionsPerKg = self.paras['emFactor'] * exp**(self.paras['emRed']) + self.paras['emLimit']
                maturity = self.paras['emLimit']/emissionsPerKg
                emissions = emissionsPerKg * weight * electrProdFactor
                return emissions, maturity
                    
        elif self.label == 'public':
            def emissionFn(self, market):
                emissions2012 = market.para['initEmPublic'] - 8. # corrected to match value of 2012
                pt2030  = self.paras['pt2030']  
                ptLimit = self.paras['ptLimit']
                rate = math.log((1-ptLimit)/(pt2030-ptLimit))/18.
                year = max(0., (market.time-market.burnIn)/12.)
                factor = (1-ptLimit)*emissions2012 * math.exp(7*rate)
                maturity = ptLimit / ((1-ptLimit)* math.exp(rate*(7-year)) + ptLimit)
                emissions = factor*math.exp(-rate*year) + ptLimit*emissions2012
                return emissions, maturity
                                 
        elif self.label == 'shared':
            def emissionFn(self, market):
                stockElShare = market.goods[GREEN].getGlobalStock() / \
                               (market.goods[BROWN].getGlobalStock()+market.goods[GREEN].getGlobalStock())
                electricShare = 0.1 + 0.9*stockElShare                    
                weight = self.paras['weight']
                emissionsPerKg = (1-electricShare)* market.goods[BROWN].properties['emissions']/market.para['weightB'] + \
                                    electricShare * market.goods[GREEN].properties['emissions']/market.para['weightG']
                emissions = emissionsPerKg * weight
                
                #maturity = float(self.currStock) / max(0.1,sum(market.goods[i].currStock for i in range(market.__nMobTypes__)))   # maturity is market share of car sharing
                maturity = (1-1/self.progress)
                return emissions, maturity
                
        else:
            def emissionFn(self, market):
                emissions = self.properties['emissions']
                
                #maturity = float(self.currStock) / max(0.1,sum(market.goods[i].currStock for i in range(market.__nMobTypes__)))   # maturity is market share of none
                maturity = (1-1/self.progress)
                return emissions, maturity
        
        self.emissionFunction = emissionFn


                
    def updateEmissionsAndMaturity(self, market):

        emissions, maturity = self.emissionFunction(self, market)        

        self.properties['emissions'] = emissions
        self.maturity = maturity

#    def updateMaturity(self):
#        self.maturity = self.getExperience() / self.overallExperience
        
    def updateTechnicalProgress(self):
        """
        Updates the growth rate and progress
        """
        # only endogeneous experience, there may also be exogenous experience, from market
        self.experience += self.lastGlobalSales[self.goodID]  
        self.addToOverallExperience(self.lastGlobalSales[self.goodID])  
        
        # use last step global sales for update of the growth rate and technical progress
        if self.globalStock[self.goodID] > 0:
            self.currGrowthRate = 1 + (self.lastGlobalSales[self.goodID]) / float(self.globalStock[self.goodID])
        else:
            self.currGrowthRate = 0
        self.progress *= 1 + (self.lastGlobalSales[self.goodID]) / float(self.experience + self.initExperience)
        
        
        
    def updateSales(self, production=None):
        """
#        Computes the sales
#        """ 
        
        # if production is not given, the internal sales is used
        if production is None:
            self.currLocalSales = self.salesModel()
        else:
            self.currLocalSales = production
            
        
#        self.technicalProgress = self.technicalProgress * (self.currGrowthRate)**self.slope
#        for prop in self.properties.keys():
#            self.properties[prop] = (self.initialProperties[prop][0] / self.technicalProgress) + self.initialProperties[prop][1]
#
#        self.maturity       =    1 - (1 / self.technicalProgress)
#
#        #update experience
#        self.experience += self.lastGlobalSales[self.goodID]

        
    def step(self, market, doTechProgress=True):
        """ replaces the old stock by the current Stock and computes the 
        technical progress
        """
        self.updateSales()
        
        if doTechProgress:
            self.updateTechnicalProgress()
                
        self.oldStock = self.currStock
        self.updateEmissionsAndMaturity(market)                
                
            
        

    def buy(self,quantity=1):
        """
        Update of the internal stock
        """
        self.currStock +=1
        return list(self.properties.values())

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
        sales = max([0,self.currStock - self.oldStock])
        sales = sales + self.oldStock * self.replacementRate
        return sales
        
    def getProperties(self):
        return list(self.properties.values())

    def getProgress(self):
        return self.technicalProgress

    def getMaturity(self):
        return self.maturity

    def getExperience(self):
        return self.experience + self.initExperience

    def getGrowthRate(self):
        return self.currGrowthRate

    def getGlobalStock(self):
        return self.globalStock[self.goodID]
        
class Market():
    """
    Market class that mangages goood, technical progress and stocks.
    """
    def __init__(self, earth, properties, propRelDev=0.01, time = 0, burnIn=0):
        #import global variables
        self.globalRecord        = earth.getGlobalRecord()
        self.comm                = core.comm
        self.glob                = earth.getGlobals()
        self._graph              = earth.getGraph()
        self.para                = earth.getParameters()
        self.getParameter        = earth.getParameters

        self.time                = time
        self.date                = earth.date
        self.properties          = properties                 # (currently: emission, fixedCosts, operatingCosts)

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
        self.goods               = dict()                     # keys: goodIDs, values: goods
       # self.sales               = list()
        
        self.experienceBrownExo = list()
        self.experienceGreenExo = list()
        self.experienceBrownStart = 0.
        self.experienceGreenStart = 0.
        self.germany = False

        #adding market globals
        self.glob.registerValue('sales' , np.asarray([0]),'sum')
        self.glob.registerStat('meanEmm' , np.asarray([0]*len(properties)),'mean')
        self.glob.registerStat('stdEmm' , np.asarray([0]*len(properties)),'std')
        self.glob.registerStat('meanFiC' , np.asarray([0]*len(properties)),'mean')
        self.glob.registerStat('stdFiC' , np.asarray([0]*len(properties)),'std')
        self.glob.registerStat('meanOpC' , np.asarray([0]*len(properties)),'mean')
        self.glob.registerStat('stdOpC' , np.asarray([0]*len(properties)),'std')
        

    def updateGlobalStock(self):
        globalStock = np.zeros(self.para['nMobTypes'])
        for re in self.para['regionIDList']:
            reStock = self.glob.globalValue['stock_' + str(re)]
            globalStock += reStock
        
        self.goods[0].updateGlobalStock(globalStock, core.mpiRank)
        
    def updateSales(self):
        
        # push current Sales to globals for sync
        #ToDo: check this implementation of class variables
        sales = np.asarray([good.currLocalSales for good in self.goods.values()])
        self.glob.updateLocalValues('sales', sales * self.para['reductionFactor'])
            
        # pull sales from last time step to oldSales for technical chance
        self.goods[0].updateGlobalSales(self.glob.globalValue['sales'])
            
    def getNMobTypes(self):
        return self.__nMobTypes__

    def getTypes(self):
        return self.mobilityLables

    def initialCarInit(self):
        # actually puts the car on the market
        for label, propertyTuple, _, goodID in self.mobilityInitDict['start']:
            self.addGood2Market(label, propertyTuple, goodID)

    def getCurrentMaturity(self):
        return [self.goods[iGood].getMaturity() for iGood in list(self.goods.keys())]

    def getCurrentExperience(self):
        return [self.goods[iGood].getExperience() for iGood in list(self.goods.keys())]


    def computeStatistics(self):
        self.mean['emissions']      = self.glob.globalValue['meanEmm']
        self.mean['costs']          = self.glob.globalValue['meanFiC']
        self.mean['operatingCosts'] = self.glob.globalValue['meanOpC']
        
        self.std['emissions']       = self.glob.globalValue['stdEmm']
        self.std['costs']           = self.glob.globalValue['stdFiC']
        self.std['operatingCosts']  = self.glob.globalValue['stdOpC']


        if self.std['emissions'] == 0.:
            self.std['emissions'] = 1.

        lg.debug('Mean properties- mean: ' + str(self.mean) + ' std: ' + str(self.std))##OPTPRODUCTION

    def setInitialStatistics(self, typeQuantities):
        total = sum(typeQuantities[goodID] for goodID in range(self.__nMobTypes__))
        shares = np.zeros(self.__nMobTypes__)

        self.mean = dict()
        self.std  = dict()

        for goodID in range(self.__nMobTypes__):
            shares[goodID] = typeQuantities[goodID]/total

        for prop in self.properties:
            propMean = sum(shares[goodID]*self.mobilityProp[goodID][prop] for goodID in range(self.__nMobTypes__))
            propVar = sum(shares[goodID]*(self.mobilityProp[goodID][prop])**2 for goodID in range(self.__nMobTypes__))-propMean**2
            propStd = math.sqrt(propVar)
            self.mean[prop] = propMean
            self.std[prop] = propStd


    def initExogenousExperience(self, scenario):
        experienceWorldBrown = self.para['experienceWorldBrown']
        experienceWorldGreen = self.para['experienceWorldGreen']
        self.experienceBrownStart = experienceWorldBrown[0]
        self.experienceGreenStart = experienceWorldGreen[0]
                                                   
        if scenario == 6:
            self.germany = True 
            experienceBrownExo = [experienceWorldBrown[i]-self.para['experienceGerBrown'][i] for i in range(len(experienceWorldBrown))]
            experienceGreenExo = [experienceWorldGreen[i]-self.para['experienceGerGreen'][i] for i in range(len(experienceWorldGreen))]                               
        else:
            experienceBrownExo = experienceWorldBrown
            experienceGreenExo = experienceWorldGreen
        
        for i in range(1,len(experienceBrownExo)):
            diffBrown = experienceBrownExo[i]-experienceBrownExo[i-1]
            diffGreen = experienceGreenExo[i]-experienceGreenExo[i-1]
            for j in range(12):
                self.experienceBrownExo.append(experienceBrownExo[i-1]+diffBrown*float(j)/12.)
                self.experienceGreenExo.append(experienceGreenExo[i-1]+diffGreen*float(j)/12.)

    
    def initPrices(self):
        for good in list(self.goods.values()):
            if good.label == 'brown': 
                exponent = self.para['priceReductionB'] * self.para['priceRedBCorrection']
                good.properties['fixedCosts'] = self.para['initPriceBrown']
                expInMio = self.experienceBrownStart/1000000.       
                good.priceFactor   = good.properties['fixedCosts'] / (expInMio**exponent)
                good.properties['operatingCosts'] = self.para['initOperatingCostsGreen']  # 0.1
                
            elif good.label == 'green':                 
                exponent = self.para['priceReductionG'] * self.para['priceRedGCorrection']          
                good.properties['fixedCosts'] = self.para['initPriceGreen']
                expInMio = self.experienceGreenStart/1000000.       
                good.priceFactor   = good.properties['fixedCosts'] / (expInMio**exponent)
                good.properties['operatingCosts'] = self.para['initOperatingCostsGreen'] # 0.06
                
            elif good.label == 'public':
                good.properties['fixedCosts'] = self.para['initPricePublic']
                good.properties['operatingCosts'] = self.para['initOperatingCostsPublic'] # 0.25
                 
            elif good.label == 'shared':
                good.properties['fixedCosts'] = self.para['initPriceShared']
                good.properties['operatingCosts'] = self.para['initOperatingCostsGreen']  # 0.45
                
            else:
                good.properties['fixedCosts'] = self.para['initPriceNone']
                good.properties['operatingCosts'] = self.para['initOperatingCostsGreen'] 
  
    
    def updatePrices(self):
        yearIdx = self.time-self.burnIn        
        for good in list(self.goods.values()):
            if good.label == 'brown': 
                exponent = self.para['priceReductionB'] * self.para['priceRedBCorrection']
                factor = good.priceFactor
                if self.germany:
                    exp = self.experienceBrownExo[yearIdx] + good.experience
                else:
                    exp = self.experienceBrownExo[yearIdx]
                expInMio = exp/1000000.  
                good.properties['fixedCosts'] = factor * expInMio**exponent 
                
            elif good.label == 'green':                 
                exponent = self.para['priceReductionG'] * self.para['priceRedGCorrection']
                factor = good.priceFactor
                if self.germany:
                    exp = self.experienceGreenExo[yearIdx] + good.experience
                else:
                    exp = self.experienceGreenExo[yearIdx]

                expInMio = exp/1000000.       
                good.properties['fixedCosts'] = factor * expInMio**exponent 
                
            elif good.label == 'public':
                if self.date[1] > 2017:
                    good.properties['fixedCosts'] *= .99**(1./12)
                    
            elif good.label == 'shared':
                if good.properties['fixedCosts'] > 0.8*min(self.goods[0].properties['fixedCosts'],self.goods[1].properties['fixedCosts']) :
                    good.properties['fixedCosts'] = 0.8*min(self.goods[0].properties['fixedCosts'],self.goods[1].properties['fixedCosts'])

        
    def ecology(self, emissions):      
        try:
            ecology = 1. / (1.+math.exp((emissions-self.mean['emissions'])/self.std['emissions']))
        except:
            ecology = 0.
        return ecology


    def ecologyVectorized(self, emissions):
        ecology = 1. / (1.+np.exp((emissions-self.mean['emissions'])/self.std['emissions']))
        return ecology

    def step(self, world):      
        # check if a new car is entering the market
        if self.time in list(self.mobilityInitDict.keys()):

            for mobTuple in self.mobilityInitDict[self.time]:
                for label, propertyDict, _, goodID in  mobTuple:
                    self.addGood2Market(label, propertyDict, goodID)
        
        self.updateGlobalStock()
        
        #self.updateSales() done in earth.step
        
        # only do technical change after the burn in phase
        doTechProgress = self.time > self.burnIn
            
        for good in list(self.goods.values()):
            good.step(self, doTechProgress)
        
        if doTechProgress:
            self.updatePrices()   
            
        lg.info( 'sales in market: ' + str(self.glob.globalValue['sales']))


        self.computeInnovation()
        self.currMobProps = self.getMobProps()

        if doTechProgress:                             ##OPTPRODUCTION
            lg.debug('techProgress: ' + str([self.glob.globalValue['sales'][iGood] for iGood in list(self.goods.keys())]))##OPTPRODUCTION
             
            
        if core.mpiRank == 0:
            self.globalRecord['growthRate'].set(self.time, [self.goods[iGood].getGrowthRate() for iGood in list(self.goods.keys())])
            for iGood in list(self.goods.keys()):
                self.globalRecord['prop_' + self.goods[iGood].label].set(self.time, self.goods[iGood].getProperties())
            self.globalRecord['allTimeProduced'].set(self.time, self.getCurrentExperience())
            self.globalRecord['maturities'].set(self.time, self.getCurrentMaturity())

        lg.debug('new value of allTimeProduced: ' + str(self.getCurrentExperience()))##OPTPRODUCTION
        # reset sales
        #self.glob.globalValue['sales'] = self.glob.globalValue['sales']*0

        self.minPrice = min([good.getProperties()[0] for good in self.goods.values()])

        #compute new statistics
        self.computeStatistics()

        self.time += 1

    def computeInnovation(self):
        self.innovation = 1 - (normalize(np.asarray(self.getCurrentExperience()))**.5)
        

#    def initGood(self, label, propDict, initTimeStep, slope, initialProgress, allTimeProduced):
    def initGood(self, initTimeStep, label, propDict, **kwProperties):
        goodID = self.__nMobTypes__
        self.__nMobTypes__ +=1
        self.glob.globalValue['sales'] = np.asarray([0]*self.__nMobTypes__)
        # stf: das update verstehe ich nicht, da werden doch die glichen werte gesetzt, oder?
        self.glob.updateLocalValues('sales', np.asarray([0]*self.__nMobTypes__))
        self.stockByMobType.append(0)
        self.mobilityTypesToInit.append(label)
        self.goods[goodID] = Good(label, propDict, **kwProperties)

        if initTimeStep not in list(self.mobilityInitDict.keys()):
            self.mobilityInitDict[initTimeStep] = [[label, propDict, initTimeStep, goodID]] #, allTimeProduced]]
        else:
            self.mobilityInitDict[initTimeStep].append([label, propDict, initTimeStep, goodID]) #, allTimeProduced])


        self.computeInnovation()
        self.currMobProps = self.getMobProps()
        
        return goodID


    def addGood2Market(self, label, propertyDict, goodID):
        #add brand to the market
        self.stockByMobType[goodID]     = 0
        self.mobilityProp[goodID]       = propertyDict
        self.mobilityLables[goodID]     = label
        self.obsDict[self.time][goodID] = list()

    def remGood(self,label):
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
        




class Infrastructure():
    """
    Model for the development  of the charging infrastructure.
    """
    
    def __init__(self, earth, potMap, potFactor, immiFactor, dampFactor):
        
        # factor for immitation of existing charging stations
        self.immitationFactor = immiFactor
        #factor for dampening over-development of infrastructure
        self.dampenFactor     = dampFactor 
        
        self.potentialMap = potMap[earth.cellMapIds] ** potFactor# basic proxi variable for drawing new charging stations
        # normalizing as probablity
        self.potentialMap = self.potentialMap / np.sum(self.potentialMap)
        self.potentialMap = normalize(self.potentialMap)
        # share of new stations that are build in the are of this process
        self.shareStationsOfProcess = np.sum(potMap[earth.cellMapIds]) / np.nansum(potMap)
        if np.isnan(self.shareStationsOfProcess):
            self.shareStationsOfProcess = 0
        lg.debug('Share of new station for this process: ' + str(self.shareStationsOfProcess))##OPTPRODUCTION
        
        self.currStatMap = np.zeros_like(potMap) # map for the stations this year
        self.nextStatMap = np.zeros_like(potMap) # map for the stations next year
        
        self.carsPerStation = 10. # number taken from assumptions of the German government
        self.mapLoaded = True
        self.sigPara = 2.56141377e+02, 3.39506037e-2 # calibarted parameters

    # scurve for fitting 
    @staticmethod
    def sigmoid(x, x0, k):
        y = (1. / (1. + np.exp(-k*(x-x0)))) * 1e6
        return y

    def loadData(self, earth):
        
        if self.mapLoaded:
            if earth.date[0] == 1: # First month of the year
                
                if earth.para['scenario'] == 6:
                    nextName = earth.para['resourcePath'] + 'charge_stations_' +str(earth.date[1]+1) + '_186x219.npy'
                    currName = earth.para['resourcePath'] + 'charge_stations_' +str(earth.date[1]) + '_186x219.npy'
                else:
                    nextName = earth.para['resourcePath'] + 'charge_stations_' +str(earth.date[1]+1) + '.npy'
                    currName = earth.para['resourcePath'] + 'charge_stations_' +str(earth.date[1]) + '.npy'
    
                if os.path.isfile(nextName):
                    self.currStatMap = np.load(currName)
                    self.nextStatMap = np.load(nextName)
                else:
                    # switch flag, so that for future steps, new stations are gernerated
                    self.mapLoaded = False
                    
                    return None
                
                
            nextYearfactor = (earth.date[0]-1)/12
            currStations = nextYearfactor * self.nextStatMap + (1-nextYearfactor) * self.currStatMap
            
            return currStations
        else:
            return None

    def step(self, earth):
        
        
        if earth.para['scenario'] in [0,1]:
            #scenario small or medium
            lg.info('New station generated for: ' + str(earth.date))
            self.growthModel(earth)
            
        elif earth.para['scenario'] in [2,6]:
            # scenario ger or leun
            
            if earth.papi.comm.rank == 0:
                currStations = self.loadData(earth)
            else:
                currStations = None
            currStations = earth.papi.comm.bcast(currStations,root=0) 
        
            if currStations is not None:
            
                lg.info('New station loaded for: ' + str(earth.date))
                self.setStations(earth, currStations)
            else:
                lg.info('New station generated for: ' + str(earth.date))
                self.growthModel(earth)
    
    def setStations(self, earth, newStationsMap):
        newValues = newStationsMap[earth.cellMapIds]
        earth.setAttrOfAgentType('chargStat', newValues, agTypeID=CELL)
        
    def growthModel(self, earth):
        # if not given exogeneous, a fitted s-curve is used to evalulate the number
        # of new charging stations
        
        timeStep = earth.timeStep - earth.para['burnIn']
        nNewStations = self.sigmoid(np.asarray([timeStep-1, timeStep]), *self.sigPara) 
        nNewStations = np.diff(nNewStations) * self.shareStationsOfProcess / earth.para['spatialRedFactor']
            
        if earth.date[1] > 2020 and earth.para['linearCharging'] == 1:
            nNewStations = 2000. * self.shareStationsOfProcess / earth.para['spatialRedFactor']
        
        deviationFactor = (100. + (np.random.randn() *3)) / 100.
        nNewStations = int(nNewStations * deviationFactor)
        lg.debug('Adding ' + str(nNewStations) + ' new stations')##OPTPRODUCTION
        
        #get the current number of charging stations
        currNumStations  = earth.getAttrOfAgentType('chargStat', agTypeID=CELL)
        greenCarsPerCell = earth.getAttrOfAgentType('carsInCell',agTypeID=CELL)[:,GREEN]+1. 
        
        #immition factor (related to hotelings law that new competitiors tent to open at the same location)
        
        if np.sum(currNumStations) == 0:
            # new stations are only generated based on the potential map
            propability = self.potentialMap
        else:
            # new stations are generated based on the combination of 
            # potential, immitation and a dampening factor
            propImmi = (currNumStations)**self.immitationFactor
            propImmi = propImmi / np.nansum(propImmi)
            
            
            # dampening factor that applies for infrastructure that is not used and
            # reduces the potential increase of charging stations
            overSupplyFactor = 1
            demand  = greenCarsPerCell * earth.para['reductionFactor'] / (self.carsPerStation * overSupplyFactor)
            supply  = currNumStations

            #dampFac  = np.divide(demand, supply, out=np.zeros_like(demand)+1, where=supply!=0) ** self.dampenFactor
            dampFac = np.divide(demand,supply, out=np.ones_like(demand), where=supply!=0)

            #dampFac[np.isnan(dampFac)] = 1
            dampFac[dampFac > 1] = 1
            
            lg.debug('Dampening growth rate for ' + str(np.sum(dampFac < 1)) + ' cells with')   ##OPTPRODUCTION
            lg.debug(str(currNumStations[dampFac < 1]))                                         ##OPTPRODUCTION
            lg.debug('charging stations per cell - by factor of:')                              ##OPTPRODUCTION
            lg.debug(str(dampFac[dampFac < 1]))                                                 ##OPTPRODUCTION
            
            propability = (propImmi + self.potentialMap) * dampFac #* (usageMap[nonNanIdx] / currMap[nonNanIdx]*14.)**2
            propability = propability / np.sum(propability)
            
      
        if np.sum(np.isnan(propability)) >0:
            import pdb
            pdb.set_trace()
        randIdx = np.random.choice(list(range(len(currNumStations))), int(nNewStations), p=propability)
        
        uniqueRandIdx, count = np.unique(randIdx,return_counts=True)
        
        currNumStations[uniqueRandIdx] += count   
        earth.setAttrOfAgentType('chargStat', currNumStations, agTypeID=CELL)

    

# %% --- entity classes ---
class Person(Agent, Parallel):
    __slots__ = ['gID', 'nID']
    
    def __init__(self, world, **kwProperties):
        Agent.__init__(self, world, **kwProperties)
        Parallel.__init__(self, world, **kwProperties)

    def isAware(self, mobNewPeriod):
        # method that returns if the Persion is aktively searching for information
        return ((self.attr['lastAction'] - mobNewPeriod/10.) / mobNewPeriod)  > random.random()


    def register(self, world, parentEntity=None, liTypeID=None):

        Agent.register(self, world, parentEntity, liTypeID)
        self.loc = parentEntity.loc
        self.loc.peList.append(self.nID)
        self.hh = parentEntity
        self.hh.addAdult(self)

    
    def weightFriendExperience(self, world, commUtilPeers, weights):        
        friendUtil = commUtilPeers[:,self['mobType']]
        ownUtil    = self.attr['util'][0]

        prop = normalizedGaussian(friendUtil, ownUtil, world.para['utilObsError'])
        prior = normalize(weights)
        assert not any(np.isnan(prior)) ##OPTPRODUCTION
        post = normalize(prior * prop)
        
        sumWeights = sum1D(post)
        if not(np.isnan(sumWeights) or np.isinf(sumWeights)):
            if sumWeights > 0:
                self.setAttrOfLink('weig', post, liTypeID=CON_PP)
                #self['ESSR'] =  (1. / sumSquared1D(post)) / nFriends
                
                if sumWeights < 0.99: ##OPTPRODUCTION
                    import pdb        ##OPTPRODUCTION
                    pdb.set_trace()   ##OPTPRODUCTION
            return post, None

        else:
            lg.debug( 'post values:')
            lg.debug([ value for value in post])
            #lg.debug('diff values:')
            #lg.debug([value for value in diff])
            lg.debug('friendUtil values:')
            lg.debug([value for value in friendUtil])

            #return weights, self.attr['ESSR']


    def socialize(self, weights, friendIDs):

#        weights, linksRef, friendIDs = self.getAttrOfLink('weig', liTypeID=CON_PP)
        nContacts = len(weights)
        nDrops    = int(nContacts/5)
        
        if np.sum(weights < 0.001) > nDrops:
            dropIds = np.asarray(friendIDs)[np.argsort(weights)[:nDrops].tolist()].tolist()
        else:
             dropIds = np.asarray(friendIDs)[weights < 0.001]
        nDrops =  len(dropIds)
        if nDrops > 0:
            self.remLinks(friendIDs=dropIds, liTypeID=CON_PP)
        
        # add t new connections
        currContacts = self.getPeerIDs(CON_PP)

        frList, linkList           = self.getRandomNewContacts(nDrops, currContacts)

        if len(linkList) > 0:
        # update links
            lg.debug('adding contact links')
            
            for friendID in frList:
                self.addLink(friendID, liTypeID=CON_PP, weig=1.0/nContacts)


    def getRandomNewContacts(self, nContacts, currentContacts):
        
        cellConnWeights, linkIds, cellIds = self.loc.getConnectedCells()
        cell = self.getPeer(np.random.choice(cellIds))
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
        ownPref    = self['preferences']
        ownIncome  = self.hh['income']


        #get spatial weights to all connected cells
        cellConnWeights, linkIds, cellIds = self.loc.getConnectedCells()
        personIdsAll = list()
        nPers = list()
        cellWeigList = list()
        for cellWeight, cellIdx in zip(cellConnWeights, cellIds):

            cellWeigList.append(cellWeight)
            personIds = world.getAgent(cellIdx).getPersons()

            personIdsAll.append(personIds)
            nPers.append(len(personIds))

        # return nothing if too few candidates
        if not isInit and nPers > nContacts:
            lg.info('ID: ' + str(self.nID) + ' failed to generate friend')
            return [],[],[]

        
        #import pdb
        #pdb.set_trace()
        
        MAX_FRIEND_CANDIDATES = 200.
        if np.sum(nPers) > MAX_FRIEND_CANDIDATES:
            persWeig = np.asarray(nPers)
            persWeig = persWeig*cellConnWeights
            persWeig = persWeig/ np.sum(persWeig)
            nPers = (persWeig*MAX_FRIEND_CANDIDATES).astype(np.int32)
            personIds = list()
            for nP, subList in zip(nPers, personIdsAll):
                personIds.extend(np.random.choice(subList, nP))
        else:
            personIds = [item for sublist in personIdsAll for item in sublist]
        personIdsAll =  personIds   
        #setup of indices of columes:
        idxColSp = 0
        idxColIn = 1
        idxColPr = list(range(idxColIn+1,world.nPriorities()+idxColIn+1))
        weightData = np.zeros([np.sum(nPers),len(idxColPr) +2])

        idx = 0
        for nP, we in zip(nPers, cellWeigList):
            weightData[idx:idx+nP,idxColSp] = we
            idx = idx+ nP
        del idx

        if world.isParallel:
            hhIDs = [world.glob2Loc(x) for x in world.getAttrOfAgents('hhID', localIDList=personIdsAll)]
        else:
            hhIDs = world.getAttrOfAgents('hhID', localIDList=personIdsAll).tolist()
        weightData[:,idxColIn] = abs(world.getAttrOfAgents('income', localIDList=hhIDs) - ownIncome)
        weightData[:,idxColPr] = world.getAttrOfAgents('preferences', localIDList=personIdsAll)


        for i in idxColPr:
            weightData[:,i]  = (weightData[:,i] - ownPref[i-2])**2

        weightData[:,idxColPr[0]] = np.sum(weightData[:,idxColPr], axis=1)

        #nullIds  = weightData== 0

        #weight = inverse of distance
        weightData = np.divide(1.,weightData, out=np.zeros_like(weightData), where=weightData!=0)
        #weightData = 1./weightData
        #weightData[nullIds] = 0

        # normalization per row
        weightData[:,:3] = weightData[:,:3] / np.sum(weightData[:,:3],axis=0)
        #np.sum(weightData[:,:3] / np.sum(weightData[:,:3],axis=0),axis=0)

        #combining weights to the first row
        weightData[:,0] = np.prod(weightData[:,:3],axis=1)

        #normalizing final row
        weightData[:,0] = weightData[:,0] / np.sum(weightData[:,0],axis=0)

        if np.sum(weightData[:,0]>0) < nContacts:
            lg.info( "nID: " + str(self.nID) + ": Reducting the number of friends at " + str(self.loc['pos']))
            lg.info( "population = " + str(self.loc['population']) + " surrounding population: " +str(np.sum(self.loc.getAttrOfPeers('population',CON_LL)[0])))

            nContacts = min(np.sum(weightData[:,0]>0)-1,nContacts)

        if nContacts < 1:                                                       ##OPTPRODUCTION
            lg.info('ID: ' + str(self.nID) + ' failed to generate friend')      ##OPTPRODUCTION

        else:
            # adding contacts
            ids = np.random.choice(weightData.shape[0], nContacts, replace=False, p=weightData[:,0])
            contactList = [ personIdsAll[idx] for idx in ids ]
            connList   = [(self.nID, personIdsAll[idx]) for idx in ids]
            assert len(ids) <= world.para['maxFriends']
            
        if isInit and world.getParameters()['addYourself'] and addYourself:
            #add yourself as a friend
            contactList.append(self.nID)
            connList.append((self.nID,self.nID))

        weigList   = [1./len(connList)]*len(connList)
        
        nGhosts = 0
        if world.debug:                                                                 ##OPTPRODUCTION
            for peId in contactList:                                                    ##OPTPRODUCTION
                if isinstance(world.getAgent(peId), GhostPerson):                        ##OPTPRODUCTION
                    nGhosts += 1                                                        ##OPTPRODUCTION
        #lg.debug('At location ' + str(self.loc.attr['pos']) + 'Ratio of ghost peers: ' + str(float(nGhosts) / len(contactList))) ##OPTPRODUCTION
        
        return contactList, connList, weigList


    def computeCommunityUtility(self,earth, weights, commUtilPeers):
        #get weights from friends
        #weights, links = self.getAttrOfLink('weig', liTypeID=CON_PP)
        commUtil = self.attr['commUtil'][0] # old value
        selfUtil = self.attr['selfUtil'][0]
        # compute weighted mean of all friends
        if earth.para['weightConnections']:
            commUtil += np.dot(weights, commUtilPeers)
        else:
            commUtil += np.mean(commUtilPeers,axis=0)

        mobType  = self['mobType']
        
        
        # adding weighted selfUtil selftrust
        commUtil[mobType] += selfUtil[mobType] * earth.para['selfTrust']
        # stf: die teilerei hier finde ich komisch, warum werden z.B. für alle mobTypes / 2 geteilt?
        # ich hätte einfach eine Zeile mit "commUtil[mobType] /= 2" erwartet
        commUtil /= 2.
        commUtil[mobType] /= (earth.para['selfTrust']+2)/2.


        if len(selfUtil) != earth.para['nMobTypes'] or len(commUtil.shape) == 0 or commUtil.shape[0] != earth.para['nMobTypes']:       ##OPTPRODUCTION
            print('nID: ' + str(self.nID))                                       ##OPTPRODUCTION
            print('error: ')                                                     ##OPTPRODUCTION
            print('communityUtil: ' + str(commUtil))                             ##OPTPRODUCTION
            print('selfUtil: ' + str(self['selfUtil']))                 ##OPTPRODUCTION
            print('nlinks: ' + str(len(weights)))                                  ##OPTPRODUCTION

            return                                                              ##OPTPRODUCTION
        
        self.set('commUtil', commUtil)

        
        

    def imitate(self, utilPeers, weights, mobTypePeers):
        #pdb.set_trace()
        if self['preferences'][INNO] > .15 and random.random() > .98:
            self.imitation = [np.random.choice(self['commUtil'].shape[0])]
        else:

            if np.sum(~np.isfinite(utilPeers)) > 0:
                lg.info('Warning for utilPeers:')
                lg.info(str(utilPeers))
            # weight of the fitness (quality) of the memes
            sumUtilPeers = sum1D(utilPeers)
            if sumUtilPeers > 0:
                w_fitness = utilPeers / sumUtilPeers
            else:
                w_fitness = np.ones_like(utilPeers) / utilPeers.shape[0]
            
            # weight of reliability of the information (evolving over time)
            #w_reliability = weights

            # combination of weights for random drawing
#            w_full = w_fitness * w_reliability 
#            w_full = w_full / np.sum(w_full)
            w_full = normalize(prod1D(w_fitness,weights))  
            self.imitation =  np.random.choice(mobTypePeers, 2, p=w_full)
        

    def step(self, earth):
#        earth = core.earth
        #load data
        peerIDs         = self.getPeerIDs(liTypeID=CON_PP)
        
        nPeers          = len(peerIDs)
        
        commUtilPeers   = Person.cacheCommUtil[:nPeers,:]
        commUtilPeers[:]= self.getAttrOfPeers('commUtil', CON_PP)
        utilPeers       = Person.cacheUtil[:nPeers]
        utilPeers[:]    = self.getAttrOfPeers('util', CON_PP)
        mobTypePeers    = Person.cacheMobType[:nPeers]
        mobTypePeers[:] = self.getAttrOfPeers('mobType', CON_PP)
        weights         = Person.cacheWeights[:nPeers]
        weights[:], _, friendIDs= self.getAttrOfLink('weig', CON_PP)

        
        
        
        if earth.para['weightConnections'] and random.random() > self['util']: 
            # weight friends
            self.weightFriendExperience(earth, commUtilPeers, weights)


            # compute similarity
            #weights = np.asarray(self.getAttrOfLink('weig', liTypeID=CON_PP)[0])
            #preferences = np.asarray(self.getAttrOfPeers('preferences',CON_PP)[0])

            #average = np.average(preferences, axis= 0, weights=weights)
            #self.attr['peerBubbleHeterogeneity'] = np.sum(np.sqrt(np.average((preferences-average)**2, axis=0, weights=weights)))
        
        self.computeCommunityUtility(earth, weights, commUtilPeers) 
        
        if self.isAware(earth.para['mobNewPeriod']):
            self.imitate(utilPeers, weights, mobTypePeers)
        else:
            self.imitation = [-1]
        

        if self['mobType']>1:
            good = earth.market.goods[self['mobType']]
            self['prop'] =[good.properties['emissions'],good.properties['fixedCosts'], good.properties['operatingCosts']]

        # socialize
#        if np.random.rand() >0.99:
#            self.socialize(weights, friendIDs)



class GhostPerson(GhostAgent):

    def __init__(self, world, mpiOwner, nID=None, **kwProperties):
        GhostAgent.__init__(self, world, mpiOwner, nID, **kwProperties)

    def register(self, world, parentEntity=None, liTypeID=None):

        GhostAgent.register(self, world, parentEntity, liTypeID)
        self.loc = parentEntity.loc
        self.loc.peList.append(self.nID)
        self.hh = parentEntity

class GhostHousehold(GhostAgent):

    def __init__(self, world, mpiOwner, nID=None, **kwProperties):
        GhostAgent.__init__(self, world, mpiOwner, nID, **kwProperties)

    def register(self, world, parentEntity=None, liTypeID=None):

        GhostAgent.register(self, world, parentEntity, liTypeID)

        #self.queueConnection(locID,CON_LH)
        self.loc = parentEntity
        self.loc.hhList.append(self.nID)

class Household(Agent, Parallel):
    __slots__ = ['gID', 'nID']
    

        
    def __init__(self, world, **kwProperties):
        Agent.__init__(self, world, **kwProperties)
        Parallel.__init__(self, world, **kwProperties)

        if world.getParameters()['util'] == 'cobb':
            self.utilFunc = cobbDouglasUtilNumba
        elif world.getParameters()['util'] == 'ces':
            self.utilFunc = self.CESUtil
        self.computeTime = 0
        


    @staticmethod
    def cobbDouglasUtil(x, alpha):
        utility = 1.
        
        for i in range(len(x)):
            utility *= (100.*x[i])**alpha[i] 
        #if np.isnan(utility) or np.isinf(utility):  ##DEEP_DEBUG
        #    import pdb                              ##DEEP_DEBUG
        #    pdb.set_trace()                         ##DEEP_DEBUG

        # assert the limit of utility
        #assert utility > 0 and utility <= factor     ##DEEP_DEBUG

        return utility / 100.

    
    @staticmethod
    def cobbDouglasUtilArray(x, alpha):
        #utility = 1.
        
        
        utility = np.prod((100. * x) ** alpha) / 100.
        
        # assert the limit of utility
        #assert utility > 0 and utility <= factor      ##DEEP_DEBUG

        return utility
    

    @staticmethod
    def CESUtil(x, alpha):
        uti = 0.
        s = 3.    # elasticity of substitution, has to be float!
        factor = 100.
        for i in range(len(x)):
            uti += (alpha[i]*(factor * x[i])**(s-1))**(1/s)
            #print uti
        utility = uti**(s/(s-1))
        #if  np.isnan(utility) or np.isinf(utility): ##DEEP_DEBUG
        #    import pdb                              ##DEEP_DEBUG
        #    pdb.set_trace()                         ##DEEP_DEBUG

        # assert the limit of utility
        #assert utility > 0 and utility <= factor ##DEEP_DEBUG

        return utility / factor


    #def registerAtLocation(self, world,x,y, agTypeID, liTypeID):
    def register(self, world, parentEntity=None, liTypeID=None):


        Agent.register(self, world, parentEntity, liTypeID)


        #self.queueConnection(locID,CON_LH)
        self.loc = parentEntity
        self.loc.hhList.append(self.nID)

    def addAdult(self, personInstance):
        """adding reference to the person to household"""
        self.adults.append(personInstance)
        
#    def setAdultNodeList(self, world):
#        adultIdList = [adult.nID for adult in self.adults]
#        self.adultNodeList = world.graph.vs[adultIdList]

    def evalUtility(self, world, actionTaken=False):
        """
        Method to evaluate the utilty of all persons and the overall household
        """
        hhUtility = 0
        for adult in self.adults:

            utility = self.utilFunc(adult['consequences'], adult['preferences'])
            #assert not( np.isnan(utility) or np.isinf(utility)), utility ##OPTPRODUCTION

            #adult.node['expUtilNew'][adult.node['mobType']] = utility + np.random.randn()* world.para['utilObsError']/10
            adult['util'] = utility


            if actionTaken:
                # self-util is only saved if an action is taken
                adult.attr['selfUtil'][0, adult['mobType']] = utility

            hhUtility += utility

        self['util'] =  hhUtility

        return hhUtility

    def evalExpectedUtility(self, earth, getInfoList):

        actionIdsList   = list()
        eUtilsList      = list()

        for i,adult in enumerate(self.adults):

            if getInfoList[i]: #or (earth.time < earth.para['burnIn']):
                adult.computeCommunityUtility(earth)
                actionIds = [-1] + list(range(earth.para['nMobTypes']))

                eUtils = [adult.attr['util'][0]] + adult.attr['commUtil'][0]

            else:
                actionIds, eUtils = [-1], [adult.attr['util'][0]]

            actionIdsList.append(actionIds)
            eUtilsList.append(eUtils)



        if len(actionIdsList) == 0:
            return None, None

        elif len(actionIdsList) > 6:                            # to avoid the problem of too many possibilities (if more than 7 adults)
            minNoAction = len(actionIdsList) - 6                # minum number of adults not to take action
            #import pdb
            #pdb.set_trace()
            #print 'Action List: ',actionIdsList
            while len([x for x in actionIdsList if x == [-1]]) < minNoAction:
                randIdx = np.random.randint(len(actionIdsList))
                actionIdsList[randIdx] = [-1]
                eUtilsList[randIdx] =  [adult['util']]#[ eUtilsList[randIdx][0] ]
            #print 'large Household'

        combActions = core.cartesian(actionIdsList)
        overallUtil = np.sum(core.cartesian(eUtilsList),axis=1,keepdims=True)

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
            
            
            person.set('mobType', int(actionIdx))

            person.set('prop', properties)
            if earth.time <  earth.para['omniscientBurnIn']:
                person.set('lastAction', random.randint(0, int(1.5*earth.para['mobNewPeriod'])))
            else:
                person.set('lastAction', 0)

            nJourneys = person['nJourneys'].tolist()
            emissionsPerKm = properties[EMISSIONS] * earth.market.para['reductionFactor']# in kg/km
                        
            #TODO optimize code
            personalEmissions = 0.
            for nTrips, avgKm in zip(nJourneys, MEAN_KM_PER_TRIP): 
                personalEmissions += float(nTrips) * avgKm * emissionsPerKm  # in g Co2
                
            # add personal emissions to household sum
            #emissions[actionIdx] += personalEmissions / 1000. # in kg Co2
            person['emissions'] = personalEmissions / 1000. # in kg Co2
            
            #calculate costs
            personalCosts = properties[FIXEDCOSTS]
            for nTrips, avgKm in zip(nJourneys, MEAN_KM_PER_TRIP): 
                personalCosts += float(nTrips) * avgKm * properties[OPERATINGCOSTS] 
            person['costs'] = personalCosts
            # add cost of mobility to the expenses
            self['expenses'] += personalCosts


    def undoActions(self, world, persons):
        """
        Method to undo actions
        """
        for adult in persons:
            mobType = adult.get('mobType')
            self.loc.remFromTraffic(mobType)

            # remove cost of mobility to the expenses
            self['expenses'] -= adult['costs']

            world.market.sellCar(mobType)

    def testConsequences(self, earth, actionIds):
        
        consMat = np.zeros([actionIds.shape[0], actionIds.shape[1], earth.nPriorities()])
        

        
        hhCarBonus = 0.2
        mobProperties = earth.market.currMobProps
        convCell      = self.loc['convenience']
        income        = self['income']
        averDist      = np.mean([np.dot(ad['nJourneys'], MEAN_KM_PER_TRIP) for ad in self.adults])

        
        for ix, actions in enumerate(actionIds):
            
            #convenience
            consMat[ix,:,CONV] = convCell[actions]
            if any(actions < 2):
                consMat[ix,actions==2,CONV] += hhCarBonus
                consMat[ix,actions==4,CONV] += hhCarBonus
                
            #ecology
            consMat[ix,:,ECO] = earth.market.ecologyVectorized(mobProperties[actions,EMISSIONS])
            
            #money
            consMat[ix,:,MON] = max(1e-5, 1 - (sum1D(mobProperties[actions,FIXEDCOSTS]) + averDist*sum1D(mobProperties[actions,OPERATINGCOSTS])) / income)
            
            #immitation
            consMat[ix,:,INNO] = earth.market.innovation[actions]
            
            
        return consMat


    def calculateConsequences(self, market):

        carInHh = False
        # at least one car in househould?
        # stf: !=2 ist unschön. Was ist in der neuen Version mit Car sharing?
        if any([adult['mobType'] !=2 for adult in self.adults]):
            carInHh = True

        # calculate money consequence
        money = min(1., max(1e-5, 1 - self['expenses'] / self['income']))


        
        hhLocation = self.loc
        
        for adult in self.adults:
            hhCarBonus = 0.

            #get action of the person
            actionIdx = adult['mobType']
            mobProps  = adult['prop']

#            if (actionIdx != 2):
#                decay = 1- (1/(1+math.exp(-0.1*(adult.get('lastAction')-market.para['mobNewPeriod']))))
#            else:
#                decay = 1.
            
            #calculate emissions per cell


            if (actionIdx > 2) and carInHh:
                hhCarBonus = 0.2

            convenience = hhLocation['convenience'][actionIdx] + hhCarBonus

            if convenience > 1.:##OPTPRODUCTION
                convenience = 1. ##OPTPRODUCTION
                lg.info('Warning: Conveniences exeeded 1.0')##OPTPRODUCTION

            # calculate ecology:
            ecology   = market.ecology(mobProps[EMISSIONS])

            #calculate innovation
            innovation = market.innovation[actionIdx]
            

            
            
            # assuring that consequences are within 0 and 1
            for consequence in [convenience, ecology, money, innovation]:     ##OPTPRODUCTION
                if not((consequence <= 1) and (consequence >= 0)):            ##OPTPRODUCTION
                    print(consequence)                                         ##OPTPRODUCTION      
                    import pdb                                                ##OPTPRODUCTION
                    pdb.set_trace()                                           ##OPTPRODUCTION  
                assert (consequence <= 1) and (consequence >= 0)              ##OPTPRODUCTION


            adult['consequences'][:] = [convenience, ecology, money, innovation]


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
                oldMobType.append(adult['mobType'])
                oldProp.append(adult['prop'])
                oldLastAction.append(adult['lastAction'])
            oldExpenses = self['expenses']
            oldUtil = copy.copy(self['util'])


            # try all mobility combinations
            for combinationIdx in range(len(combinedActions)):
                self.attr['expenses'] = 0
                averDist      = np.mean([np.dot(ad['nJourneys'], MEAN_KM_PER_TRIP) for ad in self.adults])
                for adultIdx, adult in enumerate(self.adults):
                    
                    mobChoice = combinedActions[combinationIdx][adultIdx]
                    if mobChoice == -1:     # no action taken
                        adult['mobType'] = oldMobType[adultIdx]
                        adult['prop'] = oldProp[adultIdx]
                        adult['lastAction'] = oldLastAction[adultIdx]
                        mobilityProperties = oldProp[adultIdx]
                        #print(mobilityProperties)
                    else:
                        adult['mobType'] = mobChoice
                        mobilityProperties = market.goods[adult['mobType']].getProperties()
                        
                        adult['prop'] = mobilityProperties
                        if earth.time <  earth.para['burnIn']:
                            adult['lastAction'] = random.randint(0, int(1.5* earth.para['mobNewPeriod']))
                        else:
                            adult['lastAction'] = 0
                    operatingCosts = averDist*mobilityProperties[OPERATINGCOSTS]
                    fixedCosts     = mobilityProperties[FIXEDCOSTS]
                    self['expenses'] +=  operatingCosts + fixedCosts #TODO running costs

                self.calculateConsequences(market)
                utility = self.evalUtility(earth)
                utilities.append(utility)

            # reset old node values
            for adultIdx, adult in enumerate(self.adults):
                adult.set('mobType', oldMobType[adultIdx])
                adult.set('prop', oldProp[adultIdx])
                adult.set('lastAction', oldLastAction[adultIdx])
            self.set('expenses', oldExpenses)

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
                    lg.debug(self.attr)                                                                        ##OPTPRODUCTION
                    lg.debug('properties: ' + str([adult['prop'] for adult in self.adults]))           ##OPTPRODUCTION
                    lg.debug('consequence: '+ str([adult['consequences'] for adult in self.adults]))   ##OPTPRODUCTION
                    lg.debug('preferences: '+ str([adult['preferences'] for adult in self.adults]))    ##OPTPRODUCTION
                    lg.debug('utility: ' +    str([adult['util']  for adult in self.adults]))          ##OPTPRODUCTION
        else:
            actionTaken = False

        return actionTaken


    def decisionFunction(self, oldUtil, expNewUtil):
        if oldUtil == 0:
            return True
        else:
            return  expNewUtil / oldUtil > 1.05 and (expNewUtil / oldUtil ) - 1 > random.random()

    def possibleActions(self, earth, persGetInfoList , forcedTryAll = False):
        actionsList = list()
        nMobTypes = earth.market.getNMobTypes()

        for adultIdx, adult in enumerate(self.adults):
            if forcedTryAll or (earth.time < earth.para['burnIn']) or persGetInfoList[adultIdx]:
                actionsList.append([-1]+list(range(nMobTypes)))
            else:
                actionsList.append([-1])
        if len(actionsList) > 6:                            # to avoid the problem of too many possibilities (if more than 7 adults)
            minNoAction = len(actionsList) - 6              # minum number of adults not to take action
            while len([x for x in actionsList if x == [-1]]) < minNoAction:
                randIdx = np.random.randint(len(actionsList))
                actionsList[randIdx] = [-1]
            #print 'large Household'

        possibilities = core.cartesian(actionsList)
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
#        weig = np.asarray(overallUtil) - np.min(np.asarray(overallUtil))
#        weig =weig / np.sum(weig)
        weig = normalize(np.asarray(overallUtil) - np.min(np.asarray(overallUtil)))
        propActionIdx = np.random.choice(list(range(len(weig))), p=weig)
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
            ids = np.random.choice(list(range(len(actorIds))),6,replace=False)
            actionOptions = [actionOptions[idx] for idx in ids]
            actorIds      = [actorIds[idx] for idx in ids]
#        else:
#            actorIds = None
        combinedActionsOptions = core.cartesian(actionOptions)
        
        #tt2 = time.time()
        consMat = self.testConsequences(earth, core.cartesian(actionOptions)) #shape [nOptions x nPersons x nConsequences]
        #print 'testConsequences : ' +str(time.time() -tt2)
        #tt2 = time.time()
        #consMat = self.testConsequences2(earth, core.cartesian(actionOptions)) #shape [nOptions x nPersons x nConsequences]
        #print 'testConsequences2: ' +str(time.time() -tt2)
        utilities = np.zeros(consMat.shape[0])
        #print 'done'
        #prioMat = self.adultNodeList['preferences'] # [nPers x nPriorities]
        prioMat = self.getAttrOfPeers('preferences', liTypeID=CON_HP)
        if actorIds is None:
            for iAction in range(consMat.shape[0]):
                for iPers in range(len(self.adults)):
                    utilities[iAction] += self.utilFunc(consMat[iAction,iPers,:], prioMat[iPers])
        else:
            for iAction in range(consMat.shape[0]):
                for ii, iPers in enumerate(actorIds):
                    utilities[iAction] += self.utilFunc(consMat[iAction,ii,:], prioMat[iPers])
                
        bestOpt = combinedActionsOptions[np.argmax(utilities)]
        
        if np.max(utilities) > self['util'] * earth.para['hhAcceptFactor']:
            return bestOpt, np.max(utilities), actorIds
        else:
            return None, None, None

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
        
        if actorIds is not None:
            self.undoActions(earth, [self.adults[idx] for idx in actorIds])
            self.takeActions(earth, [self.adults[idx] for idx in actorIds], bestOpt)
        
        self.calculateConsequences(earth.market)
        
        self.evalUtility(earth, actionTaken=True)
            
        self.computeTime += time.time() - tt

    def step(self, earth):
        
        tt = time.time()

        actionTaken = False
        doCheckMobAlternatives = False

        if earth.time < earth.para['burnIn']:
            doCheckMobAlternatives = True
            persGetInfoList = [True] * len(self.adults) # list of persons that gather information about new mobility options

        else:
            if len(self.adults)*earth.market.minPrice < self['income']: #TODO

                #self.attr['nPers']
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
                    if self.attr['util'] == 0:
                        actionTaken = True
                    elif self.decisionFunction(self.attr['util'], expectedUtil): #or (earth.time < earth.para['burnIn']):
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

        if len(self.adults)*earth.market.minPrice < self.attr['income']:

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

    def __init__(self, world, agTypeID = 'ag', xPos = np.nan, yPos = np.nan):
        Household.__init__(self, world, agTypeID , xPos, yPos)
        self.writer = core.Writer(world, str(self.nID) + '_diary')
        raise('do not use - or update')

        #self.writer.write(



class Cell(Location, Parallel):

    def __init__(self, earth, **kwProperties):
        kwProperties.update({'population': 0, 'convenience': [0.,0.,0.,0.,0.], 'carsInCell':[0,0,0,0,0], 'regionId':0})
        Location.__init__(self, earth, **kwProperties)
        Parallel.__init__(self, earth, **kwProperties)
        self.hhList = list()
        self.peList = list()
        self.carsToBuy = 0
        self.deleteQueue =1
        self.cellSize      = 1.
        self.convFunctions = list()
        self.redFactor     = earth.para['reductionFactor']


    def initCellMemory(self, memoryLen, memeLabels):
        """
        deprectated
        """
        from collections import deque
        self.deleteQueue = deque([list()]*(memoryLen+1))
        self.currDelList = list()
        self.obsMemory   = core.Memory(memeLabels)


    def getConnectedCells(self):
        """ 
        ToDo: check if not deprecated 
        """
        self.weights, linkReferences, connectedNodes = self.getAttrOfLink('weig',liTypeID=CON_LL)
        return self.weights, linkReferences, connectedNodes



    def getHHs(self):
        return self.hhList

    def getPersons(self):
        return self.peList

    def getConnLoc(self,liTypeID=1):
        return self.getAgentOfCell(liTypeID=1)


    def addToTraffic(self,mobTypeID):
        """
        
        """
        self.attr['carsInCell'][:,int(mobTypeID)] += 1


    def remFromTraffic(self,mobTypeID):
        self.attr['carsInCell'][0,int(mobTypeID)] -= 1




    def getX(self, choice):
        return copy.copy(self.xCell[choice,:])

    def selfTest(self, world):
        """
        Not used in the simulations, but for testing purposes.
        
        """
        #self.attr['population'] = population #/ float(world.getParameters('reductionFactor'))

        convAll = self.calculateConveniences(world.getParameters(), world.market.getCurrentMaturity())

#        for x in convAll:                             ##OPTPRODUCTION
#            if np.isinf(x) or np.isnan(x) or x == 0:  ##OPTPRODUCTION
#                import pdb                            ##OPTPRODUCTION
#                pdb.set_trace()                       ##OPTPRODUCTION
        #self.attr['population'] = 0
        return convAll, self.attr['popDensity']

    def calculateConveniences(self, parameters, currentMaturity):
        """
        Calculation of convenience for all goods + the electric infrastructure.
        
        returns list of conveniences Range: [0,1]
        
        ToDo:
            - integration in goods?
            - seperation of infrastructure and convenience of goods
        
        """
        convAll = list()

        popDensity = np.float(self.get('popDensity'))
        for i, funcCall in enumerate(self.convFunctions):
            convAll.append(funcCall(popDensity, parameters, currentMaturity[i], self))
        
        homeChargingConv = parameters['homeChargConv']
        #convenience of electric mobility is additionally dependen on the infrastructure
        convAll[GREEN] *= homeChargingConv + (self.electricInfrastructure() * (1.-homeChargingConv))
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
        
        nStation      = self.getAttrOfPeers('chargStat',CON_LL)
        #print nStation
        if sum(nStation) == 0:
            return 0.
        
        weights, _, _ = self.getAttrOfLink('weig',CON_LL)
        
        if greenMeanCars is None:
            
            carsInCells   = self.getAttrOfPeers('carsInCell',CON_LL) * self.redFactor
            greenMeanCars = sum1D(carsInCells[:,GREEN]*weights)    
        

        
        
        # part of the convenience that is related to the capacity that is used
        avgStatPerCell = sum1D(nStation *  weights)
        
        capacityUse = greenMeanCars / (avgStatPerCell * 200.)
        
        if capacityUse > 100:
            useConv = 0.
        else:
            useConv = 1. /  math.exp(capacityUse)
        
        # part of the convenience that is related to a minimal value of charging
        # stations
        minRequirement = 1.
        
        if avgStatPerCell < minRequirement:
            statMinRequ = 1. / math.exp((1.-avgStatPerCell)**2 / .2)
        else:
            statMinRequ = 1.
        
        # overall convenience is product og both 
        eConv = statMinRequ * useConv
#        if self.get('chargStat') == 26:
#            import pdb
#            pdb.set_trace()
        assert (eConv >= 0) and (eConv <= 1) ##OPTPRODUCTION
        return eConv
    
    
    def aggregateEmission(self, earth):
        mobTypesPers = earth.getAttrOfAgents('mobType', self.peList)
        emissionsCell = np.zeros(5)
        emissionsPers = earth.getAttrOfAgents('emissions', self.peList)
        for mobType in range(5):
            idx = mobTypesPers == mobType
            emissionsCell[mobType] = emissionsPers[idx].sum()
        
            if mobType == GREEN:
                # elecric car is used -> compute estimate of power consumption
                # for the current model 0.6 kg / kWh
                electricConsumption = emissionsPers[idx].sum() / 0.6
                self.set('electricConsumption', electricConsumption)
        
        self.set('emissions', emissionsCell) 
        
    def step(self, parameters, currentMaturity):
        """
        Step method for cells
        """
        self.attr['convenience'] *= 0.
        self.attr['emissions'] *= 0.
        self.set('electricConsumption', 0.)
        convAll = self.calculateConveniences(parameters,currentMaturity)
        self.attr['convenience'][:] =  convAll


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

#    def updateHHList_old(self, graph):  # toDo agTypeID is not correct anymore
#        """
#        updated method for the household list, which is required since
#        ghost cells are not active on their own
#        """
#        agTypeID = graph.class2NodeType[Household]
#        hhIDList = self.getPeerIDs(agTypeID)
#        self.hhList = graph.vs[hhIDList]

#    def updatePeList_old(self, graph):  # toDo agTypeID is not correct anymore
#        """
#        updated method for the people list, which is required since
#        ghost cells are not active on their own
#        """
#        agTypeID = graph.class2NodeType[Person] 
#        hhIDList = self.getPeerIDs(agTypeID)
#        self.hhList = graph.vs[hhIDList]

class Opinion():
    """
    Creates preferences for households, given their properties
    ToDO:
        - Update the method + more sophisticate method maybe using the DLR Data
    """
    def __init__(self, world):
        self.charAge            = world.getParameters()['charAge']
        self.indiRatio          = world.getParameters()['individualPrio']
        self.minIncomeEco       = world.getParameters()['minIncomeEco']
        self.convIncomeFraction = world.getParameters()['charIncome']

    def getPref(self, age, sex, nKids, nPers, income, radicality):


        # priority of ecology
        ce = 0.5
        if sex == 2:
            ce +=1.5
        if income > self.minIncomeEco:
            rn = random.random()
            if random.random() > 0.9:
                ce += 3.
            elif rn > 0.6:
                ce += 2.
        elif income > 2 * self.minIncomeEco:
            if random.random() > 0.8:
                ce+=4.


        ce = float(ce)**2

        # priority of convinience
        cc = 0
        cc += nKids
        cc += income/self.convIncomeFraction/2
        if sex == 1:
            cc +=1

        cc += 2* float(age)/self.charAge
        cc = float(cc)**2

        # priority of money
        cm = 0
        cm += 2* self.convIncomeFraction/income * nPers
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
                ci = random.random()*5
            else:
                ci = random.random()*2
        else:
            if income>self.minIncomeEco:
                ci = random.random()*3
            else:
                ci = random.random()*1
        
        ci += (50. /age)**2                
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
        
#        print 'age: ' + str(age)    
#        print 'income: ' + str(income)
#        print 'sex: ' + str(sex)
#        print 'nKids: ' + str(nKids)
#        print 'nPers: ' + str(nPers)
#        print 'preferences:  convenience, ecology, money, innovation'
#        print 'preferences: ' + str(pref)
#        print '##################################'
        return tuple(pref)

# %% --- main ---
if __name__ == "__main__":
    pass
