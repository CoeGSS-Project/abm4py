#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://www.globalclimateforum.org

CAR INNOVATION MARKET MODEL
-- AGENT CLASS FILE --

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

from lib_gcfabm import Agent, Location
from class_auxiliary import Memory, Writer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Enum
_tll = 1 # loc - loc
_tlh = 2 # loc - household
_thh = 3 # household, household




class Household(Agent):

    def __init__(self, world, nodeType = 'ag', xPos = np.nan, yPos = np.nan):
        Agent.__init__(self, world, nodeType,  xPos, yPos)
        self.obs = dict()
        self.utilList =list()
        self.car = dict()


    def registerAgent(self, world):

        self.register(world)    
        world.nAgents += 1
#    def getObservations(self, world, labelList, mat = False):
#        """
#        Method to load the agents registered observatiosn from the observation
#        stack
#        returns 
#          a) list of array of actions and consequences a,x(a)
#          b) list of array of actions and utilities a,u(x(a))
#        """
#        
#        if len(self.obs) == 0:
#            return None
#        obsDf = pd.DataFrame(columns=labelList)
#        for key in self.obs.keys():
#            idx = self.obs[key]
#            obsDf = obsDf.append(world.entDict[key].obsDf.ix[idx,labelList])
#            #obsDf = pd.concat([obsDf, world.entDict[key].obsDf.ix[idx][labelList]])
#        return obsDf
    
    def getObservationsMat(self, world, labelList):
        if len(self.obs) == 0:
            return None
        mat = np.zeros([0,len(labelList)])
        for key in self.obs.keys():
            idxList, timeList = self.obs[key]
            idxList = [x for x,y in zip(idxList,timeList) if world.time - y < world.memoryTime  ]
            timeList = [x for x in timeList if world.time - x< world.memoryTime  ]
            self.obs[key] = idxList, timeList
            mat = np.vstack(( mat, world.entDict[key].obsMemory.getMeme(idxList,labelList)))
        return mat
    
    def learnConsequences(self):
        """ 
        Method to train the relation between actions and consequences
        from observations. 
        returns p(x|a) as ????
        """
        pass
    
    def linearReg(self, world):
        nObs = sum([len(self.obs[x][0]) for x in self.obs.keys()])
        if nObs < 10:
            return None
        #obsDf    = self.getObservations(world, world.props + ['utility'])
        obsMat    = self.getObservationsMat(world, ['hhID'] + world.props + ['utility'])
        
        if obsMat.shape[0] == 0:
            return None
        
        from sklearn import linear_model
        
        
        regr = linear_model.LinearRegression()
        
        weighted = True
        if weighted:
                
            weights, edges = self.getConnProp('weig',_thh,mode='IN')
            sources = [self.graph.es[edge].source for edge in edges]
            srcDict =  dict(zip(sources,weights))
            
            regr.fit(obsMat[:,1:-1], obsMat[:,-1],map(srcDict.__getitem__,obsMat[:,0].tolist()))
        else:
            regr.fit(obsMat[:,1:-1], obsMat[:,-1])
            
        return regr
    
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
                
            weights, edges = self.getConnProp('weig',_thh,mode='IN')
            sources = [self.graph.es[edge].source for edge in edges]
            #targDict = {self.graph.es[edge].target: edge for edge in edges}
            #targDict = dict((self.graph.es[edge].target, edge) for edge in edges)
            srcDict =  dict(zip(sources,weights))
            for i, id_ in enumerate(carIDs):
                
                tmp[i,obsMat[:,-1] == id_] = map(srcDict.__getitem__,obsMat[obsMat[:,-1] == id_,0].tolist())
        else:
            for i, id_ in enumerate(carIDs):
                tmp[i,obsMat[:,-1] == id_] = 1
            
        avgUtil = np.dot(obsMat[:,1],tmp.T) / np.sum(tmp,axis=1)
        maxid = np.argmax(avgUtil)
        return carIDs[maxid], avgUtil[maxid]
    
    def weightFriendExperience(self, world):
        friendUtil, edgeIDs = self.getNeighNodeValues( 'util' ,edgeType= _thh)
        carLabels, _        = self.getNeighNodeValues( 'label' ,edgeType= _thh)
        #friendID            = self.getOutNeighNodes(edgeType=_thh)
        ownLabel = self.getValue('label')
        ownUtil  = self.getValue('util')
        
        idxT = np.where(carLabels==ownLabel)[0]
        indexedEdges = [ edgeIDs[x] for x in idxT]
        
        
        
        diff = np.asarray(friendUtil)[idxT] - ownUtil
        prop = np.exp(-(diff**2) / 0.001)
        prop = prop / np.sum(prop)
        #TODO  try of an bayesian update - check for right math
        
        onlyEqualCars = True
        if onlyEqualCars:
        #### only weighting agents with same cars
            prior = np.asarray(world.getEdgeValues(indexedEdges,'weig'))
            post = prior * prop 
            sumPrior = np.sum(prior)
            post = post / np.sum(post) * sumPrior
            world.setEdgeValues(indexedEdges,'weig',post)
            neig = [self.graph.es[x].target for x in indexedEdges]
            diff = [np.sum(np.abs(np.asarray(self.graph.vs[x]['preferences']) - np.asarray(self.graph.vs[self.nID]['preferences']))) for x in neig]
#            plt.scatter(diff,prop)
#            for y,z in zip(diff,prop):
#                if y > .5 and z > .8:
#                    print 1
        else:
        #### reducting also weight of owners of other cars -> factor .99
            idxF = np.where(carLabels!=ownLabel)[0]
            #otherEdges = [ edgeIDs[x] for x in idxF]
       
            prior = np.asarray(world.getEdgeValues(edgeIDs,'weig'))
            post = prior
            sumPrior = np.sum(prior)
            post[idxT] = prior[idxT] * prop 
            post[idxF] = prior[idxT] * .99
            post = post / np.sum(post) * sumPrior
            world.setEdgeValues(edgeIDs,'weig',post)
        
    def connectGeoNode(self, world):
        # todo: change to access world.queueEdge   
        # connect agent to its spatial location
        geoNodeID = int(self.graph.IdArray[int(self.x),int(self.y)])
        self.queueConnection(geoNodeID,_tlh)         
        #eID = self.graph.get_eid(self.nID,geoNodeID)
        #self.graph.es[eID]['type'] = _tlh
        self.loc = world.entDict[geoNodeID]        
        self.loc.agList.append(self.nID)
        
    def socialize(self,world):
        """
        Method to make/delete or alter the agents social connections
        details to be defined
        """
        ownPref = world.graph.vs[self.nID]['preferences']
        weights, eList, CellList = self.loc.getConnCellsPlus()
        cumWeights = np.cumsum(weights)
        #print sum(weights)
        idx = np.argmax(cumWeights > np.random.random(1)) 

        #eList   = self.loc.getConnLoc()
        cellID = CellList[idx]
        #agList = world.entDict[cellID].getAgentOfCell(2)
        agList = world.entDict[cellID].getAgents()
        if len(agList) > 0:
            newFrID = np.random.choice(agList)
            diff = 0
            for x,y in zip (world.graph.vs[newFrID]['preferences'], ownPref ):
                diff += abs(x-y)
                
            if not newFrID == self.nID and diff < self.tolerance:
                self.addConnection(newFrID,3)
                return True
            else:
                return False
    
    def generateFriends(self,world, nFriends):
        """
        Method to make/delete or alter the agents social connections
        details to be defined
        """
        friendList = list()
        connList   = list()
        ownPref = world.graph.vs[self.nID]['preferences'] #TODO change to lib-cal
        weights, eList, CellList = self.loc.getConnCellsPlus()
        cumWeights = np.cumsum(weights)
        #print sum(weights)
        iFriend = 0
        i = 0
        while iFriend < nFriends:
            
            idx = np.argmax(cumWeights > np.random.random(1)) 
    
            #eList   = self.loc.getConnLoc()
            cellID = CellList[idx]
            #print cellID
            #agList = world.entDict[cellID].getAgentOfCell(2)
            agList = world.entDict[cellID].getAgents()
            if len(agList) > 0:

                newFrID = np.random.choice(agList)
                diff = 0
                for x,y in zip (world.graph.vs[newFrID]['preferences'], ownPref ): #TODO change to lib-cal
                    diff += abs(x-y)
                    
                if not newFrID == self.nID and newFrID not in friendList and  diff < self.tolerance:   
                    friendList.append(newFrID)
                    connList.append((self.nID,newFrID))
                    #connList.append((self.nID,newFrID))
                    iFriend +=1
                i +=1
                if i > 1000:
                    break
        
            if len(connList) > 0:
                weigList = [1./len(connList)]*len(connList)
            else:
                weigList = []
        return friendList, connList, weigList

            
    def evalUtility(self, world, props =None):
        if props is None:
            props = self.car['prop']
            
        # alternative a (dominator)           
        #x = world.market.getPropPercentiles(props) 
        
        #alternative b (tanh)
        normalizedDeviation = world.market.statistics[1] - world.market.statistics[0]
        normalizedDeviation[normalizedDeviation==0] = 1 # prevent division by zero
        
        x = tuple(0.5 + 0.5 * np.tanh((props - world.market.statistics[0]) / normalizedDeviation))
        self.setValue('preceivedProps',x)
        
        util = world.og.getUtililty(x,self.getValue('preferences'))         
        self.graph.vs[self.nID]['util'] = util #TODO change to lib-cal
        if np.isnan(util):
            print 1
        return util      
                        
    def buyCar(self,world, label):
        """
        Method to execute the optimal choice, observe its real consequences and
        tell others in your network about it.
        """
        # add sale to record
        #world.record.loc[world.time, world.rec["sales"][1][self.prefTyp]] += 1
        world.globalRec['sales'].addIdx(world.time, 1 ,self.prefTyp) 
        carID, properties = world.market.buyCar(label, self.nID)
        self.loc.addToTraffic(label)
        self.car['ID']    = carID
        self.car['prop']  = properties
        self.car['obsID'] = None
        self.car['label'] = label
        self.car['age']   = 0
        self.setValue('label',label)
        
    def shareExperience(self, world):
        #util = self.evalUtility(world, self.car['prop'])
        
        # save util based on label
        world.market.obsDict[world.time][self.car['label']].append(self.util)
        #self.setValue('util',util)
        self.car['obsID'] = self.loc.registerObs(self.nID, self.car['prop'], self.util, self.car['label'])
        #world.record.loc[world.time,world.rec["avgUtilPref"][1][self.prefTyp]] += self.graph.vs[self.nID]['util']
        
        
        world.globalRec['avgUtil'].addIdx(world.time, self.graph.vs[self.nID]['util'] ,[0, self.prefTyp+1]) 
        
        #self.utilList.append(util)
        #print 'agent' + str(self.nID)
        for neig in self.getOutNeighNodes(_thh):
            agent = world.entDict[neig]
            agent.tell(self.loc.nID,self.car['obsID'], world.time)
            #print neig, self.loc.nID,obsID
    
    def sellCar(self,world, carID):
        world.market.sellCar(carID)
        self.loc.remFromTraffic(self.car['label'])
    
    def tell(self, locID, obsID, time):
        if locID in self.obs:
            self.obs[locID][0].append(obsID)
            self.obs[locID][1].append(time)
        else:
            self.obs[locID] = [obsID], [time]

    def step(self, world):
        self.car['age']  +=1
        utilUpdated = False
        if self.car['age'] > world.carNewPeriod and np.random.rand(1)>.5: 
                    
            choice = self.optimalChoice(world)  
            
            if choice is not None:
                if choice[1] > self.utilList[-1] *1.2:
                    self.sellCar(world, self.car['ID'])
                    self.buyCar(world, choice[0])
                elif np.random.rand(1)>.75:
                    regr = self.linearReg(world)
                    if regr is not None:
                        df = pd.DataFrame.from_dict(world.market.brandProp)
                        extUtil = regr.predict(df.values.T)
                        
                        if extUtil[extUtil.argmax()] > self.utilList[-1]:
                            label = df.columns[extUtil.argmax()]
                            self.sellCar(world, self.car['ID'])
                            self.buyCar(world, label)
                            util = self.evalUtility(world)
                            self.utilList.append(util)
                            utilUpdated = True
                            # update prior expectations of observation
                            self.weightFriendExperience(world)
        if not utilUpdated:
            util = self.evalUtility(world)
            self.utilList.append(util)
        self.shareExperience(world)

class Reporter(Household):
    
    def __init__(self, world, nodeType = 'ag', xPos = np.nan, yPos = np.nan):
        Household.__init__(self, world, nodeType , xPos, yPos)
        self.writer = Writer(world, str(self.nID) + '_diary')

    def evalUtility(self, world, props =None):
        if props is None:
            props = self.car['prop']
        
        # alternative a (dominator)           
        #x = world.market.getPropPercentiles(props) 
        
        #alternative b (tanh)
        normalizedDeviation = world.market.statistics[1] - world.market.statistics[0]
        normalizedDeviation[normalizedDeviation==0] = 1 # prevent division by zero
        x = 0.5 + 0.5 * np.tanh((props - world.market.statistics[0]) / normalizedDeviation)
        self.setValue('preceivedProps',x)
        
        util = world.og.getUtililty(x,self.getValue('preferences'))         
        self.graph.vs[self.nID]['util'] = util
        self.writer.write(world.market.brandLabels[self.car['label']] + str( x) + str(util))
        if np.isnan(util):
            print "util is nan"
        return util   
    
    def getObservationsMat(self, world, labelList):
        if len(self.obs) == 0:
            return None
        mat = np.zeros([0,len(labelList)])
        for key in self.obs.keys():
            idxList, timeList = self.obs[key]
            idxList = [x for x,y in zip(idxList,timeList) if world.time - y < world.memoryTime  ]
            timeList = [x for x in timeList if world.time - x< world.memoryTime  ]
            self.obs[key] = idxList, timeList
            mat = np.vstack(( mat, world.entDict[key].obsMemory.getMeme(idxList,labelList)))
            
        self.writer.write('Observation: ')
        if 'label' in labelList:
            labelIdx = labelList.index("label")
            for rowIdx in  xrange(mat.shape[0]):
                row = mat[rowIdx]
                self.writer.write(world.market.brandLabels[row[labelIdx]] + '-' + str(row) + ' ')
        else:
            for rowIdx in  xrange(mat.shape[0]):
                row = mat[rowIdx]
                self.writer.write( str(row) + ' ')
        self.writer.write('-------------------------')
               

        return mat
    
    def optimalChoice(self,world):
        """ 
        Method for searching the optimal choice that lead to the highest
        expected utility
        return a_opt = arg_max (E(u(a)))
        """
        import time 
        if len(self.obs) == 0:
            return None
        
        obsMat    = self.getObservationsMat(world,['utility','label'])

       
        if obsMat.shape[0] == 0:
            return None
        
        ids       = np.unique(obsMat[:,-1])
        
        tmp = np.zeros([len(ids),obsMat.shape[0]])
        for i, id_ in enumerate(ids):
            tmp[i] = obsMat[:,-1] == id_
        avgUtil = np.dot(obsMat[:,0],tmp.T) / np.sum(tmp,axis=1)    
            
        maxIdx = np.argmax(avgUtil)
        
        outStr = 'Average Observation: '
        for ID, util in zip(ids, avgUtil):
            outStr +=  world.market.brandLabels[ID] + '-' + str(util) + ' '
        self.writer.write(outStr)
        return ids[maxIdx], avgUtil[maxIdx]
        
    def step(self, world):
        self.car['age']  +=1
        if self.car['age'] > 5 and np.random.rand(1)>.5: 
            
            choice = self.optimalChoice(world)  
            
            if choice is not None:
                if choice[1] > self.utilList[-1]:
                    self.sellCar(world, self.car['ID'])
                    self.buyCar(world, choice[0])
                    self.writer.write('[a] Buying ' + world.market.brandLabels[choice[0]] + ' with expUtil: ' + str(choice[1]))
                elif np.random.rand(1)>.75:
                    regr = self.linearReg(world)
                    if regr is not None:
                        df = pd.DataFrame.from_dict(world.market.brandProp)
                        extUtil = regr.predict(df.values.T)
                        
                        if extUtil[extUtil.argmax()] > self.utilList[-1]:
                            carTyp = df.columns[extUtil.argmax()]
                            self.sellCar(world, self.car['ID'])
                            self.buyCar(world, carTyp)
                            self.writer.write('[b] Buying ' + world.market.brandLabels[carTyp] + ' with expUtil: ' + str(extUtil[extUtil.argmax()]))
                else:
                    self.writer.write('not doing regression')
            else:
                self.writer.write('too few observations')
        else:
            self.writer.write('car is new (' + str(self.car['age']) + ')' )
                
        self.util = self.evalUtility(world)
        
        self.writer.write('Actual util is: ' +str(self.util))
        self.utilList.append(self.util)
        self.shareExperience(world)        
            
class Cell(Location):
    
    def __init__(self, Earth,  xPos, yPos):
        Location.__init__(self, Earth,  xPos, yPos)
        self.agList = list()
        self.carsToBuy = 0
        self.deleteQueue =1
        self.currID = 0
        self.traffic = dict()
    
    def initCellMemory(self, memoryLen, memeLabels):
        from collections import deque
        self.deleteQueue = deque([list()]*memoryLen)
        self.currDelList = list()
        self.obsMemory   = Memory(memeLabels)
    
    def getConnCellsPlus(self):
        self.weights, self.eIDs = self.getConnProp('weig')
        self.connNodeList = [self.graph.es[x].target for x in self.eIDs ]
        
        #remap function to only return the values 
        self.getConnCellsPlus = self.returnConnWeights
        return self.weights, self.eIDs, self.connNodeList
    
    def returnConnWeights(self):
        return self.weights, self.eIDs, self.connNodeList
    
    def getAgents(self):
        #return self.getAgentOfCell(edgeType=1)
        return self.agList
    
    def getConnLoc(self,edgeType=1):
        return self.getAgentOfCell(edgeType=1)
    
    def returnCarPercentile(self):
        """ 
        Methods that returns the differnet percentiles for a given car choice
        """
        pass
    
    def registerObs(self, hhID, prop, util, label):
        """
        Adds a car to the cell pool of observations
        """
        #prop.append(util)
        #obsID = self.currID
        meme = prop + [util, label, hhID]
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

    def addToTraffic(self,label):
        self.traffic[label] += 1
        
    def remFromTraffic(self,label):
        self.traffic[label] -= 1
        
    def step(self):
        """
        Manages the deletion og obersvation after a while
        """
        
        self.deleteQueue.append(self.currDelList) # add current list to the queue for later
        delList = self.deleteQueue.popleft()      # takes the list to delete
        for obsID in delList:
            self.obsMemory.remMeme(obsID)         # removes the obs from memory
        #self.obsDf.drop(delList,inplace=True)     # removes the obs from memory
        self.currDelList = list()                 # restarts the list for the next step
        
        #write cell traffic to graph
        self.setValue('carsInCell', tuple(self.traffic.values()))
        
        
        
        
        
        