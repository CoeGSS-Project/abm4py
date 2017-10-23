#!/usr/bin/env python2
# -*- coding: UTF-8-*-
"""


Copyright (c) 2017
Global Climate Forum e.V.
http://www.globalclimateforum.org

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


Philosophy:
    
Classes are only the hull of methods around an graph node with its connections. 
Entities can only alter out-connections by themselves (out egbes belong to the source node)
Entities should therefore by fully defined by their global ID and local ID and
the underlying graph that contains all properties    

Thus, as far as possible save any property of all entities in the graph

Communication is managed via the spatial location 

Thus, the function "location.registerEntity" does initialize ghost copies
    
TODOs:

soon:
    - copy important graph functon to world and agument it to fit with caching and queuing
    - redirect all output to the correct folder
    - IO of connections and their attributes
    - per node collecting of communting and waiting times
    - reach usage of 100 parallell processes (96)
    - partitioning of the cell grid via metis or some other graph partioner
    - debug flag to control additional output
    - allow movements of agents - transfer between nodes
    - (optional) write attribute names to hdf5 file
    
later:
    - movement of agents between processes
    - implement mpi communication of string attributes
    - implement output of string attributes
    - reach usage of 1000 parallell processes (960) -> how to do with only 3800 Locations??
    - multipthreading within locatons??

  
"""
from mpi4py import  MPI
from os.path import expanduser
home = expanduser("~")
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
import logging as lg
#lg.info('Directory: ' +  dir_path)
import sys
import socket
if socket.gethostname() in ['gcf-VirtualBox', 'ThinkStation-D30']:
    sys.path = [dir_path + '/h5py/build/lib.linux-x86_64-2.7'] + sys.path 
    sys.path = [dir_path + '/mpi4py/build/lib.linux-x86_64-2.7'] + sys.path 
    

import igraph as ig
import numpy as np

import time 
from bunch import Bunch
#import array
#from pprint import pprint

from class_graph import WorldGraph
import class_auxiliary as aux # Record, Memory, Writer, cartesian

class Queue():

    def __init__(self, world):
        self.graph = world.graph
        self.edgeDict       = dict()
        self.edgeProperties = dict()
        
        self.currNodeID     = None
        self.nodeList       = list()
        self.nodeTypeList   = dict()
        self.nodeProperties = dict()        
        self.edgeDeleteList = list()
        
    def addVertex(self, nodeType, gID, **kwProperties):
        
        
        
        if len(self.nodeList) == 0:
            #print 'setting current nID: ' + str(self.graph.vcount())
            self.currNodeID = self.graph.vcount()

        # adding of the data
        nID = self.currNodeID
        self.nodeList.append(nID)               # nID to general list
        #kwProperties.update({'nID': nID})
        kwProperties.update({ 'type': nodeType, 'gID':gID})
        
        if nodeType not in self.nodeProperties.keys():
            # init of new nodeType
            propertiesOfType = self.graph.nodeProperies[nodeType]
            #print propertiesOfType    
            #print kwProperties.keys()
            assert all([prop in propertiesOfType for prop in kwProperties.keys()]) # check that all given properties are registered
            
            self.nodeProperties[nodeType] = dict()
            self.nodeTypeList[nodeType]   =list()
            
            #self.nodeProperties[nodeType]['nID'] = list()
            for prop in kwProperties.keys():
                #print prop
                self.nodeProperties[nodeType][prop] = list()
                

        
        self.nodeTypeList[nodeType].append(self.currNodeID) # nID to type list
        self.currNodeID += 1                                # moving index 
        
        for prop, value in kwProperties.iteritems():
            self.nodeProperties[nodeType][prop].append(value)  # properties        
            
            
        return nID, self.nodeProperties[nodeType]

    def dequeueVertices(self, world):
        
        if len(self.nodeList) ==0:
            return
        assert self.nodeList[0] == self.graph.vcount() # check that the queuing idx is consitent
        #print self.graph.vcount(), self.nodeList[0]
        #adding all nodes first
        self.graph.add_vertices(len(self.nodeList))
        
        # adding data per type
        for nodeType in self.nodeTypeList.keys():

            nodeSeq = self.graph.vs[self.nodeTypeList[nodeType]]    # node sequence of specified type
            #print 'indices' + str(nodeSeq.indices)
            nodeSeq['type'] = nodeType                                # adding type to graph
            #print 'sequ:' + str(nodeSeq['type'])
            
            for prop in self.nodeProperties[nodeType].keys():
                #print prop
                #print nodeType
                #print self.nodeProperties[nodeType][prop]
                nodeSeq[prop] = self.nodeProperties[nodeType][prop] # adding properties 
                
            #print 'nodeTypeList:' +str (self.nodeTypeList[nodeType])
            for entity in [world.entList[i] for i in self.nodeTypeList[nodeType]]:
                
                #print 'entity.nID:' + str(entity.nID)
                entity.node = self.graph.vs[entity.nID]
                #print 'node index:'  + str(entity.node.index)
                #entity.register(world)
                #print 'assert ' + str((entity.nID, entity.node.index))
                assert entity.nID == entity.node.index
        
        # reset queue
        self.currNodeID     = None
        self.nodeList       = list()
        self.nodeTypeList   = dict()
        self.nodeProperties = dict()  

    def addEdges(self, edgeList, **kwProperties):
        edgeType = kwProperties['type']
        if edgeType not in self.edgeDict.keys():
            self.edgeDict[edgeType]         = list()
            self.edgeProperties[edgeType]   = dict()
            for prop in kwProperties.keys():
                #print prop
                self.edgeProperties[edgeType] [prop] = list()
        self.edgeDict[edgeType].extend(edgeList)
        for propKey in kwProperties.keys():
            
            if not isinstance(kwProperties[propKey], list):
               self.edgeProperties[edgeType][propKey].extend ([kwProperties[propKey]]* len(edgeList) ) 
            else :
                assert len(kwProperties[propKey]) == len(edgeList)
                self.edgeProperties[edgeType][propKey].extend(kwProperties[propKey])
        
    def addEdge(self, source, target, **kwProperties):
        
        edgeType = kwProperties['type']
        #possible init
        if edgeType not in self.edgeDict.keys():
            self.edgeDict[edgeType]         = list()
            self.edgeProperties[edgeType]   = dict()
            for prop in kwProperties.keys():
                #print prop
                self.edgeProperties[edgeType] [prop] = list()
            
        # add edge source-target-tuple            
        self.edgeDict[edgeType].append((source, target))

        # add properties
        #self.edgeProperties[edgeType]['type'].append(edgeType)
        for propKey in kwProperties.keys():
            self.edgeProperties[edgeType][propKey].append(kwProperties[propKey])


    def dequeueEdges(self, world):

        if len(self.edgeDict) ==0:
            return
        #print "dequeuing edges"
        #print self.edgeDict.keys()
        for edgeType in self.edgeDict.keys():
            #print 'creating edges: ' + str(self.edgeDict[edgeType])
            eStart = self.graph.ecount()
            self.graph.add_edges(self.edgeDict[edgeType])
            for prop in self.edgeProperties[edgeType].keys():
                self.graph.es[eStart:][prop] = self.edgeProperties[edgeType][prop]
            

#        for node in world.entList:
#            node.__updateEdges__()

        # empty queue
        self.edgeDict       = dict()
        self.edgeProperties = dict()  
        
        # if all queues are done, set complete flag
        #if len(self.graph.edgeQueues) == 0:
        #    self.graph.edgesComplete = True

    
    def dequeueEdgeDeleteList(self, world):
        self.graph.delete_edges(self.edgeDeleteList)
        self.edgeDeleteList = list()

class Cache():
        
    def __init__(self, graph, nID):
        self.graph       = graph
        self.nID         = nID
        self.edgesAll    = None
        self.edgesByType = dict()
        
        self.peersAll    = None
        self.peersByType = dict()
   
    def __reCachePeers__(self, edgeType=None):
        
        eList  = self.graph.incident(self.nID,mode="out")
        if edgeType is not None:
            
            edges = self.graph.es[eList].select(type=edgeType)
            peersIDs = [edge.target for edge in edges] 
            self.peersByType[edgeType] = self.graph.vs[peersIDs]
        else:
            edges = self.graph.es[eList].select(type_ne=0)
            peersIDs = [edge.target for edge in edges] 
            self.peersAll = self.graph.vs[peersIDs]
    
          
    def __checkPeerCache__(self, edgeType):
        # check if re-caching is required
        if edgeType is None:
            if self.peersAll is None:
                self.__reCachePeers__()
        else:        
            if edgeType not in self.peersByType.keys():
                self.__reCachePeers__(edgeType)
        
    def __reCacheEdges__(self, edgeType=None):
        """ privat function that re-caches all edges of the node"""
        
        # out edges by type
        if edgeType is not None:
            # re-cache only certain type
            self.edgesByType[edgeType] = self.edgesAll.select(type=edgeType)
        else:
            # all out edges
            self.edgesAll    = self.graph.es[self.graph.incident(self.nID,'out')].select(type_ne=0)
            
    def __checkEdgeCache__(self, edgeType):
    
        # check if re-caching is required
        if self.edgesAll is None:
            self.__reCacheEdges__()
                
        if edgeType is not None:   
            # check if re-caching is required
            if edgeType not in self.edgesByType.keys():
                self.__reCacheEdges__(edgeType)    
            
    def getEdgeValues(self, prop, edgeType=None):
        """ 
        privat function to access the values of pre-cached edges
        if necessary the edges are re-cached.
        """
        # check if re-caching is required
        self.__checkEdgeCache__(edgeType)
        
        if edgeType is None:
                       
            edges = self.edgesAll
            return edges[prop], edges
        else:        

            edges = self.edgesByType[edgeType]
            return edges[prop], edges              

    def setEdgeValues(self, prop, values, edgeType=None):
        """ 
        privat function to access the values of pre-cached edges
        if necessary the edges are re-cached.
        """
        # check if re-caching is required
        self.__checkEdgeCache__(edgeType)
        
        if edgeType is None:

            
            edges = self.edgesAll
            edges[prop] = values
        else:                       
            edges = self.edgesByType[edgeType]
            edges[prop] = values

    def getEdges(self, edgeType=None):
        """ 
        privat function to access the values of pre-cached edges
        if necessary the edges are re-cached.
        """
        # check if re-caching is required
        self.__checkEdgeCache__(edgeType)
        
        if edgeType is None:
            
            return self.edgesAll
        else:        
                
            return self.edgesByType[edgeType]

    def getPeerValues(self, prop, edgeType=None):
        # check if re-caching is required
        self.__checkPeerCache__(edgeType)
    
        if edgeType is None:

            return self.peersAll[prop], self.peersAll
        else:        
            return self.peersByType[edgeType][prop], self.peersByType[edgeType]                
    
    def setPeerValues(self, prop, values, edgeType=None):
        # check if re-caching is required
        self.__checkPeerCache__(edgeType)
        
        if edgeType is None:
            self.peersAll[prop] = values
        else:
            self.peersByType[edgeType][prop] = values
            
    def getPeers(self, edgeType=None):
        # check if re-caching is required
        self.__checkPeerCache__(edgeType)
    
        if edgeType is None:

            return self.peersAll
        else:        
            return self.peersByType[edgeType]  
    
    def getPeerIDs(self, edgeType=None):

        self.__checkPeerCache__(edgeType)
        
        if edgeType is None:
            return self.peersAll.indices
        else:        
            return self.peersByType[edgeType].indices
           
    def resetPeerCache(self,edgeType=None):
        self.peersAll = None
        if edgeType is None:
            self.peersByType = dict()
        else:
            try:
                del self.peersByType[edgeType]
            except:
                pass    

    def resetEdgeCache(self,edgeType=None):
        self.edgesAll = None
        if edgeType is None:
            self.edgesByType = dict()
        else:
            try:
                del self.edgesByType[edgeType]
            except:
                pass
            
    
################ ENTITY CLASS #########################################    
# general ABM entity class for objects connected with the graph

class Entity():

    
        
    def __init__(self, world, nID = None, **kwProperties):
        nodeType =  world.graph.class2NodeType[self.__class__]
        self.graph  = world.graph
        self.gID    = self.__getGlobID__(world)
     
        
        # create instance from existing node
        if nID is not None:
            
            self.nID = nID
            self.node = self.graph.vs[nID]
            #print 'nID:' + str(nID) + ' gID: ' + str(self.node['gID'])
            self.gID = self.node['gID']
            return
        
        # create instance newly
        
        self.nID, self.node = world.addVertex(nodeType, self.gID, **kwProperties)
        
            
        if world.caching:
            self.cache  = Cache(self.graph, self.nID)

            # definition of access functions
            self.getPeerValues = self.cache.getPeerValues
            self.setPeerValues = self.cache.setPeerValues
            self.getPeers = self.cache.getPeers
            self.getPeerIDs = self.cache.getPeerIDs
            
            self.getEdgeValues = self.cache.getEdgeValues
            self.setEdgeValues = self.cache.setEdgeValues
            self.getEdges = self.cache.getEdges
            
        else:
            self.cache = None
                   


    
    def setPeerValues(self, prop, values, nodeType=None):
        nodeDict = self.node.neighbors(mode='out')
        if nodeType is None:
            neighbourIDs = [node.index for node in nodeDict]
        else:
            neighbourIDs = [node.index for node in nodeDict if node['type'] == nodeType]
            
        for node in nodeDict:
            self.graph.vs[neighbourIDs][prop] = values

    def getPeerIDs(self, edgeType=None):
        eList  = self.graph.incident(self.nID,mode="out")
    
        if edgeType is not None:
            edges = self.graph.es[eList].select(type=edgeType)
        else:
            edges = self.graph.es[eList].select(type_ne=0)
            
        return [edge.target for edge in edges]
    
    def getPeerSeq(self, edgeType=None):
        return self.graph.vs[self.getPeerIDs(edgeType)]
         
    def getPeerValues(self, prop, edgeType=None):
        return self.getPeerSeq(edgeType)[prop], self.getPeerSeq(edgeType)
        
    
    def getEdgeValues(self, prop, edgeType=None):
        """ 
        privat function to access the values of  edges
        """
        eList  = self.graph.incident(self.nID,mode="out")
    
        if edgeType is not None:
            edges = self.graph.es[eList].select(type=edgeType)
        else:
            edges = self.graph.es[eList].select(type_ne=0)
    
        return edges[prop], edges
    
    def setEdgeValues(self, prop, values, edgeType=None):
        """ 
        privat function to access the values of  edges
        """
        eList  = self.graph.incident(self.nID,mode="out")

        if edgeType is not None:
            edges = self.graph.es[eList].select(type=edgeType)
        else:
            edges = self.graph.es[eList].select(type_ne=0)

        edges[prop] = values
        
    def getEdges(self, edgeType=None):
        """ 
        privat function to access the values of  edges
        """
        eList  = self.graph.incident(self.nID,mode="out")

        if edgeType is not None:
            edges = self.graph.es[eList].select(type=edgeType)
        else:
            edges = self.graph.es[eList].select(type_ne=0)

        return edges
    

    def getNeigbourhood(self, order):
        neigIDList = self.graph.neighborhood(self.nID, order)
        neigbours = []
        for neigID in neigIDList:
            neigbours.append(self.graph.vs[neigID])
        return neigbours, neigIDList
     
        
    def queueConnection(self, friendID, edgeType, **kwpropDict):
        kwpropDict.update({'type': edgeType})
        self.queue.addEdge(self.nID,friendID, **kwpropDict)    


    def addConnection(self, friendID, edgeType, **kwpropDict):
        kwpropDict.update({'type': edgeType})
        self.graph.add_edge(self.nID,friendID, **kwpropDict)  
        if self.cache:
            self.cache.edgesALL = None
            self.cache.peersALL = None
            del self.cache.edgesByType[edgeType] 
            self.cache.peersByType = dict()  # TODO less efficient

    def remConnection(self, friendID=None, edgeID=None):
    
        if edgeID is None and friendID:
            eID = self.graph.get_eid(self.nID,friendID)
        
        edgeType = self.graph.es[eID]['type']
        self.graph.es[eID]['type'] = 0 # inactive
        if self.cache:
            self.cache.edgesALL = None
            del self.cache.edgesByType[edgeType] 
            self.cache.peersByType = dict()

    def remConnections(self, friendIDs=None, edgeIDs=None):
        
        if edgeIDs is None and friendIDs:
            
            eIDs = [ self.graph.get_eid(self.nID,friendID) for friendID in friendIDs]

        edgeTypes = np.unique(np.asarray(self.graph.es[eIDs]['type']))
        self.graph.es[eIDs]['type'] = 0           
            
        # inactive
        if self.cache:
            self.cache.edgesALL = None
            for edgeType in edgeTypes:
                del self.cache.edgesByType[edgeType]
                del self.cache.peersByType[edgeType]
        

    def setValue(self,prop,value):
        self.node[prop] = value
        
    def getValue(self,prop):
        return self.node[prop]
    
    def addValue(self, prop, value, idx = None):
        if idx is None:
            self.node[prop] += value
        else:
            self.node[prop][idx] += value
    
    def delete(self, world):
        
        
        #self.graph.delete_vertices(nID) # not really possible at the current igraph lib
        # Thus, the node is set to inactive and removed from the iterator lists
        # This is due to the library, but the problem is general and pose a challenge.
        world.graph.vs[self.nID]['type'] = 0 #set to inactive
        
        
        world.deRegisterNode()
        
        # get all edges - in and out        
        eIDList  = self.graph.incident(self.nID)
        #set edges to inactive     
        self.graph[eIDList]['type'] = 0 


    def register(self, world, parentEntity=None, edgeType=None, ghost=False):
        nodeType = world.graph.class2NodeType[self.__class__]
        world.registerNode(self, nodeType, ghost)

        if parentEntity is not None:
            self.mpiPeers = parentEntity.registerChild(world, self, edgeType)


        
class Agent(Entity):
    
    def __getGlobID__(self,world):
        return world.globIDGen.next()
        
    def __init__(self, world, **kwProperties):
        if 'nID' not in kwProperties.keys():
            nID = None
        else:
            nID = kwProperties['nID']
        Entity.__init__(self, world, nID, **kwProperties)
        self.mpiOwner =  int(world.mpi.rank)
        
    def registerChild(self, world, entity, edgeType):
        if edgeType is not None:
            #print edgeType
            world.addEdge(entity.nID,self.nID, type=edgeType)         
        entity.loc = self
        
        if len(self.mpiPeers) > 0: # node has ghosts on other processes
            for mpiPeer in self.mpiPeers:
                #print 'adding node ' + str(entity.nID) + ' as ghost'
                nodeType = world.graph.class2NodeType[entity.__class__]
                world.mpi.queueSendGhostNode( mpiPeer, nodeType, entity, self)
        
        return self.mpiPeers
        

    def getLocationValue(self,prop):
    
        return self.loc.node[prop] 
    

    def move(self):
        """ not yet implemented"""
        pass
        
            
class GhostAgent(Entity):

    def __init__(self, world, owner, nID=None, **kwProperties):
        Entity.__init__(self, world, nID, **kwProperties)
        self.mpiOwner =  int(owner)
        
    def register(self, world, parentEntity=None, edgeType=None):
        Entity.register(self, world, parentEntity, edgeType, ghost= True)
        
        
    def __getGlobID__(self,world):
        
        return None # global ID need to be acquired via MPI communication
        
    def getLocationValue(self,prop):
    
        return self.loc.node[prop] 
    


    def registerChild(self, world, entity, edgeType):
        world.addEdge(entity.nID,self.nID, type=edgeType)        
            
################ LOCATION CLASS #########################################      
class Location(Entity):

    def __getGlobID__(self,world):
        return world.globIDGen.next()
    
    def __init__(self, world, **kwProperties):
        if 'nID' not in kwProperties.keys():
            nID = None
        else:
            nID = kwProperties['nID']
        
        
        Entity.__init__(self,world, nID, **kwProperties)
        self.mpiOwner = int(world.mpi.rank)
        self.mpiPeers = list()
        
    

    def registerChild(self, world, entity, edgeType=None):
        world.addEdge(entity.nID,self.nID, type=edgeType)               
        entity.loc = self        
        
        if len(self.mpiPeers) > 0: # node has ghosts on other processes
            for mpiPeer in self.mpiPeers:
                #print 'adding node ' + str(entity.nID) + ' as ghost'
                nodeType = world.graph.class2NodeType[entity.__class__]
                world.mpi.queueSendGhostNode( mpiPeer, nodeType, entity, self)
        
        return self.mpiPeers
        
class GhostLocation(Entity):
    
    def __getGlobID__(self,world):
        
        return None
    
    def __init__(self, world, owner, nID=None, **kwProperties):

        Entity.__init__(self,world, nID, **kwProperties)
        #self.graph.vs[self.nID]['pos']= (xPos,yPos)
        self.mpiOwner = int(owner)
        self.queuing = world.queuing

    def register(self, world, parentEntity=None, edgeType=None):
        Entity.register(self, world, parentEntity, edgeType, ghost= True)
        
    def registerChild(self, world, entity, edgeType=None):
        world.addEdge(entity.nID,self.nID, type=edgeType)              
        entity.loc = self        
        
################ WORLD CLASS #########################################            

class World:
    #%% World sub-classes
    
    class Globals(dict):
        """ This class manages global variables that are assigned on all processes
        and are synced via mpi. Global variables need to be registered together with
        the aggregation method they ase synced with, .e.g. sum, mean, min, max,...
        
        #TODO
        - implement mean, deviation, std as reduce operators
        
        
        """
        def __init__(self, world):
            
            self.comm = world.mpi.comm
            #self.dprint = world.dprint
            
            # simple reductions
            self.reduceDict = dict()
            self.operations = dict()
            
            self.operations['sum'] = MPI.SUM
            self.operations['prod'] = MPI.PROD
            self.operations['min'] = MPI.MIN
            self.operations['max'] = MPI.MAX
            
            #staticical reductions/aggregations
            self.statsDict      = dict()
            self.values         = dict()
            self.nValues        = dict()
            self.statOperations = dict()
            self.statOperations['mean'] = np.mean
            self.statOperations['std'] = np.std
            self.statOperations['var'] = np.std
            
            
            #self.operations['std'] = MPI.Op.Create(np.std)
            
        #%% simple global reductions       
        def registerValue(self, globName, value, reduceType):
            self[globName] = value
            if reduceType not in self.reduceDict.keys():
                self.reduceDict[reduceType] = list()
            self.reduceDict[reduceType].append(globName)
            
        def syncReductions(self):

            for redType in self.reduceDict.keys():
                
                op = self.operations[redType]
                #print op
                for globName in self.reduceDict[redType]:
                    #print globName
                    self[globName] = self.comm.allreduce(self[globName],op)

    
        #%% statistical global reductions/aggregations
        def registerStat(self, globName, values, statType):
            #statfunc = self.statOperations[statType]
            
            assert statType in ['mean', 'std','var']
            
            
            if not isinstance(values, (list, tuple,np.ndarray)):
                values = [values]
            values = np.asarray(values)
            
            
            self.values[globName]   = values
            self.nValues[globName] = len(values)  
            if statType == 'mean':
                self[globName]          = np.mean(values)
            elif statType == 'std':
                self[globName]          = np.std(values) 
            elif statType == 'var':
                self[globName]          = np.var(values) 

            if statType not in self.statsDict.keys():
                self.statsDict[statType] = list()
            self.statsDict[statType].append(globName)
        
        def updateStatValues(self, globName, values):
            self.values[globName]   = values
            self.nValues[globName] = len(values)    
            
        def syncStats(self):
            for redType in self.statsDict.keys():
                if redType == 'mean':
                    
                    for globName in self.statsDict[redType]:
                        tmp = [(np.mean(self.values[globName]), self.nValues[globName])]* self.comm.size # out data list  of (mean, size)
                        
                        tmp = np.asarray(self.comm.alltoall(tmp))
                        
                        #self.dprint('var[tmp] after alltoall:' + str( tmp ))
                        globValue = np.sum(np.prod(tmp,axis=1)) # means * size
                        globSize = np.sum(tmp[:,1])             # sum(size)
                        self[globName] = globValue/ globSize    # glob mean
                    
                elif redType == 'std':
                    for globName in self.statsDict[redType]:
                        
                        locSTD = [np.std(self.values[globName])] * self.comm.size
                        locSTD = np.asarray(self.comm.alltoall(locSTD))
                        lg.debug('loc std: ' + str(locSTD))
                        
                        tmp = [(np.mean(self.values[globName]), self.nValues[globName])]* self.comm.size # out data list  of (mean, size)
                        tmp = np.asarray(self.comm.alltoall(tmp))
                        
                        locMean = tmp[:,0]
                        lg.debug('loc mean: ' + str(locMean))
                        
                        locNVar = tmp[:,1]
                        lg.debug('loc number of var: ' + str(locNVar))
                        
                        globMean = np.sum(np.prod(tmp,axis=1)) / np.sum(locNVar)
                        lg.debug('global mean: ' + str( globMean ))
                        
                        diffSqrMeans = (locMean - globMean)**2
                        
                        deviationOfMeans = np.sum(locNVar * diffSqrMeans)
                        
                        globVariance = (np.sum( locNVar * locSTD**2) + deviationOfMeans) / np.sum(locNVar)
                        
                        self[globName] = np.sqrt(globVariance)

                elif redType == 'var':
                    for globName in self.statsDict[redType]:
                        
                        locSTD = [np.std(self.values[globName])]* self.comm.size
                        locSTD = np.asarray(self.comm.alltoall(locSTD))
                        #print 'loc std: ',locSTD
                        
                        tmp = [(np.mean(self.values[globName]), self.nValues[globName])]* self.comm.size # out data list  of (mean, size)
                        tmp = np.asarray(self.comm.alltoall(tmp))
                        
                        locMean = tmp[:,0]
                        #print 'loc mean: ', locMean
                        
                        locNVar = tmp[:,1]
                        #print 'loc number of var: ',locNVar
                        
                        globMean = np.sum(np.prod(tmp,axis=1)) / np.sum(locNVar)
                        #print 'global mean: ', globMean
                        
                        diffSqrMeans = (locMean - globMean)**2
                        
                        deviationOfMeans = np.sum(locNVar * diffSqrMeans)
                        
                        globVariance = (np.sum( locNVar * locSTD**2) + deviationOfMeans) / np.sum(locNVar)
                        
                        self[globName] = globVariance

        def sync(self):

            self.syncStats()            
            self.syncReductions()
            
                
    class IO():  
        
        class synthInput():
            """ MPI conform loading of synthetic population data"""
            pass # ToDo
        
        
        class Record():
            """ This calls manages the translation of different graph attributes to
            the output format as a numpy array. Vectora of values automatically get
            assigned the propper matrix dimensions and indices. 
            
            So far, only integer and float are supported
            """
            def __init__(self, nAgents, agIds, nAgentsGlob, loc2GlobIdx, nodeType, timeStepMag):
                self.ag2FileIdx = agIds
                self.nAgents = nAgents
                self.nAttr = 0
                self.attributeList = list()
                self.attrIdx = dict()
                self.header = list()
                self.timeStep = 0
                self.nAgentsGlob = nAgentsGlob
                self.loc2GlobIdx = loc2GlobIdx
                self.nodeType    = nodeType
                self.timeStepMag = timeStepMag
                
            
            def addAttr(self, name, nProp):
                attrIdx = range(self.nAttr,self.nAttr+nProp)
                self.attributeList.append(name)
                self.attrIdx[name] = attrIdx
                self.nAttr += len(attrIdx)
                self.header += [name] * nProp
                
            def initStorage(self):
                self.data = np.zeros([self.nAgents,self.nAttr ], dtype=np.float32)
                
            def addData(self, timeStep, graph):
                self.timeStep = timeStep
                for attr in self.attributeList:
                    if len(self.attrIdx[attr]) == 1:
                        self.data[:,self.attrIdx[attr]] = np.expand_dims(graph.vs[self.ag2FileIdx][attr],1)
                    else:
                        self.data[:,self.attrIdx[attr]] = graph.vs[self.ag2FileIdx][attr]
                            
            def writeData(self, h5File):
                #print self.header
                path = '/' + str(self.nodeType)+ '/' + str(self.timeStep).zfill(self.timeStepMag)
                #print 'IO-path: ' + path
                self.dset = h5File.create_dataset(path, (self.nAgentsGlob,self.nAttr), dtype='f')
                self.dset[self.loc2GlobIdx[0]:self.loc2GlobIdx[1],] = self.data
        
        #%% Init of the IO class                 
        def __init__(self, world, nSteps, outputPath = ''): # of IO
            import h5py  
            self.outputPath  = outputPath
            self.graph       = world.graph
            #self.timeStep   = world.timeStep
            self.h5File      = h5py.File(outputPath + '/nodeOutput.hdf5', 
                                         'w', 
                                         driver='mpio', 
                                         comm=world.mpi.comm, 
                                         libver='latest')
            self.comm        = world.mpi.comm
            self.outData     = dict()
            self.timeStepMag = int(np.ceil(np.log10(nSteps)))
              
            
        def initNodeFile(self, world, nodeTypes):
            """ Initialized the internal data structure for later I/O""" 
            lg.info('start init of the node file')
            
            for nodeType in nodeTypes:
                world.mpi.comm.Barrier()
                tt = time.time()
                lg.info(' NodeType: ' +str(nodeType))
                #if world.mpi.comm.rank == 0:
                self.h5File.create_group(str(nodeType))
                
                    #group = self.latMinMax = node._f_getattr('LATMINMAX')
                lg.info( 'group created in ' + str(time.time()-tt)  + ' seconds'  )
                tt = time.time()
                
                nAgents = len(world.nodeDict[nodeType])
                self.nAgentsAll = np.empty(1*self.comm.size,dtype=np.int)
                
                self.nAgentsAll = self.comm.alltoall([nAgents]*self.comm.size)    
                
                lg.info( 'nAgents exchanged in  ' + str(time.time()-tt)  + ' seconds'  )
                tt = time.time()
                
                lg.info('Number of all agents' + str( self.nAgentsAll ))
                
                nAgentsGlob = sum(self.nAgentsAll)
                cumSumNAgents = np.zeros(self.comm.size+1).astype(int)
                cumSumNAgents[1:] = np.cumsum(self.nAgentsAll)
                loc2GlobIdx = (cumSumNAgents[self.comm.rank], cumSumNAgents[self.comm.rank+1])
                #print loc2GlobIdx
                lg.info( 'loc2GlobIdx exchanged in  ' + str(time.time()-tt)  + ' seconds'  )
                tt = time.time()
                
                rec = self.Record(nAgents, world.nodeDict[nodeType], nAgentsGlob, loc2GlobIdx, nodeType, self.timeStepMag)
                self.attributes = world.graph.nodeProperies[nodeType][:]
                self.attributes.remove('type')
                


                lg.info( 'record created in  ' + str(time.time()-tt)  + ' seconds')
                
                
                for attr in self.attributes:
                    #check if first property of first entity is string
                    entProp = world.graph.vs[rec.ag2FileIdx[0]][attr]
                    if not isinstance(entProp,str):
                        
                        
                        if isinstance(entProp,(list,tuple)):
                            # add mutiple fields
                            nProp = len(self.graph.vs[rec.ag2FileIdx[0]][attr])   
                        else:
                            #add one field
                            nProp = 1
                            
                        rec.addAttr(attr, nProp)                         
                
                tt = time.time()
                # allocate storage
                rec.initStorage()
                self.outData[nodeType] = rec
                lg.info( 'storage allocated in  ' + str(time.time()-tt)  + ' seconds'  )
                
        def gatherNodeData(self, timeStep):                
            """ Transfers data from the graph to the I/O storage"""
            for typ in self.outData.keys():
                self.outData[typ].addData(timeStep, self.graph)
        
        def writeDataToFile(self):
            """ Writing data to hdf5 file"""
            for typ in self.outData.keys():
                self.outData[typ].writeData(self.h5File)
        
        def initEdgeFile(self, edfeTypes):
            pass
        
        def finalizeAgentFile(self):
            self.h5File.close()
            lg.info( 'Agent file closed')
            from class_auxiliary import saveObj
            
            for nodeType in self.outData.keys():
                record = self.outData[nodeType]
                #np.save(self.para['outPath'] + '/agentFile_type' + str(typ), self.agentRec[typ].recordNPY, allow_pickle=True)
                saveObj(record.attrIdx, (self.outputPath + '/attributeList_type' + str(nodeType)))
    class Mpi():
        
        
        def __init__(self, world, mpiComm=None):
            
            self.world = world
            if mpiComm is None:
                self.comm = MPI.COMM_WORLD
            else:
                self.comm = mpiComm
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
                
            self.peers    = list()     # list of ranks of all processes that have ghost duplicates of this process   
            
            self.ghostNodeQueue = dict()
            #self.ghostEdgeQueue = dict()
            
            self.ghostNodeSend  = dict()     # ghost vertices on this process that receive information
            self.ghostNodeRecv  = dict()     # vertices on this process that provide information to ghost nodes on other process
            self.buffer         = dict()
            self.messageSize    = dict()
            self.sendReqList    = list()
            #self.ghostEdgeIn  = dict()     # ghost edves on this process that receive information
            #self.ghostEdgeOut = dict()     # edges on this process that provide information to ghost nodes on other process
            
            self.reduceDict = dict()
            world.send = self.comm.send
            world.recv = self.comm.recv    
            
            world.isend = self.comm.isend    
            world.irecv = self.comm.irecv
            
            self.__clearBuffer__()
        
        #%% Privat functions
        def __clearBuffer__(self):
            self.a2aBuff = []
            for x in range(self.comm.size):
                self.a2aBuff.append([])
            #print self.a2aBuff
            #print len(self.a2aBuff)
            
        def __add2Buffer__(self, mpiPeer, data):
            #print 'adding to:',mpiPeer
            #print 'mpiPeer',mpiPeer
            self.a2aBuff[mpiPeer].append(data)
            
        def __all2allSync__(self):
            recvBuffer = self.comm.alltoall(self.a2aBuff)
            self.__clearBuffer__()
            return recvBuffer
        
        
        def __packData__(self, nodeType, mpiPeer, nodeSeq, propList, connList=None):

            nNodes = len(nodeSeq)
            dataPackage = list()
            dataPackage.append((nNodes,nodeType))
            for prop in propList:
                dataPackage.append(nodeSeq[prop])                            
            
            if connList is not None:
                dataPackage.append(connList)

                        
            return dataPackage
  
        def initCommunicationViaLocations(self, ghostLocationList, locNodeType): 
            
            tt = time.time()
            # acquire the global IDs for the ghostNodes
            mpiRequest = dict()
            mpiReqIDList = dict()
            #print self.world.graph.IDArray
            for ghLoc in ghostLocationList:
                owner = ghLoc.mpiOwner
                #print owner
                x,y   = ghLoc.node['pos']
                if owner not in mpiRequest:
                    mpiRequest[owner]   = (list(), 'gID')
                    mpiReqIDList[owner] = list()
                    
                mpiRequest[owner][0].append( (x,y) ) # send x,y-pairs for identification
                mpiReqIDList[owner].append(ghLoc.nID)
            #print 'rank' + str(self.rank) + 'mpiReqIDList: ' + str(mpiReqIDList)
            
            for mpiDest in mpiRequest.keys():
                
                if mpiDest not in self.peers:
                    self.peers.append(mpiDest)
                
                # send request of global IDs
                #print str(self.rank) + ' asks from ' + str(mpiDest) + ' - ' + str(mpiRequest[mpiDest])
                #self.comm.send(mpiRequest[mpiDest], dest=mpiDest)
                    self.__add2Buffer__(mpiDest, mpiRequest[mpiDest])
            
            #print 'requestOut:'
            #pprint(self.a2aBuff)
            requestIn = self.__all2allSync__()
            #print 'requestIn:'
            #pprint(requestIn)
            
            
            for mpiDest in mpiRequest.keys():
                
                self.ghostNodeRecv[locNodeType, mpiDest] = self.world.graph.vs[mpiReqIDList[mpiDest]]
                
                # receive request of global IDs
                #incRequest = self.comm.recv(source=mpiDest)
                incRequest = requestIn[mpiDest][0]
                #pprint(incRequest)
                iDList = [int(self.world.graph.IDArray[xx, yy]) for xx, yy in incRequest[0]]
                #print str(self.rank) + ' - idlist:' + str(iDList)
                self.ghostNodeSend[locNodeType, mpiDest] = self.world.graph.vs[iDList]
                #self.ghostNodeOut[locNodeType, mpiDest] = self.world.graph.vs[iDList]
                #print str(self.rank) + ' - gIDs:' + str(self.ghostNodeOut[locNodeType, mpiDest]['gID'])
                
                for entity in [self.world.entList[i] for i in iDList]:
                    entity.mpiPeers.append(mpiDest)
                
                # send requested global IDs
                #print str(self.rank) + ' sends to ' + str(mpiDest) + ' - ' + str(self.ghostNodeOut[locNodeType, mpiDest][incRequest[1]])
                
                self.__add2Buffer__(mpiDest,self.ghostNodeSend[locNodeType, mpiDest][incRequest[1]])
            
            requestRecv = self.__all2allSync__()
            
            for mpiDest in mpiRequest.keys():
                #self.comm.send(self.ghostNodeOut[locNodeType, mpiDest][incRequest[1]], dest=mpiDest)
                #receive requested global IDs
                globIDList = requestRecv[mpiDest][0]
                lg.info( 'receiving globIDList:' + str(globIDList))
                #print 'localDList:' + str(self.ghostNodeIn[locNodeType, mpiDest].indices)
                self.ghostNodeRecv[locNodeType, mpiDest]['gID'] = globIDList
                for nID, gID in zip(self.ghostNodeRecv[locNodeType, mpiDest].indices, globIDList):
                    #print nID, gID
                    self.world.glob2loc[gID] = nID
                    self.world.loc2glob[nID] = gID
                #self.world.mpi.comm.Barrier()
            lg.info( 'Mpi commmunication required: ' + str(time.time()-tt) + ' seconds')          
            

        #%% Nodes            
        def queueSendGhostNode(self, mpiPeer, nodeType, entity, parentEntity):
            
            if (nodeType, mpiPeer) not in self.ghostNodeQueue.keys():
                self.ghostNodeQueue[nodeType, mpiPeer] = dict()
                self.ghostNodeQueue[nodeType, mpiPeer]['nIds'] = list()
                self.ghostNodeQueue[nodeType, mpiPeer]['conn'] = list()
            
            self.ghostNodeQueue[nodeType, mpiPeer]['nIds'].append(entity.nID)
            self.ghostNodeQueue[nodeType, mpiPeer]['conn'].append(parentEntity.gID)

        

        def sendRecvGhostNodes(self, world):
            
            #%%Packing of data
            for nodeType, mpiPeer in sorted(self.ghostNodeQueue.keys()):
                    
                #get size of send array
                IDsList= self.ghostNodeQueue[(nodeType, mpiPeer)]['nIds']
                connList = self.ghostNodeQueue[(nodeType, mpiPeer)]['conn']


    
                nodeSeq = world.graph.vs[IDsList]
                
                # setting up ghost out Comm
                self.ghostNodeSend[nodeType, mpiPeer] = nodeSeq
                propList = world.graph.nodeProperies[nodeType][:]
                #print propList
                dataPackage = self.__packData__( nodeType, mpiPeer, nodeSeq,  propList, connList)
                self.__add2Buffer__(mpiPeer, dataPackage)
#                if mpiPeer not in announcements:
#                    announcements[mpiPeer] = list()
#                announcements[mpiPeer].append((nodeType, getsize(packageDict)))            
#            
#            self.__sendDataAnnouncement__(announcements) #only required for initial communication
            recvBuffer = self.__all2allSync__()

            #pprint(recvBuffer[mpiPeer])
            
            for mpiPeer in self.peers:
                if len(recvBuffer[mpiPeer]) > 0: # will receive a message
                    pass  
                
                for dataPackage in recvBuffer[mpiPeer]:
      
            #%% create ghost agents from dataDict 
            
                    nNodes, nodeType = dataPackage[0]
                    
                    nIDStart= world.graph.vcount()
                    nIDs = range(nIDStart,nIDStart+nNodes)
                    world.graph.add_vertices(nNodes)
                    nodeSeq = world.graph.vs[nIDs]
                    
                    # setting up ghostIn communicator
                    self.ghostNodeRecv[nodeType, mpiPeer] = nodeSeq
                    
                    propList = world.graph.nodeProperies[nodeType][:]
                    #print propList
                    #propList.remove('nID')
                    
                
                    for i, prop in enumerate(propList):
                        nodeSeq[prop] = dataPackage[i+1]
                
                    gIDsCells = dataPackage[-1]
                    #print 'creating ' + str(len(nIDs)) + ' ghost agents of type' + str(nodeType)
                    for nID, gID in zip(nIDs, gIDsCells):
                        
                        GhostAgentClass = world.graph.nodeType2Class[nodeType][1]
                        
                        agent = GhostAgentClass(world, mpiPeer, nID=nID)
                        #print 'creating ghost with nID: ' + str(nID)
                        #world.registerNode(agent,nodeType) 
                        
                        parentEntity = world.entDict[world.glob2loc[gID]]
                        edgeType = world.graph.nodeTypes2edgeTypes[parentEntity.node['type'], nodeType]
    
                        #print 'edgeType' + str(edgeType)
                        agent.register(world, parentEntity, edgeType)
                        #cell.registerEntityAtLocation(world, agent, edgeType)
                    # create entity with nodes (queing)
                    # deque
                    # set data from buffer
            
            #self.__clearSendRequests__()
            
        
        
        def sendRecvGhostUpdate(self, nodeTypeList='all', propertyList='all'):
            
            #dataDict = dict()

            for (nodeType, mpiPeer) in self.ghostNodeSend.keys():
                if nodeTypeList == 'all' or nodeType in nodeTypeList:
                    nodeSeq = self.ghostNodeSend[nodeType, mpiPeer]
                
                    if propertyList == 'all':
                        propertyList = self.world.graph.nodeProperies[nodeType][:]
                        propertyList.remove('gID')
                        
                    dataPackage = self.__packData__(nodeType, mpiPeer, nodeSeq,  propertyList, connList=None)
                    self.__add2Buffer__(mpiPeer, dataPackage)
            recvBuffer = self.__all2allSync__()
            

            for mpiPeer in self.peers:
                if len(recvBuffer[mpiPeer]) > 0: # will receive a message
                      
                
                    for dataPackage in recvBuffer[mpiPeer]:
                        mNodes, nodeType = dataPackage[0]
                        
                        if propertyList == 'all':
                            propertyList= self.world.graph.nodeProperies[nodeType][:]
                            #print propertyList
                            propertyList.remove('gID')
                       
                        nodeSeq = self.ghostNodeRecv[nodeType, mpiPeer]
                        for i, prop in enumerate(propertyList):
                            nodeSeq[prop] = dataPackage[i+1]
           
            #self.__clearSendRequests__()
            
        def updateGhostNodes(self, nodeTypeList= 'all', propertyList='all'):
            tt = time.time()
            
            self.sendRecvGhostUpdate(nodeTypeList, propertyList)
            #self.recvGhostUpdate(nodeTypeList, propertyList)
            
            lg.debug('Ghost update required: ' + str(time.time()-tt) + ' seconds')
            
        def all2all(self, value):
            if isinstance(value,int):            
                buff = np.empty(1*self.comm.size,dtype=np.int)
                buff = self.comm.alltoall([value]*self.comm.size)
            elif isinstance(value,float):
                buff = np.empty(1*self.comm.size,dtype=np.float)
                buff = self.comm.alltoall([value]*self.comm.size)
            elif isinstance(value,str):
                buff = np.empty(1*self.comm.size,dtype=np.str)
                buff = self.comm.alltoall([value]*self.comm.size)
            else:
                
                buff = self.comm.alltoall([value]*self.comm.size)                
                
            return buff
            
    #%% INIT WORLD            
    def __init__(self, 
                 simNo,
                 outPath,
                 spatial=True, 
                 nSteps= 1, 
                 maxNodes = 1e6, 
                 debug = False, 
                 mpiComm=None, 
                 caching=True,
                 queuing=True):
    
        self.simNo    = simNo
        self.timeStep = 0
        self.para     = dict()
        self.spatial  = spatial
        self.maxNodes = int(maxNodes)
        self.globIDGen = self.__globIDGen__()
        self.nSteps   = nSteps
        self.debug    = debug
        
        self.para     = dict()
        self.queuing = queuing  # flag that indicates the vertexes and edges are queued and not added immediately
        self.caching = caching  # flat that indicate that edges and peers are cached for faster access
               
        # GRAPH
        self.graph    = WorldGraph(self, directed=True)
        self.para['outPath'] = outPath
     

        # setting additional debug output
#        if self.debug:
#            # inint pprint as additional output
#            import pprint
#            def debugPrint(*argv):
#                if not isinstance(argv, str):
#                    argv = str(argv)
#                pprint.pprint('DEBUG: ' + argv)
#            self.dprint = debugPrint
#        else:
#            # init dummy that does nothing
#            def debugDummy(*argv):
#                pass
#            self.dprint = debugDummy
        
        
        
        # list of types
        self.graph.nodeTypes = list()
        self.graph.edgeTypes = list()
        self.graph.nodeTypes2edgeTypes = dict()
        
        # dict of classes to init node automatically
        self.graph.nodeType2Class = dict()
        self.graph.class2NodeType = dict()
        
        # list of properties per type
        self.graph.nodeProperies = dict()
        self.graph.edgeProperies = dict()
        
        # queues
        if self.queuing:
            self.queue      = Queue(self)
            self.addEdge    = self.queue.addEdge
            self.addEdges   = self.queue.addEdges    
            self.addVertex  = self.queue.addVertex
        else:
            self.addEdge    = self.graph.add_edge
            self.addEdges   = self.graph.add_edges   
            self.delEdges    = self.graph.delete_edges
            self.addVertex  = self.graph.add_vertex
        # making global functions available
        
           
        

        # MPI communication
        self.mpi = self.Mpi(self, mpiComm=mpiComm)
        if self.mpi.comm.rank == 0:
            self.isRoot = True
        else:
            self.isRoot = False
        



        # IO
        self.io = self.IO(self, nSteps, self.para['outPath'])
        
        # Globally synced variables
        self.glob     = self.Globals(self)
        
 
        # enumerations
        self.enums = dict()
        
        
        # node lists and dicts
        self.nodeDict       = dict()    
        self.ghostNodeDict  = dict()    
        
        self.entList   = list()
        self.entDict   = dict()
        self.locDict   = dict()
        
        self.glob2loc = dict()  # reference from global IDs to local IDs
        self.loc2glob = dict()  # reference from local IDs to global IDs

        # inactive is used to virtually remove nodes
        self.registerNodeType('inactiv', None, None)
        self.registerEdgeType('inactiv', None, None)        
        
        
        
# GENERAL FUNCTIONS
    def __globIDGen__(self):
        i = -1
        while i < self.maxNodes:
            i += 1
            yield (self.maxNodes*(self.mpi.rank+1)) +i


    def initSpatialLayer(self, rankArray, connList, nodeType, LocClassObject=Location, GhstLocClassObject=GhostLocation):
        """
        Auiliary function to contruct a simple connected layer of spatial locations.
        Use with  the previously generated connection list (see computeConnnectionList)
        
        """
        nodeArray = ((rankArray * 0) +1)
        #print rankArray
        IDArray = nodeArray * np.nan
        #print IDArray
        # spatial extend
        xOrg = 0
        yOrg = 0
        xMax = nodeArray.shape[0]
        yMax = nodeArray.shape[1]
        ghostLocationList = list()
        self.cellMapIdxList = list()

        # create vertices 
        for x in range(nodeArray.shape[0]):
            for y in range(nodeArray.shape[1]):

                # only add an vertex if spatial location exist     
                if not np.isnan(rankArray[x,y]) and rankArray[x,y] == self.mpi.rank:

                    loc = LocClassObject(self, pos= (x, y))
                    IDArray[x,y] = loc.nID
                    self.registerLocation(loc, x, y)          # only for real cells
                    #self.registerNode(loc,nodeType)     # only for real cells
                    loc.register(self)
        #print self.locDict
        # create ghost location nodes 
        for (x,y), loc in self.locDict.items():

            srcID = loc.nID
            for (dx,dy,weight) in connList:
            
                xDst = x + dx
                yDst = y + dy
                
                # check boundaries of the destination node
                if xDst >= xOrg and xDst < xMax and yDst >= yOrg and yDst < yMax:
                    

                    if np.isnan(IDArray[xDst,yDst]) and not np.isnan(rankArray[xDst,yDst]) and rankArray[xDst,yDst] != self.mpi.rank:  # location lives on another process
                        
                        loc = GhstLocClassObject(self, owner=rankArray[xDst,yDst], pos= (xDst, yDst))
                        #print 'rank: ' +  str(self.mpi.rank) + ' '  + str(loc.nID)
                        IDArray[xDst,yDst] = loc.nID
                        
                        #self.cellMapIdxList.append(np.ravel_multi_index((xDst,yDst), dims=IDArray.shape))
                        #print IDArray
                        # so far ghost nodes are not in entDict, nodeDict, entList
                        
                        self.registerNode(loc,nodeType,ghost=True)
                        ghostLocationList.append(loc)
        self.graph.IDArray = IDArray
        
        if self.queuing:
            self.queue.dequeueVertices(self)

        fullConnectionList = list()
        fullWeightList     = list()
        #print 'rank: ' +  str(self.locDict) 
        
        for (x,y), loc in self.locDict.items():

            srcID = loc.nID
            
            weigList = list()
            destList = list()
            connectionList = list()
            
            for (dx,dy,weight) in connList:
                
                xDst = x + dx
                yDst = y + dy
                
                # check boundaries of the destination node
                if xDst >= xOrg and xDst < xMax and yDst >= yOrg and yDst < yMax:                        

                    trgID = IDArray[xDst,yDst]
                    #assert 

                    if not np.isnan(trgID):
                        destList.append(int(trgID))
                        weigList.append(weight)
                        connectionList.append((int(srcID),int(trgID)))
    
            #normalize weight to sum up to unity                    
            sumWeig = sum(weigList)
            weig    = np.asarray(weigList) / sumWeig
            
            fullConnectionList.extend(connectionList)
            fullWeightList.extend(weig)

        eStart = self.graph.ecount()
        self.graph.add_edges(fullConnectionList)
        self.graph.es[eStart:]['type'] = 1
        self.graph.es[eStart:]['weig'] = fullWeightList
        
        self.mpi.initCommunicationViaLocations(ghostLocationList, nodeType)

    def iterEdges(self, edgeType):
        for i in range(self.graph.ecount()):
            if self.graph.es[i]['type'] == edgeType:
                yield self.graph.es[i]
    
    def iterEntRandom(self,nodeType, ghosts = False, random=True):
        """ Iteration over entities of specified type. Default returns 
        non-ghosts in random order.
        """
        if isinstance(nodeType,str):
            nodeType = self.types.index(nodeType)
            
        if ghosts:
            nodeDict = self.ghostNodeDict[nodeType]
        else:
            nodeDict = self.nodeDict[nodeType]
        
        if random:
            #print 'nodeDict' + str(nodeDict)
            #print self.entList
            shuffled_list = sorted(nodeDict, key=lambda x: np.random.random())
            return [self.entList[i] for i in shuffled_list]
        else:
            return  [self.entList[i] for i in nodeDict]
        
    def iterEntAndIDRandom(self, nodeType, ghosts = False, random=True):
        """ Iteration over entities of specified type and their IDs . Default returns 
        non-ghosts in random order.
        """
        if isinstance(nodeType,str):
            nodeType = self.types.index(nodeType)
        
        if ghosts:
            nodeDict = self.ghostnodeDict[nodeType]
        else:
            nodeDict = self.nodeDict[nodeType]
            
        if random:
            shuffled_list = sorted(nodeDict, key=lambda x: np.random.random())
            return  [(self.entList[i], i) for i in shuffled_list]
        else:
            return  [(self.entList[i], i) for i in nodeDict]

    def setNodeValues(self,prop,nodeType, value):
        self.graph.vs[self.nodeDict[nodeType]][prop] = value

    def getNodeValues(self,prop,nodeType):
        return np.asarray(self.graph.vs[self.nodeDict[nodeType]][prop])

    def getEntity(self,nodeID):
        return self.entDict[nodeID]
            
    def registerNodeType(self, typeStr, AgentClass, GhostAgentClass, propertyList = ['type', 'gID']):
           
        # type is an required property
        assert 'type' and 'gID' in propertyList
        
        nodeTypeIdx = len(self.graph.nodeTypes)
        self.graph.nodeTypes.append(typeStr)
        self.graph.nodeProperies[nodeTypeIdx] = propertyList
        # same nodeType for ghost and non-ghost
        self.graph.nodeType2Class[nodeTypeIdx]      = AgentClass, GhostAgentClass
        self.graph.class2NodeType[AgentClass]       = nodeTypeIdx 
        self.graph.class2NodeType[GhostAgentClass]  = nodeTypeIdx
        self.nodeDict[nodeTypeIdx]      = list()
        self.ghostNodeDict[nodeTypeIdx] = list()
        self.enums[typeStr] = nodeTypeIdx
        return nodeTypeIdx
        
    
    def registerEdgeType(self, typeStr,  nodeType1, nodeType2, propertyList = ['type']):
        assert 'type' in propertyList # type is an required property
        
        edgeTypeIdx = len(self.graph.edgeTypes)
        self.graph.edgeTypes.append(edgeTypeIdx)
        self.graph.nodeTypes2edgeTypes[nodeType1, nodeType2] = edgeTypeIdx
        self.graph.edgeProperies[edgeTypeIdx] = propertyList
        #self.graph.queue.addEdgeType(edgeTypeIdx, propertyList)
        self.enums[typeStr] = edgeTypeIdx            

        return  edgeTypeIdx            

    def registerNode(self, agent, typ, ghost=False):
        #print 'assert' + str((len(self.entList), agent.nID))
        assert len(self.entList) == agent.nID
        self.entList.append(agent)
        self.entDict[agent.nID] = agent
        self.glob2loc[agent.gID] = agent.nID
        self.loc2glob[agent.nID] = agent.gID
        
        if ghost:
            self.ghostNodeDict[typ].append(agent.nID)
        else:
            #print typ
            self.nodeDict[typ].append(agent.nID)

    def deRegisterNode(self):
        self.entList[agent.nID] = None
        del self.entDict[agent.nID]
        del self.glob2loc[agent.gID]
        del self.loc2glob[agent.gID]
        
    def registerLocation(self, location, x, y):
        
        self.locDict[x,y] = location
    
    def setParameters(self, parameterDict):
        for key in parameterDict.keys():
            self.para[key] = parameterDict[key]
            
  
    def view(self,filename = 'none', vertexProp='none', dispProp='gID', layout=None):
        try:
            import matplotlib.cm as cm
            
            # Nodes        
            if vertexProp=='none':
                colors = iter(cm.rainbow(np.linspace(0, 1, len(self.graph.nodeTypes)+1)))   
                colorDictNode = {}
                for i in range(len(self.graph.nodeTypes)+1):
                    hsv =  next(colors)[0:3]
                    colorDictNode[i] = hsv.tolist()
                nodeValues = (np.array(self.graph.vs['type']).astype(float)).astype(int).tolist()
            else:
                maxCars = max(self.graph.vs[vertexProp])
                colors = iter(cm.rainbow(np.linspace(0, 1, maxCars+1)))
                colorDictNode = {}
                for i in range(maxCars+1):
                    hsv =  next(colors)[0:3]
                    colorDictNode[i] = hsv.tolist()
                nodeValues = (np.array(self.graph.vs[vertexProp]).astype(float)).astype(int).tolist()    
            # nodeValues[np.isnan(nodeValues)] = 0
            # Edges            
            colors = iter(cm.rainbow(np.linspace(0, 1, len(self.graph.edgeTypes))))              
            colorDictEdge = {}  
            for i in range(len(self.graph.edgeTypes)):
                hsv =  next(colors)[0:3]
                colorDictEdge[i] = hsv.tolist()
            self.graph.vs["label"] = [str(y) for x,y in zip(self.graph.vs.indices, self.graph.vs[dispProp])]  
            
            #self.graph.vs["label"] = [str(x) + '->' + str(y) for x,y in zip(self.graph.vs.indices, self.graph.vs[dispProp])]  
            edgeValues = (np.array(self.graph.es['type']).astype(float)).astype(int).tolist()
            
            visual_style = {}
            visual_style["vertex_color"] = [colorDictNode[typ] for typ in nodeValues]
            visual_style["vertex_shape"] = list()        
            for vert in self.graph.vs['type']:
                if vert == 0:
                    visual_style["vertex_shape"].append('hidden')                
                elif vert == 1:
                        
                    visual_style["vertex_shape"].append('rectangle')                
                else:
                    visual_style["vertex_shape"].append('circle')     
            visual_style["vertex_size"] = list()  
            for vert in self.graph.vs['type']:
                if vert >= 3:
                    visual_style["vertex_size"].append(15)  
                else:
                    visual_style["vertex_size"].append(15)  
            visual_style["edge_color"]   = [colorDictEdge[typ] for typ in edgeValues]
            visual_style["edge_arrow_size"]   = [.5]*len(visual_style["edge_color"])
            visual_style["bbox"] = (900, 900)
            if layout==None:
                if filename  == 'none':
                    ig.plot(self.graph, layout='fr', **visual_style)    
                else:
                    ig.plot(self.graph, filename, layout='fr',  **visual_style )  
            else:
                if filename  == 'none':
                    ig.plot(self.graph,layout=layout,**visual_style)    
                else:
                    ig.plot(self.graph, filename, layout=layout, **visual_style )  
        except:
            pass
if __name__ == '__main__':
    
    earth = World(maxNodes = 1e2, nSteps = 10)
# 
    log_file  = open('out' + str(earth.mpi.rank) + '.txt', 'w')
    sys.stdout = log_file   
    earth.glob.registerValue('test' , np.asarray(earth.mpi.comm.rank),'max')
    earth.glob.registerStat('meantest', np.random.randint(5,size=3).astype(float),'mean')
    earth.glob.registerStat('stdtest', np.random.randint(5,size=2).astype(float),'std')
    print earth.glob['test']
    print earth.glob['meantest']
    print 'mean of values: ',earth.glob.values['meantest'],'-> local maen: ',earth.glob['meantest']
    print 'std od values:  ',earth.glob.values['stdtest'],'-> local std: ',earth.glob['stdtest']
    earth.glob.sync()
    print earth.glob['test']
    print 'global mean: ', earth.glob['meantest']
    print 'global std: ', earth.glob['stdtest']
    
    
#    exit()

    #log_file = open("message.log","w")
    import sys
    

    mpiRankLayer   = np.asarray([[0, 0, 0, 0, 1],
                              [np.nan, np.nan, np.nan, 1, 1]])
    
    #landLayer = np.load('rankMap.npy')
    connList = aux.computeConnectionList(1.5)
    #print connList
    _cell    = earth.registerNodeType('cell' , AgentClass=Location, GhostAgentClass= GhostLocation, 
                                      propertyList = ['type', 
                                                      'gID',
                                                      'pos', 
                                                      'value', 
                                                      'value2'])
    
    _ag      = earth.registerNodeType('agent', AgentClass=Agent   , GhostAgentClass= GhostAgent, 
                                      propertyList = ['type', 
                                                      'gID',
                                                      'pos',
                                                      'value3'])
    
    _cLocLoc = earth.registerEdgeType('cellCell', _cell, _cell)
    _cLocAg = earth.registerEdgeType('cellAgent', _cell, _ag)
    _cAgAg = earth.registerEdgeType('AgAg', _ag, _ag)
    
    earth.initSpatialLayer(mpiRankLayer, connList, _cell, Location, GhostLocation)
    #earth.mpi.initCommunicationViaLocations(ghostLocationList)
    
    for cell in earth.iterEntRandom(_cell):
        cell.node['value'] = earth.mpi.rank
        cell.node['value2'] = earth.mpi.rank+2
        
        if cell.node['pos'][0] == 0:
            x,y = cell.node['pos']
            agent = Agent(earth, value3=np.random.randn(),pos=(x+ np.random.randn()*.1,  y + np.random.randn()*.1))
            print 'agent.nID' + str(agent.nID)
            agent.register(earth, cell, _cLocAg)
            #cell.registerEntityAtLocation(earth, agent,_cLocAg)
            
    earth.queue.dequeueVertices(earth)
    earth.queue.dequeueEdges(earth)
#            if agent.node['nID'] == 10:
#                agent.addConnection(8,_cAgAg)
    
    #earth.mpi.syncNodes(_cell,['value', 'value2'])
    earth.mpi.updateGhostNodes([_cell])
    print earth.graph.vs.attribute_names()
    print str(earth.mpi.rank) + ' values' + str(earth.graph.vs['value'])
    print str(earth.mpi.rank) + ' values2: ' + str(earth.graph.vs['value2'])
    
    print earth.mpi.ghostNodeRecv
    print earth.mpi.ghostNodeSend
    
    print earth.graph.vs.attribute_names()
    
    print str(earth.mpi.rank) + ' ' + str(earth.nodeDict[_ag])
    
    print str(earth.mpi.rank) + ' SendQueue ' + str(earth.mpi.ghostNodeQueue)
    
    earth.mpi.sendRecvGhostNodes(earth)
    #earth.mpi.recvGhostNodes(earth)

    earth.queue.dequeueVertices(earth)
    earth.queue.dequeueEdges(earth)
    
    cell.getConnNodeIDs(nodeType=_cell, mode='out')
    earth.view(str(earth.mpi.rank) + '.png', layout=ig.Layout(earth.graph.vs['pos']))
    
    print str(earth.mpi.rank) + ' ' + str(earth.graph.vs.indices)
    print str(earth.mpi.rank) + ' ' + str(earth.graph.vs['value3'])
    
    for agent in earth.iterEntRandom(_ag):
        agent.node['value3'] = earth.mpi.rank+ agent.nID
    
    earth.mpi.updateGhostNodes([_ag])

    earth.io.initNodeFile(earth, [_cell, _ag])
    
    earth.io.gatherNodeData(0)
    earth.io.writeDataToFile()
    
    print str(earth.mpi.rank) + ' ' + str(earth.graph.vs['value3'])
