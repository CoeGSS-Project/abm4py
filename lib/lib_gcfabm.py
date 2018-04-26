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

sooner:
    - IO of connections and their attributes
    - MPI communication with numpy arrays (seems much faster)
    - DOCUMENTATION
    - caching not only of out-connections?!
    - re-think communication model (access restrictions)
        - documentation of it
    - re-think role of edge types and strong connection with connected
      node types

later:
    - movement of agents between processes
    - implement mpi communication of string attributes
    - implement output of string attributes
    - reach usage of 1000 parallell processes (960) -> how to do with only 3800 Locations??
        - other resolution available!
        - share locatons between processes


"""

import sys
sys.path = ['../h5py/build/lib.linux-x86_64-2.7'] + sys.path
import mpi4py
mpi4py.rc.threads = False
sys_excepthook = sys.excepthook
def mpi_excepthook(v, t, tb):
    sys_excepthook(v, t, tb)
    mpi4py.MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook

from mpi4py import MPI
import h5py
import logging as lg
import sys
import igraph as ig
import numpy as np
import time
import random as rd
from bunch import Bunch
from class_graph import ABMGraph
import class_auxiliary as aux

ALLOWED_MULTI_VALUE_TYPES = (list, tuple, np.ndarray)


def assertUpdate(graph, prop, nodeType):
    
    # if is serial
    if not(graph.isParallel):##OPTPRODUCTION
        return##OPTPRODUCTION
    
    # if is static
    if prop in graph.nodeTypes[nodeType].staProp: ##OPTPRODUCTION
        return##OPTPRODUCTION
    
    # if is updated
    if prop in graph.ghostTypeUpdated[nodeType]:  ##OPTPRODUCTION
        return##OPTPRODUCTION
    
    raise('Error while accessing non-updated property')##OPTPRODUCTION
    pass

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
            self.currNodeID = self.graph.nCount()

        # adding of the data
        nID = self.currNodeID
        self.nodeList.append(nID) # nID to general list
        #kwProperties.update({'nID': nID})
        kwProperties.update({ 'gID':gID})

        if nodeType not in self.nodeProperties.keys():
            # init of new nodeType
            propertiesOfType = self.graph.getPropOfNodeType(nodeType,kind='all')
            #print propertiesOfType
            #print kwProperties.keys()
            assert all([prop in propertiesOfType for prop in kwProperties.keys()]) # check that all given properties are registered ##OPTPRODUCTION

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
        assert self.nodeList[0] == self.graph.vcount() # check that the queuing idx is consitent ##OPTPRODUCTION
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
                entity._node = self.graph.vs[entity.nID]
                # redireciton of internal functionality:
                entity.getValue = entity._node.__getitem__
                entity.setValue = entity._node.__setitem__
                #print 'node index:'  + str(entity.node.index)
                #entity.register(world)
                #print 'assert ' + str((entity.nID, entity.node.index))
                assert entity.nID == entity._node.index                        ##OPTPRODUCTION

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
                assert len(kwProperties[propKey]) == len(edgeList)             ##OPTPRODUCTION
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
    """
    As default only out peers and out connections are in the cache. 
    """
    def __init__(self, graph, nID, nodeType):
        self.graph       = graph
        self.nID         = nID
        self.edgesAll    = None
        self.edgesByType = dict()
        self.nodeType    = nodeType
        self.peersAll    = None
        self.peersByType = dict()
        self.getPeerValues2 = self.peersByType.__getitem__

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
            
            # check if re-caching is required
            if self.edgesAll is None:
                self.edgesAll          = self.graph.es[self.graph.incident(self.nID,'out')].select(type_ne=0)
            # re-cache only certain type                
            self.edgesByType[edgeType] = self.edgesAll.select(type=edgeType)
        else:
            # all out edges
            self.edgesAll              = self.graph.es[self.graph.incident(self.nID,'out')].select(type_ne=0)

    def __checkEdgeCache__(self, edgeType):

        if edgeType is None:
            self.__reCacheEdges__()
        else:
            # check if re-caching is required
            if edgeType not in self.edgesByType.keys():
                self.__reCacheEdges__(edgeType)
                
    def setEdgeCache(self, idList, edgeType):
        """
        If you know what you do, you can use this method to set the cache manualy to save time
        """
        self.edgesByType[edgeType] = self.graph.es[idList]

    def setPeerCache(self, idList, nodeType):
        """
        If you know what you do, you can use this method to set the cache manualy to save time
        """
        self.peersByType[nodeType] = self.graph.vs[idList]
        
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
        
        nodeType  = self.graph.edge2NodeType[edgeType][1]
        
        assertUpdate(self.graph, prop, nodeType)
        
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

def deco(fun):
    def helper(arg):
        return fun(arg)[0]
    return helper

class Entity():
    """
    Most basic class from which agents of different type are derived
    """
    __slots__ = ['gID', 'nID']
    

    def __init__(self, world, nID = -1, **kwProperties):
        nodeType =  world.graph.class2NodeType[self.__class__]

        if not hasattr(self, '_graph'):
            self.setGraph(world.graph)

        if world.queuing:        
            if not hasattr(self, '_queue'):
                self.setQueue(world.queue)            
        #self._graph = world.graph

        self.gID    = self.getGlobID(world)
        kwProperties['gID'] = self.gID

        # create instance from existing node
        if nID is not -1:
            if nID is None:
                raise()

            self.nID = nID
            #print nID , world.mpi.rank
            
            self._node = self._graph.getNodeView(nID)
            self.getValue = deco(self._node.__getitem__)
            self.setValue = self._node.__setitem__
            #print 'nID:' + str(nID) + ' gID: ' + str(self._node['gID'])
            self.gID = self.getValue('gID')
            # redireciton of internal functionality:
            self.__getitem__ = deco(self._node.__getitem__)
            self.__setitem__ = self._node.__setitem__
            return

        # create instance newly
        self.nID, self.dataID, self._node = world.addVertex(nodeType,  **kwProperties)
        self.nodeType = nodeType
        
        # redireciton of internal functionality:
        self.getValue = deco(self._node.__getitem__)
        self.setValue = self._node.__setitem__
        self.__getitem__ = deco(self._node.__getitem__)
        self.__setitem__ = self._node.__setitem__
        
        if world.caching:
            self._cache  = Cache(self._graph, self.nID, nodeType)

            # definition of access functions
            self.getPeerValues = self._cache.getPeerValues
            #self.getPeerValues2 = self._cache.getPeerValues2
            self.setPeerValues = self._cache.setPeerValues
            self.getPeers      = self._cache.getPeers
            self.getPeerIDs    = self._cache.getPeerIDs
            self.setPeerCache  = self._cache.setPeerCache

            self.getEdgeValues = self._cache.getEdgeValues
            self.setEdgeValues = self._cache.setEdgeValues
            self.getEdges      = self._cache.getEdges
            self.setEdgeCache  = self._cache.setEdgeCache

        else:
            self._cache = None

    

    
    @classmethod
    def setGraph(cls, graph):
        """ Makes the class variable _graph available at the first init of an entity"""
        cls._graph = graph





    def getPeerIDs(self, edgeType=None, nodeType=None, mode='out'):
        
        if edgeType is None:
            edgeType = earth.graph.node2EdgeType[self.nodeType, nodeType]
        #print edgeType
        if mode=='out':            
            eList, nodeList  = self._graph.outgoing(self.nID, edgeType)
        elif mode == 'in':
            eList, nodeList  = self._graph.incomming(self.nID, edgeType)
        
        return nodeList

        

#    def getPeers(self, edgeType=None):
#        return self._graph.vs[self.getPeerIDs(edgeType)]
#        return self._graph.getOutNodeValues(self, self.nID, eTypeID, attr=None)

    def getPeerValues(self, prop, edgeType=None):
        """
        Access the attributes of all connected nodes of an specified nodeType
        or connected by a specfic edge type
        """
        return self._graph.getOutNodeValues(self.nID, edgeType, attr=prop)

    def setPeerValues(self, prop, values, edgeType=None, nodeType=None):
        """
        Set the attributes of all connected nodes of an specified nodeType
        or connected by a specfic edge type
        """
        self._graph.setOutNodeValues(self.nID, edgeType, prop, values)
                                   

    def getEdgeValues(self, prop, edgeType):
        """
        privat function to access the values of  edges
        """
        (eTypeID, dataID), nIDList  = self._graph.outgoing(self.nID, edgeType)

        edgesValues = self._graph.getEdgeSeqAttr(label=prop, 
                                                 eTypeID=eTypeID, 
                                                 dataIDs=dataID)

        

        return edgesValues, nIDList

    def setEdgeValues(self, prop, values, edgeType=None):
        """
        privat function to access the values of  edges
        """
        (eTypeID, dataID), _  = self._graph.outgoing(self.nID, edgeType)
        
        self._graph.setEdgeSeqAttr(label=prop, 
                                   values=values,
                                   eTypeID=eTypeID, 
                                   dataIDs=dataID)

    def getEdgeIDs(self, edgeType=None):
        """
        privat function to access the values of  edges
        """
        eList, _  = self._graph.outgoing(self.nID, edgeType)
        return eList

    def getNeigbourhood(self, order):
        raise DeprecationWarning('not supported right now')
        raise NameError('sorry')
#        neigIDList = self._graph.neighborhood(self.nID, order)
#        neigbours = []
#        for neigID in neigIDList:
#            neigbours.append(self._graph.vs[neigID])
#        return neigbours, neigIDList


#    def queueConnection(self, friendID, edgeType, **kwpropDict):
#        kwpropDict.update({'type': edgeType})
#        self._queue.addEdge(self.nID,friendID, **kwpropDict)


    def addConnection(self, friendID, edgeType, **kwpropDict):
        """
        Adding a new connection to another node
        Properties must be provided in the correct order and structure
        """
        self._graph.addEdge(edgeType, self.nID, friendID, attributes = tuple(kwpropDict.values()))


    def remConnection(self, friendID=None, edgeID=None):
        """
        Removing a connection to another node
        """
        self._graph.remEdge(source=self.nID, target=friendID, eTypeID=edgeID)

    def remConnections(self, friendIDs=None, edgeIDs=None):
        raise DeprecationWarning('not supported right now')
        raise NameError('sorry')


    def addValue(self, prop, value, idx = None):
        raise DeprecationWarning('Will be deprecated in the future')
        if idx is None:
            self._node[prop] += value
        else:
            self._node[prop][idx] += value

    def delete(self, world):
        raise DeprecationWarning('not supported right now')
        raise NameError('sorry')

        #self._graph.delete_vertices(nID) # not really possible at the current igraph lib
        # Thus, the node is set to inactive and removed from the iterator lists
        # This is due to the library, but the problem is general and pose a challenge.
        world.graph.vs[self.nID]['type'] = 0 #set to inactive


        world.deRegisterNode()

        # get all edges - in and out
        eIDList  = self._graph.incident(self.nID)
        #set edges to inactive
        self._graph[eIDList]['type'] = 0


    def register(self, world, parentEntity=None, edgeType=None, ghost=False):
        nodeType = world.graph.class2NodeType[self.__class__]
        world.registerNode(self, nodeType, ghost)

        if parentEntity is not None:
            self.mpiPeers = parentEntity.registerChild(world, self, edgeType)



class Agent(Entity):



    def __init__(self, world, **kwProperties):
        if 'nID' not in kwProperties.keys():
            nID = -1
        else:
            nID = kwProperties['nID']
        Entity.__init__(self, world, nID, **kwProperties)
        self.mpiOwner =  int(world.mpi.rank)

    def getGlobID(self,world):
        return world.globIDGen.next()

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

    def getNClosePeers(self, 
                       world, 
                       nContacts,
                       edgeType, 
                       currentContacts = None, 
                       addYourself = True):
        """
        Method to generate a preliminary friend network that accounts for
        proximity in space
        #ToDO add to easyUI
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
        sourceList  = list()
        targetLis   = list()
        #ownPref    = self._node['preferences']
        #ownIncome  = self.hh._node['income']


        #get spatial weights to all connected cells
        cellConnWeights, cellIds = self.loc.getConnectedLocation()
        personIdsAll = list()
        nPers = list()
        cellWeigList = list()
        
        
        for cellWeight, cellIdx in zip(cellConnWeights, cellIds):

            cellWeigList.append(cellWeight)
            
            personIds = world.getEntity(cellIdx).peList #doto do more general
            
            #personIds = world.getEntity(cellIdx).getPersons()

            personIdsAll.extend(personIds)
            nPers.append(len(personIds))

        # return nothing if too few candidates
        if not isInit and nPers > nContacts:
            lg.info('ID: ' + str(self.nID) + ' failed to generate friend')
            return [],[],[]

        #setup of spatial weights
        weightData = np.zeros(np.sum(nPers))

        idx = 0
        for nP, we in zip(nPers, cellWeigList):
            weightData[idx:idx+nP] = we
            idx = idx+ nP
        del idx

        #normalizing final row
        weightData /= np.sum(weightData,axis=0)

        if np.sum(weightData>0) < nContacts:
            lg.info( "nID: " + str(self.nID) + ": Reducting the number of friends at " + str(self.loc.getValue('pos')))
            nContacts = min(np.sum(weightData>0)-1,nContacts)

        if nContacts < 1:                                                       ##OPTPRODUCTION
            lg.info('ID: ' + str(self.nID) + ' failed to generate friend')      ##OPTPRODUCTION

        else:
            # adding contacts
            ids = np.random.choice(weightData.shape[0], nContacts, replace=False, p=weightData)
            contactList = [ personIdsAll[idx] for idx in ids ]
            targetList  = [ personIdsAll[idx] for idx in ids]
        sourceList = [self.nID] * len(ids)
        
        if isInit and addYourself:
            #add yourself as a friend
            contactList.append(self.nID)
            sourceList.append(self.nID)
            targetList.append(self.nID)

        weigList   = [1./len(sourceList)]*len(sourceList)
        return contactList, (sourceList, targetList), weigList




class GhostAgent(Entity):
    
    def __init__(self, world, owner, nID=-1, **kwProperties):
        Entity.__init__(self, world, nID, **kwProperties)
        self.mpiOwner =  int(owner)

    def register(self, world, parentEntity=None, edgeType=None):
        Entity.register(self, world, parentEntity, edgeType, ghost= True)
        

    def getGlobID(self,world):

        return None # global ID need to be acquired via MPI communication

    def getLocationValue(self, prop):

        return self.loc.node[prop]



    def registerChild(self, world, entity, edgeType):
        world.addEdge(entity.nID,self.nID, type=edgeType)

        
################ LOCATION CLASS #########################################
class Location(Entity):

    def getGlobID(self,world):
        return world.globIDGen.next()

    def __init__(self, world, **kwProperties):
        if 'nID' not in kwProperties.keys():
            nID = -1
        else:
            nID = kwProperties['nID']


        Entity.__init__(self,world, nID, **kwProperties)
        self.mpiOwner = int(world.mpi.rank)
        self.mpiPeers = list()



    def registerChild(self, world, entity, edgeType=None):
        world.addEdge(edgeType, entity.nID,self.nID )
        entity.loc = self

        if len(self.mpiPeers) > 0: # node has ghosts on other processes
            for mpiPeer in self.mpiPeers:
                #print 'adding node ' + str(entity.nID) + ' as ghost'
                nodeType = world.graph.class2NodeType[entity.__class__]
                world.mpi.queueSendGhostNode( mpiPeer, nodeType, entity, self)

        return self.mpiPeers
    
    def getConnectedLocation(self, edgeType=1):
        """ 
        ToDo: check if not deprecated 
        """
        self.weights, nodeIDList = self.getEdgeValues('weig',edgeType=edgeType)
        
        return self.weights,  nodeIDList

class GhostLocation(Entity):
    
    def getGlobID(self,world):

        return -1

    def __init__(self, world, owner, nID=-1, **kwProperties):

        Entity.__init__(self,world, nID, **kwProperties)
        self.mpiOwner = int(owner)
        self.queuing = world.queuing

    def register(self, world, parentEntity=None, edgeType=None):
        Entity.register(self, world, parentEntity, edgeType, ghost= True)

    def registerChild(self, world, entity, edgeType=None):
        world.addEdge(edgeType, entity.nID,self.nID)
        
        entity.loc = self

 
#    def updateAgentList(self, graph, edgeType):  # toDo nodeType is not correct anymore
#        """
#        updated method for the agents list, which is required since
#        ghost cells are not active on their own
#        """
#        
#        hhIDList = self.getPeerIDs(edgeType)
#        return graph.vs[hhIDList]
################ WORLD CLASS #########################################

class World:
    #%% World sub-classes

    class Globals(dict):
        """ This class manages global variables that are assigned on all processes
        and are synced via mpi. Global variables need to be registered together with
        the aggregation method they ase synced with, .e.g. sum, mean, min, max,...

        
        #TODO
        - enforce the setting (and reading) of global stats
        - implement mean, deviation, std as reduce operators


        """
        


        def __init__(self, world):
            self.world = world
            self.comm  = world.mpi.comm

            # simple reductions
            self.reduceDict = dict()
            
            # MPI operations
            self.operations = dict()
            self.operations['sum']  = MPI.SUM
            self.operations['prod'] = MPI.PROD
            self.operations['min']  = MPI.MIN
            self.operations['max']  = MPI.MAX

            #staticical reductions/aggregations
            self.statsDict       = dict()
            self.localValues     = dict()
            self.nValues         = dict()
            self.updated         = dict()

            # self implemented operations
            statOperations         = dict()
            statOperations['mean'] = np.mean
            statOperations['std']  = np.std
            statOperations['var']  = np.std
            #self.operations['std'] = MPI.Op.Create(np.std)

        #%% simple global reductions
        def registerValue(self, globName, value, reduceType):
            self[globName] = value
            self.localValues[globName] = value
            try:
                self.nValues[globName] = len(value)
            except:
                self.nValues[globName] = 1
            if reduceType not in self.reduceDict.keys():
                self.reduceDict[reduceType] = list()
            self.reduceDict[reduceType].append(globName)
            self.updated[globName] = True

        def syncReductions(self):

            for redType in self.reduceDict.keys():

                op = self.operations[redType]
                #print op
                for globName in self.reduceDict[redType]:

                    # enforce that data is updated
                    assert  self.updated[globName] is True    ##OPTPRODUCTION
                    
                    # communication between all proceees
                    self[globName] = self.comm.allreduce(self.localValues[globName],op)
                    self.updated[globName] = False
                    lg.debug('local value of ' + globName + ' : ' + str(self.localValues[globName]))##OPTPRODUCTION
                    lg.debug(str(redType) + ' of ' + globName + ' : ' + str(self[globName]))##OPTPRODUCTION

        #%% statistical global reductions/aggregations
        def registerStat(self, globName, values, statType):
            #statfunc = self.statOperations[statType]

            assert statType in ['mean', 'std', 'var']    ##OPTPRODUCTION


            if not isinstance(values, ALLOWED_MULTI_VALUE_TYPES):
                values = [values]
            values = np.asarray(values)


            self.localValues[globName]  = values
            self.nValues[globName]      = len(values)
            if statType == 'mean':
                self[globName]          = np.mean(values)
            elif statType == 'std':
                self[globName]          = np.std(values)
            elif statType == 'var':
                self[globName]          = np.var(values)

            if statType not in self.statsDict.keys():
                self.statsDict[statType] = list()
            self.statsDict[statType].append(globName)
            self.updated[globName] = True
            

        def updateLocalValues(self, globName, values):
            self.localValues[globName]     = values
            self.nValues[globName]         = len(values)
            self.updated[globName]         = True

        def syncStats(self):
            for redType in self.statsDict.keys():
                if redType == 'mean':

                    for globName in self.statsDict[redType]:
                        
                        # enforce that data is updated
                        assert  self.updated[globName] is True    ##OPTPRODUCTION
                        
                        # sending data list  of (local mean, size)
                        tmp = [(np.mean(self.localValues[globName]), self.nValues[globName])]* self.comm.size 

                        # communication between all proceees
                        tmp = np.asarray(self.comm.alltoall(tmp))

                        lg.debug('####### Mean of ' + globName + ' #######')       ##OPTPRODUCTION
                        lg.debug('loc mean: ' + str(tmp[:,0]))                     ##OPTPRODUCTION
                        # calculation of global mean
                        globValue = np.sum(np.prod(tmp,axis=1)) # means * size
                        globSize  = np.sum(tmp[:,1])             # sum(size)
                        self[globName] = globValue/ globSize    # glob mean
                        lg.debug('Global mean: ' + str( self[globName] ))   ##OPTPRODUCTION
                        self.updated[globName] = False
                        
                elif redType == 'std':
                    for globName in self.statsDict[redType]:

                        # enforce that data is updated
                        assert  self.updated[globName] is True    ##OPTPRODUCTION
                        
                        # local calulation
                        locSTD = [np.std(self.localValues[globName])] * self.comm.size
                        locSTD = np.asarray(self.comm.alltoall(locSTD))
                        lg.debug('####### STD of ' + globName + ' #######')              ##OPTPRODUCTION
                        lg.debug('loc std: ' + str(locSTD))                       ##OPTPRODUCTION

                        # sending data list  of (local mean, size)
                        tmp = [(np.mean(self.localValues[globName]), self.nValues[globName])]* self.comm.size 
                        
                        # communication between all proceees
                        tmp = np.asarray(self.comm.alltoall(tmp))


                        # calculation of the global std
                        locMean = tmp[:,0]
                        
                        lg.debug('loc mean: ' + str(locMean))                     ##OPTPRODUCTION

                        locNVar = tmp[:,1]
                        lg.debug('loc number of var: ' + str(locNVar))            ##OPTPRODUCTION

                        globMean = np.sum(np.prod(tmp,axis=1)) / np.sum(locNVar)  
                        lg.debug('global mean: ' + str( globMean ))               ##OPTPRODUCTION

                        diffSqrMeans = (locMean - globMean)**2

                        deviationOfMeans = np.sum(locNVar * diffSqrMeans)

                        globVariance = (np.sum( locNVar * locSTD**2) + deviationOfMeans) / np.sum(locNVar)

                        self[globName] = np.sqrt(globVariance)
                        lg.debug('Global STD: ' + str( self[globName] ))   ##OPTPRODUCTION
                        self.updated[globName] = False
                        
                elif redType == 'var':
                    for globName in self.statsDict[redType]:

                        # enforce that data is updated
                        assert  self.updated[globName] is True    ##OPTPRODUCTION
                        
                        # calculation of local mean
                        locSTD = [np.std(self.localValues[globName])] * self.comm.size
                        locSTD = np.asarray(self.comm.alltoall(locSTD))
                        

                        # out data list  of (local mean, size)
                        tmp = [(np.mean(self.localValues[globName]), self.nValues[globName])]* self.comm.size 
                        tmp = np.asarray(self.comm.alltoall(tmp))

                        locMean = tmp[:,0]
                        #print 'loc mean: ', locMean

                        lg.debug('####### Variance of ' + globName + ' #######')              ##OPTPRODUCTION
                        lg.debug('loc mean: ' + str(locMean))               ##OPTPRODUCTION
                        locNVar = tmp[:,1]
                        #print 'loc number of var: ',locNVar

                        globMean = np.sum(np.prod(tmp,axis=1)) / np.sum(locNVar)
                        #print 'global mean: ', globMean
                        
                        diffSqrMeans = (locMean - globMean)**2
                        lg.debug('global mean: ' + str( globMean )) ##OPTPRODUCTION

                        deviationOfMeans = np.sum(locNVar * diffSqrMeans)

                        globVariance = (np.sum( locNVar * locSTD**2) + deviationOfMeans) / np.sum(locNVar)

                        self[globName] = globVariance
                        lg.debug('Global variance: ' + str( self[globName] ))  ##OPTPRODUCTION
                        self.updated[globName] = False

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

            def initStorage(self, dtype):
                #print dtype
                self.data = np.zeros([self.nAgents,self.nAttr ], dtype=dtype)

            def addData(self, timeStep, nodeData):
                self.timeStep = timeStep
                self.data = nodeData[self.ag2FileIdx][self.attributeList]
#                for attr in self.attributeList:
#                    if len(self.attrIdx[attr]) == 1:
#                        self.data[:,self.attrIdx[attr]] = np.expand_dims(graph.vs[self.ag2FileIdx][attr],1)
#                    else:
#                        self.data[:,self.attrIdx[attr]] = graph.vs[self.ag2FileIdx][attr]

            def writeData(self, h5File, folderName=None):
                #print self.header
                if folderName is None:
                    path = '/' + str(self.nodeType)+ '/' + str(self.timeStep).zfill(self.timeStepMag)
                    #print 'IO-path: ' + path
                    self.dset = h5File.create_dataset(path, (self.nAgentsGlob,), dtype=self.data.dtype)
                    self.dset[self.loc2GlobIdx[0]:self.loc2GlobIdx[1],] = self.data
                else:
                    path = '/' + str(self.nodeType)+ '/' + folderName
                    #print 'IO-path: ' + path
                    self.dset = h5File.create_dataset(path, (self.nAgentsGlob,), dtype=self.data.dtype)
                    self.dset[self.loc2GlobIdx[0]:self.loc2GlobIdx[1],] = self.data

        

        #%% Init of the IO class
        def __init__(self, world, nSteps, outputPath = ''): # of IO

            self.outputPath  = outputPath
            self._graph       = world.graph
            #self.timeStep   = world.timeStep
            self.h5File      = h5py.File(outputPath + '/nodeOutput.hdf5',
                                         'w',
                                         driver='mpio',
                                         comm=world.mpi.comm)
                                         #libver='latest',
                                         #info = world.mpi.info)
            self.comm        = world.mpi.comm
            self.dynamicData = dict()
            self.staticData  = dict() # only saved once at timestep == 0
            self.timeStepMag = int(np.ceil(np.log10(nSteps)))


        def initNodeFile(self, world, nodeTypes):
            """
            Initializes the internal data structure for later I/O
            """
            lg.info('start init of the node file')

            for nodeType in nodeTypes:
                world.mpi.comm.Barrier()
                tt = time.time()
                lg.info(' NodeType: ' +str(nodeType))
                group = self.h5File.create_group(str(nodeType))

                group.attrs.create('dynamicProps', world.graph.getPropOfNodeType(nodeType, 'dyn')['names'])
                group.attrs.create('staticProps', world.graph.getPropOfNodeType(nodeType, 'sta')['names'])

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

                lg.info( 'loc2GlobIdx exchanged in  ' + str(time.time()-tt)  + ' seconds'  )
                tt = time.time()


                # static data
                staticRec  = self.Record(nAgents, 
                                         world.dataDict[nodeType], 
                                         nAgentsGlob, 
                                         loc2GlobIdx, 
                                         nodeType, 
                                         self.timeStepMag)
                
                attrInfo   = world.graph.getPropOfNodeType(nodeType, 'sta')
                attributes = attrInfo['names']
                sizes      = attrInfo['sizes']
                
                attrDtype = world.graph.getDTypeOfNodeType(nodeType, 'sta')
                
                lg.info('Static record created in  ' + str(time.time()-tt)  + ' seconds')

                for attr, nProp in zip(attributes, sizes):
                    #print attr
                    #check if first property of first entity is string
                    try:
                         
                        entProp = self._graph.getNodeSeqAttr(label=attr, nTypeID=nodeType, dataIDs=staticRec.ag2FileIdx[0])
                    except ValueError:
                        #print attr
                        raise ValueError(str(attr) + ' not found')
                    if not isinstance(entProp,str):

                        #todo - check why not np.array allowed
#                        if isinstance(entProp,ALLOWED_MULTI_VALUE_TYPES):
#                            # add mutiple fields
#                            nProp = len(self._graph.nodes[staticRec.ag2FileIdx[0]][attr])
#                        else:
#                            #add one field
#                            nProp = 1

                        staticRec.addAttr(attr, nProp)

                tt = time.time()
                # allocate storage
                staticRec.initStorage(attrDtype)
                #print attrInfo
                
                self.staticData[nodeType] = staticRec
                lg.info( 'storage allocated in  ' + str(time.time()-tt)  + ' seconds'  )

                # dynamic data
                dynamicRec = self.Record(nAgents, 
                                         world.dataDict[nodeType], 
                                         nAgentsGlob, 
                                         loc2GlobIdx, 
                                         nodeType, 
                                         self.timeStepMag)

                attrInfo   = world.graph.getPropOfNodeType(nodeType, 'dyn')
                attributes = attrInfo['names']
                sizes      = attrInfo['sizes']

                attrDtype = world.graph.getDTypeOfNodeType(nodeType, 'dyn')

                lg.info('Dynamic record created in  ' + str(time.time()-tt)  + ' seconds')


                for attr, nProp in zip(attributes, sizes):
                    #check if first property of first entity is string
                    entProp = self._graph.getNodeSeqAttr(attr, 
                                                         nTypeID=nodeType,
                                                         dataIDs=staticRec.ag2FileIdx[0])
                    if not isinstance(entProp,str):


#                        if isinstance(entProp, ALLOWED_MULTI_VALUE_TYPES):
#                            # add mutiple fields
#                            nProp = len(self._graph.vs[dynamicRec.ag2FileIdx[0]][attr])
#                        else:
#                            #add one field
#                            nProp = 1

                        dynamicRec.addAttr(attr, nProp)

                tt = time.time()
                # allocate storage
                dynamicRec.initStorage(attrDtype)
                self.dynamicData[nodeType] = dynamicRec
                
                #lg.info( 'storage allocated in  ' + str(time.time()-tt)  + ' seconds'  )
                
                self.writeDataToFile(0, nodeType, static=True)
                
            lg.info( 'static data written to file in  ' + str(time.time()-tt)  + ' seconds'  )

        def writeDataToFile(self, timeStep, nodeTypes, static=False):
            """
            Transfers data from the graph to record for the I/O
            and writing data to hdf5 file
            """
            if isinstance(nodeTypes,int):
                nodeTypes = [nodeTypes]
            
            for nodeType in nodeTypes:
                if static:
                    #for typ in self.staticData.keys():
                    self.staticData[nodeType].addData(timeStep, self._graph.nodes[nodeType])
                    self.staticData[nodeType].writeData(self.h5File, folderName='static')
                else:
                    #for typ in self.dynamicData.keys():
                    self.dynamicData[nodeType].addData(timeStep, self._graph.nodes[nodeType])
                    self.dynamicData[nodeType].writeData(self.h5File)

                   

        def initEdgeFile(self, edgeTypes):
            """
            ToDo
            """
            pass

        def finalizeAgentFile(self):
            """
            finalizing the agent files - closes the file and saves the
            attribute files
            ToDo: include attributes in the agent file
            """

            for nodeType in self.dynamicData.keys():
                group = self.h5File.get('/' + str(nodeType))
                record = self.dynamicData[nodeType]
                for attrKey in record.attrIdx.keys():
                    group.attrs.create(attrKey, record.attrIdx[attrKey])

            for nodeType in self.staticData.keys():
                group = self.h5File.get('/' + str(nodeType))
                record = self.staticData[nodeType]
                for attrKey in record.attrIdx.keys():
                    group.attrs.create(attrKey, record.attrIdx[attrKey])


            self.h5File.close()
            lg.info( 'Agent file closed')
            from class_auxiliary import saveObj

            for nodeType in self.dynamicData.keys():
                record = self.dynamicData[nodeType]
                #np.save(self.para['outPath'] + '/agentFile_type' + str(typ), self.agentRec[typ].recordNPY, allow_pickle=True)
                saveObj(record.attrIdx, (self.outputPath + '/attributeList_type' + str(nodeType)))
    
    class Mpi():
        """
        MPI communication module that controles all communcation between
        different processes.
        ToDo: change to communication using numpy
        """

        def __init__(self, world, mpiComm=None):

            self.world = world
            if mpiComm is None:
                self.comm = MPI.COMM_WORLD
            else:
                self.comm = mpiComm
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

            self.info = MPI.Info.Create()
            self.info.Set("romio_ds_write", "disable")
            self.info.Set("romio_ds_read", "disable")

            self.peers    = list()     # list of ranks of all processes that have ghost duplicates of this process

            self.ghostNodeQueue = dict()
            #ID list of ghost nodes that recieve updates from other processes
            self.mpiRecvIDList = dict()           
            #ID list of ghost nodes that send updates from other processes
            self.mpiSendIDList = dict()
            
            self.buffer         = dict()
            self.messageSize    = dict()
            self.sendReqList    = list()

            self.reduceDict = dict()
            world.send = self.comm.send
            world.recv = self.comm.recv

            world.isend = self.comm.isend
            world.irecv = self.comm.irecv

            self._clearBuffer()
            
            self.world.graph.ghostTypeUpdated = dict()
            
            self.world.graph.isParallel =  self.size > 1
                
#            for nodeType in world.graph.nodeTypes.keys():
#                # lists all attributes of the nodeType which have been updated in the last step
#                self.world.graph.self.ghostTypeUpdated[nodeType] = list()

        #%% Privat functions
        def _clearBuffer(self):
            """
            Method to clear all2all buffer
            """
            self.a2aBuff = []
            for x in range(self.comm.size):
                self.a2aBuff.append([])


        def _add2Buffer(self, mpiPeer, data):
            """
            Method to add data to all2all data to buffer
            """
            self.a2aBuff[mpiPeer].append(data)

        def _all2allSync(self):
            """
            Privat all2all communication method
            """
            recvBuffer = self.comm.alltoall(self.a2aBuff)
            self._clearBuffer()
            return recvBuffer


        def _packData(self, nodeType, mpiPeer, propList, connList=None):
            """
            Privat method to pack all data for MPI transfer
            """
            dataSize = 0
            nNodes = len(self.mpiSendIDList[(nodeType,mpiPeer)])
            
            dataPackage = dict()
            dataPackage['nNodes']  = nNodes
            dataPackage['nTypeID'] = nodeType
            #print self.mpiSendIDList[(nodeType,mpiPeer)]
            dataPackage['data'] = self.world.graph.getNodeSeqAttr(label=propList, lnIDs=self.mpiSendIDList[(nodeType,mpiPeer)] )
            dataSize += np.prod(dataPackage['data'].shape)
            if connList is not None:
                dataPackage['connectedNodes'] = connList
                dataSize += len(connList)
#            for prop in propList:
#                
#                
#                dataPackage.append(self.world.graph.getNodeSeqAttr( self.mpiSendIDList[(nodeType,mpiPeer)], label=prop))
#                dataSize += len(self.world.graph.getNodeSeqAttr( self.mpiSendIDList[(nodeType,mpiPeer)], label=prop))
            
#                dataPackage.append(connList)
#                dataSize += len(connList)
                
            lg.debug('package size: ' + str(dataSize))
            return dataPackage, dataSize



        def _updateGhostNodeData(self, nodeTypeList= 'dyn', propertyList= 'dyn'):
            """
            Privat method to update the data between processes for existing ghost nodes
            """
            tt = time.time()
            messageSize = 0
            
            for (nodeType, mpiPeer) in self.mpiSendIDList.keys():
                #lg.info('here')
                if nodeTypeList == 'all' or nodeType in nodeTypeList:
                    #IDList = self.ghostNodeSend[nodeType, mpiPeer]

                    if propertyList in ['all', 'dyn', 'sta']:
                        propertyList = self.world.graph.getPropOfNodeType(nodeType, kind=propertyList)['names']
                        #del propertyList['gID']
                        
                    lg.debug('MPIMPIMPIMPI -  Updating ' + str(propertyList) + ' for nodeType ' + str(nodeType) + 'MPIMPIMPI')
                    dataPackage ,packageSize = self._packData(nodeType, mpiPeer, propertyList, connList=None)
                                                        
                    messageSize = messageSize + packageSize
                    self._add2Buffer(mpiPeer, dataPackage)

            syncPackTime = time.time() -tt

            tt = time.time()
            recvBuffer = self._all2allSync()
            pureSyncTime = time.time() -tt

            tt = time.time()
            
            for mpiPeer in self.peers:
                if len(recvBuffer[mpiPeer]) > 0: # will receive a message


                    for dataPackage in recvBuffer[mpiPeer]:
                        nNodes   = dataPackage['nNodes']
                        nodeType = dataPackage['nTypeID']

                        if propertyList == 'all':
                            propertyList= self.world.graph.nodeProperies[nodeType][:]
                            #print propertyList
                            propertyList.remove('gID')
                        #print type(self.ghostNodeRecv[nodeType, mpiPeer])
#                        nodeSeq = self.ghostNodeRecv[nodeType, mpiPeer]
#                        for i, prop in enumerate(propertyList):
#                           #nodeSeq[prop] = dataPackage[i+1]
#                            self.world.graph.setNodeSeqAttr(self.mpiRecvIDList[(nodeType, mpiPeer)], 
#                                                            label=propertyList, 
#                                                            values=dataPackage['data'][prop])    
                        #print dataPackage['data']
                        self.world.graph.setNodeSeqAttr(label=propertyList, 
                                                        values=dataPackage['data'],
                                                        lnIDs=self.mpiRecvIDList[(nodeType, mpiPeer)])                        
                        
            syncUnpackTime = time.time() -tt

            lg.info('Sync times - ' +
                    ' pack: ' + str(syncPackTime) + ' s , ' +
                    ' comm: ' + str(pureSyncTime) + ' s , ' +
                    ' unpack: ' + str(syncUnpackTime) + ' s , ')
            return messageSize

        def initCommunicationViaLocations(self, ghostLocationList, locNodeType):
            """
            Method to initialize the communication based on the spatial
            distribution
            """

            tt = time.time()
            # acquire the global IDs for the ghostNodes
            mpiRequest = dict()
            

            
            lg.debug('ID Array: ' + str(self.world.graph.IDArray))##OPTPRODUCTION
            for ghLoc in ghostLocationList:
                owner = ghLoc.mpiOwner
                #print owner
                x,y   = ghLoc.getValue('pos')
                if owner not in mpiRequest:
                    mpiRequest[owner]   = (list(), 'gID')
                    self.mpiRecvIDList[(locNodeType, owner)] = list()

                mpiRequest[owner][0].append( (x,y) ) # send x,y-pairs for identification
                self.mpiRecvIDList[(locNodeType, owner)].append(ghLoc.nID)
            lg.debug('rank ' + str(self.rank) + ' mpiRecvIDList: ' + str(self.mpiRecvIDList))##OPTPRODUCTION

            for mpiDest in mpiRequest.keys():

                if mpiDest not in self.peers:
                    self.peers.append(mpiDest)

                    # send request of global IDs
                    lg.debug( str(self.rank) + ' asks from ' + str(mpiDest) + ' - ' + str(mpiRequest[mpiDest]))##OPTPRODUCTION
                    #self.comm.send(mpiRequest[mpiDest], dest=mpiDest)
                    self._add2Buffer(mpiDest, mpiRequest[mpiDest])

            lg.debug( 'requestOut:' + str(self.a2aBuff))##OPTPRODUCTION
            requestIn = self._all2allSync()
            lg.debug( 'requestIn:' +  str(requestIn))##OPTPRODUCTION


            for mpiDest in mpiRequest.keys():

                #self.ghostNodeRecv[locNodeType, mpiDest] = self.world.graph.vs[mpiRecvIDList[mpiDest]] #sequence

                # receive request of global IDs
                lg.debug('receive request of global IDs from:  ' + str(mpiDest))##OPTPRODUCTION
                #incRequest = self.comm.recv(source=mpiDest)
                incRequest = requestIn[mpiDest][0]
                
                #pprint(incRequest)
                lnIDList = [int(self.world.graph.IDArray[xx, yy]) for xx, yy in incRequest[0]]
                #print lnIDList
                lg.debug( str(self.rank) + ' -sendIDlist:' + str(lnIDList))##OPTPRODUCTION
                self.mpiSendIDList[(locNodeType,mpiDest)] = lnIDList
                #self.ghostNodeSend[locNodeType, mpiDest] = self.world.graph.vs[IDList]
                #self.ghostNodeOut[locNodeType, mpiDest] = self.world.graph.vs[iDList]
                
                lg.debug( str(self.rank) + ' - gIDs:' + str(self.world.graph.getNodeSeqAttr('gID', lnIDList)))##OPTPRODUCTION

                for entity in [self.world.entDict[i] for i in lnIDList]:
                    entity.mpiPeers.append(mpiDest)

                # send requested global IDs
                lg.debug( str(self.rank) + ' sends to ' + str(mpiDest) + ' - ' + str(self.mpiSendIDList[(locNodeType,mpiDest)]))##OPTPRODUCTION

                x = self.world.graph.getNodeSeqAttr(label=incRequest[1], lnIDs= lnIDList )
                #print 'global IDS' + str(x)
                #print type(x)
                #print x.shape
                self._add2Buffer(mpiDest,self.world.graph.getNodeSeqAttr(label=incRequest[1], lnIDs=lnIDList))

            requestRecv = self._all2allSync()

            for mpiDest in mpiRequest.keys():
                #self.comm.send(self.ghostNodeOut[locNodeType, mpiDest][incRequest[1]], dest=mpiDest)
                #receive requested global IDs
                globIDList = requestRecv[mpiDest][0]
                
                #self.ghostNodeRecv[locNodeType, mpiDest]['gID'] = globIDList
                #print self.mpiRecvIDList[(locNodeType, mpiDest)]
                self.world.graph.setNodeSeqAttr(label=['gID'], values=globIDList, lnIDs=self.mpiRecvIDList[(locNodeType, mpiDest)])
                lg.debug( 'receiving globIDList:' + str(globIDList))##OPTPRODUCTION
                lg.debug( 'localDList:' + str(self.mpiRecvIDList[(locNodeType, mpiDest)]))##OPTPRODUCTION
                for nID, gID in zip(self.mpiRecvIDList[(locNodeType, mpiDest)], globIDList):
                    #print nID, gID
                    self.world._glob2loc[gID] = nID
                    self.world._loc2glob[nID] = gID
                #self.world.mpi.comm.Barrier()
            lg.info( 'Mpi commmunication required: ' + str(time.time()-tt) + ' seconds')

        def transferGhostNodes(self, world):
            """
            Privat method to initially transfer the data between processes and to create
            ghost nodes from the received data
            """

            messageSize = 0
            #%%Packing of data
            for nodeType, mpiPeer in sorted(self.ghostNodeQueue.keys()):

                #get size of send array
                IDsList= self.ghostNodeQueue[(nodeType, mpiPeer)]['nIds']
                connList = self.ghostNodeQueue[(nodeType, mpiPeer)]['conn']

                self.mpiSendIDList[(nodeType,mpiPeer)] = IDsList

                #nodeSeq = world.graph.vs[IDsList]

                # setting up ghost out communication
                #self.ghostNodeSend[nodeType, mpiPeer] = IDsList
                
                propList = world.graph.getPropOfNodeType(nodeType, kind='all')['names']
                #print propList
                dataPackage, packageSize = self._packData( nodeType, mpiPeer,  propList, connList)
                self._add2Buffer(mpiPeer, dataPackage)
                messageSize = messageSize + packageSize
            recvBuffer = self._all2allSync()

            lg.info('approx. MPI message size: ' + str(messageSize * 24. / 1000. ) + ' KB')

            for mpiPeer in self.peers:
                if len(recvBuffer[mpiPeer]) > 0: # will receive a message
                    pass

                for dataPackage in recvBuffer[mpiPeer]:

            #%% create ghost agents from dataDict

                    nNodes   = dataPackage['nNodes']
                    nodeType = dataPackage['nTypeID']

#                    nIDStart= world.graph.vcount()
#                    nIDs = range(nIDStart,nIDStart+nNodes)
#                    world.graph.add_vertices(nNodes)
#                    nodeSeq = world.graph.vs[nIDs]

                    
                    IDsList = world.addVertices(nodeType, nNodes)
                    # setting up ghostIn communicator
                    self.mpiRecvIDList[(nodeType, mpiPeer)] = IDsList
                    #self.ghostNodeRecv[nodeType, mpiPeer] = IDsList

                    propList = world.graph.getPropOfNodeType(nodeType, kind='all')['names']
                    propList.append('gID')
                    #if self.world.mpi.rank == 0:
                        #print dataPackage
                    
                        #print dataPackage[-1]
#                    for i, prop in enumerate(propList):
#                        #nodeSeq[prop] = dataPackage[i+1]
#                        #print prop
#                        self.world.graph.setNodeSeqAttr(label=[prop], 
#                                                        values=dataPackage['data'][[prop]],
#                                                        lnIDs=self.mpiRecvIDList[(nodeType, mpiPeer)])                        
                    self.world.graph.setNodeSeqAttr(label=propList, 
                                                    values=dataPackage['data'],
                                                    lnIDs=self.mpiRecvIDList[(nodeType, mpiPeer)])                        


                    gIDsParents = dataPackage['connectedNodes']
                    #print gIDsParents
                    

                    # creating entities with parentEntities from connList (last part of data package: dataPackage[-1])
                    for nID, gID in zip(self.mpiRecvIDList[(nodeType, mpiPeer)], gIDsParents):

                        GhostAgentClass = world.graph.nodeType2Class[nodeType][1]

                        agent = GhostAgentClass(world, mpiPeer, nID=nID)


                        parentEntity = world.entDict[world._glob2loc[gID]]
                        edgeType = world.graph.node2EdgeType[parentEntity.nodeType, nodeType]


                        agent.register(world, parentEntity, edgeType)


            lg.info('################## Ratio of ghost agents ################################################')
            for nodeTypeIdx in world.graph.nodeTypes.keys():
                nodeType = world.graph.nodeTypes[nodeTypeIdx].typeStr
                if len(world.nodeDict[nodeTypeIdx]) > 0:
                    nGhostsRatio = float(len(world.ghostNodeDict[nodeTypeIdx])) / float(len(world.nodeDict[nodeTypeIdx]))
                    lg.info('Ratio of ghost agents for type "' + nodeType + '" is: ' + str(nGhostsRatio))
            lg.info('#########################################################################################')




        def updateGhostNodes(self, nodeTypeList= 'all', propertyList='all'):
            """
            Method to update ghost node data on all processes
            """
            
            if self.comm.size == 1:
                return None
            tt = time.time()

            if nodeTypeList == 'all':
                nodeTypeList = self.world.graph.nodeTypes
            messageSize = self._updateGhostNodeData(nodeTypeList, propertyList)

            if self.world.timeStep == 0:
                lg.info('Ghost update (of approx size ' +
                     str(messageSize * 24. / 1000. ) + ' KB)' +
                     ' required: ' + str(time.time()-tt) + ' seconds')
            else:                                                           ##OPTPRODUCTION
                lg.debug('Ghost update (of approx size ' +                  ##OPTPRODUCTION
                         str(messageSize * 24. / 1000. ) + ' KB)' +         ##OPTPRODUCTION
                         ' required: ' + str(time.time()-tt) + ' seconds')  ##OPTPRODUCTION
            
            if nodeTypeList == 'all':
                nodeTypeList = self.world.graph.nodeTypes
            
            
            for nodeType in nodeTypeList:
                self.world.graph.ghostTypeUpdated[nodeType] = list()
                if propertyList in ['all', 'dyn', 'sta']:        
                    propertyList = self.world.graph.getPropOfNodeType(nodeType, kind=propertyList)['names']
                
                
                for prop in propertyList:
                    self.world.graph.ghostTypeUpdated[nodeType].append(prop)
                
#                if len(self.world.ghostNodeDict[nodeType]) > 0:
#                    firstGhostID = self.world.ghostNodeDict[nodeType][0]
#                    ghost = self.world.entDict[firstGhostID]
#                    ghost.setGhostUpdate(self.world.graph.ghostTypeUpdated[nodeType])
                
        def queueSendGhostNode(self, mpiPeer, nodeType, entity, parentEntity):

            if (nodeType, mpiPeer) not in self.ghostNodeQueue.keys():
                self.ghostNodeQueue[nodeType, mpiPeer] = dict()
                self.ghostNodeQueue[nodeType, mpiPeer]['nIds'] = list()
                self.ghostNodeQueue[nodeType, mpiPeer]['conn'] = list()

            self.ghostNodeQueue[nodeType, mpiPeer]['nIds'].append(entity.nID)
            self.ghostNodeQueue[nodeType, mpiPeer]['conn'].append(parentEntity.gID)



        def all2all(self, value):
            """
            This method is a quick communication implementation that allows +
            sharing data between all processes

            """
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

    class Random():

        def __init__(self, world):
            self.world = world # make world availabel in class random

        def entity(nChoice, entType):
            ids = np.random.choice(earth.nodeDict[entType],nChoice,replace=False)
            return [earth.entDict[idx] for idx in ids]


    #%% INIT WORLD
    def __init__(self,
                 simNo,
                 outPath,
                 spatial=True,
                 nSteps= 1,
                 maxNodes = 1e6,
                 maxEdges = 1e6,
                 debug = False,
                 mpiComm=None,
                 caching=True,
                 queuing=True):

        self.simNo    = simNo
        self.timeStep = 0
        self.para     = dict()
        self.spatial  = spatial
        self.maxNodes = int(maxNodes)
        self.globIDGen = self._globIDGen()
        self.nSteps   = nSteps
        self.debug    = debug

        self.para     = dict()
        self.queuing = queuing  # flag that indicates the vertexes and edges are queued and not added immediately
        self.caching = caching  # flat that indicate that edges and peers are cached for faster access

        # GRAPH
        self.graph    = ABMGraph(self, maxNodes, maxEdges)
        self.para['outPath'] = outPath

        
        self.globalRecord = dict() # storage of global data

        # queues
        if self.queuing:
            self.queue      = Queue(self)
            self.addEdge    = self.queue.addEdge
            self.addEdges   = self.queue.addEdges
            self.addVertex  = self.queue.addVertex
            
        else:
            self.addEdge        = self.graph.addEdge
            self.addEdges       = self.graph.addEdges
            self.delEdges       = self.graph.delete_edges
            self.addVertex      = self.graph.addNode
            self.addVertices    = self.graph.addNodes
        # MPI communication
        self.mpi = self.Mpi(self, mpiComm=mpiComm)
        lg.debug('Init MPI done')##OPTPRODUCTION
        if self.mpi.comm.rank == 0:
            self.isRoot = True
        else:
            self.isRoot = False

        # IO
        self.io = self.IO(self, nSteps, self.para['outPath'])
        lg.debug('Init IO done')##OPTPRODUCTION
        # Globally synced variables
        self.graph.glob     = self.Globals(self)
        lg.debug('Init Globals done')##OPTPRODUCTION

        # enumerations
        self.enums = dict()


        # node lists and dicts
        self.nodeDict       = dict()
        self.ghostNodeDict  = dict()
        
        # dict of list that provides the storage place for each agent per nodeType
        self.dataDict       = dict()

        self.entList   = list()
        self.entDict   = dict()
        self.locDict   = dict()

        self._glob2loc = dict()  # reference from global IDs to local IDs
        self._loc2glob = dict()  # reference from local IDs to global IDs

        # inactive is used to virtually remove nodes
        #self.registerNodeType('inactiv', None, None)
        #self.registerEdgeType('inactiv', None, None)


    def _globIDGen(self):
        i = -1
        while i < self.maxNodes:
            i += 1
            yield (self.maxNodes*(self.mpi.rank+1)) +i

# GENERAL FUNCTIONS

    def glob2loc(self, idx):
        return self._glob2loc[idx]

    def loc2glob(self, idx):
        return self._loc2glob[idx]

    

    def getLocationDict(self):
        """
        The locationDict contains all instances of locations that are
        accessed by (x,y) coordinates
        """
        return self.locDict

    def getNodeDict(self, nodeType):
        """
        The nodeDict contains all instances of different entity types
        """
        return self.nodeDict[nodeType]

    def getParameter(self,paraName=None):
        """
        Returns a dictionary of all simulations parameters
        """
        if paraName is not None:
            return self.para[paraName]
        else:
            return self.para

    def setParameter(self, paraName, paraValue):
        """
        This method is used to set parameters of the simulation
        """
        self.para[paraName] = paraValue

    def setParameters(self, parameterDict):
        """
        This method allows to set multiple parameters at once
        """
        for key in parameterDict.keys():
            self.setParameter(key, parameterDict[key])


    def getNodeValues(self, prop, nodeType=None, idxList=None):
        """
        Method to read values of node sequences at once
        Return type is numpy array
        Only return non-ghost agent properties
        """
        #if nodeType:
        #    assertUpdate(self.graph, prop, nodeType)
        
        if idxList:
            return np.asarray(self.graph.vs[idxList][prop])
        elif nodeType:
            return np.asarray(self.graph.vs[self.nodeDict[nodeType]][prop])

    def setNodeValues(self, prop, valueList, nodeType=None, idxList=None):
        """
        Method to read values of node sequences at once
        Return type is numpy array
        """
        if idxList:
            self.graph.vs[idxList][prop] = valueList
        elif nodeType:
            self.graph.vs[self.nodeDict[nodeType]][prop] = valueList

#    def getNodeData(self, propName, nodeType=None):
#        """
#        Method to retrieve all properties of all entities of one nodeType
#        """
#        nodeIdList = self.nodeDict[nodeType]
#
#        return np.asarray(self.graph.vs[nodeIdList][propName])


    def getEdgeData(self, propName, edgeType=None):
        """
        Method to retrieve all properties of all entities of one edgeType
        """
        return self.graph.es.select(type=edgeType)[propName]
    
  
    def getEntity(self, nodeID=None, globID=None):
        """
        Methode to retrieve a certain instance of an entity by the nodeID
        """
        if nodeID is not None:
            return self.entDict[nodeID]
        if globID is not None:
            return self.entDict[self._glob2loc[globID]]


    #TODO add init for non-spatial init of communication
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
        lg.debug('rank array: ' + str(rankArray)) ##OPTPRODUCTION
        # tuple of idx array of cells that correspond of the spatial input maps 
        self.cellMapIds = np.where(rankArray == self.mpi.rank)

        # create vertices
        for x in range(nodeArray.shape[0]):
            for y in range(nodeArray.shape[1]):

                # only add an vertex if spatial location exist
                if not np.isnan(rankArray[x,y]) and rankArray[x,y] == self.mpi.rank:

                    loc = LocClassObject(self, pos = [x, y])
                    IDArray[x,y] = loc.nID
                    
                    self.registerLocation(loc, x, y)          # only for real cells
                    #self.registerNode(loc,nodeType)     # only for real cells
                    loc.register(self)

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
                        
                        self.registerNode(loc,nodeType,ghost=True) #so far ghost nodes are not in entDict, nodeDict, entList
                        
                        #self.registerLocation(loc, xDst, yDst)
                        ghostLocationList.append(loc)
        self.graph.IDArray = IDArray

        if self.queuing:
            self.queue.dequeueVertices(self)

        fullSourceList      = list()
        fullTargetList      = list()
        fullWeightList          = list()
        #nConnection  = list()
        #print 'rank: ' +  str(self.locDict)

        for (x,y), loc in self.locDict.items():

            srcID = loc.nID
            
            weigList = list()
            destList = list()
            sourceList = list()
            targetList = list()
            for (dx,dy,weight) in connList:

                xDst = x + dx
                yDst = y + dy

                # check boundaries of the destination node
                if xDst >= xOrg and xDst < xMax and yDst >= yOrg and yDst < yMax:

                    trgID = IDArray[xDst,yDst]
                    #assert

                    if not np.isnan(trgID): #and srcID != trgID:
                        destList.append(int(trgID))
                        weigList.append(weight)
                        sourceList.append(int(srcID))
                        targetList.append(int(trgID))

            #normalize weight to sum up to unity
            sumWeig = sum(weigList)
            weig    = np.asarray(weigList) / sumWeig
            #print loc.nID
            #print connectionList
            fullSourceList.extend(sourceList)
            fullTargetList.extend(targetList)
            #nConnection.append(len(connectionList))
            fullWeightList.extend(weig)


            
        #eStart = self.graph.ecount()
        #print fullSourceList
        #print fullTargetList
        
        self.graph.addEdges(1, fullSourceList, fullTargetList, weig=fullWeightList)


#        eStart = 0
#        ii = 0
#        for _, loc in tqdm.tqdm(self.locDict.items()):
#        #for cell, cellID in self.iterEntAndIDRandom(1, random=False):
#            loc.setEdgeCache(range(eStart,eStart + nConnection[ii]), 1)
#            #assert loc._graph.es[loc._graph.incident(loc.nID,'out')].select(type_ne=0).indices == range(eStart,eStart + nConnection[ii])
#            eStart += nConnection[ii]
#            ii +=1
            
        lg.debug('starting initCommunicationViaLocations')##OPTPRODUCTION
        self.mpi.initCommunicationViaLocations(ghostLocationList, nodeType)
        lg.debug('finished initCommunicationViaLocations')##OPTPRODUCTION

    def iterEdges(self, edgeType):
        """
        Iteration over edges of specified type. Default returns
        non-ghosts in order of creation.
        """
        for i in range(self.graph.ecount()):
            if self.graph.es[i]['type'] == edgeType:
                yield self.graph.es[i]

    def iterEntRandom(self,nodeType, ghosts = False, random=True):
        """
        Iteration over entities of specified type. Default returns
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
            shuffled_list = sorted(nodeDict, key=lambda x: rd.random())
            return [self.entDict[i] for i in shuffled_list]
        else:
            return  [self.entDict[i] for i in nodeDict]

    def iterEntAndIDRandom(self, nodeType, ghosts = False, random=True):
        """
        Iteration over entities of specified type and their IDs . Default returns
        non-ghosts in random order.
        """
        if isinstance(nodeType,str):
            nodeType = self.types.index(nodeType)

        if ghosts:
            nodeDict = self.ghostnodeDict[nodeType]
        else:
            nodeDict = self.nodeDict[nodeType]

        if random:
            shuffled_list = sorted(nodeDict, key=lambda x: rd.random())
            return  [(self.entList[i], i) for i in shuffled_list]
        else:
            return  [(self.entList[i], i) for i in nodeDict]



    def registerNodeType(self, typeStr, AgentClass, GhostAgentClass, staticProperies = [], dynamicProperies = []):
        """
        Method to register a node type:
        - Registers the properties of each nodeType for other purposes, e.g. I/O
        of these properties
        - update of convertions dicts:
            - class2NodeType
            - nodeType2Class
        - creations of access dictionaries
            - nodeDict
            - ghostNodeDict
        - enumerations
        """
        
        # type is an required property
        #assert 'type' and 'gID' in staticProperies              ##OPTPRODUCTION

        nodeTypeIdx = len(self.graph.nodeTypes)+1

        self.graph.addNodeType(nodeTypeIdx, 
                               typeStr, 
                               AgentClass,
                               GhostAgentClass,
                               staticProperies, 
                               dynamicProperies)
        self.nodeDict[nodeTypeIdx]      = list()
        self.dataDict[nodeTypeIdx]      = list()
        self.ghostNodeDict[nodeTypeIdx] = list()
        self.enums[typeStr] = nodeTypeIdx
        return nodeTypeIdx


    def registerEdgeType(self, typeStr,  nodeType1, nodeType2, staticProperies = [], dynamicProperies=[]):
        """
        Method to register a edge type:
        - Registers the properties of each edgeType for other purposes, e.g. I/O
        of these properties
        - update of convertions dicts:
            - node2EdgeType
            - edge2NodeType
        - update of enumerations
        """
        
        #assert 'type' in staticProperies # type is an required property             ##OPTPRODUCTION

        edgeTypeIdx = len(self.graph.edgeTypes)+1
        self.graph.addEdgeType(edgeTypeIdx, typeStr, staticProperies, dynamicProperies, nodeType1, nodeType2)
        self.enums[typeStr] = edgeTypeIdx

        return  edgeTypeIdx

    def registerNode(self, agent, typ, ghost=False):
        """
        Method to register instances of nodes
        -> update of:
            - entList
            - endDict
            - _glob2loc
            - _loc2glob
        """
        #print 'assert' + str((len(self.entList), agent.nID))
        #assert len(self.entList) == agent.nID                                  ##OPTPRODUCTION
        self.entList.append(agent)
        self.entDict[agent.nID] = agent
        self._glob2loc[agent.gID] = agent.nID
        self._loc2glob[agent.nID] = agent.gID

        if ghost:
            self.ghostNodeDict[typ].append(agent.nID)
        else:
            #print typ
            self.nodeDict[typ].append(agent.nID)
            self.dataDict[typ].append(agent.dataID)

    def deRegisterNode(self):
        """
        Method to remove instances of nodes
        -> update of:
            - entList
            - endDict
            - _glob2loc
            - _loc2glob
        """
        self.entList[agent.nID] = None
        del self.entDict[agent.nID]
        del self._glob2loc[agent.gID]
        del self._loc2glob[agent.gID]
            
        
        self.nodeDict[self.nodeType].remove(agent.nID)
        self.dataDict[self.nodeType].remove(agent.dataID)

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
            
    def registerLocation(self, location, x, y):

        self.locDict[x,y] = location

    def resetEdgeCache(self):
        self._cache.resetEdgeCache()

    def resetNodeCache(self):
        self._cache.resetNodeCache()

    def returnMpiComm(self):
        return self.mpi.comm

    def returnGraph(self):
        return self.graph

    def returnGlobalRecord(self):
        return self.globalRecord

    def returnGlobals(self):
        return self.graph.glob
    
    def finalize(self):
        """
        Method to finalize records, I/O and reporter
        """
        from class_auxiliary import saveObj

        # finishing reporter files
#        for writer in self.reporter:
#            writer.close()

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
                    
    def view(self,filename = 'none', vertexProp='none', dispProp='gID', layout=None):
        try:
            raise DeprecationWarning('not supported right now')
            raise NameError('sorry')
        
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
        
class easyUI():
    """ 
    Easy-to-use user interace that provides high-level methods or functions to improve 
    user friendliness of the library
    """
    def __init__(earth):
        pass

        
if __name__ == '__main__':
    
    # MINIMAL FUNCTION TEST
    mpiComm =mpi4py.MPI.COMM_WORLD
    mpiRank = mpiComm.Get_rank()
    mpiSize = mpiComm.Get_size()
    
    lg.basicConfig(filename='log_R' + str(mpiRank),
                filemode='w',
                format='%(levelname)7s %(asctime)s : %(message)s',
                datefmt='%m/%d/%y-%H:%M:%S',
                level=lg.DEBUG)    
    
    outputPath = '.'
    simNo= 0
    earth = World(simNo,
                  outputPath,
                  nSteps=10,
                  maxNodes=1e4,
                  maxEdges=1e4,
                  debug=True,
                  mpiComm=mpiComm,
                  caching=False,
                  queuing=False)



        
    log_file  = open('out' + str(earth.mpi.rank) + '.txt', 'w')
    sys.stdout = log_file
    earth.graph.glob.registerValue('test' , earth.mpi.comm.rank,'max')
    earth.graph.glob.registerStat('meantest', np.random.randint(5,size=3).astype(float),'mean')
    earth.graph.glob.registerStat('stdtest', np.random.randint(5,size=2).astype(float),'std')
    print earth.graph.glob['test']
    print earth.graph.glob['meantest']
    print 'mean of values: ',earth.graph.glob['meantest'],'-> local maen: ',earth.graph.glob['meantest']
    print 'std od values:  ',earth.graph.glob['stdtest'],'-> local std: ',earth.graph.glob['stdtest']
    earth.graph.glob.sync()
    print earth.graph.glob['test']
    print 'global mean: ', earth.graph.glob['meantest']
    print 'global std: ', earth.graph.glob['stdtest']



    import sys
    from class_auxiliary import computeConnectionList

    mpiRankLayer   = np.asarray([[0, 0, 0, 0, 1],
                              [np.nan, np.nan, np.nan, 1, 1]])
    if mpiComm.size == 1:
        mpiRankLayer = mpiRankLayer * 0
    
    #landLayer = np.load('rankMap.npy')
    connList = computeConnectionList(1.5)
    #print connList
    CELL    = earth.registerNodeType('cell' , AgentClass=Location, GhostAgentClass= GhostLocation,
                                      staticProperies = [('gID', np.int32, 1),
                                                         ('pos', np.int16, 2)],
                                      dynamicProperies = [('value', np.float32, 1),
                                                          ('value2', np.float32, 1)])

    AG      = earth.registerNodeType('agent', AgentClass=Agent   , GhostAgentClass= GhostAgent,
                                      staticProperies   = [('gID', np.int32, 1),
                                                           ('pos', np.int16, 2)],
                                      dynamicProperies  = [('value3', np.float32, 1)])

    C_LOLO = earth.registerEdgeType('cellCell', CELL, CELL, [('weig', np.float32, 1)])
    C_LOAG = earth.registerEdgeType('cellAgent', CELL, AG)
    C_AGAG = earth.registerEdgeType('AgAg', AG, AG, [('weig', np.float32, 1)])

    earth.initSpatialLayer(mpiRankLayer, connList, CELL, Location, GhostLocation)
    #earth.mpi.initCommunicationViaLocations(ghostLocationList)

    for cell in earth.iterEntRandom(CELL):
        cell._node['value'] = earth.mpi.rank
        cell._node['value2'] = earth.mpi.rank+2

        if cell.getValue('pos')[0] == 0:
            x,y = cell.getValue('pos')
            agent = Agent(earth, value3=np.random.randn(),pos=(x,  y))
            print 'agent.nID' + str(agent.nID)
            agent.register(earth, cell, C_LOAG)
            #cell.registerEntityAtLocation(earth, agent,_cLocAg)

    #earth.queue.dequeueVertices(earth)
    #earth.queue.dequeueEdges(earth)
#            if agent.node['nID'] == 10:
#                agent.addConnection(8,_cAgAg)

    #earth.mpi.syncNodes(CELL,['value', 'value2'])
    earth.mpi.updateGhostNodes([CELL])
    print earth.graph.getPropOfNodeType(CELL, 'names')
    print str(earth.mpi.rank) + ' values' + str(earth.graph.nodes[CELL]['value'])
    print str(earth.mpi.rank) + ' values2: ' + str(earth.graph.nodes[CELL]['value2'])

    #print earth.mpi.ghostNodeRecv
    #print earth.mpi.ghostNodeSend

    print earth.graph.getPropOfNodeType(AG, 'names')

    print str(earth.mpi.rank) + ' ' + str(earth.nodeDict[AG])

    print str(earth.mpi.rank) + ' SendQueue ' + str(earth.mpi.ghostNodeQueue)

    earth.mpi.transferGhostNodes(earth)
    #earth.mpi.recvGhostNodes(earth)

    #earth.queue.dequeueVertices(earth)
    #earth.queue.dequeueEdges(earth)

    cell.getPeerIDs(nodeType=CELL, mode='out')
    earth.view(str(earth.mpi.rank) + '.png', layout=ig.Layout(earth.graph.nodes[CELL]['pos'].tolist()))

    print str(earth.mpi.rank) + ' ' + str(earth.graph.nodes[AG].indices)
    print str(earth.mpi.rank) + ' ' + str(earth.graph.nodes[AG]['value3'])

    for agent in earth.iterEntRandom(AG):
        agent['value3'] = earth.mpi.rank+ agent.nID
        assert agent.getValue('value3') == earth.mpi.rank+ agent.nID

    earth.mpi.updateGhostNodes([AG])

    earth.io.initNodeFile(earth, [CELL, AG])

    earth.io.writeDataToFile(0, [CELL, AG])

    print str(earth.mpi.rank) + ' ' + str(earth.graph.nodes[AG]['value3'])

    #%% testing agent methods 
    peerList = cell.getPeerIDs(C_LOLO)
    writeValues = np.asarray(range(len(peerList))).astype(np.float32)
    cell.setPeerValues('value', writeValues, C_LOLO )
    readValues = cell.getPeerValues('value', C_LOLO)
    assert all(readValues == writeValues)
    assert all(earth.graph.getNodeSeqAttr('value', peerList,) == writeValues)
    print 'Peer values write/read successful'
    edgeList = cell.getEdgeIDs(C_LOLO)
    writeValues = np.random.random(len(edgeList[1])).astype(np.float32)
    cell.setEdgeValues('weig',writeValues, C_LOLO)
    readValues, _  = cell.getEdgeValues('weig',C_LOLO)
    assert all(readValues == writeValues)
    print 'Edge values write/read successful'
    
    friendID = earth.nodeDict[AG][0]
    agent.addConnection(friendID, C_AGAG, weig=.51)
    assert earth.graph.isConnected(agent.nID, friendID, C_AGAG)
    readValue, _ = agent.getEdgeValues('weig',C_AGAG)
    assert readValue[0] == np.float32(0.51)
    
    agent.remConnection(friendID, C_AGAG)
    assert not(earth.graph.isConnected(agent.nID, friendID, C_AGAG))
    print 'Adding/removing connection successfull'
    
    value = agent['value3']
    agent['value3'] +=1
    assert agent['value3'] == value +1
    assert earth.graph.getNodeAttr('value3', agent.nID) == value +1
    print 'Value access and increment sucessful'
    
    #%%
    pos = (0,4)
    cellID = earth.graph.IDArray[pos]
    cell40 = earth.entDict[cellID]
    agentID = cell40.getPeerIDs(edgeType=C_LOAG, mode='in')
    connAgent = earth.entDict[agentID[0]]
    assert all(cell40['pos'] == connAgent['pos'])
    
    if earth.mpi.rank == 1:
        cell40['value'] = 32.0
        connAgent['value3'] = 43.2
    earth.mpi.updateGhostNodes([CELL])
    earth.mpi.updateGhostNodes([AG],['value3'])
    
    
    buff =  earth.mpi.all2all(cell40['value'])
    assert buff[0] == buff[1]
    print 'ghost update of cells successful (all attributes) '
    
    buff =  earth.mpi.all2all(connAgent['value3'])
    assert buff[0] == buff[1]
    print 'ghost update of agents successful (specific attribute)'