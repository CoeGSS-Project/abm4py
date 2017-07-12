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
Entities should therefore by fully defined by their global ID and local ID and
the underlying graph that contains all properties    

Thus, as far as possible save any property of all entities in the graph

Communication is managed via the spatial location 

Thus, the function "location.registerEntity" does initialize ghost copies
    
    
"""
from __future__ import division

import sys
sys.path = ['/home/gcf/python/synEarth/agModel/h5py/build/lib.linux-x86_64-2.7'] + sys.path 

import igraph as ig
import numpy as np
from mpi4py import MPI
import time 

from class_auxiliary import computeConnectionList

class Queue():

    def __init__(self, world):
        self.graph = world.graph
        self.edgeList       = dict()
        self.edgeProperties = dict()
        
        self.currNodeID     = None
        self.nodeList       = list()
        self.nodeTypeList   = dict()
        self.nodeProperties = dict()        
        
        
    def addVertex(self, nodeType, **kwProperties):
        #print kwProperties.keys()
        if len(self.nodeList) == 0:
            self.currNodeID = len(self.graph.vs)

        # adding of the data
        nID = self.currNodeID
        self.nodeList.append(nID)               # nID to general list
        #kwProperties.update({'nID': nID})
        
        if nodeType not in self.nodeProperties.keys():
            # init of new nodeType
            propertiesOfType = self.graph.nodeProperies[nodeType]
            #print propertiesOfType
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
            
        return nID

    def dequeueVertices(self, world):
        
        if len(self.nodeList) ==0:
            return
        assert self.nodeList[0] == self.graph.vcount() # check that the queuing idx is consitent
        #print self.nodeProperties
        #adding all nodes first
        self.graph.add_vertices(len(self.nodeList))
        
        # adding data per type
        for nodeType in self.nodeTypeList.keys():

            nodeSeq = self.graph.vs[self.nodeTypeList[nodeType]]    # node sequence of specified type
            nodeSeq['type'] = nodeType                                # adding type to graph
            
            for prop in self.nodeProperties[nodeType].keys():
                print prop
                print nodeType
                print self.nodeProperties[nodeType][prop]
                nodeSeq[prop] = self.nodeProperties[nodeType][prop] # adding properties 

        for i, entity in enumerate([world.entList[i] for i in self.nodeList]):
            entity.node = self.graph.vs[self.nodeList[i]]
            assert entity.nID == entity.node.index
        # reset queue
        self.currNodeID     = None
        self.nodeList       = list()
        self.nodeTypeList   = dict()
        self.nodeProperties = dict()  

                    
    def addEdge(self, source, target, **kwProperties):
        
        edgeType = kwProperties['type']
        #possible init
        if edgeType not in self.edgeList.keys():
            self.edgeList[edgeType]         = list()
            self.edgeProperties[edgeType]   = dict()
            for prop in kwProperties.keys():
                #print prop
                self.edgeProperties[edgeType] [prop] = list()
            
        # add edge source-target-tuple            
        self.edgeList[edgeType].append((source, target))

        # add properties
        #self.edgeProperties[edgeType]['type'].append(edgeType)
        for propKey in kwProperties.keys():
            self.edgeProperties[edgeType][propKey].append(kwProperties[propKey])


    def dequeueEdges(self):

        if len(self.edgeList) ==0:
            return
        print "dequeuing edges"
        print self.edgeList.keys()
        for edgeType in self.edgeList.keys():
            print 'creating edges: ' + str(self.edgeList[edgeType])
            eStart = self.graph.ecount()
            self.graph.add_edges(self.edgeList[edgeType])
            for prop in self.edgeProperties[edgeType].keys():
                self.graph.es[eStart:][prop] = self.edgeProperties[edgeType][prop]
            

        #for node in self.entList:
        #    node.updateEdges()

        # empty queue
        self.edgeList       = list()
        self.edgeProperties = dict()  
        
        # if all queues are done, set complete flag
        #if len(self.graph.edgeQueues) == 0:
        #    self.graph.edgesComplete = True

    

      
################ ENTITY CLASS #########################################    
# general ABM entity class for objects connected with the graph

class Entity():
        
    def __init__(self, world, nID = None, **kwProperties):
        nodeType =  world.graph.class2NodeType[self.__class__]
        #len(world.graph.nodeTypes) >=  nodeType
        self.graph  = world.graph
        self.queue  = world.queue
        self.gID    = self.__getGlobID__(world)
        self.edges  = dict()
       
        # create instance from existing node
        if nID is not None:
            
            self.nID = nID
            self.node = self.graph.vs[nID]
            self.node['nID']  = nID
            print 'nID:' + str(nID) + ' gID: ' + str(self.node['gID'])
            return
        
        # create instance newly
        if world.queuing:
            # add vertex to queue
            kwProperties.update({'type': nodeType, 'gID':self.gID})    
            self.nID = world.queue.addVertex(nodeType, **kwProperties)
            self.node = None
        else:
            # add vertex to graph
            self.nID  = len(self.graph.vs)
            kwProperties.update({'nID':self.nID, 'type': nodeType, 'gID':self.gID})
            self.graph.add_vertex( **kwProperties)
            self.node = self.graph.vs[self.nID]            # short cuts for value access
            
            
        self.edges = dict()                



    def __updateEdges__(self):
        #TODO re-think this approach            
        self.edgesAll = self.graph.es[self.graph.incident(self.nID,'out')]
        for typ in self.graph.edgeTypes:
            self.edges[typ] = self.edgesAll.select(type=typ)


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
        self.__updateEdges__()
            
    def remConnection(self, friendID,edgeType):
        eID = self.graph.get_eid(self.nID,friendID)
        self.graph.delete_edges(eID)
        self.__updateEdges__()

    def setValue(self,prop,value):
        self.node[prop] = value
        
    def getValue(self,prop):
        return self.node[prop]
    
    def addValue(self, prop, value, idx = None):
        if idx is None:
            self.node[prop] += value
        else:
            self.node[prop][idx] += value
    
    def delete(self,Earth):
        nID = self.nID
        
        #self.graph.delete_vertices(nID) # not really possible at the current igraph lib
        # Thus, the node is set to inactive and removed from the iterator lists
        # This is due to the library, but the problem is general and pose a challenge.
        Earth.graph.vs[nID]['type'] = 0
        Earth.nodeDict[self.type].remove(nID)
        #remove edges        
        eIDSeq = self.graph.es.select(_target=self.nID).indices        
        self.graph.delete_edges(eIDSeq)
        eIDSeq = self.graph.es.select(_source=self.nID).indices        
        self.graph.delete_edges(eIDSeq)

    def getEdgeValues(self, prop, edgeType=None, mode="out"):
        values = [] 
        edges  = []
        eList  = self.graph.incident(self.nID,mode)

        if edgeType is not None:
            for eIdx in eList:
                if self.graph.es[eIdx]['type'] == edgeType:
                    values.append(self.graph.es[eIdx][prop])   
                    edges.append(eIdx)
        else:
            for edge in eList:
                values.append(self.graph.es[eIdx][prop])        
                edges.append(eIdx)
        return values, edges
        
    def getEdges(self, edgeType=0):
        edges = self.edges[edgeType]
        return edges
    
    def getEdgeValuesFast(self, prop, edgeType=0):
        edges = self.edges[edgeType]
        return edges[prop], edges

    def getConnNodes(self, nodeType=0, mode='out'):
        if mode is None:
            neigbours = self.node.neighbors()
        else:
            neigbours = self.node.neighbors(mode)
    
        return [x for x in neigbours if x['type'] == nodeType]

    def getConnNodeIDs(self, nodeType=0, mode='out'):
        if mode is None:
            neigbours = self.node.neighbors()
        else:
            neigbours = self.node.neighbors(mode)
    
        return [x.index for x in neigbours if x['type'] == nodeType]
        
    def getConnNodeValues(self, prop, nodeType=0, mode='out'):
        nodeDict = self.node.neighbors(mode)
        neighbourIDs     = list()
        values          = list()

        for node in nodeDict:
            if node['type'] == nodeType:
                neighbourIDs.append(node.index)   
                values.append(node[prop])
        
        return values, neighbourIDs


        
class Agent(Entity):
    
    def __getGlobID__(self,world):
        return world.globIDGen.next()
        
    def __init__(self, world, **kwProperties):
        Entity.__init__(self, world, **kwProperties)
        self.mpiOwner =  int(world.mpi.rank)

    def getLocationValue(self,prop):
    
        return self.loc.node[prop] 
    
    def register(self, world):
            world.registerNode(self, self.node['type'], ghost=False)
            
class GhostAgent(Entity):
    
    def __getGlobID__(self,world):
        
        return None # global ID need to be aquired via MPI communication
        
    def __init__(self, world, owner, nID = None):
        Entity.__init__(self, world, nID)
        self.mpiOwner =  int(owner)

    def getLocationValue(self,prop):
    
        return self.loc.node[prop] 
    
    def register(self, world):
            world.registerNode(self, self.node['type'], ghost=True)
            
################ LOCATION CLASS #########################################      
class Location(Entity):

    def __getGlobID__(self,world):
        return world.globIDGen.next()
    
    def __init__(self, world, **kwProperties):
        Entity.__init__(self,world, **kwProperties)
        #self.graph.vs[self.nID]['pos']= 
        self.mpiOwner = int(world.mpi.rank)
        self.mpiPeers = list()
        self.queuing = world.queuing
    
    def register(self, world):
        world.registerNode(self, self.node['type'], ghost=False)
    
    def registerEntity(self, world, entity, nodeType, edgeType):
        print edgeType
        if self.queuing:
            world.queue.addEdge(entity.nID,self.nID, type=edgeType)         
        else:
            world.graph.add_edge(entity.nID,self.nID, type=edgeType)         
        entity.loc = self
        
        if len(self.mpiPeers) > 0: # node has ghosts on other processes
            #nodeType = entity.node['type']
            for mpiPeer in self.mpiPeers:
                world.mpi.queueSendGhostNode( mpiPeer, nodeType, entity)

        
        
class GhostLocation(Entity):
    
    def __getGlobID__(self,world):
        
        return None
    
    def __init__(self, world, pos,  owner):
        Entity.__init__(self,world,pos=pos)
        #self.graph.vs[self.nID]['pos']= (xPos,yPos)
        self.mpiOwner = int(owner)
        self.queuing = world.queuing

    def register(self, world):
        world.registerNode(self, self.node['type'], ghost=True)
        
    def registerEntity(self, world, entity, edgeType):
        print edgeType
        if self.queuing:
            world.queue.addEdge(entity.nID,self.nID, type=edgeType)         
        else:
            world.graph.add_edge(entity.nID,self.nID, type=edgeType)         
        entity.loc = self        
        
################ WORLD CLASS #########################################            
class World:
    #%% World sub-classes
    class IO():  
        
        
        
        class Record():
            
            def __init__(self, nAgents, agIds, nAgentsGlob, loc2GlobIdx, nodeType):
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
                
                
            
            def addAttr(self, name, nProp):
                attrIdx = range(self.nAttr,self.nAttr+nProp)
                self.attributeList.append(name)
                self.attrIdx[name] = attrIdx
                self.nAttr += len(attrIdx)
                self.header += [name] * nProp
                
            def initStorage(self):
                self.data = np.zeros([self.nAgents,self.nAttr ])
                
            def addData(self, timeStep, graph):
                self.timeStep = timeStep
                for attr in self.attributeList:
                    if len(self.attrIdx[attr]) == 1:
                        self.data[:,self.attrIdx[attr]] = np.expand_dims(graph.vs[self.ag2FileIdx][attr],1)
                    else:
                        self.data[:,self.attrIdx[attr]] = graph.vs[self.ag2FileIdx][attr]
                            
            def writeData(self, h5File):
                print self.header
                path = '/' + str(self.nodeType)+ '/' + str(self.timeStep)
                self.dset = h5File.create_dataset(path, (self.nAgentsGlob,self.nAttr), dtype='f')
                self.dset[self.loc2GlobIdx,] = self.data
                         
        def __init__(self, world):
            import h5py  
            
            self.graph      = world.graph
            self.timeStep   = world.timeStep
            self.h5File    =  h5py.File('nodeOutput.hdf5', 'w', driver='mpio', comm=world.mpi.comm)
            self.comm       = world.mpi.comm
            self.outData    = dict()
            
              
            
        def initNodeFile(self, world, nodeTypes):
            """ Initialized the internal data structure for later I/O""" 
            
            for nodeType in nodeTypes:
                self.h5File.create_group(str(nodeType))
                
                nAgents = len(world.nodeDict[nodeType])
                self.nAgentsAll = np.empty(1*self.comm.size,dtype=np.int)

                self.nAgentsAll = self.comm.alltoall([nAgents]*self.comm.size)    
                print self.nAgentsAll 
                
                nAgentsGlob = sum(self.nAgentsAll)
                cumSumNAgents = np.zeros(self.comm.size+1).astype(int)
                cumSumNAgents[1:] = np.cumsum(self.nAgentsAll)
                loc2GlobIdx = range(cumSumNAgents[self.comm.rank], cumSumNAgents[self.comm.rank+1])
                print loc2GlobIdx
                
                rec = self.Record(nAgents, world.nodeDict[nodeType], nAgentsGlob, loc2GlobIdx, nodeType)
                self.attributes = world.graph.nodeProperies[nodeType][:]
                self.attributes.remove('type')

                
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
                # allocate storage
                rec.initStorage()
                self.outData[nodeType] = rec
                
        def gatherNodeData(self):                
            """ Transfers data from the graph to the I/O storage"""
            for typ in self.outData.keys():
                self.outData[typ].addData(self.timeStep, self.graph)
        
        def writeDataToFile(self):
            """ Writing data to hdf5 file"""
            for typ in self.outData.keys():
                self.outData[typ].writeData(self.h5File)
        
        def initEdgeFile(self, edfeTypes):
            pass
            
    class Mpi():
        
        def __init__(self, world):
            
            self.world = world
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.peers    = list()     # list of ranks of all processes that have ghost duplicates of this process   
            
            self.ghostNodeQueue = dict()
            #self.ghostEdgeQueue = dict()
            
            self.ghostNodeIn  = dict()     # ghost vertices on this process that receive information
            self.ghostNodeOut = dict()     # vertices on this process that provide information to ghost nodes on other process
            
            #self.ghostEdgeIn  = dict()     # ghost edves on this process that receive information
            #self.ghostEdgeOut = dict()     # edges on this process that provide information to ghost nodes on other process
            
            world.send = self.comm.send
            world.recv = self.comm.recv    
            
            world.isend = self.comm.isend    
            world.irecv = self.comm.irecv
        #%% Privat functions
        def __packData__(self, packageDict, nodeType, mpiPeer, nodeSeq, propList, connList=None):

            nNodes = len(nodeSeq)
            packageDict[nodeType, mpiPeer] = list()
            packageDict[nodeType, mpiPeer].append(nNodes)
            for prop in propList:
                packageDict[nodeType, mpiPeer].append(nodeSeq[prop])                            
            
            if connList is not None:
                packageDict[nodeType, mpiPeer].append(connList)
            return packageDict
                
            
        def __sendDataAnnouncement__(self, announcements):    
            #%% Sending of announcments and data
            for mpiPeer in self.peers:
                # send what nodetypes to expect
                if mpiPeer in announcements.keys():
                    self.comm.isend(announcements[mpiPeer], dest=mpiPeer, tag=9999)
                else:
                    self.comm.isend([], dest=mpiPeer, tag=9999)
                    
        def __sendData__(self, packageDict):
            for nodeType, mpiPeer in packageDict:
                self.comm.isend(packageDict[nodeType, mpiPeer], dest=mpiPeer, tag=nodeType)
                
        #%% Puplic functions        

        def syncNodes(self, nodeType, propertyList='all'):
            tt = time.time()
            buffers = dict()
            
            if propertyList=='all':
                print self.graph.nodeProperies
                propertyList = self.world.graph.nodeProperies[nodeType][:]
                propertyList.remove('type')
                propertyList.remove('gID')
            else:
                if not isinstance(propertyList,list):
                    propertyList = [propertyList]
                
                assert all([prop in self.world.graph.nodeProperies[nodeType] for prop in propertyList])
                
    
                
            for peer in self.peers:
                
                for i, prop in enumerate(propertyList):
                    #check if out sync is registered for nodetype and pee
                    if (nodeType, peer) in self.ghostNodeOut:
                        print self.ghostNodeOut[(nodeType, peer)][prop]
                        self.comm.isend(self.ghostNodeOut[(nodeType, peer)][prop], peer, tag=i)
                
                    #check if in sync is registered for nodetype and peer
                    if (nodeType, peer) in self.ghostNodeIn:
                        buffers[peer,i] = self.comm.irecv(source=peer, tag=i)
                        #data = buf.wait()
                
            for key in buffers.keys():
                print str(self.rank) + ' ' + 'key:' + str(key)
                
                data = buffers[key].wait()
                print 'data ' + str(data)
                print self.ghostNodeIn
                self.ghostNodeIn[(nodeType, key[0])][propertyList[key[1]]] = data
            print 'MPI commmunication required: ' + str(time.time()-tt) + ' seconds'
        
        def initCommunicationViaLocations(self, ghostLocationList):        
            tt = time.time()
            # acquire the global IDs for the ghostNodes
            mpiRequest = dict()
            mpiReqIDList = dict()
            print self.world.graph.IDArray
            for ghLoc in ghostLocationList:
                owner = ghLoc.mpiOwner
                x,y   = ghLoc.node['pos']
                if owner not in mpiRequest:
                    mpiRequest[owner]   = (list(), 'gID')
                    mpiReqIDList[owner] = list()
                    
                mpiRequest[owner][0].append( (x,y) ) # send x,y-pairs for identification
                mpiReqIDList[owner].append(ghLoc.nID)
            print 'rank' + str(self.rank) + 'mpiReqIDList: ' + str(mpiReqIDList)
            
            for mpiDest in mpiRequest.keys():
                
                if mpiDest not in self.peers:
                    self.peers.append(mpiDest)
                
                # send request of global IDs
                print str(self.rank) + ' asks from ' + str(mpiDest) + ' - ' + str(mpiRequest[mpiDest])
                self.comm.send(mpiRequest[mpiDest], dest=mpiDest)
                
                
                self.ghostNodeIn[_cell, mpiDest] = self.world.graph.vs[mpiReqIDList[mpiDest]]
    
                # receive request of global IDs
                incRequest = self.comm.recv(source=mpiDest)
                iDList = [int(self.world.graph.IDArray[xx, yy]) for xx, yy in incRequest[0]]
                print str(self.rank) + ' - idlist:' + str(iDList)
                
                self.ghostNodeOut[_cell, mpiDest] = self.world.graph.vs[iDList]
                print str(self.rank) + ' - gIDs:' + str(self.ghostNodeOut[_cell, mpiDest]['gID'])
                for entity in [self.world.entList[i] for i in iDList]:
                    entity.mpiPeers.append(mpiDest)
                
                # send requested global IDs
                print str(self.rank) + ' sends to ' + str(mpiDest) + ' - ' + str(self.ghostNodeOut[_cell, mpiDest][incRequest[1]])
                self.comm.send(self.ghostNodeOut[_cell, mpiDest][incRequest[1]], dest=mpiDest)
                #receive requested global IDs
                globIDList = self.comm.recv(source=mpiDest)
                print 'receiving:'
                print 'globIDList:' + str(globIDList)
                print 'localDList:' + str(self.ghostNodeIn[_cell, mpiDest].indices)
                self.ghostNodeIn[_cell, mpiDest]['gID'] = globIDList
                for nID, gID in zip(self.ghostNodeIn[_cell, mpiDest].indices, globIDList):
                    print nID, gID
                    self.world.glob2loc[gID] = nID
                    self.world.loc2glob[nID] = gID
            print 'Mpi commmunication required: ' + str(time.time()-tt) + ' seconds'            


        #%% Nodes            
        def queueSendGhostNode(self, mpiPeer, nodeType, entity):
            
            if (nodeType, mpiPeer) not in self.ghostNodeQueue.keys():
                self.ghostNodeQueue[nodeType, mpiPeer] = dict()
                self.ghostNodeQueue[nodeType, mpiPeer]['nIds'] = list()
                self.ghostNodeQueue[nodeType, mpiPeer]['conn'] = list()
            
            self.ghostNodeQueue[nodeType, mpiPeer]['nIds'].append(entity.nID)
            self.ghostNodeQueue[nodeType, mpiPeer]['conn'].append(entity.loc.gID)

        

        def sendGhostNodes(self, world):
            
            
            #%%Packing of data
            packageDict = dict()        # indexed by (nodeType, mpiDest)
            announcements = dict()      # indexed by (mpiDest) containing nodeTypes
            
            for nodeType, mpiPeer in self.ghostNodeQueue.keys():
                
                if mpiPeer not in announcements:
                    announcements[mpiPeer] = list()
                announcements[mpiPeer].append(nodeType)
                    
                #get size of send array
                IDsList= self.ghostNodeQueue[(nodeType, mpiPeer)]['nIds']
                connList = self.ghostNodeQueue[(nodeType, mpiPeer)]['conn']
    
                nodeSeq = world.graph.vs[IDsList]
                
                # setting up ghost out Comm
                self.ghostNodeOut[nodeType, mpiPeer] = nodeSeq
                propList = world.graph.nodeProperies[nodeType][:]
                print propList
                packageDict = self.__packData__(packageDict, nodeType, mpiPeer, nodeSeq,  propList, connList)
            
            
            self.__sendDataAnnouncement__(announcements) #only required for initial communication
            self.__sendData__( packageDict)

        
        def recvGhostNodes(self, world):
            #%%Reading announcements
            announcements = dict()
            requestDict  = dict()
            # get expected nodetypes
            for mpiPeer in self.peers:
                announcements[mpiPeer] = self.comm.irecv(source=mpiPeer, tag=9999).wait()
                
            for mpiPeer in self.peers:
                if len(announcements[mpiPeer]) > 0: # will receive a message
                    
                    for nodeType in announcements[mpiPeer]:
                        
                        requestDict[nodeType, mpiPeer] = self.comm.irecv(source=mpiPeer, tag=nodeType)
            
            #%% Receiving of data
            dataDict = dict()
            for nodeType, mpiPeer in requestDict.keys():
                dataDict[nodeType, mpiPeer] = requestDict[nodeType, mpiPeer].wait()
                print 'data: ' + str(dataDict)
            return dataDict
        
            #%% create ghost agents from dataDict 
            
            for (nodeType, mpiSrc), data in dataDict.iteritems():
                
                nNodes = data[0]
                
                nIDStart= world.graph.vcount()
                IDs = range(nIDStart,nIDStart+nNodes)
                world.graph.add_vertices(nNodes)
                nodeSeq = world.graph.vs[IDs]
                
                # setting up ghostIn communicator
                self.ghostNodeIn[nodeType, mpiPeer] = nodeSeq
                
                propList = world.graph.nodeProperies[nodeType][:]
                print propList
                #propList.remove('nID')
                
                for i, prop in enumerate(propList):
                    nodeSeq[prop] = data[i+1]
                
                gIDsCells = data[-1]
                
                for nID, gID in zip(IDs, gIDsCells):
                    
                    GhostAgentClass = world.graph.nodeClass[nodeType][1]
                    
                    agent = GhostAgentClass(world, nodeType, mpiSrc, nID)
                    earth.registerNode(agent,_ag) 
                    cell = world.entDict[world.glob2loc[gID]]
                    cell.registerEntity(earth, agent,_cLocAg)
                # create entity with nodes (queing)
                # deque
                # set data from buffer
        
        def sendGhostUpdate(self, nodeTypeList='all', propertyList='all'):
            
            dataDict = dict()
            
            for (nodeType, mpiPeer) in self.ghostNodeOut.keys():
                if nodeTypeList == 'All' or nodeType in nodeTypeList:
                    nodeSeq = self.ghostNodeOut[nodeType, mpiPeer]
                
                    if propertyList == 'all':
                        propertyList = self.world.graph.nodeProperies[nodeType][:]
                        propertyList.remove('gID')
                        
                    dataDict = self.__packData__(dataDict, nodeType, mpiPeer, nodeSeq,  propertyList, connList)
            
            self.__sendData__( dataDict)
        
        def recvGhostUpdate(self, nodeTypeList= 'all', propertyList='all'):
            
            dataDict  = dict()
            requestDict = dict()
            
            for nodeType, mpiPeer in self.ghostNodeIn.keys():
                if nodeTypeList == 'All' or nodeType in nodeTypeList:
                    requestDict[nodeType, mpiPeer] = self.comm.irecv(source=mpiPeer, tag=nodeType)
                    
                    dataDict[nodeType, mpiPeer] = requestDict[nodeType, mpiPeer].wait()
                    
                    
                
            for (nodeType, mpiSrc), data in dataDict.iteritems():
                
                if propertyList == 'all':
                    propertyList= self.world.graph.nodeProperies[nodeType][:]
                    print propertyList
                    propertyList.remove('gID')
               
                nodeSeq = self.ghostNodeIn[nodeType, mpiPeer]
                for i, prop in enumerate(propertyList):
                    nodeSeq[prop] = data[i+1]
           
        def updateGhostNodes(self, nodeTypeList= 'all'):
            tt = time.time()
            self.sendGhostUpdate(nodeTypeList)
            self.recvGhostUpdate(nodeTypeList)
            print 'Ghost update required: ' + str(time.time()-tt) + ' seconds'    
            
        
            
    #%% INIT WORLD            
    def __init__(self,spatial=True, maxNodes = 1e6):
        
        self.timeStep = 0
        
        self.para     = dict()
        self.spatial  = spatial
        self.maxNodes = int(maxNodes)
        self.globIDGen = self.__globIDGen__()
        
        self.queuing = True     # flag that indicates the vertexes and edges are queued and not added immediately
        self.para     = dict()
               
        #GRAPH
        self.graph    = ig.Graph(directed=True)
       
        
        
        # list of types
        self.graph.nodeTypes = list()
        self.graph.edgeTypes = list()
        
        # dict of classes to init node automatically
        self.graph.nodeType2Class = dict()
        self.graph.class2NodeType = dict()
        
        # list of properties per type
        self.graph.nodeProperies = dict()
        self.graph.edgeProperies = dict()
        # queues
        self.queue = Queue(self)

        # MPI communication
        self.mpi = self.Mpi(self)
        
        # IO
        self.io = self.IO(self)
        
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
        self.registerEdgeType('inactiv')        
        
        
        
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

        # create vertices 
        for x in range(nodeArray.shape[0]):
            for y in range(nodeArray.shape[1]):

                # only add an vertex if spatial location exist     
                if not np.isnan(rankArray[x,y]) and rankArray[x,y] == self.mpi.rank:

                    loc = LocClassObject(self, pos= (x, y))
                    IDArray[x,y] = loc.nID
                    self.registerLocation(loc, x, y)          # only for real cells
                    self.registerNode(loc,nodeType)     # only for real cells
        
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
                        
                        loc = GhstLocClassObject(self, pos= (xDst, yDst), owner=rankArray[xDst,yDst],)
                        #print 'rank: ' +  str(self.mpi.rank) + ' '  + str(loc.nID)
                        IDArray[xDst,yDst] = loc.nID
                        print IDArray
                        # so far ghost nodes are not in entDict, nodeDict, entList
                        #self.registerLocation(loc, x, y)
                        self.registerNode(loc,nodeType,ghost=True)
                        ghostLocationList.append(loc)
        self.graph.IDArray = IDArray
        
        self.queue.dequeueVertices(self)

        fullConnectionList = list()
        fullWeightList     = list()
        print 'rank: ' +  str(self.locDict) 
        
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
        
        return ghostLocationList

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
            nodeDict = self.ghostnodeDict[nodeType]
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

    def getEntity(self,nodeID):
        return self.entDict[nodeID]
            
    def registerNodeType(self, typeStr, AgentClass, GhostAgentClass, propertyList = ['type', 'gID']):
           
        # type is an required property
        assert 'type' and 'gID' in propertyList
        
        nodeTypeIdx = len(self.graph.nodeTypes)
        self.graph.nodeTypes.append(typeStr)
        self.graph.nodeProperies[nodeTypeIdx] = propertyList
        # same nodeType for ghost and non-ghost
        self.graph.nodeType2Class[nodeTypeIdx] = AgentClass, GhostAgentClass
        self.graph.class2NodeType[AgentClass]       = nodeTypeIdx 
        self.graph.class2NodeType[GhostAgentClass]  = nodeTypeIdx
        self.nodeDict[nodeTypeIdx]      = list()
        self.ghostNodeDict[nodeTypeIdx] = list()
        self.enums[typeStr] = nodeTypeIdx
        return nodeTypeIdx
        
    
    def registerEdgeType(self, typeStr, propertyList = ['type']):
        assert 'type' in propertyList # type is an required property
        
        edgeTypeIdx = len(self.graph.edgeTypes)
        self.graph.edgeTypes.append(typeStr)
        self.graph.edgeProperies[edgeTypeIdx] = propertyList
        #self.graph.queue.addEdgeType(edgeTypeIdx, propertyList)
        self.enums[typeStr] = edgeTypeIdx            

        return  edgeTypeIdx            

    def registerNode(self, agent, typ, ghost=False):
        self.entList.append(agent)
        self.entDict[agent.nID] = agent
        self.glob2loc[agent.gID] = agent.nID
        self.loc2glob[agent.nID] = agent.gID
        
        if ghost:
            self.ghostNodeDict[typ].append(agent.nID)
        else:
            self.nodeDict[typ].append(agent.nID)
        
    def registerLocation(self, location, x, y):
        
        self.locDict[x,y] = location
    
    def setParameters(self, parameterDict):
        for key in parameterDict.keys():
            self.para[key] = parameterDict[key]
            
    
    
    


        
    
    def view(self,filename = 'none', vertexProp='none'):
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
        
        self.graph.vs["label"] = self.graph.vs.indices
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
        if filename  == 'none':
            ig.plot(self.graph,**visual_style)    
        else:
            ig.plot(self.graph, filename, layout=ig.Layout(self.graph.vs['pos']), **visual_style )  
       
if __name__ == '__main__':
    
    earth = World()
    log_file  = open('out' + str(earth.mpi.rank) + '.txt', 'w')
    #log_file = open("message.log","w")
    import sys
    sys.stdout = log_file

    mpiRankLayer   = np.asarray([[0, 0, 0, 0, 1],
                              [np.nan, np.nan, np.nan, 1, 1]])
    
    #landLayer = np.load('rankMap.npy')
    connList = computeConnectionList(1.5)
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
    _cLocLoc = earth.registerEdgeType('cellCell')
    _cLocAg = earth.registerEdgeType('cellAgent')
    _cAgAg = earth.registerEdgeType('AgAg')
    ghostLocationList = earth.initSpatialLayer(mpiRankLayer, connList, _cell, Location, GhostLocation)
    earth.mpi.initCommunicationViaLocations(ghostLocationList)
    
    for cell in earth.iterEntRandom(_cell):
        cell.node['value'] = earth.mpi.rank
        cell.node['value2'] = earth.mpi.rank+2
        
        if cell.node['pos'][0] == 0:
            x,y = cell.node['pos']
            agent = Agent(earth, value3=np.random.randn(),pos=(x+ np.random.randn()*.1,  y + np.random.randn()*.1))
            earth.registerNode(agent,_ag) 
            cell.registerEntity(earth, agent,_ag,_cLocAg)
            
    earth.queue.dequeueVertices(earth)
    earth.queue.dequeueEdges()
#            if agent.node['nID'] == 10:
#                agent.addConnection(8,_cAgAg)
    
    #earth.mpi.syncNodes(_cell,['value', 'value2'])
    earth.mpi.updateGhostNodes([_cell])
    print earth.graph.vs.attribute_names()
    print str(earth.mpi.rank) + ' values' + str(earth.graph.vs['value'])
    print str(earth.mpi.rank) + ' values2: ' + str(earth.graph.vs['value2'])
    
    print earth.mpi.ghostNodeIn
    print earth.mpi.ghostNodeOut
    
    print earth.graph.vs.attribute_names()
    
    print str(earth.mpi.rank) + ' ' + str(earth.nodeDict[_ag])
    
    print str(earth.mpi.rank) + ' SendQueue ' + str(earth.mpi.ghostNodeQueue)
    
    earth.mpi.sendGhostNodes(earth)
    dataDict = earth.mpi.recvGhostNodes(earth)
   
    cell.getConnNodeIDs(nodeType=_cell, mode='out')
    earth.view(str(earth.mpi.rank) + '.png')
    
    print str(earth.mpi.rank) + ' ' + str(earth.graph.vs.indices)
    print str(earth.mpi.rank) + ' ' + str(earth.graph.vs['value3'])
    
    for agent in earth.iterEntRandom(_ag):
        agent.node['value3'] = earth.mpi.rank+ agent.nID
    
    earth.mpi.updateGhostNodes([_ag])

    earth.io.initNodeFile(earth, [_cell, _ag])
    
    earth.io.gatherNodeData()
    earth.io.writeDataToFile()
    
    print str(earth.mpi.rank) + ' ' + str(earth.graph.vs['value3'])