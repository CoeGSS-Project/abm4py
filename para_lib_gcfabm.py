#!/usr/bin/env python2
# -*- coding: UTF-8-*-
"""
G R A P H    M O D U L E 

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
    
    
"""
from __future__ import division
import igraph as ig
import numpy as np
from mpi4py import MPI
import time 

from class_auxiliary import computeConnectionList
class Queue():

    def __init__(self, world):
        self.graph = world.graph
        self.edges          = dict()
        self.edgeProperties = dict()
        self.nodes          = dict()    
        self.nodeProperties = dict()
        
        
    def addEdgeType(self, edgeTypeIdx, propertyList):
        self.edges[edgeTypeIdx]             = list() #(nodeTuple list,)
        self.edgeProperties[edgeTypeIdx]    = dict()
        for prop in propertyList:
            self.edgeProperties[edgeTypeIdx][prop] = list()
            
            
    def addNodeType(self, edgeTypeIdx, propertyList):
        pass
        
################ ENTITY CLASS #########################################    
# general ABM entity class for objects connected with the graph

class Entity():
        
    def __init__(self, world, nodeType):
        len(world.graph.nodeTypes) >=  nodeType
        self.graph= world.graph
        self.nID  = len(self.graph.vs)
        self.gID  = self.__getGlobID__(world)
        self.graph.add_vertex(self.nID, type=nodeType, gID=self.gID)
        # short cuts for value access
        self.node = self.graph.vs[self.nID]
        self.edges = dict()                
        
class Agent(Entity):
    
    def __getGlobID__(self,world):
        return world.globIDGen.next()
        
    def __init__(self, world, nodeType, owner, xPos = np.nan, yPos = np.nan):
        Entity.__init__(self, world, nodeType)
        self.mpiOwner =  int(owner)

class GhostAgent(Entity):
    
    def __getGlobID__(self,world):
        
        #TODO add request for global ID
        return None
        
    def __init__(self, world, nodeType, owner):
        Entity.__init__(self, world, nodeType)
        self.mpiOwner =  int(owner)

################ LOCATION CLASS #########################################      
class Location(Entity):

    def __getGlobID__(self,world):
        return world.globIDGen.next()
    
    def __init__(self, world, nodeType, owner, xPos, yPos):
        Entity.__init__(self,world,nodeType)
        self.graph.vs[self.nID]['pos']= (xPos,yPos)
        self.mpiOwner = int(owner)
        
class GhostLocation(Entity):
    
    def __getGlobID__(self,world):
        
        #TODO add request for global ID
        return None
    
    def __init__(self, world, nodeType, xPos, yPos, owner):
        Entity.__init__(self,world,nodeType)
        self.graph.vs[self.nID]['pos']= (xPos,yPos)
        self.mpiOwner = int(owner)
        
        
################ WORLD CLASS #########################################            
class World:
    
    def __init__(self,spatial=True, maxNodes = 1e6):
        
        self.para     = dict()
        self.spatial  = spatial
        self.maxNodes = int(maxNodes)
        self.globIDGen = self.__globIDGen__()
        
               
        #GRAPH
        self.graph    = ig.Graph(directed=True)
        # list of types
        self.graph.nodeTypes = list()
        self.graph.edgeTypes = list()
        # list of properties per type
        self.graph.nodeProperies = dict()
        self.graph.edgeProperies = dict()
        # queues
        self.graph.queue = Queue(self)

        # MPI communication
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.graph.mpiIn  = dict()
        self.graph.mpiOut = dict()
        self.graph.mpiPeers = list()        

        
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
        self.registerNodeType('inactiv')
        self.registerEdgeType('inactiv')        
        
# GENERAL FUNCTIONS
    def __globIDGen__(self):
        i = -1
        while i < self.maxNodes:
            i += 1
            yield (self.maxNodes*(self.rank+1)) +i

    def iterNodesRandom(self,nodeType, ghosts = False, random=True):
        if isinstance(nodeType,str):
            nodeType = self.types.index(nodeType)
            
        if ghosts:
            nodeDict = self.ghostnodeDict[nodeType]
        else:
            nodeDict = self.nodeDict[nodeType]
        if random:
            print 'nodeDict' + str(nodeDict)
            #print self.entList
            shuffled_list = sorted(nodeDict, key=lambda x: np.random.random())
            return [self.entList[i] for i in shuffled_list]
        else:
            return  [self.entList[i] for i in nodeDict]
        
            
    def registerNodeType(self, typeStr, propertyList = ['type', 'gID']):
           
        # type is an required property
        assert 'type' and 'gID' in propertyList
        
        nodeTypeIdx = len(self.graph.nodeTypes)
        self.graph.nodeTypes.append(typeStr)
        self.graph.nodeProperies[nodeTypeIdx] = propertyList
        self.nodeDict[nodeTypeIdx]= list()
        
        self.enums[typeStr] = nodeTypeIdx
        return nodeTypeIdx
        
    
    def registerEdgeType(self, typeStr, propertyList = ['type']):
        assert 'type' in propertyList # type is an required property
        
        edgeTypeIdx = len(self.graph.edgeTypes)
        self.graph.edgeTypes.append(typeStr)
        self.graph.nodeProperies[edgeTypeIdx] = propertyList
        self.graph.queue.addEdgeType(edgeTypeIdx, propertyList)
        self.enums[typeStr] = edgeTypeIdx            

        return  edgeTypeIdx            

    def registerNode(self, agent, typ, ghost=False):
        self.entList.append(agent)
        self.entDict[agent.nID] = agent
        if ghost:
            self.ghostNodeDict[typ].append(agent.nID)
        else:
            self.nodeDict[typ].append(agent.nID)
        
    def registerLocation(self, location):
        
        self.locDict[location.node['pos']] = location
            
    def initSpatialLayerMpi(self, rankArray, connList, nodeType, LocClassObject=Location, GhstLocClassObject=GhostLocation):
        """
        Auiliary function to contruct a simple connected layer of spatial locations.
        Use with  the previously generated connection list (see computeConnnectionList)
        
        """
        nodeArray = ((rankArray * 0) +1)
        #print nodeArray
        self.graph.IdArray = nodeArray * np.nan
        self.graph.IdArray[nodeArray == 1] = xrange(int(np.nansum(nodeArray)))
        IDArray = self.graph.IdArray.astype(int)
        IDArray[np.isnan(rankArray)] = -1
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
                if not np.isnan(rankArray[x,y]):
                    if rankArray[x,y] == self.rank:

                        loc = LocClassObject(self,nodeType, xPos=x, yPos=y, owner=rankArray[x,y])
                        self.registerLocation(loc)          # only for real cells
                        self.registerNode(loc,nodeType)     # only for real cells
                    else:
                        loc = GhstLocClassObject(self, nodeType, xPos=x, yPos=y, owner=rankArray[x,y],)
                        # so far ghost nodes are not in entDict, nodeDict, entList
                        self.registerNode(loc,nodeType)
                        ghostLocationList.append(loc)

        fullConnectionList = list()
        fullWeightList     = list()

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
                    

                    if trgID !=-1:
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
        
        
        tt = time.time()
        # acquire the global IDs for the ghostNodes
        mpiReq = dict()
        for ghLoc in ghostLocationList:
            owner = ghLoc.mpiOwner
            x,y   = ghLoc.node['pos']
            if owner not in mpiReq:
                mpiReq[owner] = (list(), 'gID')
            mpiReq[owner][0].append(IDArray[x,y])
            
        for mpiDest in mpiReq.keys():
            
            if mpiDest not in self.graph.mpiPeers:
                self.graph.mpiPeers.append(mpiDest)
            # send request of global IDs
            self.comm.send(mpiReq[mpiDest], dest=mpiDest)
            
            self.graph.mpiIn[_cell, mpiDest] = self.graph.vs[mpiReq[mpiDest][0]]
            
            # receive request of global IDs
            incRequest = self.comm.recv(source=mpiDest)
            self.graph.mpiOut[_cell, mpiDest] = self.graph.vs[incRequest[0]]
            
            # send requested global IDs
            self.comm.send(self.graph.vs[incRequest[0]][incRequest[1]], dest=mpiDest)
            #receive requested global IDs
            globIDList = self.comm.recv(source=mpiDest)
            
            self.graph.vs[mpiReq[mpiDest][0]]['gID'] = globIDList
        print 'Mpi commmunication required: ' + str(time.time()-tt) + ' seconds'

    def syncNodesMpi(self, nodeType, propertyList='all'):
        tt = time.time()
        buffers = dict()
        
        if not isinstance(propertyList,list):
            propertyList = [propertyList]
            
        for peer in self.graph.mpiPeers:
            
            for i, prop in enumerate(propertyList):
                #check if out sync is registered for nodetype and pee
                if (nodeType, peer) in self.graph.mpiOut:
                    #print self.graph.mpiOut[(nodeType, peer)][prop]
                    self.comm.isend(self.graph.mpiOut[(nodeType, peer)][prop], peer, tag=i)
            
                #check if in sync is registered for nodetype and peer
                if (nodeType, peer) in self.graph.mpiIn:
                    buffers[peer,i] = self.comm.irecv(source=peer, tag=i)
                    #data = buf.wait()
            
        for key in buffers.keys():
            data = buffers[key].wait()
            #print data
            self.graph.mpiOut[(nodeType, key[0])][propertyList[key[1]]] = data
        print 'MPI commmunication required: ' + str(time.time()-tt) + ' seconds'
        
    
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
        
        self.graph.vs["label"] = self.graph.vs["name"]
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
            ig.plot(self.graph, filename, **visual_style)  
       
if __name__ == '__main__':
    
    earth = World()
    
    landLayer   = np.asarray([[0, 0, 1],
                         [np.nan, 1, 1]])
    
    #landLayer = np.load('rankMap.npy')
    connList = computeConnectionList(1.5)
    #print connList
    _cell    = earth.registerNodeType('cell')
    _cLocLoc = earth.registerEdgeType('cellCell')
    earth.initSpatialLayerMpi(landLayer, connList, _cell, Location, GhostLocation)
    
    
    for cell in earth.iterNodesRandom(_cell):
        cell.node['value'] = earth.rank
        cell.node['value2'] = earth.rank+2
    #print earth.graph.vs['gID']
#    for i in range(10):
#        print earth.globIDGen.next()   
    earth.view(str(earth.rank) + '.png')    
    #print earth.graph.mpiOut
    #print earth.graph.mpiIn
    
    earth.syncNodesMpi(_cell,['value', 'value2'])
    
    print earth.graph.vs['value']
    print earth.graph.vs['value2']