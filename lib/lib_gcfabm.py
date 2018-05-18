#!/usr/bin/env python3
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

import logging as lg
import sys
import numpy as np

from class_graph import ABMGraph
import core




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



################ ENTITY CLASS #########################################
# general ABM entity class for objects connected with the graph

def firstElementDeco(fun):
    """ 
    Decorator that returns the first element
    ToDo: if possible find better way
    """
    def helper(arg):
        return fun(arg)[0]
    return helper


class Entity():
    """
    Most basic class from which agents of different type are derived
    """
    __slots__ = ['gID', 'nID']
    

    def __init__(self, world, nID = -1, **kwProperties):
        if world is None:
            return
        
        nodeType =  world.graph.class2NodeType[self.__class__]
        
        if not hasattr(self, '_graph'):
            self._setGraph(world.graph)

        self.gID    = self.getGlobID(world)
        kwProperties['gID'] = self.gID

        # create instance from existing node
        if nID is not -1:


            self.nID = nID
            self.data, self.dataID = self._graph.getNodeView(nID)
            self.gID = self.data['gID'][0]
        
        else:
            self.nID, self.dataID, self.data = world.addVertex(nodeType,  **kwProperties)
            
        self.nodeType = nodeType
        # redireciton of internal functionality:
        self.get = firstElementDeco(self.data.__getitem__)
        self.set = self.data.__setitem__
        
    
    @classmethod
    def _setGraph(cls, graph):
        """ Makes the class variable _graph available at the first init of an entity"""
        cls._graph = graph


    def dataView(self, key=None):
        if key is None:
            return self.data[0].view()
        else:
            return self.data[0,key].view()


    def getPeerIDs(self, edgeType=None, nodeType=None, mode='out'):
        
        if edgeType is None:
            
            edgeType = earth.graph.node2EdgeType[self.nodeType, nodeType]
        else:
            assert nodeType is None
        #print edgeType
        if mode=='out':            
            eList, nodeList  = self._graph.outgoing(self.nID, edgeType)
        elif mode == 'in':
            eList, nodeList  = self._graph.incomming(self.nID, edgeType)
        
        return nodeList

        
    def getPeerValues(self, prop, edgeType=None):
        """
        Access the attributes of all connected nodes of an specified nodeType
        or connected by a specfic edge type
        """
        return self._graph.getOutNodeValues(self.nID, edgeType, attr=prop)

    def setPeerValues(self, prop, values, edgeType=None, nodeType=None, force=False):
        """
        Set the attributes of all connected nodes of an specified nodeType
        or connected by a specfic edge type
        """
        if not force:
            raise Exception
        else:
            import warnings
            warnings.warn('This is violating the current rules and data get lost')

            self._graph.setOutNodeValues(self.nID, edgeType, prop, values)
                                   

    def getEdgeValues(self, prop, edgeType):
        """
        privat function to access the values of  edges
        """
        (eTypeID, dataID), nIDList  = self._graph.outgoing(self.nID, edgeType)

        edgesValues = self._graph.getEdgeSeqAttr(label=prop, 
                                                 eTypeID=eTypeID, 
                                                 dataIDs=dataID)

        

        return edgesValues, (eTypeID, dataID), nIDList

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


    def addConnection(self, friendID, edgeType, **kwpropDict):
        """
        Adding a new connection to another node
        Properties must be provided in the correct order and structure
        """
        self._graph.addEdge(edgeType, self.nID, friendID, attributes = tuple(kwpropDict.values()))


    def remConnection(self, friendID=None, edgeType=None):
        """
        Removing a connection to another node
        """
        self._graph.remEdge(source=self.nID, target=friendID, eTypeID=edgeType)

    def remConnections(self, friendIDs=None, edgeType=None):
        """
        Removing mutiple connections to another node
        """        
        if friendIDs is not None:
            for friendID in friendIDs:
                self._graph.remEdge(source=self.nID, target=friendID, eTypeID=edgeType)


    def addTo(self, prop, value, idx = None):
        #raise DeprecationWarning('Will be deprecated in the future')
        if idx is None:
            self.data[prop] += value
        else:
            self.data[prop][0, idx] += value

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
        if 'nID' not in list(kwProperties.keys()):
            nID = -1
        else:
            nID = kwProperties['nID']
        Entity.__init__(self, world, nID, **kwProperties)
        self.mpiOwner =  int(world.papi.rank)

    def getGlobID(self,world):
        return next(world.globIDGen)

    def registerChild(self, world, entity, edgeType):
        if edgeType is not None:
            #print edgeType
            world.addEdge(edgeType, self.nID, entity.nID)
        entity.loc = self

        if len(self.mpiPeers) > 0: # node has ghosts on other processes
            for mpiPeer in self.mpiPeers:
                #print 'adding node ' + str(entity.nID) + ' as ghost'
                nodeType = world.graph.class2NodeType[entity.__class__]
                world.papi.queueSendGhostNode( mpiPeer, nodeType, entity, self)

        return self.mpiPeers


    def getLocationValue(self,prop):

        return self.loc.node[prop]


    def move(self):
        """ not yet implemented"""
        pass






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
        world.addEdge(edgeType, self.nID, entity.nID)

        
################ LOCATION CLASS #########################################
class Location(Entity):

    def getGlobID(self,world):
        return next(world.globIDGen)

    def __init__(self, world, **kwProperties):
        if 'nID' not in list(kwProperties.keys()):
            nID = -1
        else:
            nID = kwProperties['nID']


        Entity.__init__(self,world, nID, **kwProperties)
        self.mpiOwner = int(world.papi.rank)
        self.mpiPeers = list()



    def registerChild(self, world, entity, edgeType=None):
        world.addEdge(edgeType, self.nID, entity.nID)
        entity.loc = self

        if len(self.mpiPeers) > 0: # node has ghosts on other processes
            for mpiPeer in self.mpiPeers:
                #print 'adding node ' + str(entity.nID) + ' as ghost'
                nodeType = world.graph.class2NodeType[entity.__class__]
                world.papi.queueSendGhostNode( mpiPeer, nodeType, entity, self)

        return self.mpiPeers
    
    def getConnectedLocation(self, edgeType=1):
        """ 
        ToDo: check if not deprecated 
        """
        self.weights, _, nodeIDList = self.getEdgeValues('weig',edgeType=edgeType)
        
        return self.weights,  nodeIDList

class GhostLocation(Entity):
    
    def getGlobID(self,world):

        return -1

    def __init__(self, world, owner, nID=-1, **kwProperties):

        Entity.__init__(self,world, nID, **kwProperties)
        self.mpiOwner = int(owner)
        

    def register(self, world, parentEntity=None, edgeType=None):
        Entity.register(self, world, parentEntity, edgeType, ghost= True)

    def registerChild(self, world, entity, edgeType=None):
        world.addEdge(edgeType, self.nID, entity.nID)
        
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

    #%% INIT WORLD
    def __init__(self,
                 simNo,
                 outPath,
                 spatial=True,
                 nSteps= 1,
                 maxNodes = 1e6,
                 maxEdges = 1e6,
                 debug = False,
                 mpiComm=None):

        self.simNo    = simNo
        self.timeStep = 0
        self.para     = dict()
        
        self.maxNodes = int(maxNodes)
        self.globIDGen = self._globIDGen()
        self.nSteps   = nSteps
        self.debug    = debug

        self.para     = dict()

        # GRAPH
        self.graph    = ABMGraph(self, maxNodes, maxEdges)
        self.para['outPath'] = outPath

        self.globalRecord = dict() # storage of global data

        self.addEdge        = self.graph.addEdge
        self.addEdges       = self.graph.addEdges
        self.delEdges       = self.graph.delete_edges
        self.addVertex      = self.graph.addNode
        self.addVertices    = self.graph.addNodes
        
        # agent passing interface for communication between parallel processes
        self.papi = core.PAPI(self, mpiComm=mpiComm)
        
        lg.debug('Init MPI done')##OPTPRODUCTION
        if self.papi.comm.rank == 0:
            self.isRoot = True
        else:
            self.isRoot = False

        # IO
        self.io = core.IO(self, nSteps, self.para['outPath'])
        lg.debug('Init IO done')##OPTPRODUCTION
        # Globally synced variables
        self.graph.glob     = core.Globals(self)
        lg.debug('Init Globals done')##OPTPRODUCTION

        self.random = core.Random(self)
        if spatial:
            self.spatial  = core.Spatial(self)
        
        # enumerations
        self.enums = dict()

        # node lists and dicts
        self.nodeDict       = dict()
        self.ghostNodeDict  = dict()
        
        # dict of list that provides the storage place for each agent per nodeType
        self.dataDict       = dict()
        self.ghostDataDict  = dict()

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
            yield (self.maxNodes*(self.papi.rank+1)) +i

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
        for key in list(parameterDict.keys()):
            self.setParameter(key, parameterDict[key])


    def getNodeValues(self, label, localNodeIDList=None, nodeType=None):
        """
        Method to read values of node sequences at once
        Return type is numpy array
        Only return non-ghost agent properties
        """
        if localNodeIDList:   
            assert nodeType is None # avoid wrong usage ##OPTPRODUCTION
            return self.graph.getNodeSeqAttr(label, lnIDs=localNodeIDList)
        
        elif nodeType:           
            assert localNodeIDList is None # avoid wrong usage ##OPTPRODUCTION
            return self.graph.getNodeSeqAttr(label, lnIDs=self.nodeDict[nodeType])
        
    def setNodeValues(self, label, valueList, localNodeIDList=None, nodeType=None):
        """
        Method to read values of node sequences at once
        Return type is numpy array
        """
        if localNodeIDList:
            assert nodeType is None # avoid wrong usage ##OPTPRODUCTION
            self.graph.setNodeSeqAttr(label, valueList, lnIDs=localNodeIDList)
        
        elif nodeType:
            assert localNodeIDList is None # avoid wrong usage ##OPTPRODUCTION
            self.graph.setNodeSeqAttr(label, valueList, lnIDs=self.nodeDict[nodeType])

    def getEdgeValues(self, label, valueList, localEdgeIDList=None, edgeType=None):
        if localEdgeIDList:   
            assert edgeType is None # avoid wrong usage ##OPTPRODUCTION
            return self.graph.getNodeSeqAttr(label, lnIDs=localEdgeIDList)
        
        elif edgeType:           
            assert localEdgeIDList is None # avoid wrong usage ##OPTPRODUCTION
            return self.graph.getNodeSeqAttr(label, lnIDs=self.edgeDict[edgeType])

    def setEdgeValues(self, label, valueList, localEdgeIDList=None, edgeType=None):
        """
        Method to retrieve all properties of all entities of one edgeType
        """
        if localEdgeIDList:
            assert edgeType is None # avoid wrong usage ##OPTPRODUCTION
            self.graph.setEdgeSeqAttr(label, valueList, lnIDs=localEdgeIDList)
        
        elif edgeType:
            assert localEdgeIDList is None # avoid wrong usage ##OPTPRODUCTION
            self.graph.setEdgeSeqAttr(label, valueList, lnIDs=self.edgeDict[edgeType])
    
  
    def getEntity(self, nodeID=None, globID=None):
        """
        Methode to retrieve a certain instance of an entity by the nodeID
        """
        if nodeID is not None:
            return self.entDict[nodeID]
        if globID is not None:
            return self.entDict[self._glob2loc[globID]]


    def iterEntity(self,nodeType, ghosts = False):
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

        return  [self.entDict[i] for i in nodeDict]


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
        self.ghostDataDict[nodeTypeIdx] = list()
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
            self.ghostDataDict[typ].append(agent.dataID)
        
        else:
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
        self.globalRecord[name] = core.GlobalRecord(name, colLables, self.nSteps, title, style)

        if mpiReduce is not None:
            self.graph.glob.registerValue(name , np.asarray([np.nan]*len(colLables)),mpiReduce)
            self.globalRecord[name].glob = self.graph.glob
            
    def registerLocation(self, location, x, y):

        self.locDict[x,y] = location


    def returnApiComm(self):
        return self.papi.comm

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

        # finishing reporter files
#        for writer in self.reporter:
#            writer.close()

        if self.isRoot:
            # writing global records to file
            filePath = self.para['outPath'] + '/globals.hdf5'
            
            for key in self.globalRecord:
                self.globalRecord[key].saveCSV(self.para['outPath'])
                self.globalRecord[key].save2Hdf5(filePath)

            
            # saving enumerations
            core.saveObj(self.enums, self.para['outPath'] + '/enumerations')


            # saving enumerations
            core.saveObj(self.para, self.para['outPath'] + '/simulation_parameters')

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
    import mpi4py
    mpiComm = mpi4py.MPI.COMM_WORLD
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
                  mpiComm=mpiComm)


    #earth = World(0, '.', maxNodes = 1e2, nSteps = 10)
    print(earth.papi.comm.rank)
    if mpiComm.size > 1:
        log_file  = open('out' + str(earth.papi.rank) + '.txt', 'w')
        sys.stdout = log_file
    earth.graph.glob.registerValue('test' , np.asarray([earth.papi.comm.rank]),'max')

    earth.graph.glob.registerStat('meantest', np.random.randint(5,size=3).astype(float),'mean')
    earth.graph.glob.registerStat('stdtest', np.random.randint(5,size=2).astype(float),'std')
    print(earth.graph.glob.globalValue['test'])
    print(earth.graph.glob.globalValue['meantest'])
    print('mean of values: ',earth.graph.glob.localValues['meantest'],'-> local mean: ',earth.graph.glob.globalValue['meantest'])
    print('std od values:  ',earth.graph.glob.localValues['stdtest'],'-> local std: ',earth.graph.glob.globalValue['stdtest'])

    earth.graph.glob.sync()
    print(earth.graph.glob.globalValue['test'])
    print('global mean: ', earth.graph.glob.globalValue['meantest'])
    print('global std: ', earth.graph.glob.globalValue['stdtest'])



    import sys

    mpiRankLayer   = np.asarray([[0, 0, 0, 0, 1],
                              [np.nan, np.nan, np.nan, 1, 1]])
    if mpiComm.size == 1:
        mpiRankLayer = mpiRankLayer * 0
    
    #landLayer = np.load('rankMap.npy')
    connList = core.computeConnectionList(1.5)
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

    earth.spatial.initSpatialLayer(mpiRankLayer, connList, CELL, Location, GhostLocation)
    #earth.papi.initCommunicationViaLocations(ghostLocationList)

    for cell in earth.random.iterEntity(CELL):
        cell.data['value'] = earth.papi.rank
        cell.data['value2'] = earth.papi.rank+2

        if cell.get('pos')[0] == 0:
            x,y = cell.get('pos')
            agent = Agent(earth, value3=np.random.randn(),pos=(x,  y))
            #print 'agent.nID' + str(agent.nID)
            agent.register(earth, cell, C_LOAG)
            #cell.registerEntityAtLocation(earth, agent,_cLocAg)

    #earth.queue.dequeueVertices(earth)
    #earth.queue.dequeueEdges(earth)
#            if agent.node['nID'] == 10:
#                agent.addConnection(8,_cAgAg)

    #earth.papi.syncNodes(CELL,['value', 'value2'])
    earth.papi.updateGhostNodes([CELL])
    print(earth.graph.getPropOfNodeType(CELL, 'names'))
    print(str(earth.papi.rank) + ' values' + str(earth.graph.nodes[CELL]['value']))
    print(str(earth.papi.rank) + ' values2: ' + str(earth.graph.nodes[CELL]['value2']))

    #print earth.papi.ghostNodeRecv
    #print earth.papi.ghostNodeSend

    print(earth.graph.getPropOfNodeType(AG, 'names'))

    print(str(earth.papi.rank) + ' ' + str(earth.nodeDict[AG]))

    print(str(earth.papi.rank) + ' SendQueue ' + str(earth.papi.ghostNodeQueue))

    earth.papi.transferGhostNodes(earth)
    #earth.papi.recvGhostNodes(earth)

    #earth.queue.dequeueVertices(earth)
    #earth.queue.dequeueEdges(earth)

    cell.getPeerIDs(nodeType=CELL, mode='out')
    #earth.view(str(earth.papi.rank) + '.png', layout=ig.Layout(earth.graph.nodes[CELL]['pos'].tolist()))

    print(str(earth.papi.rank) + ' ' + str(earth.graph.nodes[AG].indices))
    print(str(earth.papi.rank) + ' ' + str(earth.graph.nodes[AG]['value3']))

    for agent in earth.random.iterEntity(AG):
        agent.data['value3'] = earth.papi.rank+ agent.nID
        assert agent.get('value3') == earth.papi.rank+ agent.nID

    earth.papi.updateGhostNodes([AG])

    earth.io.initNodeFile(earth, [CELL, AG])

    earth.io.writeDataToFile(0, [CELL, AG])

    print(str(earth.papi.rank) + ' ' + str(earth.graph.nodes[AG]['value3']))

    #%% testing agent methods 
    peerList = cell.getPeerIDs(C_LOLO)
    writeValues = np.asarray(list(range(len(peerList)))).astype(np.float32)
    cell.setPeerValues('value', writeValues, C_LOLO , force=True)
    readValues = cell.getPeerValues('value', C_LOLO)
    assert all(readValues == writeValues)
    assert all(earth.graph.getNodeSeqAttr('value', peerList,) == writeValues)
    print('Peer values write/read successful')
    edgeList = cell.getEdgeIDs(C_LOLO)
    writeValues = np.random.random(len(edgeList[1])).astype(np.float32)
    cell.setEdgeValues('weig',writeValues, C_LOLO)
    readValues, _, _  = cell.getEdgeValues('weig',C_LOLO)
    assert all(readValues == writeValues)
    print('Edge values write/read successful')
    
    friendID = earth.nodeDict[AG][0]
    agent.addConnection(friendID, C_AGAG, weig=.51)
    assert earth.graph.isConnected(agent.nID, friendID, C_AGAG)
    readValue, _, _ = agent.getEdgeValues('weig',C_AGAG)
    assert readValue[0] == np.float32(0.51)
    
    agent.remConnection(friendID, C_AGAG)
    assert not(earth.graph.isConnected(agent.nID, friendID, C_AGAG))
    print('Adding/removing connection successfull')
    
    value = agent.data['value3'].copy()
    agent.data['value3'] +=1
    assert agent.data['value3'] == value +1
    assert earth.graph.getNodeAttr('value3', agent.nID) == value +1
    print('Value access and increment sucessful')
    
    #%%
    pos = (0,4)
    cellID = earth.graph.IDArray[pos]
    cell40 = earth.entDict[cellID]
    agentID = cell40.getPeerIDs(edgeType=C_LOAG, mode='out')
    connAgent = earth.entDict[agentID[0]]
    assert np.all(cell40.data['pos'] == connAgent.data['pos'])
    
    if earth.papi.rank == 1:
        cell40.data['value'] = 32.0
        connAgent.data['value3'] = 43.2
    earth.papi.updateGhostNodes([CELL])
    earth.papi.updateGhostNodes([AG],['value3'])
    
    
    buff =  earth.papi.all2all(cell40.data['value'][0])
    assert buff[0] == buff[1]
    print('ghost update of cells successful (all attributes) ')
    
    buff =  earth.papi.all2all(connAgent.data['value3'])
    assert buff[0] == buff[1]
    print('ghost update of agents successful (specific attribute)')

