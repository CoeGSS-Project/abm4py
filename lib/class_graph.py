#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 10:31:00 2017

@author: Andreas Geiges, Global Climate Forum e.V.
"""

from igraph import Graph
import numpy as np
import itertools

class GraphQueue():

    def __init__(self):
        pass

class TypeDescription():

    def __init__(self, nodeTypeIdx, typeStr, staticProperties, dynamicProperties):

        # list of properties per type
        self.staProp = staticProperties
        self.dynProp = dynamicProperties
        self.typeIdx = nodeTypeIdx
        self.typeStr = typeStr

class BaseGraph():
    
    CHUNK_SIZE = 1000
    
    class NodeArray(np.rec.ndarray):
        
        def __new__(cls, maxNodes, nTypeID, startID, dtype):
            self = np.rec.array(np.empty(maxNodes,dtype=dtype)).view(cls)
            
            self.maxNodes       = maxNodes
            self.nType          = nTypeID
            self.getNewID       = itertools.count(startID).next
            self.nodeList       = []
            self.freeRows       = []
            
            return self
        
        def nCount(self):
            return len(self.nodeList)
        
        def add(self, attributes):
            if len(self.freeRows) > 0:
                # use and old local node ID from the free row
                dataID = self.freeRows.pop()
                lnID   = dataID + self.nType * self.maxNodes
            else:
                # generate a new local ID
                lnID   = self.getNewID()
                #print lnID
                dataID = lnID - self.nType * self.maxNodes
                #print dataID
            dataview = self[dataID:dataID+1].view() 
            if attributes is not None:
                dataview[:] = attributes
            self.nodeList.append(lnID)
            return lnID, dataview
        
        def rem(self):
            pass
            
        def set(self):
            pass
        
        def get(self):
            pass
            
    
    class EdgeArray(np.rec.ndarray):
        def __new__(cls, maxEdges, startID, dtype):
            self = np.rec.array(np.empty(maxEdges,dtype=dtype)).view(cls)
            
            self.maxEdges       = maxEdges
            
            self.getNewID       = itertools.count(startID).next
            self.edgeList       = []
            self.freeRows       = []
            self.eDict          = dict()
            self.edgesOut       = dict() # (source -> targets)
            self.edgesIn        = dict() # (target -> sources)
            return self
        
        def eCount(self):
            return len(self.edgeList)
        
    def __init__(self, maxNodesPerType, maxEdgesPerType):
        """
        This class provides the basic functions to contruct a directed graph
        with different node and edge types. 
        The max number of edges and nodes
        cannot be exceeded during execution, since it is used for pre-assigning
        storage space.
        """
        
        self.maxNodes       = maxNodesPerType
        self.maxEdges       = maxEdgesPerType
        

        self.nodeGlob2Loc   = dict()
        self.nodeLoc2Glob   = dict()
        self.nodes          = dict()
        self.edges          = dict()
        
        
        self.eDict    = dict() # (source, target -> leID)

        
        self.getNodeTypeID = itertools.count(1).next
        self.getEdgeTypeID = itertools.count(1).next
        
        #persistent nodeattributes
        self.persNodeAttr =[('gnID', np.int32, 1)]
        #persistent edge attributes
        self.persEdgeAttr =[('source', np.int32, 1), 
                            ('target', np.int32, 1)]
    #%% NODES
    def initNodeType(self, nodeName, size, attrDescriptor):
        
        nTypeID          = self.getNodeTypeID()
        dt               = np.dtype(self.persNodeAttr + attrDescriptor)
        nIDStart         = nTypeID * self.maxNodes
        self.nodes[nTypeID] = self.NodeArray(self.maxNodes, nTypeID, nIDStart, dtype=dt)
        return nTypeID
    
#    def extendNodeArray(self, nTypeID):
#        currentSize = len(self.nAttr[nTypeID])
        #self.nAttr[nTypeID][currentSize:currentSize+self.CHUNK_SIZE] = 
        

                
    def getNodeDataRef(self, lnID):
        """ calculates the node type ID from local ID"""
        if isinstance(lnID, tuple):
            nTypeID, dataID = lnID
        else:
            if isinstance(lnID, list):
                lnID = np.asarray(lnID)
                nTypeID, dataID = int(lnID[0] / self.maxNodes), lnID%self.maxNodes
                
            elif isinstance(lnID, np.ndarray):
            
                nTypeID, dataID = int(lnID[0] / self.maxNodes), lnID%self.maxNodes
            else:
                nTypeID, dataID = int(lnID / self.maxNodes), lnID%self.maxNodes
        
        return  nTypeID, dataID
    
    def addNode(self, nTypeID, attributes=None):
        
        
         if len(self.nodes[nTypeID].freeRows) > 0:
             # use and old local node ID from the free row
             dataID = self.nodes[nTypeID].freeRows.pop()
             lnID   = dataID + nTypeID * self.maxNodes
         else:
             # generate a new local ID
             lnID   = self.nodes[nTypeID].getNewID()
             #print lnID
             dataID = lnID - nTypeID * self.maxNodes
             #print dataID
         dataview = self.nodes[nTypeID][dataID:dataID+1].view() 
         if attributes is not None:
             dataview[:] = attributes
         self.nodes[nTypeID].nodeList.append(lnID)
         return lnID, dataview
#         return self.nodes[nTypeID].add(attributes)
    
   

    def remNode(self, lnID):
        nTypeID = self.getNodeType(lnID)
        dataID = lnID - nTypeID * self.maxNodes
        self.freeNodeRows.append(dataID)
        self.nodes[dataID]['gnID'] = -1
        self.nDict[nTypeID].remove(lnID)
    
    def setNodeAttr(self, lnID, label, value, nTypeID=None):
        
        nTypeID, dataID = self.getEdgeDataRef(lnID)
        self.nodes[nTypeID][label][dataID] = value
    
    def getNodeAttr(self, lnID, label=None, nTypeID=None):
        nTypeID, dataID = self.getNodeDataRef(lnID)
        if label:
            return self.nodes[nTypeID][label][dataID]
        else:
            return self.nodes[nTypeID][dataID]
        
        
    def setNodeSeqAttr(self, lnID, label, value, nTypeID=None):
       
        nTypeID, dataIDs = self.getNodeDataRef(lnID)
        self.nodes[nTypeID][label][dataIDs] = value

    def getNodeSeqAttr(self, lnIDs, label=None, nTypeID=None):
        nTypeID, dataIDs = self.getNodeDataRef(lnIDs)
        if label:
            return self.nodes[nTypeID][label][dataIDs]
        else:
            return self.nodes[nTypeID][dataIDs]

    #%% EDGES
    def initEdgeType(self, nodeName, size, attrDescriptor):
        
        eTypeID = self.getEdgeTypeID()
        dt = np.dtype(self.persEdgeAttr + attrDescriptor)
        eIDStart               = eTypeID * self.maxNodes

        self.edges[eTypeID] = self.EdgeArray(self.maxEdges, eIDStart, dtype=dt)
        return eTypeID

    def getEdgeDataRef(self, leID):
        """ calculates the node type ID and dataID from local ID"""
        if isinstance(leID, tuple):
            eTypeID, dataID = leID
        else:
            if isinstance(leID, list):
                leID = np.asarray(leID)
                eTypeID, dataID = int(leID[0] / self.maxEdges), leID%self.maxEdges
            
            elif isinstance(leID, np.ndarray):
                eTypeID, dataID = int(leID[0] / self.maxEdges), leID%self.maxEdges
            else:
                eTypeID, dataID = int(leID / self.maxEdges), leID%self.maxEdges
        
        #assert leID in self.edges[eTypeID].edgeList ##OPTPRODUCTION
        return  eTypeID, dataID
    
    def addEdge(self, eTypeID, source, target, attributes = None):
         """ 
         Adding a new connecting edge between source and target of
         the specified type
         """
         
         if len(self.edges[eTypeID].freeRows) > 0:
             # use and old local node ID from the free row
             dataID = self.edges[eTypeID].freeRows.pop()
             leID   = dataID + eTypeID * self.maxNodes
         else:
             # gernerate a new local ID
             leID   = self.edges[eTypeID].getNewID()
             #print lnID
             dataID = leID - eTypeID * self.maxNodes
             #print dataID
         dataview = self.edges[eTypeID][dataID:dataID+1].view() 
         
         #updating edge dictionaries
         self.edges[eTypeID].eDict[(source, target)] = dataID
         self.edges[eTypeID].edgeList.append(leID)
         if source in self.edges[eTypeID].edgesOut:
             self.edges[eTypeID].edgesOut[source].append(dataID)
         else:
             self.edges[eTypeID].edgesOut[source] = [dataID]
         if target in self.edges[eTypeID].edgesIn:
             self.edges[eTypeID].edgesIn[target].append(dataID)
         else:
             self.edges[eTypeID].edgesIn[target] = [dataID]     
             
         if attributes is not None:
             dataview[:] = (source, target) + attributes
         
         
         return leID, dataview

    def remEdge(self, source, target, eTypeID):
        
        dataID = self.edges[eTypeID].eDict[(source, target)]
        leID   = dataID + eTypeID * self.maxEdges 
        self.edges[eTypeID].freeRows.append(dataID)
        self.edges[eTypeID][dataID:dataID+1][:] = (-1, -1,- 1)
        del self.edges[eTypeID].eDict[(source, target)]
        self.edges[eTypeID].edgeList.remove(leID)
        self.edges[eTypeID].edgesIn[target].remove(dataID)
        self.edges[eTypeID].edgesOut[source].remove(dataID)
        
    def setEdgeAttr(self, leID, label, value, eTypeID=None):
        
        eTypeID, dataID = self.getEdgeDataRef(leID)
        
        self.edges[eTypeID][label][dataID] = value
    
    def getEdgeAttr(self, leID, label=None, eTypeID=None):

        eTypeID, dataID = self.getEdgeDataRef(leID)
            
        if label:
            return self.edges[eTypeID][dataID][label]
        else:
            return self.edges[eTypeID][dataID]
        
        
    def setEdgeSeqAttr(self, leIDs, label, value, eTypeID=None):
        
        eTypeID, dataIDs = self.getEdgeDataRef(leIDs)
        
        self.edges[eTypeID][label][dataIDs] = value
    
    def getEdgeSeqAttr(self, leIDs, label=None, eTypeID=None):
       
        eTypeID, dataIDs = self.getEdgeDataRef(leIDs)
        
        if label:
            return self.edges[eTypeID][label][dataIDs]
        else:
            return self.edges[eTypeID][dataIDs]
        
    #%% General
    def isConnected(self, source, target, eTypeID):
        """ 
        Returns if source and target is connected by an eddge of the specified
        edge type
        """
        return (source, target) in self.edges[eTypeID].eDict

    def outgoing(self, lnID, eTypeID):
        """ Returns the dataIDs of all outgoing edges of the specified type"""
        try: 
            return self.edges[eTypeID].edgesOut[lnID] 
        except:
            return []
    def incomming(self, lnID, eTypeID):
        """ Returns the dataIDs of all incoming edges of the specified type"""
        try:
            return self.edges[eTypeID].edgesIn[lnID]
        except:
            return []
        
    def nCount(self, nTypeID=None):
        """Returns the number of nodes of all or a specific node type"""
        if nTypeID is None:
            return sum([nodeType.nCount() for nodeType in self.nodes.itervalues()])
        else:
            return self.nodes[nTypeID].nCount()

    def eCount(self, eTypeID=None):
        """Returns the number of edges of all or a specific node type"""
        if eTypeID is None:
            return sum([edgeType.eCount() for edgeType in self.edges.itervalues()])
        else:
            return self.edges[eTypeID].eCount()

    def selfTest(self):
        """ 
        This function is testing the base graph class
        Not complete 
        """
        NT1 = self.initNodeType('A',
                          10000, 
                          [('f1', np.float32,4),
                           ('i2', np.int32,1),
                           ('s3', np.str,20)])
        self.addNode(NT1,(1000, [1,2,3,4],44, 'foo' ))
        
class WorldGraphNP(BaseGraph):
    """
    World graph NP is a high-level API to contrl BaseGraph
    
    """    

    def __init__(self, world, maxNodesPerType, maxEdgesPerType):

        BaseGraph.__init__(self, maxNodesPerType, maxEdgesPerType)
        self.world = world
        self.queingMode = False

        # list of types
        self.nodeTypes = dict()
        self.edgeTypes = dict()
        self.node2EdgeType = dict()
        self.edge2NodeType = dict()

        # dict of classes to init node automatically
        self.nodeType2Class = dict()
        self.class2NodeType = dict()

    def addNodeType(self, 
                    nodeTypeIdx, 
                    typeStr, 
                    AgentClass, 
                    GhostAgentClass, 
                    staticProperties, 
                    dynamicProperties):
        """ Create node type description"""
        
        nodeType = TypeDescription(nodeTypeIdx, typeStr, staticProperties, dynamicProperties)
        self.nodeTypes[nodeTypeIdx] = nodeType
        # same nodeType for ghost and non-ghost
        self.nodeType2Class[nodeTypeIdx]      = AgentClass, GhostAgentClass
        self.class2NodeType[AgentClass]       = nodeTypeIdx
        self.class2NodeType[GhostAgentClass]  = nodeTypeIdx
        self.initNodeType(typeStr, 1e5, staticProperties + dynamicProperties)

    def addEdgeType(self ,  edgeTypeIdx, typeStr, staticProperties, dynamicProperties, nodeType1, nodeType2):
        """ Create edge type description"""
        edgeType = TypeDescription(edgeTypeIdx, typeStr, staticProperties, dynamicProperties)
        self.edgeTypes[edgeTypeIdx] = edgeType
        self.node2EdgeType[nodeType1, nodeType2] = edgeTypeIdx
        self.edge2NodeType[edgeTypeIdx] = nodeType1, nodeType2
        self.initEdgeType(typeStr, 1e5, staticProperties + dynamicProperties)


    def getPropOfNodeType(self, nodeType, kind):
        if kind == 'all':
            return self.nodeTypes[nodeType].staProp + self.nodeTypes[nodeType].dynProp
        elif kind == 'sta':
            return self.nodeTypes[nodeType].staProp
        elif kind == 'dyn':
            return self.nodeTypes[nodeType].dynProp

    def add_edges(self, edgeList, **argProps):
        """ overrides graph.add_edges"""
        eStart = self.ecount()
        Graph.add_edges(self, edgeList)
        for key in argProps.keys():
            self.es[eStart:][key] = argProps[key]

    def startQueuingMode(self):
        """
        Starts queuing mode for more efficient setup of the graph
        Blocks the access to the graph and stores new vertices and eges in
        a queue
        """
        pass

    def stopQueuingMode(self):
        """
        Stops queuing mode and adds all vertices and edges from the queue.
        """
        pass


    def add_edge(self, source, target, **kwproperties):
        """ overrides graph.add_edge"""
        return Graph.add_edge(self, source, target, **kwproperties)

    def add_vertex(self, nodeType, gID, **kwProperties):
        """ overrides graph.add_vertice"""
        nID  = len(self.vs)
        kwProperties.update({'nID':nID, 'type': nodeType, 'gID':gID})
        Graph.add_vertex(self, **kwProperties)
        return nID, self.vs[nID]

    def delete_edges(self, edgeIDs=None, pairs=None):
        """ overrides graph.delete_edges"""
        if pairs:
            edgeIDs = self.get_eids(pairs)


        self.es[edgeIDs]['type'] = 0 # set to inactive

    def delete_vertex(self, nodeID):
        print 'not implemented yet'

class WorldGraph(Graph):
    """
    World graph is an agumented version of igraphs Graph that overrides some
    functions to work with the restrictions and functionality of world.
    It ensures that for example that deleting and adding of edges are recognized
    and cached.
    It also extends some of the functionalities of igraph (e.g. add_edges).
    """

    def __init__(self, world, directed=None):

        Graph.__init__(self, directed=directed)
        self.world = world
        self.queingMode = False

        # list of types
        self.nodeTypes = dict()
        self.edgeTypes = dict()
        self.node2EdgeType = dict()
        self.edge2NodeType = dict()

        # dict of classes to init node automatically
        self.nodeType2Class = dict()
        self.class2NodeType = dict()



    def addNodeType(self, nodeTypeIdx, typeStr, AgentClass, GhostAgentClass, staticProperties, dynamicProperties):
        """ Create node type description"""
        nodeType = TypeDescription(nodeTypeIdx, typeStr, staticProperties, dynamicProperties)
        self.nodeTypes[nodeTypeIdx] = nodeType
        # same nodeType for ghost and non-ghost
        self.nodeType2Class[nodeTypeIdx]      = AgentClass, GhostAgentClass
        self.class2NodeType[AgentClass]       = nodeTypeIdx
        self.class2NodeType[GhostAgentClass]  = nodeTypeIdx

    def addEdgeType(self ,  edgeTypeIdx, typeStr, staticProperties, dynamicProperties, nodeType1, nodeType2):
        """ Create edge type description"""
        edgeType = TypeDescription(edgeTypeIdx, typeStr, staticProperties, dynamicProperties)
        self.edgeTypes[edgeTypeIdx] = edgeType
        self.node2EdgeType[nodeType1, nodeType2] = edgeTypeIdx
        self.edge2NodeType[edgeTypeIdx] = nodeType1, nodeType2
        
    def getPropOfNodeType(self, nodeType, kind):
        if kind == 'all':
            return self.nodeTypes[nodeType].staProp + self.nodeTypes[nodeType].dynProp
        elif kind == 'sta':
            return self.nodeTypes[nodeType].staProp
        elif kind == 'dyn':
            return self.nodeTypes[nodeType].dynProp

    def add_edges(self, edgeList, **argProps):
        """ overrides graph.add_edges"""
        eStart = self.ecount()
        Graph.add_edges(self, edgeList)
        for key in argProps.keys():
            self.es[eStart:][key] = argProps[key]

    def startQueuingMode(self):
        """
        Starts queuing mode for more efficient setup of the graph
        Blocks the access to the graph and stores new vertices and eges in
        a queue
        """
        pass

    def stopQueuingMode(self):
        """
        Stops queuing mode and adds all vertices and edges from the queue.
        """
        pass


    def add_edge(self, source, target, **kwproperties):
        """ overrides graph.add_edge"""
        return Graph.add_edge(self, source, target, **kwproperties)

    def add_vertex(self, nodeType, gID, **kwProperties):
        """ overrides graph.add_vertice"""
        nID  = len(self.vs)
        kwProperties.update({'nID':nID, 'type': nodeType, 'gID':gID})
        Graph.add_vertex(self, **kwProperties)
        return nID, self.vs[nID]

    def delete_edges(self, edgeIDs=None, pairs=None):
        """ overrides graph.delete_edges"""
        if pairs:
            edgeIDs = self.get_eids(pairs)


        self.es[edgeIDs]['type'] = 0 # set to inactive

    def delete_vertex(self, nodeID):
        print 'not implemented yet'

if __name__ == "__main__":

    world = dict()

    graph = WorldGraph(world)

    graph.add_vertices(5)

    graph.add_edges([(1,2),(2,3)], type=[1,1], weig=[0.5,0.5])

    bigTest = 1
    bg = BaseGraph(int(1e6), int(1e6))
    bg.selfTest()
    sdf
    #%% nodes
    LOC = bg.initNodeType('location', 
                          1000, 
                          [('pos', np.float64,2),
                           ('population', np.int16,1)])
    
    lnID1, dataview1 = bg.addNode(1, (1, np.random.random(2), 10 ))
    lnID2, dataview2 = bg.addNode(1, (2, np.random.random(2), 10 ))
    lnID3, dataview3 = bg.addNode(1, (3, np.random.random(2), 20 ))
    dataview1['gnID'] = 99
    dataview2['gnID'] = 88
    bg.getNodeAttr(lnID1)
    bg.setNodeSeqAttr([lnID1, lnID2],'gnID',[12,13])
    print bg.getNodeSeqAttr([lnID1, lnID2],'gnID',[12,13])
    print bg.getNodeAttr(lnID2)
    print bg.getNodeSeqAttr(np.array([lnID1, lnID2]),'gnID')
    
    #%% edges
    LOCLOC = bg.initEdgeType('loc-loc', 
                                  1000, 
                                  [('weig', np.float64, 1)])
    
    leID1, dataview4 = bg.addEdge(LOCLOC, lnID1, lnID2, (.5,))
    leID2, dataview4 = bg.addEdge(LOCLOC, lnID1, lnID3, (.4,))
    
    print bg.getEdgeSeqAttr([leID1, leID2], 'weig')
    bg.setEdgeSeqAttr([leID1, leID2], 'weig', [.9, .1])
    print bg.getEdgeSeqAttr([leID1, leID2], 'weig')
    dataview4
    print bg.isConnected(lnID1, lnID3, LOCLOC)
    print bg.outgoing(lnID1, LOCLOC)
    print bg.eCount(LOCLOC)
    #print bg.eAttr[1].target
    bg.remEdge(lnID1, lnID3, LOCLOC)
    
    print bg.isConnected(lnID1, lnID3, LOCLOC)
    print bg.outgoing(lnID1, LOCLOC)
    print bg.edges[1][0:2]['weig']

    print bg.getEdgeAttr(leID1)
    
    print bg.eCount()
    
    if bigTest:
        from tqdm import tqdm
        nAgents = int(1e5)
        AGENT = bg.initNodeType('agent', 
                                nAgents, 
                                [('preferences', np.float64,4),])
        AGAG = bg.initEdgeType('ag_ag', 
                                  nAgents, 
                                  [('weig', np.float64, 1)])
        agArray = np.zeros(nAgents, dtype=np.int32)
        randAttr = np.random.random([nAgents,4])
        
        for i in tqdm(range(nAgents)):
            lnID, dataview = bg.addNode(AGENT, (i,randAttr[i,:]))
            agArray[i] = lnID
         
        nConnections = int(5e5)
         
        for i in tqdm(range(nConnections)):
            
            source, target = np.random.choice(agArray,2)
            bg.addEdge(AGAG, source, target, attributes = (.5,))
             
        nChecks = int(1e5)
        for i in tqdm(range(nChecks)) :
            source, target = np.random.choice(agArray,2)
            bg.isConnected(source, target, AGAG)
            
        nReadsOfNeigbours = int(1e5)
        
        for i in tqdm(range(nReadsOfNeigbours)):
            lnID  = bg.nodes[2].nodeList[i]
            eList = bg.outgoing(lnID,AGAG)
            if len(eList)>0:
                neigList  = bg.edges[AGAG]['target'][eList]
                y = bg.getEdgeSeqAttr((1, eList),'weig')
                x = bg.getNodeSeqAttr(neigList,'preferences')
                