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
    
    def __init__(self, maxNodes, maxEdges):
        
        self.maxNodes       = maxNodes
        self.maxEdges       = maxEdges
        
        self.freeNodeRows   = dict()
        self.freeEdgeRows   = dict()
        self.nodeGlob2Loc   = dict()
        self.nodeLoc2Glob   = dict()
        self.nAttr          = dict()
        self.eAttr          = dict()
        self.getNewNodeID   = dict()
        self.getNewEdgeID   = dict()
        
        self.nCount   = 0
        self.eCount   = 0
        self.eDict    = dict() # (source, target -> leID)
        self.edgesOut = dict() # (source -> targets)
        self.edgesIn  = dict() # (target -> sources)
        
        self.getNodeTypeID = itertools.count(1).next
        self.getEdgeTypeID = itertools.count(1).next
        
        #persistent nodeattributes
        self.persNodeAttr =[('gnID', np.int64, 1)]
        #persistent edge attributes
        self.persEdgeAttr =[('source', np.int64, 1), 
                            ('target', np.int64, 1)]
    #%% NODES
    def initNodeType(self, nodeName, size, attrDescriptor):
        
        nTypeID = self.getNodeTypeID()
        dt = np.dtype(self.persNodeAttr + attrDescriptor)
        nIDStart               = nTypeID * self.maxNodes
        self.nAttr[nTypeID]    = np.rec.array(np.empty(size,dtype=dt))
        self.getNewNodeID[nTypeID] = itertools.count(nIDStart).next
        self.freeNodeRows[nTypeID] = []
        return nTypeID
    
#    def extendNodeArray(self, nTypeID):
#        currentSize = len(self.nAttr[nTypeID])
        #self.nAttr[nTypeID][currentSize:currentSize+self.CHUNK_SIZE] = 
        

                
    def getNodeType(self, lnID):
        """ calculates the node type ID from local ID"""
        return  int(lnID / self.maxNodes)
    
    def addNode(self, nTypeID, attributes=None):
        
        
         if len(self.freeNodeRows[nTypeID]) > 0:
             # use and old local node ID from the free row
             dataID = self.freeNodeRows[nTypeID].pop()
             lnID   = dataID + nTypeID * self.maxNodes
         else:
             # generate a new local ID
             lnID   = self.getNewNodeID[nTypeID]()
             #print lnID
             dataID = lnID - nTypeID * self.maxNodes
             #print dataID
         dataview = self.nAttr[nTypeID][dataID:dataID+1].view() 
         if attributes is not None:
             dataview[:] = attributes
         
         return lnID, dataview
    
   

    def remNode(self, lnID):
        nTypeID = self.getNodeType(lnID)
        dataID = lnID - nTypeID * self.maxNodes
        self.freeRows.append(dataID)
        self.nAttr[dataID]['gnID'] = -1
    
    def setNodeAttr(self, lnID, label, value, nTypeID=None):
        
        if not nTypeID:
            nTypeID = self.getNodeType(lnID)
        dataID = lnID - nTypeID * self.maxNodes
        self.nAttr[nTypeID][dataID][label] = value
    
    def getNodeAttr(self, lnID, label=None, nTypeID=None):
        if not nTypeID:
            nTypeID = self.getNodeType(lnID)
        dataID = lnID - nTypeID * self.maxNodes
        if label:
            return self.nAttr[nTypeID][dataID][label]
        else:
            return self.nAttr[nTypeID][dataID]
        
        
    def setNodeSeqAttr(self, lnID, label, value, nTypeID=None):
       
        if not nTypeID:
            nTypeID = self.getNodeType(lnID[0])
        dataID = lnID - nTypeID * self.maxNodes
        self.nAttr[nTypeID][dataID][label] = value
    
    def getNodeSeqAttr(self, lnIDs, label=None, nTypeID=None):
        if not nTypeID:
            nTypeID = self.getNodeType(lnIDs[0])
        dataIDs = lnIDs - nTypeID * self.maxNodes
        if label:
            return self.nAttr[nTypeID][label][dataIDs]
        else:
            return self.nAttr[nTypeID][dataIDs]

    #%% EDGES
    def initEdgeType(self, nodeName, size, attrDescriptor):
        
        
        eTypeID = self.getEdgeTypeID()
        dt = np.dtype(self.persEdgeAttr + attrDescriptor)
        self.eAttr[eTypeID]    = np.rec.array(np.empty(size,dtype=dt))
        nIDStart               = eTypeID * self.maxNodes
        
        self.getNewEdgeID[eTypeID] = itertools.count(nIDStart).next
        self.eDict[eTypeID] = dict()
        self.edgesOut[eTypeID] = dict() 
        self.edgesIn[eTypeID]  = dict() 
        self.freeEdgeRows[eTypeID] = []
        return eTypeID

    def getEdgeType(self, leID):
        """ calculates the node type ID from local ID"""
        return  int(leID / self.maxEdges)
    
    def addEdge(self, eTypeID, source, target, attributes = None):
        
        
         if len(self.freeEdgeRows[eTypeID]) > 0:
             # use and old local node ID from the free row
             dataID = self.freeEdgeRows[eTypeID].pop()
             leID   = dataID + eTypeID * self.maxNodes
         else:
             # gernerate a new local ID
             leID   = self.getNewEdgeID[eTypeID]()
             #print lnID
             dataID = leID - eTypeID * self.maxNodes
             #print dataID
         dataview = self.eAttr[eTypeID][dataID:dataID+1].view() 
         
         #updating edge dictionaries
         self.eDict[eTypeID][(source, target)] = dataID
         if source in self.edgesOut[eTypeID]:
             self.edgesOut[eTypeID][source].append(target)
         else:
             self.edgesOut[eTypeID][source] = [target]
         if target in self.edgesIn[eTypeID]:
             self.edgesIn[eTypeID][target].append(source)
         else:
             self.edgesIn[eTypeID][target] = [source]     
             
         if attributes is not None:
             dataview[:] = (source, target) + attributes
         
         return leID, dataview

    def remEdge(self, leID=None, source=None, target=None):
        
        if leID is None:
            leID = self.eDict((source, target))
            
        eTypeID = self.getEdgeType(leID)
        self.freeEdgeRows[eTypeID].append(leID)
        self.eAttr[leID]['source','target'] = (-1, -1)
            
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


    basegraph = BaseGraph(int(1e6), int(1e6))
    
    #%% nodes
    basegraph.initNodeType('agent', 1000, [('preferences', np.float64,4)])
    
    lnID1, dataview1 = basegraph.addNode(1)
    lnID2, dataview2 = basegraph.addNode(1)
    lnID3, dataview3 = basegraph.addNode(1)
    dataview1['gnID'] = 99
    dataview2['gnID'] = 88
    basegraph.getNodeAttr(lnID1)
    print basegraph.getNodeSeqAttr(np.array([lnID1, lnID2]),'gnID')
    
    #%% edges
    AGAG = basegraph.initEdgeType('ag_ag', 
                                  1000, 
                                  [('weig', np.float64, 1)])
    leID1, dataview4 = basegraph.addEdge(AGAG, lnID1, lnID2, (.5,))
    leID2, dataview4 = basegraph.addEdge(AGAG, lnID1, lnID3, (.5,))
    dataview4
    
    basegraph.remEdge(leID=leID1)
