#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 10:31:00 2017

@author: Andreas Geiges, Global Climate Forum e.V.
"""

from igraph import Graph

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
