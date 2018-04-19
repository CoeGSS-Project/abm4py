#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 10:31:00 2017

@author: Andreas Geiges, Global Climate Forum e.V.
"""

from igraph import Graph
import numpy as np
import itertools
import functools

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
    
        def __new__(cls, maxNodes, nTypeID, dtype):
            # Input array is an already formed ndarray instance
            # We first cast to be our class type
            self = np.rec.array(np.empty(maxNodes, dtype)).view(cls)
            # add the new attribute to the created instance
            self.maxNodes       = maxNodes
            self.nType          = nTypeID
            self.getNewID       = itertools.count().next
            self.nodeList       = []
            self.freeRows       = []
            # Finally, we must return the newly created object:
            return self

        def __array_finalize__(self, obj):
            # see InfoArray.__array_finalize__ for comments
            if obj is None: return
            self.info = getattr(obj, 'info', None)
            if obj.dtype.names is not None:
                for name in obj.dtype.names:
                    self.__setattr__(name, getattr(obj, name, None))
            

        def __getattr__(self, attr):
            try: return getattr(self, attr)
            except AttributeError:
                # extend interface to all functions from numpy
                f = getattr(np, attr, None)
                if hasattr(f, '__call__'):
                    return functools.partial(f, self)
                else:
                    raise AttributeError(attr)
                  
            
        def nCount(self):
            return len(self.nodeList)
        
        def add(self, attributes):
            if len(self.freeRows) > 0:
                # use and old local node ID from the free row
                dataID  = self.freeRows.pop()
                lnID    = dataID + self.nType * self.maxNodes
            else:
                # generate a new local ID
                dataID  = self.getNewID()
                #print lnID
                lnID    = dataID + self.nType * self.maxNodes
                #print dataID
            dataview    = self[dataID:dataID+1].view() 
            
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
        def __new__(cls, maxEdges, eTypeID, dtype):
            self = np.rec.array(np.empty(maxEdges, dtype=dtype)).view(cls)
            
            self.maxEdges       = maxEdges
            self.eTypeID        = eTypeID
            self.getNewID       = itertools.count().next
            self.edgeList       = []
            self.freeRows       = []
            self.eDict          = dict()
            self.edgesOut       = dict() # (source -> leID)
            self.edgesIn        = dict() # (target -> leID)
            self.nodesOut       = dict() # (source -> target)
            self.nodesIn        = dict() # (target -> source)
            return self

        def __array_finalize__(self, obj):
            # see InfoArray.__array_finalize__ for comments
            if obj is None: return
            self.info = getattr(obj, 'info', None)
            if obj.dtype.names is not None:
                for name in obj.dtype.names:
                    self.__setattr__(name, getattr(obj, name, None))
            

        def __getattr__(self, attr):
            #try: return getattr(self, attr)
            #except AttributeError:
                # extend interface to all functions from numpy
            f = getattr(np, attr, None)
            if hasattr(f, '__call__'):
                return functools.partial(f, self)
            else:
                raise AttributeError(attr)
                
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
        
        self.maxNodes       = int(maxNodesPerType)
        self.maxEdges       = int(maxEdgesPerType)
        

        self.nodeGlob2Loc   = dict()
        self.nodeLoc2Glob   = dict()
        self.nodes          = dict()
        self.edges          = dict()
        
        
        self.eDict    = dict() # (source, target -> leID)

        
        self.getNodeTypeID = itertools.count(1).next
        self.getEdgeTypeID = itertools.count(1).next
        
        #persistent nodeattributes
        self.persNodeAttr =[('active', np.bool_,1)]
        #persistent edge attributes
        self.persEdgeAttr =[('active', np.bool_, 1),
                            ('source', np.int32, 1), 
                            ('target', np.int32, 1)]
    #%% NODES
    def initNodeType(self, nodeName, attrDescriptor):
        
        nTypeID          = self.getNodeTypeID()
        dt               = np.dtype(self.persNodeAttr + attrDescriptor)
        nIDStart         = nTypeID * self.maxNodes
        self.nodes[nTypeID] = self.NodeArray(self.maxNodes, nTypeID, dtype=dt)
        return nTypeID
    
#    def extendNodeArray(self, nTypeID):
#        currentSize = len(self.nAttr[nTypeID])
        #self.nAttr[nTypeID][currentSize:currentSize+self.CHUNK_SIZE] = 
        
    def getNodeDataRef(self, lnIDs):
        """ 
        calculates the node type ID from local ID
        ONLY on node type per call
        """
        try:
            return lnIDs[0] // self.maxNodes, lnIDs%self.maxNodes
            
        except:
            try: 
                return int(lnIDs[0] / self.maxNodes), [lnID%self.maxNodes for lnID in lnIDs]
            except:
                return lnIDs // self.maxNodes, lnIDs%self.maxNodes
        
#    def getNodeDataRef(self, lnID):
#        """ calculates the node type ID from local ID"""
#        if isinstance(lnID, tuple):
#            nTypeID, dataID = lnID
#        else:
#            if isinstance(lnID, list):
#                lnID = np.asarray(lnID)
#                nTypeID, dataID = int(lnID[0] / self.maxNodes), lnID%self.maxNodes
#                
#            elif isinstance(lnID, np.ndarray):
#            
#                nTypeID, dataID = int(lnID[0] / self.maxNodes), lnID%self.maxNodes
#            else:
#                nTypeID, dataID = int(lnID / self.maxNodes), lnID%self.maxNodes
#        
#        return  nTypeID, dataID
    
    def addNode(self, nTypeID, attributes=None, **kwProp):
        
        
         if len(self.nodes[nTypeID].freeRows) > 0:
             # use and old local node ID from the free row
             dataID = self.nodes[nTypeID].freeRows.pop()
             lnID   = dataID + nTypeID * self.maxNodes
         else:
             # generate a new local ID
             dataID   = self.nodes[nTypeID].getNewID()
             #print lnID
             lnID = dataID + nTypeID * self.maxNodes
             #print dataID
         dataview = self.nodes[nTypeID][dataID:dataID+1].view() 
         if attributes is not None:
             dataview[:] = (True,) + attributes
         else:
             dataview['active'] = True
             #print self.nodes[nTypeID].dtype.names
             
             dataview[kwProp.keys()] = tuple(kwProp.values())
             
         self.nodes[nTypeID].nodeList.append(lnID)
         
         return lnID, dataID, dataview
#         return self.nodes[nTypeID].add(attributes)
    
    def addNodes(self, nTypeID, nNodes, **kwAttr):
        """
        Method to create serveral nodes at once
        Attribures are given as list or array per key word
        """
        
        nType = self.nodes[nTypeID]
        dataIDs = np.zeros(nNodes, dtype=np.int32)
        
        if len(nType.freeRows) == 0:
            
            dataIDs[:] = [nType.getNewID() for x in range(nNodes)]  
        elif len(nType.freeRows) < nNodes:
            newIDs = [nType.getNewID() for x in range(nNodes - len(nType.freeRows))] 
            dataIDs[:] = nType.freeRows + newIDs
            nType.freeRows = []
        else:
            dataIDs[:] = nType.freeRows[:nNodes]
            nType.freeRows = nType.freeRows[nNodes:]
   
        nType.active[dataIDs] = True
        
        # adding attributes
        if kwAttr is not None:
            for attrKey in kwAttr.keys():
                nType[attrKey][dataIDs] = kwAttr[attrKey]

    def remNode(self, lnID):
        nTypeID = self.getNodeType(lnID)
        dataID = lnID - nTypeID * self.maxNodes
        self.freeNodeRows.append(dataID)
        self.nodes[dataID]['acive','gnID'] = (False, -1)
        self.nDict[nTypeID].remove(lnID)
    
    
    def isNode(self, lnIDs):
        """
        Checks if nodes are active
        only one node type per time can be checked
        """
        nTypeIDs, dataIDs  = self.getNodeDataRef(lnIDs)
        
        try: 
            return all(self.nodes[nTypeIDs]['active'][dataIDs])
        except:
            return self.nodes[nTypeIDs]['active'][dataIDs]
            
            
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

    def getNodeSeqAttr(self, lnIDs=None, label=None, nTypeID=None, dataIDs=None):
        
        if nTypeID is None:
            nTypeID, dataIDs = self.getNodeDataRef(lnIDs)
        if label:
            return self.nodes[nTypeID][dataIDs][label]
        else:
            return self.nodes[nTypeID][dataIDs]

    #%% EDGES
    def initEdgeType(self, nodeName, attrDescriptor):
        
        eTypeID = self.getEdgeTypeID()
        dt = np.dtype(self.persEdgeAttr + attrDescriptor)
        eIDStart               = eTypeID * self.maxNodes

        self.edges[eTypeID] = self.EdgeArray(self.maxEdges, eTypeID, dtype=dt)
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
    
    def get_leID(self, source, target, eTypeID):
        return self.eges[eTypeID].eDict[(source, target)]

    def addEdge(self, eTypeID, source, target, attributes = None):
        """ 
        Adding a new connecting edge between source and target of
        the specified type
        Attributes can be given optionally with the correct structured
        tuple
        """
        if not (self.isNode(source)) or not( self.isNode(target)):
            raise ValueError('Nodes do not exist')

        eType = self.edges[eTypeID]

         
        if len(self.edges[eTypeID].freeRows) > 0:
            # use and old local node ID from the free row
            dataID = eType.freeRows.pop()
            leID   = dataID + eTypeID * self.maxNodes
        else:
            # generate a new local ID
            dataID   = eType.getNewID()
            #print lnID
            leID = dataID + eTypeID * self.maxNodes
            #print dataID
        dataview = eType[dataID:dataID+1].view() 
         
        #updating edge dictionaries
        eType.eDict[(source, target)] = dataID
        eType.edgeList.append(leID)
        
        try:
            eType.edgesOut[source].append(dataID)
            eType.nodesOut[source].append(target)
        except:
            eType.edgesOut[source] = [dataID]
            eType.nodesOut[source] = [target]
        
        if target in eType.edgesIn:
            eType.edgesIn[target].append(dataID)
            eType.nodesIn[target].append(source)
        else:
            eType.edgesIn[target] = [dataID]     
            eType.nodesIn[target] = [source]
             
        if attributes is not None:
            dataview[:] = (True, source, target) + attributes
        else:
            dataview[['active', 'source', 'target']] = (True, source, target)
         
        return leID, dataID, dataview

    def addEdges(self, eTypeID, sources, targets, **kwAttr):
        """
        Method to create serveral edges at once
        Attribures are given as list or array per key word
        """
        
        if not (self.isNode(sources)) or not( self.isNode(targets)):
            raise('Nodes do not exist')
        
        nEdges = len(sources)
        
        eType = self.edges[eTypeID]
        dataIDs = np.zeros(nEdges, dtype=np.int32)
        
        if len(eType.freeRows) == 0:
            
            dataIDs[:] = [eType.getNewID() for x in range(nEdges)]  
        elif len(eType.freeRows) < nEdges:
            newIDs = [eType.getNewID() for x in range(nEdges - len(eType.freeRows))] 
            dataIDs[:] = eType.freeRows + newIDs
            eType.freeRows = []
        else:
            dataIDs[:] = eType.freeRows[:nEdges]
            eType.freeRows = eType.freeRows[nEdges:]
        
        eType.source[dataIDs] = sources
        eType.target[dataIDs] = targets    
        eType.active[dataIDs] = True
              
        #updating edge dictionaries
        leIDs = dataIDs + eTypeID * self.maxNodes
        eType.edgeList.extend(leIDs.tolist())
        for source, target, dataID in zip(sources,targets, dataIDs):
            eType.eDict[(source, target)] = dataID
            
            try:
                eType.edgesOut[source].append(dataID)
                eType.nodesOut[source].append(target)
            except:
                eType.edgesOut[source] = [dataID]
                eType.nodesOut[source] = [target]
            
            if target in eType.edgesIn:
                eType.edgesIn[target].append(dataID)
                eType.nodesIn[target].append(source)
            else:
                eType.edgesIn[target] = [dataID]     
                eType.nodesIn[target] = [source]

        
              
        if kwAttr is not None:
            for attrKey in kwAttr.keys():
                eType[attrKey][dataIDs] = kwAttr[attrKey]
#    def add_edges(self, sources, targes, attributes):

    def remEdge(self, source, target, eTypeID):
        
        eType = self.edges[eTypeID]
        
        dataID = eType.eDict[(source, target)]
        leID   = dataID + eTypeID * self.maxEdges 
        eType.freeRows.append(dataID)
        eType[dataID:dataID+1]['active'] = False
        del eType.eDict[(source, target)]
        eType.edgeList.remove(leID)
        eType.edgesIn[target].remove(dataID)
        eType.edgesOut[source].remove(dataID)
        
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
        
        #self.add_edge = BaseGraph.addEdge
        #self.add_edges = BaseGraph.addEdges

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
        self.initNodeType(typeStr, staticProperties + dynamicProperties)

    def addEdgeType(self , edgeTypeIdx, typeStr, staticProperties, dynamicProperties, nodeType1, nodeType2):
        """ Create edge type description"""
        edgeType = TypeDescription(edgeTypeIdx, typeStr, staticProperties, dynamicProperties)
        self.edgeTypes[edgeTypeIdx] = edgeType
        self.node2EdgeType[nodeType1, nodeType2] = edgeTypeIdx
        self.edge2NodeType[edgeTypeIdx] = nodeType1, nodeType2
        self.initEdgeType(typeStr, staticProperties + dynamicProperties)


    def getDTypeOfNodeType(self, nodeType, kind):
        
        if kind == 'sta':
            dtype = self.nodeTypes[nodeType].staProp
        elif kind == 'dyn':
            dtype = self.nodeTypes[nodeType].dynProp
        else:
            dtype = self.nodeTypes[nodeType].staProp + self.nodeTypes[nodeType].dynProp
        return dtype
    
    def getPropOfNodeType(self, nodeType, kind):
        
        dtype = self.getDTypeOfNodeType(nodeType, kind)
         
        info = dict()    
        info['names'] = []
        info['types'] = []
        info['sizes'] = []
        for it in dtype:
            info['names'].append(it[0])
            info['types'].append(it[1])
            info['sizes'].append(it[2])
        return info
            
#    def add_edges(self, edgeList, **argProps):
#        """ overrides graph.add_edges"""
#        eStart = self.ecount()
#        Graph.add_edges(self, edgeList)
#        for key in argProps.keys():
#            self.es[eStart:][key] = argProps[key]
#
#    def startQueuingMode(self):
#        """
#        Starts queuing mode for more efficient setup of the graph
#        Blocks the access to the graph and stores new vertices and eges in
#        a queue
#        """
#        pass
#
#    def stopQueuingMode(self):
#        """
#        Stops queuing mode and adds all vertices and edges from the queue.
#        """
#        pass
#
#
##    def add_edge(self, source, target, **kwproperties):
##        """ overrides graph.add_edge"""
##        return Graph.add_edge(self, source, target, **kwproperties)
#
#    def add_vertex(self, nodeType, gID, **kwProperties):
#        """ overrides graph.add_vertice"""
#        nID  = len(self.vs)
#        kwProperties.update({'nID':nID, 'type': nodeType, 'gID':gID})
#        Graph.add_vertex(self, **kwProperties)
#        return nID, self.vs[nID]
#
    def delete_edges(self, source, target, eTypeID):
        """ overrides graph.delete_edges"""
        edgeIDs = self.get_leID(source, target, eTypeID)


        self.remEdge(edgeIDs) # set to inactive
#
#    def delete_vertex(self, nodeID):
#        print 'not implemented yet'

    def getOutNodeValues(self, lnID, eTypeID, attr=None):
        """
        Method to read the attributes of connected nodes via a 
        specifice edge type and outward direction
        """
        nIDsOut = self.edges[eTypeID].nodesOut[lnID]
        
        nTypeID, dataIDs = self.getNodeDataRef(nIDsOut)
        
        if attr:
            return self.nodes[nTypeID][dataIDs][attr]
        else:
            return self.nodes[nTypeID][dataIDs]

    def getInNodeValues(self, lnID, eTypeID, attr=None):
        """
        Method to read the attributes of connected nodes via a 
        specifice edge type and inward direction
        """
        nIDsIn = self.edges[eTypeID].nodesIn[lnID]
        
        nTypeID, dataIDs = self.getNodeDataRef(nIDsIn)
        
        if attr:
            return self.nodes[nTypeID][dataIDs][attr]
        else:
            return self.nodes[nTypeID][dataIDs]

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
    #bg.selfTest()
    #sdf
    #%% nodes
    LOC = bg.initNodeType('location', 
                          [('gnID', np.int16, 1),
                           ('pos', np.float64,2),
                           ('population', np.int16,1)])
    
    lnID1, _, dataview1 = bg.addNode(LOC, (1, np.random.random(2), 10 ))
    lnID2, _, dataview2 = bg.addNode(LOC, (2, np.random.random(2), 10 ))
    lnID3, _, dataview3 = bg.addNode(LOC, (3, np.random.random(2), 20 ))
    lnID4, _, dataview4 = bg.addNode(LOC, (4, np.random.random(2), 20 ))
    dataview1['gnID'] = 99
    dataview2['gnID'] = 88
    bg.getNodeAttr(lnID1)
    bg.setNodeSeqAttr([lnID1, lnID2],'gnID',[12,13])
    print bg.getNodeSeqAttr([lnID1, lnID2],'gnID')
    print bg.getNodeAttr(lnID2)
    print bg.getNodeSeqAttr(np.array([lnID1, lnID2]),'gnID')
    
    #%% edges
    LOCLOC = bg.initEdgeType('loc-loc', 
                                  [('weig', np.float64, 1)])
    
    
    leID1, _, dataview4 = bg.addEdge(LOCLOC, lnID1, lnID2, (.5,))
    leID2, _, dataview4 = bg.addEdge(LOCLOC, lnID1, lnID3, (.4,))
    
    
    
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
    print bg.edges[LOC][0:2]['weig']

    print bg.getEdgeAttr(leID1)
    
    print bg.eCount()
    
    bg.addEdges(LOCLOC, [lnID1, lnID2], [lnID4, lnID4], weig=[-.1, -.112])
    
    
    if bigTest:
        from tqdm import tqdm
        nAgents = int(1e5)
        AGENT = bg.initNodeType('agent', 
                                [('gnID', np.int16, 1),
                                 ('preferences', np.float64,4),])
        AGAG = bg.initEdgeType('ag_ag', 
                                  [('weig', np.float64, 1)])
        agArray = np.zeros(nAgents, dtype=np.int32)
        randAttr = np.random.random([nAgents,4])
        
        for i in tqdm(range(nAgents)):
            lnID, _, dataview = bg.addNode(AGENT, (i,randAttr[i,:]))
            agArray[i] = lnID
         
        nConnections = int(1e5)
         
        for i in tqdm(range(nConnections)):
            
            source, target = np.random.choice(agArray,2)
            bg.addEdge(AGAG, source, target, attributes = (.5,))
        
        import time
        tt = time.time()
        weights = np.zeros(nConnections)+.3
        sources = np.random.choice(agArray, nConnections)
        targets = np.random.choice(agArray, nConnections)
        
        bg.addEdges(AGAG, sources, targets, weig=weights)
        print (str(time.time() - tt) + ' s')
        
        nChecks = int(1e5)
        for i in tqdm(range(nChecks)) :
            source, target = np.random.choice(agArray,2)
            bg.isConnected(source, target, AGAG)
            
        nReadsOfNeigbours = int(1e5)
        
        for i in tqdm(range(nReadsOfNeigbours)):
            lnID  = bg.nodes[AGENT].nodeList[i]
            eList = bg.outgoing(lnID,AGAG)
            if len(eList)>0:
                neigList  = bg.edges[AGAG]['target'][eList]
                y = bg.getEdgeSeqAttr((1, eList),'weig')
                x = bg.getNodeSeqAttr(neigList,'preferences')
                