#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://wwww.globalclimateforum.org

---- MoTMo ----
MOBILITY TRANSIOn MODEL
-- Class graph --

This file is part on GCFABM.

GCFABM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GCFABM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GCFABM.  If not, see <http://earth.gnu.org/licenses/>.
"""

import logging as lg
import numpy as np
import itertools
from numba import njit


@njit
def getRefByList(maxLen, idList):
    return idList[0] // maxLen, [lnID%maxLen for lnID in idList]
    
class TypeDescription():

    def __init__(self, agTypeIDIdx, typeStr, staticProperties, dynamicProperties):

        # list of properties per type
        
        self.staProp = staticProperties
        self.dynProp = dynamicProperties
        self.typeIdx = agTypeIDIdx
        self.typeStr = typeStr
        

    
class NodeArray(np.ndarray):
    def __new__(subtype, maxNodes, nTypeID, dtype=float, buffer=None, offset=0,
          strides=None, order=None):

        
        obj = np.ndarray.__new__(subtype, maxNodes, dtype, buffer, offset, strides,
                         order)
        obj.maxNodes       = maxNodes
        obj.nType          = nTypeID
        obj.getNewID       = itertools.count().__next__
        obj.nodeList       = []
        obj.freeRows       = []
        return obj

 
    def nCount(self):
        return len(self.nodeList)

    def indices(self):
        return self.nodeList

class EdgeArray(np.ndarray):
    """
    Data structure for edges and related informations based on a numpy array
    """    
    def __new__(subtype, maxEdges, eTypeID, dtype=float, buffer=None, offset=0,
          strides=None, order=None):

        # It also triggers a call to InfoArray.__array_finalize__
        obj = np.ndarray.__new__(subtype, maxEdges, dtype, buffer, offset, strides,
                         order)
        obj.maxEdges       = maxEdges
        obj.eTypeID        = eTypeID
        obj.getNewID       = itertools.count().__next__
        obj.edgeList       = []
        obj.freeRows       = []
        obj.eDict          = dict()
        obj.edgesOut       = dict() # (source -> leID)
        obj.edgesIn        = dict() # (target -> leID)
        obj.nodesOut       = dict() # (source -> target)
        obj.nodesIn        = dict() # (target -> source)
        return obj
                
    def eCount(self):
        return len(self.edgeList)

    def indices(self):
        return self.edgeList
              

class BaseGraph():
    """
    Graph class that is used to store data and connections of the ABM model
    """   
        
    def __init__(self, maxNodesPerType, maxEdgesPerType):
        """
        This class provides the basic functions to contruct a directed graph
        with different node and edge types. 
        The max number of edges and nodes
        cannot be exceeded during execution, since it is used for pre-assigning
        storage space.
        """
        
        self.maxNodes       = np.int64(maxNodesPerType)
        self.maxEdges       = np.int64(maxEdgesPerType)
        
        self.lnID2dataIdx   = dict()
        self.nodeGlob2Loc   = dict()
        self.nodeLoc2Glob   = dict()
        ## im nodes dict stehen keine nodes sondern nodeArrays (stf)
        self.nodes          = dict()
        ## auch im edges dict stehen keine edges (stf)
        self.edges          = dict()
        
        
        self.eDict    = dict() # (source, target -> leID)

        
        self.getNodeTypeID = itertools.count(1).__next__
        self.getEdgeTypeID = itertools.count(1).__next__
        
        #persistent node attributes
        self.persNodeAttr = [('active', np.bool_,1),
                             ('instance', np.object,1)]
        self.defaultNodeValues = (False, None,)
        #persistent edge attributes
        self.persEdgeAttr = [('active', np.bool_, 1),
                             ('source', np.int32, 1), 
                             ('target', np.int32, 1)]
        self.defaultEdgeValues = (False, None, None)
        
    #%% NODES
    def _initNodeType(self, nodeName, attrDescriptor):
        
        nTypeID          = self.getNodeTypeID()
        
        # enforcing that attributes are unique
        uniqueAttributes = []
        uniqueAttributesList = []
        for attrDesc in self.persNodeAttr + attrDescriptor:
            if attrDesc[0] not in uniqueAttributesList:
                uniqueAttributesList.append(attrDesc[0])
                uniqueAttributes.append(attrDesc)
        
        dt                  = np.dtype(uniqueAttributes)
        self.nodes[nTypeID] = NodeArray(self.maxNodes, nTypeID, dtype=dt)
        self.nodes[nTypeID]['active'] = False

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
            # array is given
            return lnIDs[0] // self.maxNodes, lnIDs%self.maxNodes

        except:
            try:
                # single ID is given
                return lnIDs // self.maxNodes, lnIDs%self.maxNodes
            except:
                # list is given
#                try:
                return getRefByList(self.maxNodes, lnIDs)
#                except:
#                    print(lnIDs)
#                   
#                    import pdb
#                    pdb.set_trace()
            


    def get_lnID(self, nTypeID):
        return self.nodes[nTypeID].nodeList

    ## when I grep for "attributes =" I get the impression that this is never used
    ## (as only Entity.__init__ calls addNode, beside some test methods)
    ## also here the kwProps are dropped when attributes are given (stf)
    def addNode(self, nTypeID, attributes=None, **kwProp):
        
        
         try:
             # use and old local node ID from the free row
             dataID = self.nodes[nTypeID].freeRows.pop()
             lnID   = dataID + nTypeID * self.maxNodes
         except:
             # generate a new local ID
             dataID   = self.nodes[nTypeID].getNewID()
             lnID = dataID + nTypeID * self.maxNodes

         dataview = self.nodes[nTypeID][dataID:dataID+1].view() 
#         if attributes is not None:
#             dataview[:] = (True,) + attributes
#         else:
         dataview['active'] = True
         dataview['ID'] = lnID
         if any(kwProp):
             dataview[list(kwProp.keys())] = tuple(kwProp.values())
         elif attributes is not None:
             dataview[:] = (True,) + attributes
         
             
         self.nodes[nTypeID].nodeList.append(lnID)
         
         self.lnID2dataIdx[lnID] = dataID
         return lnID, dataID, dataview[0]
     
    
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
   
        nType['active'][dataIDs] = True
        
        # adding attributes
        if kwAttr is not None:
            for attrKey in list(kwAttr.keys()):
                nType[attrKey][dataIDs] = kwAttr[attrKey]
        lnIDs = [dataID + nTypeID * self.maxNodes for dataID in dataIDs]  
        nType.nodeList.extend(lnIDs)
        return lnIDs

    def remNode(self, lnID):
        agTypeID, dataID = self.getNodeDataRef(lnID)
        self.nodes[agTypeID].freeRows.append(dataID)
        self.nodes[agTypeID][dataID:dataID+1][self.persitentAttributes] = self.defaultNodeValues
        self.nodes[agTypeID].nodeList.remove(lnID)
    
        for eTypeID in self.edges.keys():
            targets = self.outgoingIDs(lnID, eTypeID)
            [self.remEdge(eTypeID, lnID, target) for target in targets.copy()]
            sources = self.incommingIDs(lnID, eTypeID)
            [self.remEdge(eTypeID, source, lnID) for source in sources.copy()]

#    def remNode(self, lnID):
#        agTypeID, dataID = self.getNodeDataRef(lnID)
#        self.nodes[agTypeID].freeRows.append(dataID)
#        self.nodes[agTypeID][dataID:dataID+1][self.persitentAttributes] = self.defaultNodeValues
#        self.nodes[agTypeID].nodeList.remove(lnID)
#    
#        for eTypeID in self.edges.keys():
#            try:
#                self.remOutgoingEdges(eTypeID, lnID, self.edges[eTypeID].nodesOut[lnID].copy())
#            except:
#                pass
#            try:
#                [self.remEdge(eTypeID, source, lnID) for source in self.edges[eTypeID].nodesIn[lnID].copy()]
#            except:
#                pass
    
    def isNode(self, lnIDs):
        """
        Checks if node is active
        only one node type per time can be checked
        """
        nTypeIDs, dataIDs  = self.getNodeDataRef(lnIDs)        
        return self.nodes[nTypeIDs]['active'][dataIDs], nTypeIDs, dataIDs


    def areNodes(self, lnIDs):
        """
        Checks if nodes are active
        only one node type per time can be checked
        """
        nTypeIDs, dataIDs  = self.getNodeDataRef(lnIDs)
        
        return np.all(self.nodes[nTypeIDs]['active'][dataIDs])
                        
            
    def setAttrOfAgents(self, label, value, lnID, nTypeID=None):
        
        nTypeID, dataID = self.getEdgeDataRef(lnID)
        self.nodes[nTypeID][label][dataID] = value
    
    def getAttrOfAgents(self, label=None, lnID=None, nTypeID=None):
        nTypeID, dataID = self.getNodeDataRef(lnID)
        if label:
            return self.nodes[nTypeID][label][dataID]
        else:
            return self.nodes[nTypeID][dataID]
        
        
    def setNodeSeqAttr(self, label, values, lnIDs=None, nTypeID=None, dataIDs=None):
        """
        Nodes are either identified by list of lnIDS or (nType and dataID)
        Label is a either string or list of strings
        """
        if nTypeID is None:
            nTypeID, dataIDs = self.getNodeDataRef(lnIDs)
        
        if isinstance(label, list):
            # label is a list of string
            dtype2 = np.dtype({name:self.nodes[nTypeID]._data.dtype.fields[name] for name in label})
                
            view = np.ndarray(self.nodes[nTypeID]._data.shape, 
                              dtype2, 
                              self.nodes[nTypeID]._data, 
                              0, 
                              self.nodes[nTypeID]._data.strides) 
            #print view[dataIDs].dtype
            view[dataIDs] = values
        else:
            #label is a singel string
            self.nodes[nTypeID][label][dataIDs]= values
            
    def getNodeSeqAttr(self, label, lnIDs=None, nTypeID=None, dataIDs=None):
        """
        Nodes are either identified by list of lnIDS or (nType and dataID)
        label is a either string or list of strings
        """     
        if lnIDs==[]:
            return None
        if nTypeID is None:
            nTypeID, dataIDs = self.getNodeDataRef(lnIDs)
        
        if label:
            return self.nodes[nTypeID][label][dataIDs]
        else:
            return self.nodes[nTypeID][dataIDs]

    #%% EDGES
    def _initEdgeType(self, nodeName, attrDescriptor):
        
        eTypeID = self.getEdgeTypeID()
        dt = np.dtype(self.persEdgeAttr + attrDescriptor)
        self.edges[eTypeID] = EdgeArray(self.maxEdges, eTypeID, dtype=dt)
        
        return eTypeID

    def getEdgeDataRef(self, leID):
        """ calculates the node type ID and dataID from local ID"""
        if isinstance(leID, tuple):
            eTypeID, dataID = leID
            return  eTypeID, dataID
        else:
            try: 
                # array is given
                return leID[0] // self.maxEdges, lnIDs%self.maxEdges
    
            except:
                try:
                    # single ID is given
                    return leID // self.maxEdges, leID%self.maxEdges
                except:
                    # list is given
                    return(getRefByList(self.maxEdges, leID))
                
        return  eTypeID, dataID
    
    def get_leID(self, source, target, eTypeID):
        print ((source, target))
        return self.edges[eTypeID].eDict[(source, target)]

    def addEdge(self, eTypeID, source, target, attributes = None):
        """ 
        Adding a new connecting edge between source and target of
        the specified type
        Attributes can be given optionally with the correct structured
        tuple
        """
        sourceIsNode, srcNodeTypeID, srcDataID = self.isNode(source)
        targetIsNode, trgNodeTypeID, trgDataID = self.isNode(target)
        
        if not (sourceIsNode) or not(targetIsNode):
            raise ValueError('Nodes do not exist')

        eType = self.edges[eTypeID]

         
        try:
            # use and old local node ID from the free row
            dataID = eType.freeRows.pop()
            leID   = dataID + eTypeID * self.maxEdges
        except:
            # generate a new local ID
            dataID   = eType.getNewID()
            leID = dataID + eTypeID * self.maxEdges
            
        dataview = eType[dataID:dataID+1].view() 
         
        #updating edge dictionaries
        eType.eDict[(source, target)] = dataID
        eType.edgeList.append(leID)
        
        try:
            eType.edgesOut[source].append(dataID)
            eType.nodesOut[source][1].append(trgDataID)
        except:
            eType.edgesOut[source] = [dataID]
            eType.nodesOut[source] = trgNodeTypeID, [trgDataID]
        
        try:
            eType.edgesIn[target].append(dataID)
            eType.nodesIn[target][1].append(srcDataID)
        except:
            eType.edgesIn[target] = [dataID]     
            eType.nodesIn[target] = srcNodeTypeID, [srcDataID]
             
        if attributes is not None:
            dataview[:] = (True, source, target) + attributes
        else:
            dataview[['active', 'source', 'target']] = (True, source, target)
         
        return leID, dataID, dataview, eType.edgesOut[source]

    def addEdges(self, eTypeID, sources, targets, **kwAttr):
        """
        Method to create serveral edges at once
        Attribures are given as list or array per key word
        """
        if sources == []:
             raise(BaseException('Empty list given'))
        if not (self.areNodes(sources)) or not( self.areNodes(targets)):
            raise(BaseException('Nodes do not exist'))
        
        nEdges = len(sources)
        
        eType = self.edges[eTypeID]
        dataIDs = np.zeros(nEdges, dtype=np.int32)
        
        nfreeRows = len(eType.freeRows)

        if nfreeRows == 0:
            dataIDs[:] = [eType.getNewID() for x in range(nEdges)]  
        elif nfreeRows < nEdges:
            newIDs = [eType.getNewID() for x in range(nEdges - nfreeRows)] 
            dataIDs[:] = eType.freeRows + newIDs
            eType.freeRows = []
        else:
            dataIDs[:] = eType.freeRows[:nEdges]
            eType.freeRows = eType.freeRows[nEdges:]
        
        eType['source'][dataIDs] = sources
        eType['target'][dataIDs] = targets    
        eType['active'][dataIDs] = True
              
        #updating edge dictionaries
        leIDs = dataIDs + eTypeID * self.maxEdges
        eType.edgeList.extend(leIDs.tolist())
        for source, target, dataID in zip(sources, targets, dataIDs):
            eType.eDict[(source, target)] = dataID
            (srcNodeTypeID,  srcDataID) = self.getNodeDataRef(source)
            (trgNodeTypeID,  trgDataID) = self.getNodeDataRef(target)
        
            try:
                eType.edgesOut[source].append(dataID)
                eType.nodesOut[source][1].append(trgDataID)
            except:
                eType.edgesOut[source] = [dataID]
                eType.nodesOut[source] = trgNodeTypeID, [trgDataID]
            
            try:
                eType.edgesIn[target].append(dataID)
                eType.nodesIn[target][1].append(srcDataID)
            except:
                eType.edgesIn[target] = [dataID]     
                eType.nodesIn[target] = srcNodeTypeID, [srcDataID]
              
        if kwAttr is not None:
            for attrKey in list(kwAttr.keys()):
                eType[attrKey][dataIDs] = kwAttr[attrKey]

    def remOutgoingEdges(self, eTypeID, source, targets):
        eType = self.edges[eTypeID]
          
        eType.edgesOut[source] = []
        targets = eType.nodesOut.pop(source)
        
        for target in targets:
             dataID = eType.eDict.pop((source, target))
             leID   = dataID + eTypeID * self.maxEdges
             eType.edgeList.remove(leID)
             eType[dataID:dataID+1]['active'] = False
             eType.edgesIn[target].remove(dataID)
             eType.nodesIn[target].remove(source)
             
    def remEdge(self, eTypeID, source, target):
        eType = self.edges[eTypeID]
        
        dataID = eType.eDict.pop((source, target))
        leID   = dataID + eTypeID * self.maxEdges 
        eType.freeRows.append(dataID)
        eType[dataID:dataID+1]['active'] = False
        eType.edgeList.remove(leID)
        
        (_, srcDataID) = self.getNodeDataRef(source)
        (_, trgDataID) = self.getNodeDataRef(target)
            
        
        eType.edgesIn[target].remove(dataID)
        eType.edgesOut[source].remove(dataID)
        
        eType.nodesOut[source][1].remove(trgDataID)
        eType.nodesIn[target][1].remove(srcDataID)
#    
#    def remEdges(self, eTypeID, source, target):
#        eType = self.edges[eTypeID]
#        
#        dataID = eType.eDict[(source, target)]
#        leID   = dataID + eTypeID * self.maxEdges 
#        eType.freeRows.append(dataID)
#        eType[dataID:dataID+1]['active'] = False
#        del eType.eDict[(source, target)]
#        eType.edgeList.remove(leID)
#        eType.edgesIn[target].remove(dataID)
#        eType.edgesOut[source].remove(dataID)
#        
#        eType.nodesOut[source].remove(target)
#        eType.nodesIn[target].remove(source)
    def setEdgeAttr(self, leID, label, value, eTypeID=None):
        
        eTypeID, dataID = self.getEdgeDataRef(leID)
        
        self.edges[eTypeID][label][dataID] = value
    
    def getEdgeAttr(self, leID, label=None, eTypeID=None):

        eTypeID, dataID = self.getEdgeDataRef(leID)
            
        if label:
            return self.edges[eTypeID][dataID][label]
        else:
            return self.edges[eTypeID][dataID]
        

    def setEdgeSeqAttr(self, label, values, leIDs=None, eTypeID=None, dataIDs=None):
        
        if dataIDs is None:
            eTypeID, dataIDs = self.getEdgeDataRef(leIDs)
        else:
            assert leIDs is None    
        #print eTypeID, dataIDs
        self.edges[eTypeID][label][dataIDs] = values
    
    def getEdgeSeqAttr(self, label=None, leIDs=None, eTypeID=None, dataIDs=None):
        
        if dataIDs is None:
            assert eTypeID is None
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
            #print (eTypeID, self.edges[eTypeID].edgesOut[lnID]) , self.edges[eTypeID].nodesOut[lnID]
            return (eTypeID, self.edges[eTypeID].edgesOut[lnID]) , self.edges[eTypeID].nodesOut[lnID]
        except:
            return (None, []), (None, [])
        
    def incomming(self, lnID, eTypeID):
        """ Returns the dataIDs of all incoming edges of the specified type"""
        try:
            return (eTypeID, self.edges[eTypeID].edgesIn[lnID]), self.edges[eTypeID].nodesIn[lnID]
        except:
            return (None, []), (None, [])
        
    def outgoingIDs(self, lnID, eTypeID):
        """ Returns the dataIDs of all outgoing edges of the specified type"""
        try:
            nTypeID, dataIDs = self.edges[eTypeID].nodesOut[lnID]
            return self.nodes[nTypeID]['ID'][dataIDs]
        except:
            return []
        
    def incommingIDs(self, lnID, eTypeID):
        """ Returns the dataIDs of all outgoing edges of the specified type"""
        try:
            nTypeID, dataIDs = self.edges[eTypeID].nodesIn[lnID]
            return self.nodes[nTypeID]['ID'][dataIDs] 
        except:
            return []
        
    def outgoingInstance(self, lnID, eTypeID):
        """ Returns the dataIDs of all outgoing edges of the specified type"""
        try:
            
            nTypeID, dataIDs = self.edges[eTypeID].nodesOut[lnID]
            return self.nodes[nTypeID]['instance'][dataIDs] 
        except:
            return []

    def incommingInstance(self, lnID, eTypeID):
        """ Returns the dataIDs of all outgoing edges of the specified type"""
        try:
            nTypeID, dataIDs = self.edges[eTypeID].nodesIn[lnID]
            return self.nodes[nTypeID]['ID'][dataIDs] 
        except:
            return []
        
    def nCount(self, nTypeID=None):
        """Returns the number of nodes of all or a specific node type"""
        if nTypeID is None:
            return sum([agTypeID.nCount() for agTypeID in self.nodes.values()])
        else:
            return self.nodes[nTypeID].nCount()

    def eCount(self, eTypeID=None):
        """Returns the number of edges of all or a specific node type"""
        if eTypeID is None:
            return sum([liTypeID.eCount() for liTypeID in self.edges.values()])
        else:
            return self.edges[eTypeID].eCount()

    def selfTest(self):
        """ 
        This function is testing the base graph class
        Not complete 
        """
        NT1 = self._initNodeType('A',
                          10000, 
                          [('f1', np.float32,4),
                           ('i2', np.int32,1),
                           ('s3', np.str,20)])
        self.addNode(NT1,(1000, [1,2,3,4],44, 'foo' ))
        
class ABMGraph(BaseGraph):
    """
    World graph NP is a high-level API to contrl BaseGraph
    
    """    

    def __init__(self, world, maxNodesPerType, maxEdgesPerType):

        BaseGraph.__init__(self, maxNodesPerType, maxEdgesPerType)
        self.world = world
        self.queingMode = False

        # list of types
        self.agTypeByID = dict()
        self.agTypeByStr = dict()
        self.liTypeByID = dict()
        self.node2EdgeType = dict()
        self.edge2NodeType = dict()

        # dict of classes to init node automatically
        self.__agTypeID2Class = dict()
        self.__class2NodeType = dict()
        self.__ghostOfAgentClass   = dict()
        
        #persistent nodeattributes
        self.persitentAttributes = ['active', 'instance', 'gID', 'ID']
        self.defaultNodeValues     = (False, None, -1, -1)
        self.persNodeAttr = [('active', np.bool_,1),
                             ('instance', np.object,1),
                             ('gID', np.int32,1),
                             ('ID', np.int32,1)]


        
        
        #persistent edge attributes
        self.persEdgeAttr = [('active', np.bool_, 1),
                             ('source', np.int32, 1), 
                             ('target', np.int32, 1)]
    
        
    
    def addNodeType(self, 
                    agTypeIDIdx, 
                    typeStr, 
                    AgentClass, 
                    GhostAgentClass, 
                    staticProperties, 
                    dynamicProperties):
        """ Create node type description"""
        
        agTypeID = TypeDescription(agTypeIDIdx, typeStr, staticProperties, dynamicProperties)
        self.agTypeByID[agTypeIDIdx] = agTypeID
        self.agTypeByStr[typeStr]    = agTypeID
        # same agTypeID for ghost and non-ghost
        self.__agTypeID2Class[agTypeIDIdx]      = AgentClass, GhostAgentClass
        self.__class2NodeType[AgentClass]       = agTypeIDIdx
        if GhostAgentClass is not None:
            self.__class2NodeType[GhostAgentClass]  = agTypeIDIdx
            self.__ghostOfAgentClass[AgentClass]    = GhostAgentClass
        self._initNodeType(typeStr, staticProperties + dynamicProperties)

    def addEdgeType(self, typeStr, staticProperties, dynamicProperties, agTypeID1, agTypeID2):
        """ Create edge type description"""
        liTypeIDIdx = len(self.liTypeByID)+1
        liTypeID = TypeDescription(liTypeIDIdx, typeStr, staticProperties, dynamicProperties)
        self.liTypeByID[liTypeIDIdx] = liTypeID
        self.node2EdgeType[agTypeID1, agTypeID2] = liTypeIDIdx
        self.edge2NodeType[liTypeIDIdx] = agTypeID1, agTypeID2
        self._initEdgeType(typeStr, staticProperties + dynamicProperties)

        return liTypeIDIdx


    #%% NODE ACCESS
    def getDTypeOfNodeType(self, agTypeID, kind):
        
        if kind == 'sta':
            dtype = self.agTypeByID[agTypeID].staProp
        elif kind == 'dyn':
            dtype = self.agTypeByID[agTypeID].dynProp
        else:
            dtype = self.agTypeByID[agTypeID].staProp + self.agTypeByID[agTypeID].dynProp
        return dtype
    
    def getPropOfNodeType(self, agTypeID, kind):
        
        dtype = self.getDTypeOfNodeType(agTypeID, kind)
         
        info = dict()    
        info['names'] = []
        info['types'] = []
        info['sizes'] = []
        for it in dtype:
            info['names'].append(it[0])
            info['types'].append(it[1])
            info['sizes'].append(it[2])
        return info
            

#    def delete_edges(self, source, target, eTypeID):
#        """ overrides graph.delete_edges"""
#        edgeIDs = self.get_leID(source, target, eTypeID)
#
#        self.remEdge(edgeIDs) # set to inactive


    def getOutNodeValues(self, lnID, eTypeID, attr=None):
        """
        Method to read the attributes of connected nodes via a 
        specifice edge type and outward direction
        """
        nTypeID, dataIDs = self.edges[eTypeID].nodesOut[lnID]
        
        if attr:
            return self.nodes[nTypeID][dataIDs][attr]
        else:
            return self.nodes[nTypeID][dataIDs]

    def setOutNodeValues(self, lnID, eTypeID, attr, values):
        """
        Method to read the attributes of connected nodes via a 
        specifice edge type and outward direction
        """
        nTypeID, dataIDs = self.edges[eTypeID].nodesOut[lnID]
            
        self.nodes[nTypeID][attr][dataIDs] = values
        

    def getInNodeValues(self, lnID, eTypeID, attr=None):
        """
        Method to read the attributes of connected nodes via a 
        specifice edge type and inward direction
        """
        nTypeID, dataIDs = self.edges[eTypeID].nodesIn[lnID]
           
        if attr:
            return self.nodes[nTypeID][dataIDs][attr]
        else:
            return self.nodes[nTypeID][dataIDs]

    def setInNodeValues(self, lnID, eTypeID, attr, values):
        """
        Method to read the attributes of connected nodes via a 
        specifice edge type and inward direction
        """
        nTypeID, dataIDs = self.edges[eTypeID].nodesIn[lnID]

        self.edges[eTypeID][attr][dataIDs] = values


    #%%  EDGE ACCESS
    
    def getOutEdgeValues(self, leID, eTypeID, attr=None):
        """
        Method to read the attributes of connected nodes via a 
        specifice edge type and outward direction
        """
        dataIDs = self.edges[eTypeID].edgesOut[leID]
        
        if attr:
            return self.edges[eTypeID][dataIDs][attr]
        else:
            return self.edges[eTypeID][dataIDs]

    def setOutEdgeValues(self, leID, eTypeID, attr, values):
        """
        Method to read the attributes of connected nodes via a 
        specifice edge type and outward direction
        """
        dataIDs = self.edges[eTypeID].edgesOut[leID]
        self.edges[eTypeID][attr][dataIDs] = values
        

    def getInEdgeValues(self, leID, eTypeID, attr=None):
        """
        Method to read the attributes of connected nodes via a 
        specifice edge type and inward direction
        """
        dataIDs = self.edges[eTypeID].edgesIn[leID]
        
        #eTypeID, dataIDs = self.getNodeDataRef(nIDsIn)
        
        if attr:
            return self.edges[eTypeID][dataIDs][attr]
        else:
            return self.edges[eTypeID][dataIDs]

    def setInEdgeValues(self, leID, eTypeID, attr, values):
        """
        Method to read the attributes of connected nodes via a 
        specifice edge type and inward direction
        """
        dataIDs = self.edges[eTypeID].edgesIn[leID]
        self.edges[eTypeID][attr][dataIDs] = values
        


        
    def getNodeView(self, lnID):
        nTypeID, dataID = self.getNodeDataRef(lnID)
        return self.nodes[nTypeID][dataID:dataID+1].view()[0], dataID

    def agTypeID2Class(self, agTypeIDID):
        return self.__agTypeID2Class[agTypeIDID]
    
    def class2NodeType(self, agentClass):
        return self.__class2NodeType[agentClass]
    
    def ghostOfAgentClass(self, agentClass):
        return self.__ghostOfAgentClass[agentClass]
        

    def getAdjMatrix(self, agTypeID):
        liTypeID = self.node2EdgeType[agTypeID, agTypeID]
        
        nNodes = len(self.nodes[agTypeID].nodeList)
        
        adjMatrix = np.zeros([nNodes, nNodes])
        
        for key in self.edges[liTypeID].nodesIn.keys():
            nodeList = self.edges[liTypeID].nodesIn[key]
            
            if len(nodeList)> 0:
                source = key - self.maxNodes
                targetList = [target - self.maxNodes for target in nodeList]
                for target in targetList:
                    adjMatrix[source, target] = 1
        return adjMatrix
    
    def getAdjList(self, agTypeID):
        liTypeID = self.node2EdgeType[agTypeID, agTypeID]        
        
        eDict = self.edges[liTypeID].nodesOut
        adjList = []
        nLinks = 0
        for source in self.nodes[agTypeID].nodeList:
            if source in eDict.keys():
                targetList = [target - self.maxNodes for target in eDict[source] if target != source]
                nLinks += len(targetList)
                adjList.append(targetList)
            else:
                adjList.append([])
        return adjList, nLinks
    
if __name__ == "__main__":

    world = dict()


    bigTest = 1
    bg = ABMGraph(world, int(1e6), int(1e6))

    #%% nodes
    LOC = bg._initNodeType('location', 
                          [('gnID', np.int16, 1),
                           ('pos', np.float64,2),
                           ('population', np.int16,1)])
    
    lnID1, _, dataview1 = bg.addNode(LOC, (1, np.random.random(2), 10 ))
    lnID2, _, dataview2 = bg.addNode(LOC, (2, np.random.random(2), 10 ))
    lnID3, _, dataview3 = bg.addNode(LOC, (3, np.random.random(2), 20 ))
    lnID4, _, dataview4 = bg.addNode(LOC, (4, np.random.random(2), 20 ))
    dataview1['gnID'] = 99
    dataview2['gnID'] = 88
    bg.getAttrOfAgentType(lnID=lnID1)
    bg.setNodeSeqAttr('gnID', [12,13], [lnID1, lnID2])                        
    print(bg.getNodeSeqAttr('gnID', [lnID1, lnID2]))
    print(bg.getAttrOfAgentType(lnID=lnID2))
    print(bg.getNodeSeqAttr('gnID', np.array([lnID1, lnID2])))
    
    #%% edges
    LOCLOC = bg.addLinkType('loc-loc',  
                            staticProperties=[],
                            dynamicProperties=[('weig', np.float64, 1)],
                            agTypeID1 = LOC,
                            agTypeID2 = LOC)
                            
    
    leID1, _, dataview4 = bg.addLink(LOCLOC, lnID1, lnID2, (.5,))
    leID2, _, dataview4 = bg.addLink(LOCLOC, lnID1, lnID3, (.4,))
    
    
    
    print(bg.getEdgeSeqAttr('weig', [leID1, leID2]))
    bg.setEdgeSeqAttr('weig', [.9, .1], [leID1, leID2]) 
    print(bg.getEdgeSeqAttr('weig', [leID1, leID2]))
    dataview4
    print(bg.isConnected(lnID1, lnID3, LOCLOC))
    print(bg.outgoing(lnID1, LOCLOC))
    print(bg.eCount(LOCLOC))
    #print bg.eAttr[1]['target']
    bg.remEdge(lnID1, lnID3, LOCLOC)
    
    print(bg.isConnected(lnID1, lnID3, LOCLOC))
    print(bg.outgoing(lnID1, LOCLOC))
    print(bg.edges[LOC][0:2]['weig'])

    print(bg.getEdgeAttr(leID1))
    
    print(bg.eCount())
    
    bg.addLinks(LOCLOC, [lnID1, lnID2, lnID1], [lnID4, lnID4, lnID2], weig=[-.1, -.112, 3])
    
    x = bg.getAdjList(LOC)
    
    if bigTest:
        from tqdm import tqdm
        nAgents = int(1e5)
        AGENT = bg._initNodeType('agent', 
                                [('gnID', np.int16, 1),
                                 ('preferences', np.float64,4),])
        AGAG = bg._initEdgeType('ag_ag', 
                                  [('weig', np.float64, 1)])
        agArray = np.zeros(nAgents, dtype=np.int32)
        randAttr = np.random.random([nAgents,4])
        print('adding nodes')
        for i in tqdm(range(nAgents)):
            lnID, _, dataview = bg.addNode(AGENT, (i,randAttr[i,:]))
            agArray[i] = lnID
         
        nConnections = int(1e4)
        print('creating random edges') 
        for i in tqdm(range(nConnections)):
            
            source, target = np.random.choice(agArray,2)
            bg.addLink(AGAG, source, target, attributes = (.5,))
        
        import time
        tt = time.time()
        weights = np.zeros(nConnections)+.3
        sources = np.random.choice(agArray, nConnections)
        targets = np.random.choice(agArray, nConnections)
        
        bg.addLinks(AGAG, sources, targets, weig=weights)
        print((str(time.time() - tt) + ' s'))
        
        print('checking if nodes are connected')
        nChecks = int(1e5)
        for i in tqdm(range(nChecks)) :
            source, target = np.random.choice(agArray,2)
            bg.isConnected(source, target, AGAG)
            
        nReadsOfNeigbours = int(1e5)
        
        print('reading values of connected nodes')
        for i in tqdm(range(nReadsOfNeigbours)):
            lnID  = bg.nodes[AGENT].nodeList[i]
            (eType, dataIDs) , neigList = bg.outgoing(lnID,AGAG)
            if len(dataIDs)>0:
                neigList  = bg.edges[AGAG]['target'][dataIDs]
                y = bg.getEdgeSeqAttr('weig', eTypeID=eType, dataIDs=dataIDs)
                x = bg.getNodeSeqAttr('preferences', neigList)
        #%%
        print('writing values of sequence of nodes')    
        nNodes = 5
        nWriteOfSequence = int(1e4)
        values = np.random.randn(nWriteOfSequence+5, 4)
        for i in tqdm(range(nWriteOfSequence)):
            lnIDs = np.random.choice( bg.nodes[AGENT].nodeList[:10000],nNodes,replace=False)
            bg.setNodeSeqAttr( 'preferences', values[i:i+5,:], lnIDs=lnIDs)
