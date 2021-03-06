#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2018, Global Climate Forun e.V. (GCF)
http://www.globalclimateforum.org

This file is part of ABM4py.

ABM4py is free software: you can redistribute it and/or modify it 
under the terms of the GNU Lesser General Public License as published 
by the Free Software Foundation, version 3 only.

ABM4py is distributed in the hope that it will be useful, 
but WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License 
along with this program. If not, see <http://www.gnu.org/licenses/>. 
GNU Lesser General Public License version 3 (see the file LICENSE).
"""

import numpy as np
import itertools
from abm4py import misc
from abm4py import core


def formatPropertyDefinition(propertyList):
    """
    Checks and completes the property definition for entities and edges
    """
    for iProp in range(len(propertyList)):
        if not isinstance(propertyList[iProp], tuple):
            propertyList[iProp] = (propertyList[iProp], np.float64, 1) 
        else:
            if len(propertyList[iProp]) == 3:
                pass
                
            elif len(propertyList[iProp]) == 2:
                propertyList[iProp] = (propertyList[iProp] + (1,))
                print('Assuming a single number for ' + str(propertyList[iProp][1]))
            elif len(propertyList[iProp]) == 1:
                propertyList[iProp] = (propertyList[iProp] + (np.float64, 1,))
                print('Assuming a single float number for ' + str(propertyList[iProp][1]))
            else:
                raise(BaseException('Property format of ' + str(propertyList[iProp]) + ' not understood'))    
                
        assert isinstance(propertyList[iProp][0],str)
        assert isinstance(propertyList[iProp][1],type)
        assert isinstance(propertyList[iProp][2],int)
        
    return propertyList

try:
    from numba import njit
    @njit
    def getRefByList(maxLen, idList):
        return idList[0] // maxLen, [lnID%maxLen for lnID in idList]
except:
    print('numba import failed')

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
    """
    Node array derived from numpy ndarray
    """
    def __new__(subtype, maxNodes, nTypeID, dtype=float, buffer=None, offset=0,
          strides=None, order=None, startID = 0, nodeList=None):

        
        obj = np.ndarray.__new__(subtype, maxNodes, dtype, buffer, offset, strides,
                         order)
        obj.maxNodes       = maxNodes
        obj.nType          = nTypeID
        obj.getNewID       = itertools.count(startID).__next__
        if nodeList is None:
            obj.nodeList   = list()
        else:
            obj.nodeList   = nodeList
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
          strides=None, order=None, startID = 0, edgeList=None, eDict=None,
          edgesOut=None, edgesIn=None, nodesOut=None, nodesIn=None,
          freeRows=None):

        # It also triggers a call to InfoArray.__array_finalize__
        obj = np.ndarray.__new__(subtype, maxEdges, dtype, buffer, offset, strides,
                             order)
        obj.maxEdges       = maxEdges
        obj.eTypeID        = eTypeID
        obj.getNewID       = itertools.count(startID).__next__

        if edgeList is None:
            obj.edgeList   = list()
        else:
            obj.edgeList   = edgeList
            
        if freeRows is None:
            obj.freeRows   = list()
        else:
            obj.freeRows   = freeRows
            
        if eDict is None:
            obj.eDict = dict()
        else:
            obj.eDict = eDict
        
        if eDict is None:
            obj.eDict = dict()
        else:
            obj.eDict = eDict
            
        if edgesOut is None:
            obj.edgesOut = dict()
        else:
            obj.edgesOut = edgesOut
        
        if edgesIn is None:
            obj.edgesIn = dict()
        else:
            obj.edgesIn = edgesIn
        
        if nodesOut is None:
            obj.nodesOut = dict()
        else:
            obj.nodesOut = nodesOut     

        if nodesIn is None:
            obj.nodesIn  = dict()
        else:
            obj.nodesIn = nodesIn  
                    
        return obj
                
    def eCount(self):
        return len(self.edgeList)

    def indices(self):
        return self.edgeList
              

class BaseGraph():
    """
    This class provides the basic functions to contruct a directed._graph
    with different node and edge types. 
    The max number of edges and nodesis extended dynamically.
    """ 
        
    def __init__(self, maxNodesPerType, maxEdgesPerType):

        
        self.INIT_SIZE      = core.config.GRAPH_ARRAY_INIT_SIZE
        
        self.maxNodes       = np.int64(maxNodesPerType)
        self.maxEdges       = np.int64(maxEdgesPerType)
        
        self.lnID2dataIdx   = dict()
        self.nodeGlob2Loc   = dict()
        self.nodeLoc2Glob   = dict()
        
        # self.nodes is a dictionary of node types, each containing an array of the 
        # actual nodes of that type
        self.nodes          = dict()
        # self.edges is a dictionary of edge types, each containing an array of the 
        # actual edges of that type
        self.edges          = dict()
        
        
        self.eDict    = dict() # (source, target -> leID)

        
        self.getNodeTypeID = itertools.count(1).__next__
        self.getEdgeTypeID = itertools.count(1).__next__
        
        #persistent node attributes
        self.persNodeAttr = [('active', np.bool_,1),
                             ('instance', np.object,1)]

        #persistent edge attributes
        self.persEdgeAttr = [('active', np.bool_, 1),
                             ('source', np.int32, 1), 
                             ('target', np.int32, 1)]

        
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
        
        self.nodes[nTypeID] = NodeArray(self.INIT_SIZE, nTypeID, dtype=dt)
        self.nodes[nTypeID]['active'] = False
        self.nodes[nTypeID].currentSize = self.INIT_SIZE
        
        return nTypeID

    def _extendNodeArray(self, nTypeID, factor=None, newSize=None):
        """
        Method to increas the array for nodes to dynamically adapt to
        higher numbers of nodes
        """
        
        if newSize is None:
            currentSize = self.nodes[nTypeID].currentSize
            dt = self.nodes[nTypeID].dtype
            tmp = NodeArray(int(currentSize*factor), nTypeID, dtype=dt, startID = currentSize+1, nodeList = self.nodes[nTypeID].nodeList)
            tmp['active'] = False
            tmp[:currentSize] = self.nodes[nTypeID]
            self.nodes[nTypeID] = tmp
            
            for dataID, nodeInstance in enumerate(self.nodes[nTypeID]['instance'][:currentSize]):
                if nodeInstance is not None:
                    nodeInstance.attr = self.nodes[nTypeID][dataID:dataID+1].view()[0]

            self.nodes[nTypeID].currentSize = int(currentSize*factor)
        
        elif factor is None:
            assert newSize >=  len(self.nodeList)
            currentSize = self.nodes[nTypeID].currentSize
            
            dt = self.nodes[nTypeID].dtype
            tmp = NodeArray(int(newSize), nTypeID, dtype=dt, startID = currentSize+1, nodeList = self.nodes[nTypeID].nodeList)
            tmp['active'] = False
            tmp[:currentSize] = self.nodes[nTypeID]
            self.nodes[nTypeID] = tmp
            
            for dataID, nodeInstance in enumerate(self.nodes[nTypeID]['instance'][:currentSize]):
                if nodeInstance is not None:
                    nodeInstance.attr = self.nodes[nTypeID][dataID:dataID+1].view()[0]
                
            self.nodes[nTypeID].currentSize = int(newSize)
        else:
            raise()
        
    def getNodeDataRef(self, lnIDs):
        """ 
        calculates the node type ID from local ID
        ONLY on node type per call
        
        returns: (nodeTypeID, dataID)
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
                return getRefByList(self.maxNodes, lnIDs)
   


    def get_lnID(self, nTypeID):
        return self.nodes[nTypeID].nodeList


    def addNode(self, nTypeID, attributes=None, **kwProp):
        #TODO attributes might be dropped in the future since is seems 
        # the more complicated version of assigning attributes
        
         try:
             # use and old local node ID from the free row
             dataID = self.nodes[nTypeID].freeRows.pop()
             lnID   = dataID + nTypeID * self.maxNodes
         except:
             # generate a new local ID
             dataID   = self.nodes[nTypeID].getNewID()
             assert dataID < self.maxNodes 
             lnID = dataID + nTypeID * self.maxNodes
             
             if dataID >= self.nodes[nTypeID].currentSize:
                 self._extendNodeArray(nTypeID, core.config.EXTENTION_FACTOR)

         dataview = self.nodes[nTypeID][dataID:dataID+1].view() 

         dataview['active'] = True
         dataview['ID'] = lnID
         if any(kwProp):
             dataview[list(kwProp.keys())] = tuple(kwProp.values())
         elif attributes is not None:
             dataview[:] = (True,) + attributes
         
             
         self.nodes[nTypeID].nodeList.append(lnID)
         
         self.lnID2dataIdx[lnID] = dataID
         try:
             return lnID, dataID, dataview[0]
         except:
             import pdb 
             pdb.set_trace()
     
    
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
        
        assert max(dataIDs) < self.maxNodes
        if max(dataIDs) >= self.nodes[nTypeID].currentSize:
            extFactor = int(max(np.ceil((max(dataIDs) +1) / nType.currentSize), core.config.EXTENTION_FACTOR))
            self._extendNodeArray(nTypeID, extFactor)
            nType = self.nodes[nTypeID]
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
        self.nodes[agTypeID][dataID:dataID+1]['active'] = False
        self.nodes[agTypeID].nodeList.remove(lnID)
    
        for eTypeID in self.edges.keys():
            targetIDs = self.outgoingIDs(lnID, eTypeID)
            [self.remEdge(eTypeID, lnID, targetID) for targetID in targetIDs.copy()]
            sourceIDs = self.incommingIDs(lnID, eTypeID)
            [self.remEdge(eTypeID, sourceID, lnID) for sourceID in sourceIDs.copy()]

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
        return self.nodes[nTypeIDs]['active'][dataIDs]


    def areNodes(self, lnIDs):
        """
        Checks if nodes are active
        only one node type per time can be checked
        """
        nTypeIDs, dataIDs  = self.getNodeDataRef(lnIDs)
        
        return np.all(self.nodes[nTypeIDs]['active'][dataIDs])
                        
    def getAttrOfNodeType(self, attribute, nTypeID):

        array = self.nodes[nTypeID]
        return array[attribute][array['active']]
 
    def setAttrOfNodeType(self, attribute, values, nTypeID):

        array = self.nodes[nTypeID]
        array[attribute][array['active']] = values  
            
    def getAttrOfNodesIdx(self, attribute, nTypeID, dataIDs):
        if attribute:
            return self.nodes[nTypeID][attribute][dataIDs]
        else:
            return self.nodes[nTypeID][dataIDs]
    
    def getAttrOfNodesSeq(self, attribute, lnID):
        nTypeID, dataID = self.getNodeDataRef(lnID)
        if attribute:
            return self.nodes[nTypeID][attribute][dataID]
        else:
            return self.nodes[nTypeID][dataID]

    def setAttrOfNodesSeq(self, attribute, values, lnIDs):
        
        nTypeID, dataID = self.getNodeDataRef(lnIDs)
        self.nodes[nTypeID][attribute][dataID] = values
        
    def setNodeSeqAttr(self, attribute, values, lnIDs):
        """
        Nodes are either identified by list of lnIDS or (nType and dataID)
        Label is a either string or list of strings
        """
        nTypeID, dataIDs = self.getNodeDataRef(lnIDs)
        
        self.nodes[nTypeID][attribute][dataIDs] = values
            
    def getNodeSeqAttr(self, attribute, lnIDs):
        """
        Nodes are either identified by list of lnIDS or (nType and dataID)
        attribute is a either string or list of strings
        """     
        if lnIDs==[]:
            return None
        nTypeID, dataIDs = self.getNodeDataRef(lnIDs)
        
        if attribute:
            return self.nodes[nTypeID][attribute][dataIDs]
        else:
            return self.nodes[nTypeID][dataIDs]

    #%% EDGES
    def _initEdgeType(self, nodeName, attrDescriptor):
        
        eTypeID = self.getEdgeTypeID()
        uniqueAttributes = []
        uniqueAttributesList = []
        for attrDesc in self.persEdgeAttr + attrDescriptor:
            if attrDesc[0] not in uniqueAttributesList:
                uniqueAttributesList.append(attrDesc[0])
                uniqueAttributes.append(attrDesc)
        
        dt                  = np.dtype(uniqueAttributes)
        
        #dt = np.dtype(self.persEdgeAttr + attrDescriptor)
        self.edges[eTypeID] = EdgeArray(self.INIT_SIZE, eTypeID, dtype=dt)
        self.edges[eTypeID]['active'] = False
        self.edges[eTypeID].currentSize = self.INIT_SIZE
        return eTypeID

    def _extendEdgeArray(self, eTypeID, factor=None, newSize=None):
        """
        MEthod to increas the array for edges to dynamically adapt to
        higher numbers of edges
        """
        
        if newSize is None:
            edges = self.edges[eTypeID]
            currentSize = edges.currentSize
            dt = edges.dtype
            tmp = EdgeArray(int(currentSize*factor), eTypeID, dtype=dt, 
                            startID = currentSize+1, edgeList=edges.edgeList, eDict = edges.eDict,
                            edgesOut= edges.edgesOut, edgesIn= edges.edgesIn,
                            nodesOut= edges.nodesOut, nodesIn= edges.nodesIn,
                            freeRows=edges.freeRows)
            
            tmp['active'] = False
            tmp[:currentSize] = self.edges[eTypeID]
            self.edges[eTypeID] = tmp
            self.edges[eTypeID].currentSize = int(currentSize*factor)
        
        elif factor is None:
            assert newSize >=  len(self.edgeList)
            edges = self.edges[eTypeID]
            currentSize = edges.currentSize
            dt = edges.dtype
            tmp = EdgeArray(int(newSize), eTypeID, dtype=dt, 
                            startID = currentSize+1, edgeList=edges.edgeList, eDict = edges.eDict,
                            edgesOut= edges.edgesOut, edgesIn= edges.edgesIn,
                            nodesOut= edges.nodesOut, nodesIn= edges.nodesIn)
            tmp['active'] = False
            tmp[:currentSize] = self.edges[eTypeID]
            self.edges[eTypeID] = tmp
            self.edges[eTypeID].currentSize = int(newSize)
        else:
            print('error')#raise()
            

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

    def addEdge(self, eTypeID, sourceID, targetID, attributes = None):
        """ 
        Adding a new connecting edge between source and target of
        the specified type
        Attributes can be given optionally with the correct structured
        tuple
        """
        srcNodeTypeID, srcDataID = self.getNodeDataRef(sourceID)   
        trgNodeTypeID, trgDataID = self.getNodeDataRef(targetID)   

        if not (self.nodes[srcNodeTypeID]['active'][srcDataID]) or \
           not(self.nodes[trgNodeTypeID]['active'][trgDataID]):
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
            assert dataID < self.maxEdges
            if dataID >= self.edges[eTypeID].currentSize:
                 self._extendEdgeArray(eTypeID, core.config.EXTENTION_FACTOR)
                 eType = self.edges[eTypeID]
        
        dataview = eType[dataID:dataID+1].view()          
        #updating edge dictionaries
        eType.eDict[(sourceID, targetID)] = dataID
        eType.edgeList.append(leID)
        
        try:
            eType.edgesOut[sourceID].append(dataID)
            eType.nodesOut[sourceID][1].append(trgDataID)
        except:
            eType.edgesOut[sourceID] = [dataID]
            eType.nodesOut[sourceID] = trgNodeTypeID, [trgDataID]
        
        try:
            eType.edgesIn[targetID].append(dataID)
            eType.nodesIn[targetID][1].append(srcDataID)
        except:
            eType.edgesIn[targetID] = [dataID]     
            eType.nodesIn[targetID] = srcNodeTypeID, [srcDataID]
             
        if attributes is not None:
            dataview[:] = (True, sourceID, targetID) + attributes
        else:
            dataview[['active', 'source', 'target']] = (True, sourceID, targetID)
    

    def addEdges(self, eTypeID, sourceIDs, targetIDs, **kwAttr):
        """
        Method to create serveral edges at once
        Attribures are given as list or array per key word
        """
        if sourceIDs == []:
             raise(BaseException('Empty list given'))
        if not (self.areNodes(sourceIDs)) or not( self.areNodes(targetIDs)):
            raise(BaseException('Nodes do not exist'))
        
        nEdges = len(sourceIDs)
        
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
        
        assert max(dataIDs) < self.maxEdges
        
        if max(dataIDs) >= eType.currentSize:
            extFactor = int(max(np.ceil( (max(dataIDs)+1 )/eType.currentSize), core.config.EXTENTION_FACTOR))
            self._extendEdgeArray(eTypeID, extFactor)
            eType = self.edges[eTypeID]
            
        eType['source'][dataIDs] = sourceIDs
        eType['target'][dataIDs] = targetIDs    
        eType['active'][dataIDs] = True
              
        #updating edge dictionaries
        leIDs = dataIDs + eTypeID * self.maxEdges
        eType.edgeList.extend(leIDs.tolist())
        
        for source, target, dataID in zip(sourceIDs, targetIDs, dataIDs):
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

    def remOutgoingEdges(self, eTypeID, sourceID, targetIDs):
        eType = self.edges[eTypeID]
          
        eType.edgesOut[sourceID] = []
        targetIDs = eType.nodesOut.pop(sourceID)
        
        for targetID in targetIDs:
             dataID = eType.eDict.pop((sourceID, targetID))
             leID   = dataID + eTypeID * self.maxEdges
             eType.edgeList.remove(leID)
             eType[dataID:dataID+1]['active'] = False
             eType.edgesIn[targetID].remove(dataID)
             eType.nodesIn[targetID].remove(sourceID)
    
    def _changeSourceOfEdge(self, eTypeID, targetID, oldSourceID, newSourceID):
        """
        This function replaces the source for an existing edge and all related
        list entries
        """
        eType = self.edges[eTypeID]
        dataID = eType.eDict.pop((oldSourceID, targetID))
        eType.eDict[(newSourceID, targetID)] = dataID
        
        (_, oldSrcDataID) = self.getNodeDataRef(oldSourceID)
        (_, newSrcDataID) = self.getNodeDataRef(newSourceID)
        
        (trgNodeTypeID , trgDataID) = self.getNodeDataRef(targetID)        
        
        eType.edgesOut[oldSourceID].remove(dataID)
        eType.nodesOut[oldSourceID][1].remove(trgDataID)
        try:
            eType.edgesOut[newSourceID].append(dataID)
            eType.nodesOut[newSourceID][1].append(trgDataID)
        except:
            eType.edgesOut[newSourceID] = [dataID]
            eType.nodesOut[newSourceID] = trgNodeTypeID, [trgDataID]
                
        
        idx = misc.listFind(oldSrcDataID, eType.nodesIn[targetID][1])
        eType.nodesIn[targetID][1][idx] = newSrcDataID
    
    def _changeTargetOfEdge(self, eTypeID, sourceID, oldTargetID, newTargetID):
        """
        This function replaces the target for an existing edge and all related
        list entries
        """
        eType = self.edges[eTypeID]
        dataID = eType.eDict.pop((sourceID, oldTargetID))
        eType.eDict[(source, newTargetID)] = dataID
        
        (srcNodeTypeID, srcDataID) = self.getNodeDataRef(sourceID)
    
        (_, oldTrgDataID) = self.getNodeDataRef(oldTargetID)        
        (_, newTrgDataID) = self.getNodeDataRef(newTargetID)        
        
        eType.edgesIn[oldTargetID].remove(dataID)
        eType.nodesIn[oldTargetID][1].remove(srcDataID)
        try:
            eType.edgesIn[newTargetID].append(dataID)
            eType.nodesIn[newTargetID][1].append(srcDataID)
        except:
            eType.edgesIn[newTargetID] = [dataID]     
            eType.nodesIn[newTargetID] = srcNodeTypeID, [srcDataID]
                
        
        idx = misc.listFind(oldTrgDataID, eType.nodesOut[sourceID][1])
        eType.nodesOut[sourceID][1][idx] = newTrgDataID

        
    def remEdge(self, eTypeID, sourceID, targetID):
        eType = self.edges[eTypeID]
        
        dataID = eType.eDict.pop((sourceID, targetID))
        leID   = dataID + eTypeID * self.maxEdges 
        eType.freeRows.append(dataID)
        eType[dataID:dataID+1]['active'] = False
        eType.edgeList.remove(leID)
        
        (_, srcDataID) = self.getNodeDataRef(sourceID)
        (_, trgDataID) = self.getNodeDataRef(targetID)
            
        
        eType.edgesIn[targetID].remove(dataID)
        eType.edgesOut[sourceID].remove(dataID)
        
        eType.nodesOut[sourceID][1].remove(trgDataID)
        eType.nodesIn[targetID][1].remove(srcDataID)


    def remEdges(self, eTypeID, sourceIDs, targetIDs):
        for sourceID, targetID in zip(sourceIDs, targetIDs):
            self.remEdge(eTypeID, sourceID, targetID)
            
    def setEdgeAttr(self, leID, attribute, value, eTypeID=None):
        
        eTypeID, dataID = self.getEdgeDataRef(leID)
        
        self.edges[eTypeID][attribute][dataID] = value

    def getAttrOfEdgesByDataID(self, attribute, eTypeID, dataIDs):
        if attribute:
            return self.edges[eTypeID][attribute][dataIDs]
        else:
            return self.edges[eTypeID][dataIDs]
    
    
    def getEdgeAttr(self, leID, attribute=None, eTypeID=None):

        eTypeID, dataID = self.getEdgeDataRef(leID)
            
        if attribute:
            return self.edges[eTypeID][dataID][attribute]
        else:
            return self.edges[eTypeID][dataID]
        

    def setEdgeSeqAttr(self, attribute, values, leIDs=None, eTypeID=None, dataIDs=None):
        
        if dataIDs is None:
            eTypeID, dataIDs = self.getEdgeDataRef(leIDs)
        else:
            assert leIDs is None    
        #print eTypeID, dataIDs
        self.edges[eTypeID][attribute][dataIDs] = values
    
    def getEdgeSeqAttr(self, attribute=None, leIDs=None, eTypeID=None, dataIDs=None):
        
        if dataIDs is None:
            assert eTypeID is None
            eTypeID, dataIDs = self.getEdgeDataRef(leIDs)
        
        if attribute:
            return self.edges[eTypeID][attribute][dataIDs]
        else:
            return self.edges[eTypeID][dataIDs]
        
    #%% General
    def areConnected(self, sourceID, targetID, eTypeID):
        """ 
        Returns if source and target is connected by an eddge of the specified
        edge type
        """
        return (sourceID, targetID) in self.edges[eTypeID].eDict

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
    
    def countIncomming(self, lnID, eTypeID):
        try:
            nTypeID, dataIDs = self.edges[eTypeID].nodesIn[lnID]
            return len(dataIDs)
        except:
            return 0
    
    def countOutgoing(self, lnID, eTypeID):
        try:
            nTypeID, dataIDs = self.edges[eTypeID].nodesOut[lnID]
            return len(dataIDs)
        except:
            return 0
        
        
        
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
        This function is testing the base._graph class
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
    ABMGraph is a numpy based graph library ot handle a directed multi-graph
    with different storage layouts for edges.
    
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
        #self.defaultNodeValues     = (False, None, -2, -2)
        self.persNodeAttr = [('active', np.bool_,1),
                             ('instance', np.object,1),
                             ('gID', core.config.GID_TYPE,1),
                             ('ID', core.config.ID_TYPE,1)]


        
        
        #persistent edge attributes
        self.persEdgeAttr = [('active', np.bool_, 1),
                             ('source', core.config.ID_TYPE, 1), 
                             ('target', core.config.ID_TYPE, 1)]
    
    def _addNoneEdge(self, source):
        """
        Method to create the empty nodesOut dict and returns a pointer
        to it. Used by the GridNode trait (see traits.py)
        """
        nTypeID, _ = self.getNodeDataRef(source)
        linkTypeID = self.node2EdgeType[(nTypeID,nTypeID)]
        eType = self.edges[linkTypeID]
        eType.edgesOut[source] = []
        eType.nodesOut[source] = nTypeID, []

        return  eType.nodesOut[source]
    
    def addNodeType(self, 
                    agTypeIDIdx, 
                    typeStr, 
                    AgentClass, 
                    GhostAgentClass, 
                    staticProperties, 
                    dynamicProperties):
        """ Create node type description"""

        # adds and formats properties we need for the framework (like gID) automatically
        staticProperties  = formatPropertyDefinition(staticProperties)
        dynamicProperties = formatPropertyDefinition(dynamicProperties)
        
        
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

    def setNodeStorageSize(self, nTypeID, newSize):
        self._extendNodeArray(self, nTypeID, newSize=newSize)

    def addEdgeType(self, typeStr, staticProperties, dynamicProperties, agTypeID1, agTypeID2):
        """ Create edge type description"""
        # adds and formats properties we need for the framework (like gID) automatically
        staticProperties  = formatPropertyDefinition(staticProperties)
        dynamicProperties = formatPropertyDefinition(dynamicProperties)

        liTypeIDIdx = len(self.liTypeByID)+1
        liTypeID = TypeDescription(liTypeIDIdx, typeStr, staticProperties, dynamicProperties)
        self.liTypeByID[liTypeIDIdx] = liTypeID
        self.node2EdgeType[agTypeID1, agTypeID2] = liTypeIDIdx
        self.edge2NodeType[liTypeIDIdx] = agTypeID1, agTypeID2
        self._initEdgeType(typeStr, staticProperties + dynamicProperties)

        return liTypeIDIdx

    def setEdgeStorageSize(self, eTypeID, newSize):
        self._extendEdgeArray(self, eTypeID, newSize=newSize)

    #%% RELATIVE NODE ACCESS
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
            

    def getDTypeOfEdgeType(self, liTypeID, kind):
        
        if kind == 'sta':
            dtype = self.liTypeByID[liTypeID].staProp
        elif kind == 'dyn':
            dtype = self.liTypeByID[liTypeID].dynProp
        else:
            dtype = self.liTypeByID[liTypeID].staProp + self.agTypeByID[liTypeID].dynProp
        return dtype
    
    def getPropOfEdgeType(self, liTypeID, kind):
        
        dtype = self.getDTypeOfEdgeType(liTypeID, kind)
         
        info = dict()    
        info['names'] = []
        info['types'] = []
        info['sizes'] = []
        for it in dtype:
            info['names'].append(it[0])
            info['types'].append(it[1])
            info['sizes'].append(it[2])
        return info


    def getOutNodeValues(self, lnID, eTypeID, attr=None):
        """
        Method to read the attributes of connected nodes via a 
        specifice edge type and outward direction
        """
        try: 
            nTypeID, dataIDs = self.edges[eTypeID].nodesOut[lnID]
        except KeyError:
            # return empty list if no edges exist
            return []
        
        if attr:
            return self.nodes[nTypeID][dataIDs][attr]
        else:
            return self.nodes[nTypeID][dataIDs]
        
    
    def setOutNodeValues(self, lnID, eTypeID, attr, values):
        """
        Method to read the attributes of connected nodes via a 
        specifice edge type and outward direction
        """
        try: 
            nTypeID, dataIDs = self.edges[eTypeID].nodesOut[lnID]
        except KeyError:
            # return empty list if no edges exist
            return []
        
        self.nodes[nTypeID][attr][dataIDs] = values
        

    def getInNodeValues(self, lnID, eTypeID, attr=None):
        """
        Method to read the attributes of connected nodes via a 
        specifice edge type and inward direction
        """
        try: 
            nTypeID, dataIDs = self.edges[eTypeID].nodesIn[lnID]
        except KeyError:
            # return empty list if no edges exist
            return []
        
        if attr:
            return self.nodes[nTypeID][dataIDs][attr]
        else:
            return self.nodes[nTypeID][dataIDs]

    def setInNodeValues(self, lnID, eTypeID, attr, values):
        """
        Method to read the attributes of connected nodes via a 
        specifice edge type and inward direction
        """
        try: 
            nTypeID, dataIDs = self.edges[eTypeID].nodesIn[lnID]
        except KeyError:
            # return empty list if no edges exist
            return []
        
        self.edges[eTypeID][attr][dataIDs] = values


    #%%  EDGE ACCESS
    
    def getOutEdgeValues(self, leID, eTypeID, attr=None):
        """
        Method to read the attributes of connected nodes via a 
        specifice edge type and outward direction
        """
        try: 
            dataIDs = self.edges[eTypeID].edgesOut[leID]
        except KeyError:
            # return empty list if no edges exist
            return []
        
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
                dataIDSource = source - self.maxNodes
                targetList = [target for target in eDict[source][1] if target != dataIDSource]
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
