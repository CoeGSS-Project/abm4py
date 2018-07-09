#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""
from .graph import ABMGraph
from . import core

import logging as lg
import numpy as np

class World:

    #%% INIT WORLD
    def __init__(self,
                 simNo=None,
                 outPath='.',
                 spatial=True,
                 nSteps=1,
                 maxNodes=1e3,
                 maxLinks=1e5,
                 debug=False,
                 mpiComm=None,
                 agentOutput=False):

        if mpiComm is None:
            self.isParallel = False
            self.isRoot     = True
        else:
            self.isParallel = mpiComm.size > 1
            self.mpiRank = core.mpiRank
            self.mpiSize = core.mpiSize
            if self.isParallel:
                if self.mpiRank == 0:
                    self.isRoot = True
                else:
                    self.isRoot = False
            else:
                self.isRoot = False
                
        # ======== GRAPH ========
        self.graph    = ABMGraph(self, maxNodes, maxLinks)
        
        # determines if the frameworks runs in parallel or not
        
        
        # agent passing interface for communication between parallel processes
        self.papi = core.PAPI(self)
        self.__glob2loc = dict()  # reference from global IDs to local IDs
        self.__loc2glob = dict()  # reference from local IDs to global IDs

        # generator for IDs that are globally unique over all processe
        self.globIDGen = self._globIDGen()
        
        lg.debug('Init MPI done')##OPTPRODUCTION
                
                
          
        self.para      = dict()
        self.para['outPath'] = outPath # is not graph, move to para
        # TODO put to parameters                    
        self.agentOutput = agentOutput
        self.simNo     = simNo
        self.timeStep  = 0    
        self.maxNodes  = int(maxNodes)
        self.nSteps    = nSteps
        self.debug     = debug

        # enumerations
        self.__enums = dict()

        # node lists and dicts
        self.__nodeDict       = dict()
        self.__ghostNodeDict  = dict()
        
        # dict of list that provides the storage place for each agent per nodeTypeID
        self.__dataDict       = dict()
        self.__ghostDataDict  = dict()

        self.__entList   = list()
        self.__entDict   = dict()
        self.__locDict   = dict()

        
        
        
        
        # ======== GLOBALS ========
        # storage of global data
        self.globalRecord = dict() 
        # Globally synced variables
        self.graph.glob     = core.Globals(self)
        lg.debug('Init Globals done')##OPTPRODUCTION
        
        # ======== IO ========
        if self.agentOutput:
            self.io = core.IO(self, nSteps, self.para['outPath'])
            lg.debug('Init IO done')##OPTPRODUCTION
        
        # ======== SPATIAL LAYER ========
        if spatial:
            self.spatial  = core.Spatial(self)
        
        # ======== RANDOM ========
        self.random = core.Random(self, self.__nodeDict, self.__ghostNodeDict, self.__entDict)
        
        self.addLink        = self.graph.addLink
        self.addLinks       = self.graph.addLinks
        self.delLinks       = self.graph.remEdge

        self.addNode     = self.graph.addNode
        self.addNodes    = self.graph.addNodes

        

        


        # inactive is used to virtually remove nodes
        
        class GetEntityBy():
            def __init__ (self, world):
                self.locID = world._entByLocID
                self.globID = world._entByGlobID
                self.location = world.spatial.getLocation
        self.getNodeBy = GetEntityBy(self)

    def _globIDGen(self):
        i = -1
        while i < self.maxNodes:
            i += 1
            yield (self.maxNodes*(core.mpiRank+1)) +i

# GENERAL FUNCTIONS

    def _entByGlobID(self, globID):
        return self.__entDict[self.__glob2loc[globID]]
        
    def _entByLocID(self, locID):
        return self.__entDict[locID]

    def nNodes(self, nodeTypeID=None):
        return self.graph.nCount(nTypeID=nodeTypeID)
        
    def nLinks(self, linkTypeID):
       return self.graph.eCount(eTypeID=linkTypeID)  
        
    def getDataIDs(self, nodeTypeID):
        return self.__dataDict[nodeTypeID]

    def glob2Loc(self, globIdx):
        return self.__glob2loc[globIdx]

    def setGlob2Loc(self, globIdx, locIdx):
        self.__glob2loc[globIdx] = locIdx

    def loc2Glob(self, locIdx):
        return self.__loc2glob[locIdx]

    def setLoc2Glob(self, globIdx, locIdx):
        self.__loc2glob[locIdx] = globIdx 

    def getLocationDict(self):
        """
        The locationDict contains all instances of locations that are
        accessed by (x,y) coordinates
        """
        return self.__locDict

    def getNodeDict(self, nodeTypeID):
        """
        The nodeDict contains all instances of different entity types
        """
        return self.__nodeDict[nodeTypeID]

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


    def getNodeAttr(self, label, localNodeIDList=None, nodeTypeID=None):
        """
        Method to read values of node sequences at once
        Return type is numpy array
        Only return non-ghost agent properties
        """
        if localNodeIDList:   
            assert nodeTypeID is None # avoid wrong usage ##OPTPRODUCTION
            return self.graph.getNodeSeqAttr(label, lnIDs=localNodeIDList)
        
        elif nodeTypeID:           
            assert localNodeIDList is None # avoid wrong usage ##OPTPRODUCTION
            return self.graph.getNodeSeqAttr(label, lnIDs=self.__nodeDict[nodeTypeID])
        
    def setNodeAttr(self, label, valueList, localNodeIDList=None, nodeTypeID=None):
        """
        Method to read values of node sequences at once
        Return type is numpy array
        """
        if localNodeIDList:
            assert nodeTypeID is None # avoid wrong usage ##OPTPRODUCTION
            self.graph.setNodeSeqAttr(label, valueList, lnIDs=localNodeIDList)
        
        elif nodeTypeID:
            assert localNodeIDList is None # avoid wrong usage ##OPTPRODUCTION
            self.graph.setNodeSeqAttr(label, valueList, lnIDs=self.__nodeDict[nodeTypeID])

    def getNodeIDs(self, nodeTypeID):
        """ 
        Method to return all local node IDs for a given nodeType
        """
        return self.__nodeDict[nodeTypeID]

    def getLinkAttr(self, label, valueList, localLinkIDList=None, linkTypeID=None):
        if localLinkIDList:   
            assert linkTypeID is None # avoid wrong usage ##OPTPRODUCTION
            return self.graph.getNodeSeqAttr(label, lnIDs=localLinkIDList)
        
        elif linkTypeID:           
            assert localLinkIDList is None # avoid wrong usage ##OPTPRODUCTION
            return self.graph.getNodeSeqAttr(label, lnIDs=self.linkDict[linkTypeID])

    def setLinkAttr(self, label, valueList, localLinkIDList=None, linkTypeID=None):
        """
        Method to retrieve all properties of all entities of one linkTypeID
        """
        if localLinkIDList:
            assert linkTypeID is None # avoid wrong usage ##OPTPRODUCTION
            self.graph.setEdgeSeqAttr(label, valueList, lnIDs=localLinkIDList)
        
        elif linkTypeID:
            assert localLinkIDList is None # avoid wrong usage ##OPTPRODUCTION
            self.graph.setEdgeSeqAttr(label, valueList, lnIDs=self.linkDict[linkTypeID])
    
  
    def getNode(self, nodeID=None, globID=None, nodeTypeID=None, ghosts=False):
        """
        Method to retrieve a certain instance of an entity by the nodeID
        Selections can be done by the local nodeID, global ID and the nodetype
        and the flag ghost

        """
        if nodeID is not None:
            return self.__entDict[nodeID]
        elif globID is not None:
            return self.__entDict[self.__glob2loc[globID]]
        elif nodeTypeID is not None:
            if ghosts:
                return  [self.__entDict[agentID] for agentID in self.__ghostNodeDict[nodeTypeID]]
            else:
                return [self.__entDict[agentID] for agentID in self.__nodeDict[nodeTypeID]]

    def filterNodes(self, nodeTypeID, attr, operator, value = None, compareAttr=None):
        """
        Method for quick filtering nodes according to comparison of attributes
        allowed operators are:
            "lt" :'less than <
            "elt" :'less or equal than <=
            "gt" : 'greater than >
            "egt" : 'greater or equal than >=
            "eq" : 'equal ==
        Comparison can be made to values or another attribute
        """
        
        # get comparison value
        if compareAttr is None:
            compareValue = value
        elif value is None:
            compareValue = self.graph.getNodeSeqAttr(compareAttr, lnIDs=self.__nodeDict[nodeTypeID])
        
        # get values of all nodes
        values = self.graph.getNodeSeqAttr(attr, lnIDs=self.__nodeDict[nodeTypeID])
        
        if operator=='lt':
            boolArr = values < compareValue    
            
        elif operator=='gt':
            boolArr = values > compareValue    
            
        elif operator=='eq':
            boolArr = values == compareValue    
            
        elif operator=='elt':
            boolArr = values <= compareValue    
            
        elif operator=='egt':
            boolArr = values >= compareValue    
            
        lnIDs = np.where(boolArr)[0] + (self.maxNodes * nodeTypeID)
        
        return lnIDs
        

    def iterNodes(self, nodeTypeID=None, localIDs=None, ghosts = False):
        """
        Iteration over entities of specified type. Default returns
        non-ghosts in random order.
        """
        if nodeTypeID is None:
            nodeList = localIDs
        elif localIDs is None:
            if isinstance(nodeTypeID,str):
                nodeTypeID = self.types.index(nodeTypeID)
    
            if ghosts:
                nodeList = self.__ghostNodeDict[nodeTypeID]
            else:
                nodeList = self.__nodeDict[nodeTypeID]

        return  [self.__entDict[i] for i in nodeList]


    def setEnum(self, enumName, enumDict):
        """
        Method to add enumertations
        Dict should have integers as keys an strings as values
        """
        self.__enums[enumName] = enumDict
         
    def getEnum(self, enumName=None):
        """ 
        Returns a specified enumeration dict
        """         
        if enumName is None:
            return self.__enums.keys()
        else:
            return self.__enums[enumName]

    def registerNodeType(self, typeStr, AgentClass, GhostAgentClass=None, staticProperties = [], dynamicProperties = []):
        """
        Method to register a node type:
        - Registers the properties of each nodeTypeID for other purposes, e.g. I/O
        of these properties
        - update of convertions dicts:
            - class2NodeType
            - nodeTypeID2Class
        - creations of access dictionaries
            - nodeDict
            - ghostNodeDict
        - enumerations
        """
        
        # type is an required property
        #assert 'type' and 'gID' in staticProperties              ##OPTPRODUCTION
        
        staticProperties = core.formatPropertyDefinition(staticProperties)
        dynamicProperties = core.formatPropertyDefinition(dynamicProperties)
                
        nodeTypeIDIdx = len(self.graph.nodeTypeIDs)+1

        self.graph.addNodeType(nodeTypeIDIdx, 
                               typeStr, 
                               AgentClass,
                               GhostAgentClass,
                               staticProperties, 
                               dynamicProperties)
        self.__nodeDict[nodeTypeIDIdx]      = list()
        self.__dataDict[nodeTypeIDIdx]      = list()
        self.__ghostNodeDict[nodeTypeIDIdx] = list()
        self.__ghostDataDict[nodeTypeIDIdx] = list()

        return nodeTypeIDIdx


    def registerLinkType(self, typeStr,  nodeTypeID1, nodeTypeID2, staticProperties = [], dynamicProperties=[]):
        """
        Method to register a edge type:
        - Registers the properties of each linkTypeID for other purposes, e.g. I/O
        of these properties
        - update of convertions dicts:
            - node2EdgeType
            - edge2NodeType
        - update of enumerations
        """
        staticProperties  = core.formatPropertyDefinition(staticProperties)
        dynamicProperties = core.formatPropertyDefinition(dynamicProperties)        
        #assert 'type' in staticProperties # type is an required property             ##OPTPRODUCTION

        
        linkTypeIDIdx = self.graph.addLinkType( typeStr, 
                                               staticProperties, 
                                               dynamicProperties, 
                                               nodeTypeID1, 
                                               nodeTypeID2)
        

        return  linkTypeIDIdx

    def registerNode(self, agent, typ, ghost=False): #TODO rename agent to entity?
        """
        Method to register instances of nodes
        -> update of:
            - entList
            - endDict
            - __glob2loc
            - _loc2glob
        """
        #print 'assert' + str((len(self.__entList), agent.nID))
        #assert len(self.__entList) == agent.nID                                  ##OPTPRODUCTION
        self.__entList.append(agent)
        self.__entDict[agent.nID] = agent
        
        if self.isParallel:
            self.__glob2loc[agent.gID] = agent.nID
            self.__loc2glob[agent.nID] = agent.gID

        if ghost:
            self.__ghostNodeDict[typ].append(agent.nID)
            self.__ghostDataDict[typ].append(agent.dataID)
        
        else:
            self.__nodeDict[typ].append(agent.nID)
            self.__dataDict[typ].append(agent.dataID)

    def deRegisterNode(self, agent, ghost):
        """
        Method to remove instances of nodes
        -> update of:
            - entList
            - endDict
            - __glob2loc
            - _loc2glob
        """
        self.__entList.remove(agent)
        del self.__entDict[agent.nID]
        if self.isParallel:
            del self.__glob2loc[agent.gID]
            del self.__loc2glob[agent.nID]
        nodeTypeID =  self.graph.class2NodeType(agent.__class__)
        if ghost:
            self.__ghostNodeDict[nodeTypeID].remove(agent.nID)
            self.__ghostDataDict[nodeTypeID].remove(agent.dataID)
        else:
            self.__nodeDict[nodeTypeID].remove(agent.nID)
            self.__dataDict[nodeTypeID].remove(agent.dataID)
            #print(self.__nodeDict[nodeTypeID])
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

        self.__locDict[x,y] = location


    def returnApiComm(self):
        return self.papi.comm

    def returnGraph(self):
        return self.graph

    def returnGlobalRecord(self):
        return self.globalRecord

    def returnGlobals(self):
        return self.graph.glob
    
    def returnNodeDict(self,):
        return self.__nodeDict
    
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
            core.saveObj(self.__enums, self.para['outPath'] + '/enumerations')


            # saving enumerations
            core.saveObj(self.para, self.para['outPath'] + '/simulation_parameters')

            if self.para['showFigures']:
                # plotting and saving figures
                for key in self.globalRecord:
                    self.globalRecord[key].plot(self.para['outPath'])