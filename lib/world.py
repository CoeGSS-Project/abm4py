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
import types

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
                self.isRoot = True
        
        self.isSpatial = spatial
                
        # ======== GRAPH ========
        self.graph    = ABMGraph(self, maxNodes, maxLinks)
        
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

        # agent ID lists and dicts
        self.__agentIDsByType  = dict()
        self.__ghostIDsByType  = dict()
        
        # dict of lists that provides the storage place for each agent per agTypeID
        self.__agendDataIDsList    = dict()
        self.__ghostDataIDsByType  = dict()

        # dict and list of agent instances
        self.__agentsByType  = dict()
        self.__ghostsByType  = dict()
        
        self.__allAgentDict    = dict()
        self.__locDict   = dict()

        
        self.__nAgentsByType = dict()
        self.__nGhostsByType = dict()
        
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
        if self.isSpatial:
            self.spatial  = core.Spatial(self)
        
        # ======== RANDOM ========
        self.random = core.Random(self, self.__agentIDsByType, self.__ghostIDsByType, self.__agentsByType)
        
        
        # re-direct of graph functionality 
        self.addLink        = self.graph.addEdge
        self.addLinks       = self.graph.addEdges
        self.delLinks       = self.graph.remEdge

        self.addNode     = self.graph.addNode
        self.addNodes    = self.graph.addNodes
      
        self.getAgents  = core.AgentAccess(self)

        if core.dakota.isActive:
            self.simNo = core.dakota.simNo
            self.setParameters(core.dakota.params)


    def _globIDGen(self):
        i = -1
        while i < self.maxNodes:
            i += 1
            yield (self.maxNodes*(core.mpiRank+1)) +i

# GENERAL FUNCTIONS


    def _entByGlobID(self, globID):
        return self.__allAgentDict[self.__glob2loc[globID]]
        
    def _entByLocID(self, locID):
        return self.__allAgentDict[locID]
    
#%% General Infromation methods
    def nAgents(self, agTypeID, ghosts=False):
        if ghosts:
            return self.__nGhostsByType[agTypeID]
        else:
            return self.__nAgentsByType[agTypeID]
        
    def nLinks(self, liTypeID):
       return self.graph.eCount(eTypeID=liTypeID)  

    def getParameter(self, paraName):
        """
        Returns a single simulation parameter
        """
        return self.para[paraName]

    def getParameters(self):
        """
        Returns a dictionary of all simulations parameters 
        """
        return self.para

    def setParameter(self, paraName, paraValue):
        """
        This method is used to set parameters of the simulation
        """
        if core.dakota.isActive:
            if core.dakota.params.get(paraName):
                self.para[paraName] = core.dakota.params[paraName]
        else:
            self.para[paraName] = paraValue

    def setParameters(self, parameterDict):
        """
        This method allows to set multiple parameters at once
        """
        for key in list(parameterDict.keys()):
            self.setParameter(key, parameterDict[key])

    # saving enumerations
    def saveParameters(self, fileName= 'simulation_parameters'):
        pass
        # core.saveObj(self.para, self.para['outPath'] + '/' + fileName)
       
    def getEnums(self):
        """ 
        Returns a specified enumeration dict
        """         
        return self.__enums

    def saveEnumerations(self, fileName= 'enumerations'):
        # saving enumerations
        #core.saveObj(self.__enums, self.para['outPath'] + '/' + fileName)
        pass

#%% Global Agent scope
        

    def getAttrOfAgents(self, attribute, localIDList):
        return self.graph.getNodeSeqAttr(attribute, lnIDs=localIDList)


    def getAttrOfAgentType(self, attribute, agTypeID):
        """
        Method to read attributes of node sequences at once
        gitgeturn type is numpy array
        Only return non-ghost agent properties
        """
        return self.graph.getNodeSeqAttr(attribute, lnIDs=self.__agentIDsByType[agTypeID])
    
    def getAttrOfFilteredAgents(self, attribute, agTypeID, func):
        """
        This function allows to access the attributes of a sub-selection of agents 
        that is defined  by a filter function that is action on agent properties.

        Use case: Iterate over agents with a property below a certain treshold:
        for agent in world.filterAgents(AGENT, lambda a: a['age'] < 1)
        """
        array = self.graph.nodes[agTypeID]
        mask = array['active'] & func(array)
        #maskedArray = array[mask]
        return array[attribute][mask]
    
    def setAttrOfAgents(self, attribute, valueList, localIDList):

        """
        Method to write values of node sequences at once
        Return type is numpy array
        """
        self.graph.setNodeSeqAttr(attribute, valueList, lnIDs=localIDList)

    def setAttrOfAgentType(self, attribute, valueList, agTypeID):
        """
        Method to write values of all agents of a type at once
        Return type is numpy array
        """
        self.graph.setNodeSeqAttr(attribute, valueList, lnIDs=self.__agentIDsByType[agTypeID])           
            
    def getAttrOfLinks(self, attribute, localLinkIDList):
        """
        Method to retrieve all properties of all entities in the localLinkIDList
        """
        return self.graph.getEdgeSeqAttr(attribute, lnIDs=localLinkIDList)
        
    def getAttrOfLinkType(self, attribute, liTypeID):
        """
        Method to retrieve all properties of all entities of one liTypeID
        """
        return self.graph.getEdgeSeqAttr(attribute, lnIDs=self.linkDict[liTypeID])
        
    def setAttrOfLinks(self, attribute, valueList, localLinkIDList):
        """
        Method to write values of a sequence of links at once
        Return type is numpy array
        """
        self.graph.setEdgeSeqAttr(attribute, valueList, lnIDs=localLinkIDList)
        
    def setAttrOfLinkType(self, attribute, valueList, liTypeID):
        """
        Method to write values of all links of a type at once
        Return type is numpy array
        """
        self.graph.setEdgeSeqAttr(attribute, valueList, lnIDs=self.linkDict[liTypeID])            


#%% Agent access 

    def getAgentIDs(self, agTypeID, ghosts=False):
        """ 
        Method to return all local node IDs for a given nodeType
        """
        if ghosts:
            return self.__ghostIDsByType[agTypeID]
        else:
            return self.__agentIDsByType[agTypeID]
    

    def getAgent(self, agentID):    
        return self.__allAgentDict[agentID]

    

    def filterAgents(self, func, agTypeID):
        """
        This function allows to access a sub-selection of agents that is defined 
        by a filter function that is action on agent properties.

        Use case: Iterate over agents with a property below a certain treshold:
        for agent in world.filterAgents(lambda a: a['age'] < 1, AGENT)
        """
        array = self.graph.nodes[agTypeID]
        mask = array['active'] & func(array)
        return array['instance'][mask]

    def countAgents(self, func, agTypeID):
        """
        This function allows to access a sub-selection of agents that is defined 
        by a filter function that is action on agent properties.

        Use case: Iterate over agents with a property below a certain treshold:
        for agent in world.filterAgents(lambda a: a['age'] < 1, AGENT)
        """
        array = self.graph.nodes[agTypeID]
        mask = array['active'] & func(array)
        return np.sum(mask)

    
    def getAttrOfFilteredAgentType(self, attribute, func, agTypeID):
        """
        This function allows to access a sub-selection of agents that is defined 
        by a filter function that is action on agent properties.

        Use case: Get the pos of all agents with a property below a certain treshold:
        for agent in world.filterAgents('pos', lambda a: a['age'] < 1, AGENT)
        """
        array = self.graph.nodes[agTypeID]
        mask = array['active'] & func(array)
        return array[attribute][mask]

    def setAttrsForTypeVectorized(self, agTypeID, attribute, vfunc):
        array = self.graph.nodes[agTypeID]
        array['active'][attribute] = vfunc(array[array['active']])    
    
    def setAttrsForType(self, agTypeID, func):
        """
        This function allows to manipulate the underlaying np.array directly, e.g.
        to change the attributes of all agents of a given type in a more performant way
        then with a python loop for comprehension.

        func is a function that gets an (structured) np.array, and must return an array with the 
        dimensions.

        Use case: Increase the age for all agents by one:
        def increaseAge(agents):
            agents['age'] += 1
            return agents

        setAttrsForType(AGENT, increaseAge)

        see also: setAttrsForFilteredType/Array
        """    
        array = self.graph.nodes[agTypeID]
        array['active'] = func(array['active'])

    def setAttrsForFilteredType(self, agTypeID, filterFunc, setFunc):
        """
        This function allows to manipulate a subset of the underlaying np.array directly. The 
        subset is formed by the filterFunc.

        filterFunc is a function that gets an (structed) np.array and must return an boolean array with
        the same length. For performance reasons, the function gets the complete underlaying array, 
        which can contain unused rows (the 'active' column is set to False for those rows).

        setFunc is a function that gets an array that contains the rows where the boolean array returned
        by the filterFunc is True, and must return an array with the same dimensions. 

        Use case: All agents without food are dying:

        def dieAgentDie(deadAgents):
          deadAgents['alive'] = False
          return deadAgents

        setAttrsForFilteredType(AGENT, lambda a: a['food'] <= 0, dieAgentDie)

        see also: setAttrForType and setAttrsForFilteredArray
        """    
        array = self.graph.nodes[agTypeID]
        mask = array['active'] & filterFunc(array) 
        array[mask] = setFunc(array[mask])

    def setAttrsForFilteredArray(self, array, filterFunc, setFunc):
        """
        This function allows to manipulate a subset of a given np.array directly. The 
        subset is formed by the filterFunc. That the array is given to the function explicitly 
        allows to write nested filters, but for most use cases setFilteredAttrsForType should
        be good enough.

        filterFunc is a function that gets an (structed) np.array and must return an boolean array with
        the same length.

        setFunc is a function that gets an array that contains the rows where the boolean array returned
        by the filterFunc is True, and must return an array with the same dimensions. 

        Use case: All agents without food are dying, and only agents with a good karma goes to heaven:

        def goToHeaven(angels):
            angels['heaven'] = True
            return angels

        def heavenOrHell(deadAgents):
            setAttrsForFilteredArray(deadAgents, lambda a: a['karma'] > 10 , goToHeaven)
            deadAgents['alive'] = False
            return deadAgents

        setAttrsForFilteredType(AGENT, lambda a: a['food'] <= 0, heavenOrHell)
        
        see also: setAttrForType and setAttrsForFilteredType
        """    
        mask = array['active'] & filterFunc(array) 
        array[mask] = setFunc(array[mask])


    def deleteAgentsIf(self, agTypeID, filterFunc):
        array = self.graph.nodes[agTypeID]
        filtered = array[array['active'] == True & filterFunc(array)]
        if len(filtered) > 0:
            for aID in np.nditer(filtered['gID']):
                agent = self.getNode(globID = int(aID))
                agent.delete(self)

    #%% Local and global IDs            
    def getDataIDs(self, agTypeID):
        return self.__agendDataIDsList[agTypeID]

    def glob2Loc(self, globIdx):
        return self.__glob2loc[globIdx]

    def setGlob2Loc(self, globIdx, locIdx):
        self.__glob2loc[globIdx] = locIdx

    def loc2Glob(self, locIdx):
        return self.__loc2glob[locIdx]

    def setLoc2Glob(self, globIdx, locIdx):
        self.__loc2glob[locIdx] = globIdx 




#%% Register methods
    def __addIterNodeFunction(self, agTypeStr, nodeTypeId):
        name = "iter" + agTypeStr.capitalize()
        source = """def %NAME%(self):
                        return [ self._World__allAgentDict[i] for i in self._World__agentIDsByType[%NODETYPEID%] ]
        """.replace("%NAME%", name).replace("%NODETYPEID%", str(nodeTypeId))
        exec(compile(source, "", "exec"))
        setattr(self, name, types.MethodType(locals()[name], self))

        
    def registerAgentType(self, agTypeStr, AgentClass, GhostAgentClass=None, staticProperties = [], dynamicProperties = []):
        """
        Method to register a node type:
        - Registers the properties of each agTypeID for other purposes, e.g. I/O
        of these properties
        - update of convertions dicts:
            - class2NodeType
            - agTypeID2Class
        - creations of access dictionaries
            - nodeDict
            - ghostNodeDict
        - enumerations
        """
        
        # type is an required property
        #assert 'type' and 'gID' in staticProperties              ##OPTPRODUCTION
        
        # add properties we need for the framework (like gID) automatically (stf)
        staticProperties = core.formatPropertyDefinition(staticProperties)
        dynamicProperties = core.formatPropertyDefinition(dynamicProperties)
                
        agTypeIDIdx = len(self.graph.agTypeByID)+1

#        if self.isParallel:
#            globalIDset = False
#            for item in staticProperties:
#                if item[0] == 'gID':
#                    globalIDset = True
#            if not globalIDset:
#                staticProperties = [('gID', np.int32, 1)] + staticProperties
#            print(staticProperties)
                
        self.graph.addNodeType(agTypeIDIdx, 
                               agTypeStr, 
                               AgentClass,
                               GhostAgentClass,
                               staticProperties, 
                               dynamicProperties)

        self.__addIterNodeFunction(agTypeStr, agTypeIDIdx)
        
        
        # agent instance lists
        self.__agentsByType[agTypeIDIdx]   = list()
        self.__ghostsByType[agTypeIDIdx]   = list()
        
        # agent ID lists
        self.__agentIDsByType[agTypeIDIdx]  = list()
        self.__ghostIDsByType[agTypeIDIdx]  = list()
        
        # data idx lists per type       
        self.__agendDataIDsList[agTypeIDIdx]    = list()
        self.__ghostDataIDsByType[agTypeIDIdx]  = list()

        # number of agens per type
        self.__nAgentsByType[agTypeIDIdx] = 0
        self.__nGhostsByType[agTypeIDIdx] = 0

        return agTypeIDIdx


    def registerLinkType(self, agTypeStr,  agTypeID1, agTypeID2, staticProperties = [], dynamicProperties=[]):
        """
        Method to register a edge type:
        - Registers the properties of each liTypeID for other purposes, e.g. I/O
        of these properties
        - update of convertions dicts:
            - node2EdgeType
            - edge2NodeType
        - update of enumerations
        """
        staticProperties  = core.formatPropertyDefinition(staticProperties)
        dynamicProperties = core.formatPropertyDefinition(dynamicProperties)        
        #assert 'type' in staticProperties # type is an required property             ##OPTPRODUCTION

        
        liTypeIDIdx = self.graph.addEdgeType( agTypeStr, 
                                               staticProperties, 
                                               dynamicProperties, 
                                               agTypeID1, 
                                               agTypeID2)
        

        return  liTypeIDIdx

    def registerAgent(self, agent, ghost=False): 
        """
        Method to register instances of Agents
        """
        self.__allAgentDict[agent.nID] = agent
        
        if self.isParallel:
            self.__glob2loc[agent.gID] = agent.nID
            self.__loc2glob[agent.nID] = agent.gID

        if ghost:
            self.__ghostIDsByType[agent.agTypeID].append(agent.nID)
            self.__ghostDataIDsByType[agent.agTypeID].append(agent.dataID)
            self.__ghostsByType[agent.agTypeID].append(agent)
            self.__nGhostsByType[agent.agTypeID] +=1
        else:
            self.__agentIDsByType[agent.agTypeID].append(agent.nID)
            self.__agendDataIDsList[agent.agTypeID].append(agent.dataID)
            self.__agentsByType[agent.agTypeID].append(agent)
            self.__nAgentsByType[agent.agTypeID] +=1


            
    def deRegisterAgent(self, agent, ghost):
        """
        Method to remove instances of agents
        -> update of:
            - entList
            - endDict
            - __glob2loc
            - _loc2glob
        """
        
        del self.__allAgentDict[agent.nID]
        if self.isParallel:
            del self.__glob2loc[agent.gID]
            del self.__loc2glob[agent.nID]
        agTypeID =  self.graph.class2NodeType(agent.__class__)
        self.__agentsByType[agTypeID].remove(agent)
        if ghost:
            self.__ghostIDsByType[agTypeID].remove(agent.nID)
            self.__ghostDataIDsByType[agTypeID].remove(agent.dataID)
            self.__nGhostsByType[agTypeID] -=1
        else:
            self.__agentIDsByType[agTypeID].remove(agent.nID)
            self.__agendDataIDsList[agTypeID].remove(agent.dataID)
            self.__nAgentsByType[agTypeID] -=1
            #print(self.__agentIDsByType[agTypeID])
    
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


#%% Access of privat variables
        
    def getParallelCommInterface(self):
        """
        returns the parallel agent passing interface (see core.py)
        """
        return self.papi.comm

    def getGraph(self):
        """
        returns the agent graph instance
        """
        return self.graph

    def getGlobalRecord(self):
        """
        returns global record class
        """
        return self.globalRecord

    def getGlobals(self):
        """
        returns the global variables defined in the graph
        """
        return self.graph.glob

    def getLocationDict(self):
        """
        The locationDict contains all instances of locations that are
        accessed by (x,y) coordinates
        """
        return self.__locDict

    def getAgentDict(self):
        """
        The nodeDict contains all instances of different entity types
        """
        return self.__agentIDsByType

    def getAgentListsByType(self):
        return self.__agentsByType, self.__ghostsByType

    def getAllAgentDict(self):
        return self.__allAgentDict
    
    def getGlobToLocDIct(self):
        return self.__glob2loc
#%% other    
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



            if self.para['showFigures']:
                # plotting and saving figures
                for key in self.globalRecord:

                    self.globalRecord[key].plot(self.para['outPath'])

        
        if self.getParameters()['writeAgentFile']:
            self.io.finalizeAgentFile()

        
if __name__ == '__main__':
    
    pass
    
