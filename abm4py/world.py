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

from .graph import ABMGraph
from . import core, misc

import logging as lg
import numpy as np
import types

class World:
    """
    The world class is the core interface for controling and structuring the
    ABM-simulation. 
    It functions as user interface and contains all other components and sub-
    classes.
    """

    def __init__(self,
                 simNo=None,
                 outPath='.',
                 nSteps=1,
                 maxNodes=1e6,
                 maxLinks=1e6,
                 debug=False,
                 agentOutput=False,
                 linkOutput=False):
        
        mpiComm = core.comm

        if mpiComm is None:
            self.isParallel = False
            self.isRoot     = True
            self.mpiRank    = 0
        else:
            self.isParallel = mpiComm.size > 1
            self.mpiRank = core.mpiRank
            self.mpiSize = core.mpiSize
            if self.isParallel:
                print('parallel mode enabled')
                if self.mpiRank == 0:
                    self.isRoot = True
                else:
                    self.isRoot = False
            else:
                self.isRoot = True
        
                
        # ======== GRAPH ========
        self._graph    = ABMGraph(self, maxNodes, maxLinks)
        
        # agent passing interface for communication between parallel processes
        self.papi = core.PAPI(self)
        self.__glob2loc = dict()  # reference from global IDs to local IDs
        self.__loc2glob = dict()  # reference from local IDs to global IDs

        # generator for IDs that are globally unique over all processe
        self.globIDGen = self._globIDGen()
        
        lg.debug('Init MPI done')###OPTPRODUCTION
                
                
          
        self.para      = dict()
        self.para['outPath'] = outPath # is not._graph, move to para
        # TODO put to parameters                    
        self.agentOutput = agentOutput
        self.linkOutput  = linkOutput
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
        
        self.__nAgentsByType = dict()
        self.__nGhostsByType = dict()
        
        # ======== GLOBALS ========
        # storage of global data
        self.globalRecord = dict() 
        # Globally synced variables
        self._graph.glob     = core.Globals(self)
        lg.debug('Init Globals done')###OPTPRODUCTION
        
        # ======== IO ========
        self.io = core.IO(self, nSteps, self.para['outPath'])
        lg.debug('Init IO done')###OPTPRODUCTION
        
        # ======== RANDOM ========
        self.random = core.Random(self, self.__agentIDsByType, self.__ghostIDsByType, self.__agentsByType)
        
        
        # re-direct of._graph functionality 
        self.addLink        = self._graph.addEdge
        self.addLinks       = self._graph.addEdges
        
        self.delLink       = self._graph.remEdge

        self.addNode     = self._graph.addNode
        self.addNodes    = self._graph.addNodes
        
        self.areConnected = self._graph.areConnected
      
        #self.getAgents  = core.AgentAccess(self)

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

#%% Register methods
    def registerGrid(self, GridNodeType, GridLinkType):
        """
        This functions registers a grid within the world. A grid consists of agents
        of the type GridNode and links that represent the spatial proximity
        """
        # ======== GRID LAYER ========
        self.grid  = core.Grid(self, GridNodeType, GridLinkType)
        
    
    def __addIterNodeFunction(self, agTypeStr, nodeTypeId):
        name = "iter" + agTypeStr.capitalize()
        source = """def %NAME%(self):
                        return [ self._World__allAgentDict[i] for i in self._World__agentIDsByType[%NODETYPEID%] ]
        """.replace("%NAME%", name).replace("%NODETYPEID%", str(nodeTypeId))
        exec(compile(source, "", "exec"))
        setattr(self, name, types.MethodType(locals()[name], self))

    def registerAgentType(self, AgentClass, GhostAgentClass=None , agTypeStr = None, staticProperties=None, dynamicProperties=None):
        """
        This function registers a node type and the defined properties of
        each agTypeID for other purposes, e.g. I/O
        
        """
        descDict = AgentClass.__descriptor__()
        
        # setting defaults if required
        if agTypeStr is None:
            agTypeStr = descDict['nameStr']
        if staticProperties is None:
            staticProperties = descDict['staticProperties'] 
        if dynamicProperties is None:
            dynamicProperties = descDict['dynamicProperties']

        # adds and formats properties we need for the framework (like gID) automatically
        staticProperties = core.formatPropertyDefinition(staticProperties)
        dynamicProperties = core.formatPropertyDefinition(dynamicProperties)
                
        agTypeIDIdx = len(self._graph.agTypeByID)+1
                
        self._graph.addNodeType(agTypeIDIdx, 
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
        Method to register a edge type and to the related properties of each 
        liTypeID for other purposes, e.g. I/O
    
        """
        if ('source', np.int32, 1) not in staticProperties:
            staticProperties.append(('source', np.int32, 1))
        if ('target', np.int32, 1) not in staticProperties:
            staticProperties.append(('target', np.int32, 1))
                             
        staticProperties  = core.formatPropertyDefinition(staticProperties)
        dynamicProperties = core.formatPropertyDefinition(dynamicProperties)        
        
        liTypeIDIdx = self._graph.addEdgeType( agTypeStr, 
                                               staticProperties, 
                                               dynamicProperties, 
                                               agTypeID1, 
                                               agTypeID2)
        
        return  liTypeIDIdx


    def addAgent(self, agent, ghost=False): 
        """
        This function registers a new instances of an agent to the simulation.
        """
        self.__allAgentDict[agent.ID] = agent
        
        if self.isParallel:
            self.__glob2loc[agent.gID] = agent.ID
            self.__loc2glob[agent.ID] = agent.gID

        if ghost:
            self.__ghostIDsByType[agent.agTypeID].append(agent.ID)
            self.__ghostDataIDsByType[agent.agTypeID].append(agent.dataID)
            self.__ghostsByType[agent.agTypeID].append(agent)
            self.__nGhostsByType[agent.agTypeID] +=1
        else:
            self.__agentIDsByType[agent.agTypeID].append(agent.ID)
            self.__agendDataIDsList[agent.agTypeID].append(agent.dataID)
            self.__agentsByType[agent.agTypeID].append(agent)
            self.__nAgentsByType[agent.agTypeID] +=1


            
    def removeAgent(self, agent, ghost):
        """
        Method to remove instances of agents from the environment
        """
        
        del self.__allAgentDict[agent.ID]
        if self.isParallel:
            del self.__glob2loc[agent.gID]
            del self.__loc2glob[agent.ID]
        agTypeID =  self._graph.class2NodeType(agent.__class__)
        self.__agentsByType[agTypeID].remove(agent)
        if ghost:
            self.__ghostIDsByType[agTypeID].remove(agent.ID)
            self.__ghostDataIDsByType[agTypeID].remove(agent.dataID)
            self.__nGhostsByType[agTypeID] -=1
        else:
            self.__agentIDsByType[agTypeID].remove(agent.ID)
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
            self._graph.glob.registerValue(name , np.asarray([np.nan]*len(colLables)),mpiReduce)
            self.globalRecord[name].glob = self._graph.glob
    
#%% General Infromation methods
    def countAgents(self, agTypeID, ghosts=False):
        """
        This function allows to count a sub-selection of links that is defined 
        by a filter function that act on links attribute.

        Use case:
        nAgents = world.countAgents(agTypeID)    
        """
        if ghosts:
            return self.__nGhostsByType[agTypeID]
        else:
            return self.__nAgentsByType[agTypeID]
        
    def countLinks(self, liTypeID):
        """
        This function allows to count a sub-selection of links that is defined 
        by a filter function that act on links attribute.

        Use case:
        nLinks = world.countLinks(liTypeID) 
        """
        return self._graph.eCount(eTypeID=liTypeID)  

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
        misc.saveObj(self.para, self.para['outPath'] + '/' + fileName)
       
    def getEnums(self):
        """ 
        Returns a specified enumeration dict
        """         
        return self.__enums

    def saveEnumerations(self, fileName= 'enumerations'):
        # saving enumerations
        misc.saveObj(self.__enums, self.para['outPath'] + '/' + fileName)
        
    def agentTypesIDs(self):
        return list(self._graph.agTypeByID.keys())

    def linkTypesIDs(self):
        return list(self._graph.liTypeByID.keys())
                
#%% Agent access 
    def getAgent(self, agentID):
        """
        Returns a single agent selcetd by its ID
        """
        return self.__allAgentDict[agentID]
    
    def getAgentsByIDs(self, localIDs=None):
        """
        Returns agents defined by the ID
        """
        return [self.__allAgentDict[i] for i in localIDs] 

    def getAgentsByGlobalID(self, globalIDs=None):
        """
        Returns agents defined by the global ID
        """
        return  [self.__allAgentDict[self.__glob2Loc[i]] for i in globalIDs]
    
        
    def getAgentsByType(self, agTypeID=None,ghosts = False):
        """
        Iteration over entities of specified type. Default returns
        non-ghosts in random order.
        """
        if ghosts:
            return  self.__ghostsByType[agTypeID].copy()
        else:
            return  self.__agentsByType[agTypeID].copy()

    def getAgentsByFilteredType(self, func, agTypeID):
        """
        This function allows to access a sub-selection of agents that is defined 
        by a filter function that is action on agent properties.

        Use case: Iterate over agents with a property below a certain treshold:
        for agent in world.filterAgents(lambda a: a['age'] < 1, AGENT)
        """
        array = self._graph.nodes[agTypeID]
        mask = array['active'] & func(array)
        return array['instance'][mask]


    def countFilteredAgents(self, func, agTypeID):
        """
        This function allows to coiunt a sub-selection of agents that is defined 
        by a filter function that act on agent attribute.

        Use case: Count  agents with a property below a certain treshold:
        nAgents = world.countFilteredAgents(lambda a: a['age'] < 1, AGENT)
        """
        array = self._graph.nodes[agTypeID]
        mask = array['active'] & func(array)
        return np.sum(mask)

    def countFilteredLinks(self, func, liTypeID):
        """
        This function allows to count a sub-selection of links that is defined 
        by a filter function that act on links attribute.

        Use case: Count over links with a property below a certain treshold:
        nLinks = world.countFilteredLinks(lambda a: a['wieight'] < 01, linkTypeID)
        """
        array = self._graph.edges[liTypeID]
        mask = array['active'] & func(array)
        return np.sum(mask)    


    def deleteAgentsIf(self, agTypeID, filterFunc):
        """
        This function allows to delete a sub-selection of agents that is defined 
        by a filter function that act on agent attribute.

        Use case: Iterate over agents with a property below a certain treshold:
        for agent in world.filterAgents(lambda a: a['age'] < 1, AGENT)
        """
        array = self._graph.nodes[agTypeID]
        filtered = array[array['active'] == True & filterFunc(array)]
        if len(filtered) > 0:
            for aID in np.nditer(filtered['gID']):
                agent = self.getNode(globID = int(aID))
                agent.delete(self)

  
    
    def getAgentIDs(self, agTypeID, ghosts=False):
        """ 
        Method to return all local node IDs for a given nodeType
        """
        if ghosts:
            return self.__ghostIDsByType[agTypeID]
        else:
            return self.__agentIDsByType[agTypeID]
        

        
    def getAgentDataIDs(self, agTypeID):
        """
        This function return the dataID of an agent. The dataID is the index in the
        numpy array, where the agents attributes are stored
        """
        return self.__agendDataIDsList[agTypeID]


#%% Global agent attribute access
        

    def getAttrOfAgents(self, attribute, localIDList):
        """
        Read attributes for a given sequence of agents
        Example: 
        attrArray = world.setAttrOfAgents('coord', [100000, 100001])
        """
        return self._graph.getNodeSeqAttr(attribute, lnIDs=localIDList)

    def setAttrOfAgents(self, attribute, valueList, localIDList):
        """
        Write attributes for a given sequence of agents
        Example: 
        world.setAttrOfAgents('coord', [(1,2), (0,1)], [100000, 100001])
        """
        self._graph.setNodeSeqAttr(attribute, valueList, lnIDs=localIDList)
    
    def getAttrOfAgentType(self, attribute, agTypeID):
        """
        Returns numpy array of the defined attribute of all agents of one type
        Example: 
        attrArray = world.getAttrOfAgentType('coord', agType=LOCATION_TYPE_ID)
        """
        return self._graph.getNodeSeqAttr(attribute, lnIDs=self.__agentIDsByType[agTypeID])
        


    def setAttrOfAgentType(self, attribute, valueList, agTypeID):
        """
        Write attributes of all agents of type at once
        Example:
        world.setAttrOfAgentType('coord', (0,0), agTypeID)
        """
        self._graph.setNodeSeqAttr(attribute, valueList, lnIDs=self.__agentIDsByType[agTypeID])
          
        
            
    def getAttrOfLinks(self, attribute, localLinkIDList):
        """
        Read the specified attribute the specified type of links
        Example:
        coordinates = world.getAttrOfAgentType('coord', [100000, 100001]) 
        """
        return self._graph.getEdgeSeqAttr(attribute, lnIDs=localLinkIDList)

    def setAttrOfLinks(self, attribute, valueList, localLinkIDList):
        """
        Write the specified attribute the specified list of links
        Example:
        world.setAttrOfAgentType('coord', [[0,0], [0,1]], [100000, 100001]) 
        """
        self._graph.setEdgeSeqAttr(attribute, valueList, lnIDs=localLinkIDList)
        
    def getAttrOfLinkType(self, attribute, liTypeID):
        """
        Read the specified attribute of all agents that are of the specified link type
        Example:
        coordinates = getAttrOfLinkType('coord', liTypeID)    
        """
        return self._graph.getEdgeSeqAttr(attribute, lnIDs=self.linkDict[liTypeID])
        
    def setAttrOfLinkType(self, attribute, valueList, liTypeID):
        """
        Write the specified attribute of all agents that are of the specified link type
        Example:
        getAttrOfLinkType('coord', coordinateList, liTypeID)   
        """
        self._graph.setEdgeSeqAttr(attribute, valueList, lnIDs=self.linkDict[liTypeID])            


#%% Vectorized functions access
    def getAttrOfFilteredAgentType(self, attribute, func, agTypeID):
        """
        This function allows to access a sub-selection of agents that is defined 
        by a filter function that is action on agent properties.

        Use case: Get the pos of all agents with a property below a certain treshold:
        for agent in world.filterAgents('pos', lambda a: a['age'] < 1, AGENT)
        """
        array = self._graph.nodes[agTypeID]
        mask = array['active'] & func(array)
        return array[attribute][mask]

    def setAttrOfFilteredAgentType(self, attribute, value, agTypeID, func):
        """
        This function allows to access the attributes of a sub-selection of agents 
        that is defined  by a filter function that is action on agent properties.

        Use case: Iterate over agents with a property below a certain treshold:
        for agent in world.filterAgents(AGENT, lambda a: a['age'] < 1)
        """
        array = self._graph.nodes[agTypeID]
        mask = array['active'] & func(array)
        array[attribute][mask] = value

    def setAttrsForTypeVectorized(self, agTypeID, attribute, vfunc, idx=None):
        """
        This function allows to alter the attributes of agents of a specified type
        based on the agents attribute.

        Use case: Iterate over agents with a property below a certain treshold:
        for agent in world.filterAgents(AGENT, lambda a: a['age'] < 1)
        """
        array = self._graph.nodes[agTypeID]
        if idx is None:
            array[attribute][array['active']] = vfunc(array[array['active']]) 
        else:
            array[attribute][:,idx][array['active']] = vfunc(array[array['active']]) 
    
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

        setAttrsForType(agTypeID, increaseAge)

        def increaseXCoord(agents):
            agents['coord'][:,0] +=1
            return agents
        
        setAttrsForType(agTypeID, increaseXCoord)
        
        see also: setAttrsForFilteredType/Array
        """    
        array = self._graph.nodes[agTypeID]
        array[array['active']] = func(array[array['active']])

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
        array = self._graph.nodes[agTypeID]
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




#%% Access of privat variables
        
    def getParallelCommInterface(self):
        """
        returns the parallel agent passing interface (see core.py)
        """
        return self.papi.comm

    def getGraph(self):
        """
        Returns the world._graph instance
        """
        return self._graph

    def getGlobalRecord(self):
        """
        Returns global record class
        """
        return self.globalRecord

    def getGlobals(self):
        """
        Returns the global variables defined in the._graph
        """
        return self._graph.glob


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
        
    
    def glob2Loc(self, globIdx):
        if self.isParallel == False:
            def replacement(globIdx):
                return globIdx
            self.glob2Loc = replacement
            return globIdx
            
        return self.__glob2loc[globIdx]

    def setGlob2Loc(self, globIdx, locIdx):
        
        
        self.__glob2loc[globIdx] = locIdx

    def loc2Glob(self, locIdx):
        return self.__loc2glob[locIdx]

    def setLoc2Glob(self, globIdx, locIdx):
        self.__loc2glob[locIdx] = globIdx 
        
        
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

        if self.getParameters()['writeLinkFile']:
            self.io.finalizeLinkFile()

if __name__ == '__main__':
    
    pass
    
