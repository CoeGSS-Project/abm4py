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

class Parallel():
    """
    This agent traits adds the required methods and variables to agents
    for parallel execution of the code.
    This contains:
        - global ID: self.gID
        - 
    """
    def __init__(self, world, nID = None, **kwProperties):
        
        # adding properties to the attributes
        gID = self.getGlobID(world)
        self.attr['gID'] = gID
        self.gID = gID
         #kwProperties['gID'] = self.gID

        self.mpiPeers = list()

    def getGlobID(self,world):
        return next(world.globIDGen)


    def registerChild(self, world, entity, liTypeID):
        """
        
        """
        # why is register child also adding a link?
        if liTypeID is not None:
            world.addLink(liTypeID, self.nID, entity.nID)

        if len(self.mpiPeers) > 0: # node has ghosts on other processes
            for mpiPeer in self.mpiPeers:
                #print 'adding node ' + str(entity.nID) + ' as ghost'
                agTypeID = world.graph.class2NodeType(entity.__class__)
                world.papi.queueSendGhostAgent( mpiPeer, agTypeID, entity, self)

        return self.mpiPeers


class GridNode():
    """
    This enhancement identifies the agent as part of a grid. Currently, it only
    registers itself in the location dictionary, found in the world
    (see world.getLocationDict())
    """
    def __init__(self, world, nID = None, **kwProperties):
        self.__getAgent = world.getAgent
        
        
    def register(self, world, parentEntity=None, liTypeID=None, ghost=False):
        
        world.registerAgent(self, ghost=ghost)
        world.registerLocation(self, *self.attr['coord'])
        self.gridPeers  = world.graph._addNoneEdge(self.attr['ID'])
        
        if parentEntity is not None:
            self.mpiPeers = parentEntity.registerChild(world, self, liTypeID)
       
    def getGridPeers(self):
        return self._graph.nodes[self.gridPeers[0]]['instance'][self.gridPeers[1]]

    def getAttrOfGridPeers(self, attribute):
        return self._graph.nodes[self.gridPeers[0]][attribute][self.gridPeers[1]]
        
        
class Mobile():
    """
    This enhancemement allows agents to move in the spatial domain. Currently
    this does not work in the parallel version
    """
 
    def __init__(self, world, nID = None, **kwProperties):
        """ assert that position is declared as an agent's attribute, since 
         moving relates to the 'pos' attribute """
        #TODO can be made more general"

        
        if world.isParallel:
            raise(BaseException('Mobile agents are not working in parallel'))
            
        assert 'coord' in kwProperties.keys()
        
        self._setLocationDict(world.getLocationDict())
        
    def move(self, newX, newY, spatialLinkTypeID):
        # remove old link
        assert (newX, newY) in self.locDict.keys()
        self['coord'] = [ newX, newY]
        self.loc.remLink(self.nID, liTypeID=spatialLinkTypeID)
       
        # add new link and location
        self.loc = self.locDict[(newX, newY)]      
        self.loc.addLink(self.nID, liTypeID=spatialLinkTypeID)

    @classmethod
    def _setLocationDict(cls, locDict):
        """ Makes the class variable _graph available at the first init of an entity"""
        cls.locDict = locDict
                      

class SuperPowers():
    """
    This agent-enhancement allows to write attributes of connected agents
    Use carefully and not in parallel mode.
    """
    
    def __init__(self, world, nID = None, **kwProperties):
        # check that the framework is not parallelized since the writing of 
        # attributes from other agents violates the consistency of parallel
        # execution
        assert world.isParallel == False

    def setPeerAttr(self, prop, values, liTypeID=None, agTypeID=None):
        """
        Set the attributes of all connected nodes of an specified agTypeID
        or connected by a specfic edge type
        """
        self._graph.setOutNodeValues(self.nID, liTypeID, prop, values)    

class Aggregator():
    """
    This is an experimental trait that overrides the addLink and remLink methods
    of the agent classes with addtional capabilities.
    
    AddLink will than also add the attrbute array of the link target to an 
    aggregationDict, which is ordered by linkTypeIDs. Similarly, remLink will
    remove the attributes again. ATTENTION: world.addLink(s), does not support
    this additional feature!!
    
    Derive a new Class like **ClassNewClass(Aggregator, Agent)**.
    
    aggregateItems
    
    """
    
    def __init__(self, world, nID = None, **kwProperties):
        self.aggegationDict = dict()
        self.__getAgent = world.getAgent
    
    def addLink(self, peerID, liTypeID, **kwpropDict):
        """
        This method adds a new connection to another node. Properties must be 
        provided in the correct order and structure, bt also 
        """
        self._graph.addEdge(liTypeID, self.nID, peerID, attributes = tuple(kwpropDict.values()))
        try:
            self.aggegationDict[liTypeID].append(self.__getAgent(peerID).attr)
        except:
            self.aggegationDict[liTypeID] = [self.__getAgent(peerID).attr]
            
    def remLink(self, peerID, liTypeID):
        """
        This method removes a link to another agent.
        """
        self._graph.remEdge(source=self.nID, target=peerID, eTypeID=liTypeID)
        self.aggegationDict[liTypeID].remove(self.__getAgent(peerID).attr)
        