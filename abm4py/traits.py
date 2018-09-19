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

class Parallel():
    """
    This agent traits adds the required methods and variables to agents
    for parallel execution of the code.
    This contains:
    """
    def __init__(self, world, nID = None, **kwProperties):
        
        # adding properties to the attributes
        if world.isParallel:
            self.gID = self.getGlobID(world)
        else:
            self.gID = self.attr['ID']

        self.attr['gID'] = self.gID
        self.mpiOwner = world.mpiRank
        self.mpiGhostRanks = list()

    def getGlobID(self,world):
        return next(world.globIDGen)


    def registerChild(self, world, entity, liTypeID):
        """
        
        """
        # why is register child also adding a link?
        if liTypeID is not None:
            world.addLink(liTypeID, self.nID, entity.nID)

        if len(self.mpiGhostRanks) > 0: # node has ghosts on other processes
            for mpiPeer in self.mpiGhostRanks:
                #print 'adding node ' + str(entity.nID) + ' as ghost'
                agTypeID = world._graph.class2NodeType(entity.__class__)
                world.papi.queueSendGhostAgent( mpiPeer, agTypeID, entity, self)

        return self.mpiGhostRanks


class GridNode():
    """
    This enhancement identifies the agent as part of a grid. Currently, it only
    registers itself in the location dictionary, found in the world
    (see world.grid.registerNode())
    """
    def __init__(self, world, nID = None, **kwProperties):
        self.__getAgent = world.getAgent
        
        
    def register(self, world, parentEntity=None, liTypeID=None, ghost=False):
        
        world.registerAgent(self, ghost=ghost)
        world.grid.registerNode(self, *self.attr['coord'])
        self.gridPeers  = world._graph._addNoneEdge(self.attr['ID'])
        
        if parentEntity is not None:
            self.mpiGhostRanks = parentEntity.registerChild(world, self, liTypeID)
       
    def getGridPeers(self):
        return self._graph.nodes[self.gridPeers[0]]['instance'][self.gridPeers[1]]

    def getAttrOfGridPeers(self, attribute):
        return self._graph.nodes[self.gridPeers][attribute][self.gridPeers[1]]
        
        
class Mobile():
    """
    This enhancemement allows agents to move in the spatial domain. Currently
    this does not work in the parallel version
    """
 
    def __init__(self, world, nID = None, **kwProperties):
        """ assert that position is declared as an agent's attribute, since 
         moving relates to the 'coord' attribute """
        #TODO can be made more general"

        
        if world.isParallel:
            raise(BaseException('Mobile agents are not working in parallel'))
            
        assert 'coord' in kwProperties.keys()
        
        self._setLocationDict(world.grid.getNodeDict())
        
    def move(self, newX, newY, spatialLinkTypeID):
        # remove old link
        assert (newX, newY) in self.locDict.keys()
        self.attr['coord'] = [ newX, newY]
        #self.loc.remLink(self.nID, liTypeID=spatialLinkTypeID)
        
        oldLocID = self.loc.nID
        self.loc = self.locDict[(newX, newY)]      
        newLocID = self.loc.nID
        self._graph._changeSourceOfEdge(eTypeID=spatialLinkTypeID, targetID=self.nID, oldSourceID=oldLocID, newSourceID=newLocID)
        # add new link and location
        #self.loc = self.locDict[(newX, newY)]      
        #self.loc.addLink(self.nID, liTypeID=spatialLinkTypeID)

    @classmethod
    def _setLocationDict(cls, locDict):
        """ Makes the class variable ._graph available at the first init of an entity"""
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


        