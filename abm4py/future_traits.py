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

class GridNode():
    """
    This enhancement allows agents to iterate over neigboring instances and thus
    funtion as collectives of agents. 
    """
    def __init__(self, world, ID = -1, **kwProperties):
        self.__getAgent = world.getAgent
        self.Neighborhood = dict()
        #self.register = GridNode.register

    def register(self, world, parentEntity=None, liTypeID=None, ghost=False):
        
        world.registerAgent(self, ghost=ghost)
        world.registerLocation(self, *self.attr['coord'])
        if parentEntity is not None:
            self.mpiPeers = parentEntity.registerChild(world, self, liTypeID)
    
    def getNeighbor(self, peerID):
        return self.__getAgent(peerID)
    
    def reComputeNeighborhood(self, liTypeID):
        #self.Neighborhood[liTypeID] = [self.getNeighbor(ID) for ID in self.getPeerIDs(liTypeID)]
        self.Neighborhood[liTypeID] = self.getAttrOfPeers('instance', liTypeID= liTypeID)
        
    def iterNeighborhood(self, liTypeID):
        return iter(self.Neighborhood[liTypeID])

class Collective():
    """
    This enhancement allows agents to iterate over member instances and thus
    funtion as collectives of agents. Examples could be a pack of wolf or 
    an household of persons.
    """
    def __init__(self, world, ID = -1, **kwProperties):
        self.__getAgent = world.getAgent
        self.groups = dict()
        
    def getMember(self, peerID):
        return self.__getAgent(agentID=peerID)
    
    def iterMembers(self, peerIDs):
        return [self.__getAgent(agentID=peerID) for peerID in peerIDs]
 
    def registerGroup(self, groupName, members):
        self.groups[groupName] = members
        
    def iterGroup(self, groupName):
        return iter(self.groups[groupName])
    
    def join(self, groupName, agent):
        self.groups[groupName].append(agent)
        
    def leave(self, groupName, agent):
        self.groups[groupName].remove(agent)  
        
        
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
    
    def __init__(self, world, ID = None, **kwProperties):
        self.aggegationDict = dict()
        self.__getAgent = world.getAgent
    
    def addLink(self, peerID, liTypeID, **kwpropDict):
        """
        This method adds a new connection to another node. Properties must be 
        provided in the correct order and structure, bt also 
        """
        self._graph.addEdge(liTypeID, self.ID, peerID, attributes = tuple(kwpropDict.values()))
        try:
            self.aggegationDict[liTypeID].append(self.__getAgent(peerID).attr)
        except:
            self.aggegationDict[liTypeID] = [self.__getAgent(peerID).attr]
            
    def remLink(self, peerID, liTypeID):
        """
        This method removes a link to another agent.
        """
        self._graph.remEdge(source=self.ID, target=peerID, eTypeID=liTypeID)
        self.aggegationDict[liTypeID].remove(self.__getAgent(peerID).attr)
        