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

class GridNode():
    """
    This enhancement allows agents to iterate over neigboring instances and thus
    funtion as collectives of agents. 
    """
    def __init__(self, world, nID = -1, **kwProperties):
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
    def __init__(self, world, nID = -1, **kwProperties):
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