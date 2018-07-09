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

from .base_agent import BaseAgent, Entity
from .enhancements import Mobil, Parallel

class Agent(BaseAgent, Mobil):
    """
    The most common agent type derives from the BaseAgent and additionally
    receives the abilty to move
    """
    def __init__(self, world, **kwProperties):
        if 'nID' not in list(kwProperties.keys()):
            nID = -1
        else:
            nID = kwProperties['nID']
        
        BaseAgent.__init__(self, world, nID, **kwProperties)
        




class GhostAgent(Entity, Parallel):
    """
    Ghost agents are only derived from Entities, since they are not full 
    agents, but passive copies of the real agents. Thus they do not have 
    the methods to act themselves.
    """
    def __init__(self, world, mpiOwner, nID=-1, **kwProperties):
        
        Entity.__init__(self, world, nID, **kwProperties)
        self.mpiOwner = int(mpiOwner)
        
        self._setGraph(world.graph)
        
        if nID is not -1:
            self.nID = nID
            self.attr, self.dataID = self._graph.getNodeView(nID)
        self.gID = self.attr['gID'][0]
        
        
    def register(self, world, parentEntity=None, liTypeID=None):
        Entity.register(self, world, parentEntity, liTypeID, ghost=True)
        

    def getGlobID(self,world):

        return None # global ID need to be acquired via MPI communication

    def registerChild(self, world, entity, liTypeID):
        world.addLink(liTypeID, self.nID, entity.nID)
