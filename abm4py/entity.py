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


class _Entity(object):
    """
    Enitity is the storage structure of all agents and contains the basic 
    methods to create and delete itself.
    """
    __slots__ = ['ID', 'dataID', 'attr']
    
    def __init__(self, world, ID=None, **kwProperties):
        
        #  the agTypeID is derived from the agent class and stored        
        self.agTypeID =  world._graph.class2NodeType(self.__class__)

        # create new node in the._graph
        if ID == None:
            self.ID, self.dataID, self.attr = world.addNode(self.agTypeID, **kwProperties)    
            
            self._setGraph(world._graph)
            self.attr['instance'] = self
            self.__getNode = world.getAgent

        else:
            self.ID = ID
            self._setGraph(world._graph)
            self.attr, self.dataID = self._graph.getNodeView(ID)
            self.attr['instance'] = self
            
            
        
    @classmethod
    def _setGraph(cls,graph):
        """ Makes the class variable ._graph available at the first init of an entity"""
        cls._graph = graph

    def register(self, world, parentEntity=None, liTypeID=None, ghost=False):
        
        world.addAgent(self, ghost=ghost)

        if parentEntity is not None:
            self.mpiGhostRanks = parentEntity.registerChild(world, self, liTypeID)

    def delete(self, world):
        """ method to delete the agent from the simulation"""
        world._graph.remNode(self.ID)
        world.removeAgent(self, ghost=False)
