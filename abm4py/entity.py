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
    __slots__ = ['nID', 'dataID', 'attr']
    
    def __init__(self, world, nID=None, **kwProperties):
        
        #  the agTypeID is derived from the agent class and stored        
        self.agTypeID =  world._graph.class2NodeType(self.__class__)

        # create new node in the._graph
        if nID == None:
            self.nID, self.dataID, self.attr = world.addNode(self.agTypeID, **kwProperties)    
            
            self._setGraph(world._graph)
            self.attr['instance'] = self
            self.__getNode = world.getAgent

#            self.get = self.attr.__getitem__
#            self.set = self.attr.__setitem__
#            self.__getitem__ = self.attr.__getitem__
#            self.__setitem__ = self.attr.__setitem__
        # connects the agent to an existing node
        else:
            self.nID = nID
            self._setGraph(world._graph)
            self.attr, self.dataID = self._graph.getNodeView(nID)
            self.attr['instance'] = self
            

#    def __getitem__(self, a):
#        return self.attr.__getitem__(a)
#
#    def __setitem__(self, a, value):
#        self.attr.__setitem__(a, value)


#    def __getattr__(self, a):
#        return self.attr[a][0]
#    
#    def __setattr__(self, a, value):
#        try:
#            self.attr[a] = value
#        except:
#            object.__setattr__(self, a, value)
            
        
    @classmethod
    def _setGraph(cls,graph):
        """ Makes the class variable ._graph available at the first init of an entity"""
        cls._graph = graph

    def register(self, world, parentEntity=None, liTypeID=None, ghost=False):
        
        world.registerAgent(self, ghost=ghost)

        if parentEntity is not None:
            self.mpiGhostRanks = parentEntity.registerChild(world, self, liTypeID)

    def delete(self, world):
        """ method to delete the agent from the simulation"""
        world._graph.remNode(self.nID)
        world.deRegisterAgent(self, ghost=False)
