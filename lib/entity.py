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

#def firstElementDeco(fun):
#    """ 
#    Decorator that returns the first element
#    ToDo: if possible find better way
#    """
#    def helper(arg):
#        return fun(arg)[0]
#    return helper

class _Entity():
    """
    Enitity is the storage structure of all agents and contains the basic 
    methods to create and delete itself.
    """
    def __init__(self, world, nID = -1, **kwProperties):
        
        #  the agTypeID is derived from the agent class and stored        
        self.agTypeID =  world.graph.class2NodeType(self.__class__)

        # create new node in the graph
        if nID == -1:
            self.nID, self.dataID, self.attr = world.addNode(self.agTypeID, **kwProperties)    
            self._setGraph(world.graph)
            self['instance'] = self
            self.__getNode = world.getAgent

    def __getitem__(self, a):
        return self.attr.__getitem__(a)[0]

    def __setitem__(self, a, value):
        self.attr.__setitem__(a, value)


#    def __getattr__(self, a):
#        return self.attr[a][0]
#    
#    def __setattr__(self, a, value):
#        try:
#            self.attr[a] = value
#        except:
#            object.__setattr__(self, a, value)
            
        
    @classmethod
    def _setGraph(cls, graph):
        """ Makes the class variable _graph available at the first init of an entity"""
        cls._graph = graph

    def register(self, world, parentEntity=None, liTypeID=None, ghost=False):
        
        world.registerAgent(self, self.agTypeID, ghost=ghost)

        if parentEntity is not None:
            self.mpiPeers = parentEntity.registerChild(world, self, liTypeID)

    def delete(self, world):
        """ method to delete the agent from the simulation"""
        world.graph.remNode(self.nID)
        world.deRegisterAgent(self, ghost=False)

    


        

    
