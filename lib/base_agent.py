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
from .traits import Parallel, Mobile, Collective

def firstElementDeco(fun):
    """ 
    Decorator that returns the first element
    ToDo: if possible find better way
    """
    def helper(arg):
        return fun(arg)[0]
    return helper

class Entity():
    """
    Enitity is the stroage structure of all agents and contains the basic 
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

            self.get = firstElementDeco(self.attr.__getitem__)
            self.set = self.attr.__setitem__
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

class BaseAgent(Entity):
    """
    Most basic class from which agents of different type are derived
    """

    def __init__(self, world, nID = -1, **kwProperties):
        # init of the Entity class to init storage
        Entity.__init__(self, world, nID, **kwProperties)

        # redireciton of internal functionality:
        self.get = firstElementDeco(self.attr.__getitem__)
        self.set = self.attr.__setitem__
        self.__getNode = world.getAgent
    
    
    def getPeerIDs(self, liTypeID=None, agTypeID=None, mode='out'):
        """
        This method returns the IDs of all agents that are connected with a 
        certain link type or of a specified nodeType
        As default, only outgoing connctions are considered, but can be changed
        by setting mode ='in'.
        """
        if liTypeID is None:
            
            liTypeID = self._graph.node2EdgeType[self.agTypeID, agTypeID]
        else:
            assert agTypeID is None
        
        
        if mode=='out':            
            eList, nodeList  = self._graph.outgoing(self.nID, liTypeID)
        elif mode == 'in':
            eList, nodeList  = self._graph.incomming(self.nID, liTypeID)
        
        return nodeList


    def getPeerAttr(self, prop, liTypeID):
        """
        This method returns the attributes of all connected nodes connected 
        by a specfic edge type.
        """
        return self._graph.getOutNodeValues(self.nID, liTypeID, attr=prop)


                                   

    def getLinkAttr(self, prop, liTypeID):
        """
        This method accesses the values of outgoing links
        """
        (eTypeID, dataID), nIDList  = self._graph.outgoing(self.nID, liTypeID)

        edgesValues = self._graph.getEdgeSeqAttr(label=prop, 
                                                 eTypeID=eTypeID, 
                                                 dataIDs=dataID)

        

        return edgesValues, (eTypeID, dataID), nIDList

    def setLinkAttr(self, prop, values, liTypeID=None):
        """
        This method changes the attributes of outgoing links
        """
        (eTypeID, dataID), _  = self._graph.outgoing(self.nID, liTypeID)
        
        self._graph.setEdgeSeqAttr(label=prop, 
                                   values=values,
                                   eTypeID=eTypeID, 
                                   dataIDs=dataID)

    def getLinkIDs(self, liTypeID=None):
        """
        This method changes returns the linkID of all outgoing links of a 
        specified type
        """
        eList, _  = self._graph.outgoing(self.nID, liTypeID)
        return eList


    def addLink(self, friendID, liTypeID, **kwpropDict):
        """
        This method adds a new connection to another node. Properties must be 
        provided in the correct order and structure
        """
        self._graph.addLink(liTypeID, self.nID, friendID, attributes = tuple(kwpropDict.values()))

    def remLink(self, friendID=None, liTypeID=None):
        """
        This method removes a link to another agent.
        """
        self._graph.remEdge(source=self.nID, target=friendID, eTypeID=liTypeID)

    def remLinks(self, friendIDs=None, liTypeID=None):
        """
        Removing mutiple links to other agents.
        """        
        if friendIDs is not None:
            for friendID in friendIDs:
                self._graph.remEdge(source=self.nID, target=friendID, eTypeID=liTypeID)


        

    
