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
from .enhancements import Parallel, Mobil, Collective

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
        
        #  the nodeTypeID is derived from the agent class and stored        
        self.nodeTypeID =  world.graph.class2NodeType(self.__class__)

        # create new node in the graph
        if nID == -1:
            self.nID, self.dataID, self.attr = world.addNode(self.nodeTypeID,  **kwProperties)    
            self._setGraph(world.graph)
            #print(self.attr['gID'][0])
        
        
    @classmethod
    def _setGraph(cls, graph):
        """ Makes the class variable _graph available at the first init of an entity"""
        cls._graph = graph

    def register(self, world, parentEntity=None, linkTypeID=None, ghost=False):
        
        world.registerNode(self, self.nodeTypeID, ghost=ghost)

        if parentEntity is not None:
            self.mpiPeers = parentEntity.registerChild(world, self, linkTypeID)

    def delete(self, world):
        """ method to delete the agent from the simulation"""
        world.graph.remNode(self.nID)
        world.deRegisterNode(self, ghost=False)

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
        self.__getNode = world.getNode
    
    
    def getPeerIDs(self, linkTypeID=None, nodeTypeID=None, mode='out'):
        """
        This method returns the IDs of all agents that are connected with a 
        certain link type or of a specified nodeType
        As default, only outgoing connctions are considered, but can be changed
        by setting mode ='in'.
        """
        if linkTypeID is None:
            
            linkTypeID = self._graph.node2EdgeType[self.nodeTypeID, nodeTypeID]
        else:
            assert nodeTypeID is None
        
        
        if mode=='out':            
            eList, nodeList  = self._graph.outgoing(self.nID, linkTypeID)
        elif mode == 'in':
            eList, nodeList  = self._graph.incomming(self.nID, linkTypeID)
        
        return nodeList

        
    def getPeerAttr(self, prop, linkTypeID=None):
        """
        This method returns the attributes of all connected nodes connected 
        by a specfic edge type.
        """
        return self._graph.getOutNodeValues(self.nID, linkTypeID, attr=prop)


                                   

    def getLinkAttr(self, prop, linkTypeID):
        """
        This method accesses the values of outgoing links
        """
        (eTypeID, dataID), nIDList  = self._graph.outgoing(self.nID, linkTypeID)

        edgesValues = self._graph.getEdgeSeqAttr(label=prop, 
                                                 eTypeID=eTypeID, 
                                                 dataIDs=dataID)

        

        return edgesValues, (eTypeID, dataID), nIDList

    def setLinkAttr(self, prop, values, linkTypeID=None):
        """
        This method changes the attributes of outgoing links
        """
        (eTypeID, dataID), _  = self._graph.outgoing(self.nID, linkTypeID)
        
        self._graph.setEdgeSeqAttr(label=prop, 
                                   values=values,
                                   eTypeID=eTypeID, 
                                   dataIDs=dataID)

    def getLinkIDs(self, linkTypeID=None):
        """
        This method changes returns the linkID of all outgoing links of a 
        specified type
        """
        eList, _  = self._graph.outgoing(self.nID, linkTypeID)
        return eList


    def addLink(self, friendID, linkTypeID, **kwpropDict):
        """
        This method adds a new connection to another node. Properties must be 
        provided in the correct order and structure
        """
        self._graph.addLink(linkTypeID, self.nID, friendID, attributes = tuple(kwpropDict.values()))

    def remLink(self, friendID=None, linkTypeID=None):
        """
        This method removes a link to another agent.
        """
        self._graph.remEdge(source=self.nID, target=friendID, eTypeID=linkTypeID)

    def remLinks(self, friendIDs=None, linkTypeID=None):
        """
        Removing mutiple links to other agents.
        """        
        if friendIDs is not None:
            for friendID in friendIDs:
                self._graph.remEdge(source=self.nID, target=friendID, eTypeID=linkTypeID)


        

    
