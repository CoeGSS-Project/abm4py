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
    Most basic class from which agents of different type are derived
    """
    __slots__ = ['gID', 'nID']
    

    def __init__(self, world, nID = -1, **kwProperties):
        if world is None:
            return
                
        self.nodeTypeID =  world.graph.class2NodeType(self.__class__)
        
        if not hasattr(self, '_graph'):
            self._setGraph(world.graph)

        self.gID    = self.getGlobID(world)
        kwProperties['gID'] = self.gID

        # create instance from existing node
        if nID is not -1:
            self.nID = nID
            self.attr, self.dataID = self._graph.getNodeView(nID)
            self.gID = self.attr['gID'][0]
        
        else:
            self.nID, self.dataID, self.attr = world.addNode(self.nodeTypeID,  **kwProperties)
            

        # redireciton of internal functionality:
        self.get = firstElementDeco(self.attr.__getitem__)
        self.set = self.attr.__setitem__
        self.__getNode = world.getNode
    
    @classmethod
    def _setGraph(cls, graph):
        """ Makes the class variable _graph available at the first init of an entity"""
        cls._graph = graph


#    def attrView(self, key=None):
#        if key is None:
#            return self.attr[0].view()
#        else:
#            return self.attr[0,key].view()

    def getPeer(self, peerID):
        return self.__getNode(nodeID=peerID)
    
    def getPeers(self, peerIDs):
        return [self.__getNode(nodeID=peerID) for peerID in peerIDs]
    
    def getPeerIDs(self, linkTypeID=None, nodeTypeID=None, mode='out'):
        
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
        Access the attributes of all connected nodes of an specified nodeTypeID
        or connected by a specfic edge type
        """
        return self._graph.getOutNodeValues(self.nID, linkTypeID, attr=prop)

#    def setPeerAttr(self, prop, values, linkTypeID=None, nodeTypeID=None, force=False):
#        """
#        Set the attributes of all connected nodes of an specified nodeTypeID
#        or connected by a specfic edge type
#        """
#        if not force:
#            raise Exception
#        else:
#            #import warnings
#            #warnings.warn('This is violating the current rules and data get lost')
#
#            self._graph.setOutNodeValues(self.nID, linkTypeID, prop, values)
                                   

    def getLinkAttr(self, prop, linkTypeID):
        """
        private function to access the values of  edges
        """
        (eTypeID, dataID), nIDList  = self._graph.outgoing(self.nID, linkTypeID)

        edgesValues = self._graph.getEdgeSeqAttr(label=prop, 
                                                 eTypeID=eTypeID, 
                                                 dataIDs=dataID)

        

        return edgesValues, (eTypeID, dataID), nIDList

    def setLinkAttr(self, prop, values, linkTypeID=None):
        """
        private function to access the values of  edges
        """
        (eTypeID, dataID), _  = self._graph.outgoing(self.nID, linkTypeID)
        
        self._graph.setEdgeSeqAttr(label=prop, 
                                   values=values,
                                   eTypeID=eTypeID, 
                                   dataIDs=dataID)

    def getLinkIDs(self, linkTypeID=None):
        """
        private function to access the values of  edges
        """
        eList, _  = self._graph.outgoing(self.nID, linkTypeID)
        return eList


    def addLink(self, friendID, linkTypeID, **kwpropDict):
        """
        Adding a new connection to another node
        Properties must be provided in the correct order and structure
        """
        self._graph.addLink(linkTypeID, self.nID, friendID, attributes = tuple(kwpropDict.values()))

    def remLink(self, friendID=None, linkTypeID=None):
        """
        Removing a connection to another node
        """
        self._graph.remEdge(source=self.nID, target=friendID, eTypeID=linkTypeID)

    def remLinks(self, friendIDs=None, linkTypeID=None):
        """
        Removing mutiple connections to another node
        """        
        if friendIDs is not None:
            for friendID in friendIDs:
                self._graph.remEdge(source=self.nID, target=friendID, eTypeID=linkTypeID)


    def addToAttr(self, prop, value, idx = None):
        if idx is None:
            self.attr[prop] += value
        else:
            self.attr[prop][0, idx] += value

    def delete(self, world):
        world.graph.remNode(self.nID)
        world.deRegisterNode(self, ghost=False)
        

    def register(self, world, parentEntity=None, linkTypeID=None, ghost=False):
        
        world.registerNode(self, self.nodeTypeID, ghost=False)

        if parentEntity is not None:
            self.mpiPeers = parentEntity.registerChild(world, self, linkTypeID)
