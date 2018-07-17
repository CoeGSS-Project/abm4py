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

from .entity import _Entity
from .traits import Parallel

class Agent(_Entity):
    """
    The most common agent type derives from the BaseAgent and additionally
    receives the abilty to move
    """
    def __init__(self, world, nID = None, **kwProperties):
        self.__getNode = world.getAgent
#        if 'nID' not in list(kwProperties.keys()):
#            nID = None
#        else:
#            nID = kwProperties['nID']

        # init of the Entity class to init storage
        _Entity.__init__(self, world, nID, **kwProperties)

    def getPeers(self, liTypeID):
        return self._graph.getOutNodeValues(self.nID, liTypeID, attr='instance')
    
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
            peerIDs  = self._graph.outgoingIDs(self.nID, liTypeID)
        elif mode == 'in':
            peerIDs  = self._graph.incommingIDs(self.nID, liTypeID)
        
        return peerIDs

    def getLinkIDs(self, liTypeID):
        """
        This method changes returns the linkID of all outgoing links of a 
        specified type
        """
        eList, _  = self._graph.outgoing(self.nID, liTypeID)
        return eList
    
        
    def getAttrOfPeers(self, attribute, liTypeID):
        """
        This method returns the attributes of all connected nodes connected 
        by a specfic edge type.
        """
        return self._graph.getOutNodeValues(self.nID, liTypeID, attr=attribute)


#self.weights, linkReferences, connectedNodes = self.getAttrOfLink('weig',liTypeID=CON_LL)                                   

#    def getAttrOfLink(self, attribute, liTypeID):
#        """
#        This method accesses the values of outgoing links
#        """
#        (eTypeID, dataID), (nTypeID, dataIDs)  = self._graph.outgoing(self.nID, liTypeID)
#        
#        nIDList = self._graph.nodes[nTypeID]['ID'][dataIDs]
#        
#        edgesValues = self._graph.getEdgeSeqAttr(label=attribute, 
#                                                 eTypeID=eTypeID, 
#                                                 dataIDs=dataID)
#
#        
#
#        return edgesValues, (eTypeID, dataID), nIDList

    def getAttrOfLink(self, attribute, liTypeID):
        """
        This method accesses the values of outgoing links
        (stf) Improve documentation and/or simplify function
        """
        return self._graph.getOutEdgeValues(self.nID, liTypeID, attribute)
        
    def setAttrOfLink(self, attribute, values, liTypeID):
        """
        This method changes the attributes of outgoing links
        """
        self._graph.setOutEdgeValues(self.nID, liTypeID, attribute, values)
        

    def addLink(self, peerID, liTypeID, **kwpropDict):
        """
        This method adds a new connection to another node. Properties must be 
        provided in the correct order and structure
        """
        self._graph.addEdge(liTypeID, self.nID, peerID, attributes = tuple(kwpropDict.values()))

    def remLink(self, peerID, liTypeID):
        """
        This method removes a link to another agent.
        """
        self._graph.remEdge(source=self.nID, target=peerID, eTypeID=liTypeID)

    def remLinks(self, peerIDs, liTypeID):
        """
        Removing mutiple links to other agents.
        """        
        [self._graph.remEdge(source=self.nID, target=peerID, eTypeID=liTypeID) for peerID in peerIDs]
                


class GhostAgent(_Entity, Parallel):
    """
    Ghost agents are only derived from Entities, since they are not full 
    agents, but passive copies of the real agents. Thus they do not have 
    the methods to act themselves.
    """
    def __init__(self, world, mpiOwner, nID=None, **kwProperties):
        
        _Entity.__init__(self, world, nID, **kwProperties)
        self.mpiOwner = int(mpiOwner)       
        self.gID = self.attr['gID'][0]
        
        
    def register(self, world, parentEntity=None, liTypeID=None):
        _Entity.register(self, world, parentEntity, liTypeID, ghost=True)
        

    def getGlobID(self,world):

        return None # global ID need to be acquired via MPI communication

    def registerChild(self, world, entity, liTypeID):
        world.addLink(liTypeID, self.nID, entity.nID)

    def delete(self, world):
        """ method to delete the agent from the simulation"""
        world.graph.remNode(self.nID)
        world.deRegisterAgent(self, ghost=True)
