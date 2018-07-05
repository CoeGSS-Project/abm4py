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

from .entity import Entity


class Agent(Entity):

    def __init__(self, world, **kwProperties):
        if 'nID' not in list(kwProperties.keys()):
            nID = -1
        else:
            nID = kwProperties['nID']
        
        Entity.__init__(self, world, nID, **kwProperties)
        self.mpiOwner =  int(world.mpiRank)

    def getGlobID(self,world):
        return next(world.globIDGen)

    def registerChild(self, world, entity, linkTypeID):
        """
        
        """
        if linkTypeID is not None:
            #print linkTypeID
            world.addLink(linkTypeID, self.nID, entity.nID)
        
        #entity.loc = self

        if len(self.mpiPeers) > 0: # node has ghosts on other processes
            for mpiPeer in self.mpiPeers:
                #print 'adding node ' + str(entity.nID) + ' as ghost'
                nodeTypeID = world.graph.class2NodeType(entity.__class__)
                world.papi.queueSendGhostNode( mpiPeer, nodeTypeID, entity, self)

        return self.mpiPeers


#    def getLocationAttr(self,prop):
#
#        return self.loc.node[prop]


    def _moveSpatial(self, newPosition):
        pass
        
    def _moveNormal(self, newPosition):
        self.attr['pos'] = newPosition
        
    def move(self):
        """ not yet implemented"""
        pass






class GhostAgent(Entity):
    
    def __init__(self, world, mpiOwner, nID=-1, **kwProperties):
        Entity.__init__(self, world, nID, **kwProperties)
        self.mpiOwner = int(mpiOwner)

    def register(self, world, parentEntity=None, linkTypeID=None):
        Entity.register(self, world, parentEntity, linkTypeID, ghost=True)
        

    def getGlobID(self,world):

        return None # global ID need to be acquired via MPI communication



    def registerChild(self, world, entity, linkTypeID):
        world.addLink(linkTypeID, self.nID, entity.nID)
