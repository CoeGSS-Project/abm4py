#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:51:31 2018

@author: gcf
"""
from .base_agent import BaseAgent, Entity
from .enhancements import Neighborhood

class Location(BaseAgent, Neighborhood):

    def getGlobID(self,world):
        return next(world.globIDGen)

    def __init__(self, world, **kwProperties):
        if 'nID' not in list(kwProperties.keys()):
            nID = -1
        else:
            nID = kwProperties['nID']


        BaseAgent.__init__(self, world, nID, **kwProperties)
        Neighborhood.__init__(self, world, nID, **kwProperties)
    
#    def getConnectedLocation(self, linkTypeID=1):
#        """ 
#        ToDo: check if not deprecated 
#        """
#        self.weights, _, nodeIDList = self.getLinkAttr('weig',linkTypeID=linkTypeID)
#        
#        return self.weights,  nodeIDList

class GhostLocation(Entity):
    
    def getGlobID(self,world):

        return -1

    def __init__(self, world, owner, nID=-1, **kwProperties):

        Entity.__init__(self,world, nID, **kwProperties)
        self.mpiOwner = int(owner)
        
        self._setGraph(world.graph)
        if nID is not -1:
            self.nID = nID
            self.attr, self.dataID = self._graph.getNodeView(nID)
        self.gID = self.attr['gID'][0]

    def register(self, world, parentEntity=None, linkTypeID=None):
        Entity.register(self, world, parentEntity, linkTypeID, ghost= True)

    def registerChild(self, world, entity, linkTypeID=None):
        world.addLink(linkTypeID, self.nID, entity.nID)
        
        entity.loc = self