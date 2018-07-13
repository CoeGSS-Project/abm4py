#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:51:31 2018

@author: gcf
"""
from .agent import  Agent
from .traits import Parallel

class Location(Agent):

    def getGlobID(self,world):
        return next(world.globIDGen)

    def __init__(self, world, **kwProperties):
        if 'nID' not in list(kwProperties.keys()):
            nID = -1
        else:
            nID = kwProperties['nID']


        Agent.__init__(self, world, nID, **kwProperties)
    

class GhostLocation(Agent, Parallel):
    
    def getGlobID(self,world):

        return -1

    def __init__(self, world, owner, nID=-1, **kwProperties):

        Agent.__init__(self,world, nID, **kwProperties)
        self.mpiOwner = int(owner)
        
        self._setGraph(world.graph)
        if nID is not -1:
            self.nID = nID
            self.attr, self.dataID = self._graph.getNodeView(nID)
            self['instance'] = self
        self.gID = self.attr['gID'][0]

    def register(self, world, parent_Entity=None, liTypeID=None):
        Agent.register(self, world, parent_Entity, liTypeID, ghost= True)

    def registerChild(self, world, entity, liTypeID=None):
        world.addLink(liTypeID, self.nID, entity.nID)
        
        #entity.loc = self
        
    def delete(self, world):
        """ method to delete the agent from the simulation"""
        world.graph.remNode(self.nID)
        world.deRegisterAgent(self, ghost=True)        