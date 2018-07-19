#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:51:31 2018

@author: gcf
"""
import numpy as np
from .agent import  Agent
from .traits import Parallel, GridNode

class Location(GridNode, Agent):


    def __init__(self, world, **kwProperties):
        if 'nID' not in list(kwProperties.keys()):
            nID = None
        else:
            nID = kwProperties['nID']


        Agent.__init__(self, world, nID, **kwProperties)
        GridNode.__init__(self, world, nID, **kwProperties)

    def __descriptor__():
        """
        This desriptor defines the agent attributes that are saved in the 
        agent graph an can be shared/viewed by other agents and acessed via 
        the global scope of the world class.
        All static and dynamic attributes can be accessed by the agent by:
            1) agent.get('attrLabel') / agent.set('attrLabel', value)
            2) agent.attr['attrLabel']
            3) agent.attr['attrLabel']
        """
        classDesc = dict()
        classDesc['nameStr'] = 'Location'
        # Static properites can be re-assigned during runtime, but the automatic
        # IO is only logging the initial state
        classDesc['staticProperties'] =  [('coord', np.int32, 2)]          
        # Dynamic properites can be re-assigned during runtime and are logged 
        # per defined time step intervall (see core.IO)
        classDesc['dynamicProperties'] = []     
        return classDesc

    def getGlobID(self,world):
        return next(world.globIDGen)
    

class GhostLocation(Agent, Parallel):
    

   
    def __init__(self, world, mpiOwner, nID=None, **kwProperties):
        
        Agent.__init__(self, world, nID, **kwProperties)
        
        self.mpiOwner = int(mpiOwner)       
        self.gID = self.attr['gID']

    def getGlobID(self,world):

        return -1        

    def register(self, world, parent_Entity=None, liTypeID=None):
        Agent.register(self, world, parent_Entity, liTypeID, ghost= True)

    def registerChild(self, world, entity, liTypeID=None):
        world.addLink(liTypeID, self.nID, entity.nID)
        
        #entity.loc = self
        
    def delete(self, world):
        """ method to delete the agent from the simulation"""
        world.graph.remNode(self.nID)
        world.deRegisterAgent(self, ghost=True)        