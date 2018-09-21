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

import numpy as np
from .agent import  Agent
from .traits import Parallel, GridNode

class Location(GridNode, Agent):


    def __init__(self, world, **kwProperties):
        if 'ID' not in list(kwProperties.keys()):
            ID = None
        else:
            ID = kwProperties['ID']


        Agent.__init__(self, world, ID, **kwProperties)
        GridNode.__init__(self, world, ID, **kwProperties)

    def __descriptor__():
        """
        This desriptor defines the agent attributes that are saved in the 
        agent._graph an can be shared/viewed by other agents and acessed via 
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
    

   
    def __init__(self, world, mpiOwner, ID=None, **kwProperties):
        
        Agent.__init__(self, world, ID, **kwProperties)
        
        self.mpiOwner = int(mpiOwner)       
        self.gID = self.attr['gID']

    def getGlobID(self,world):

        return -1        

    def register(self, world, parent_Entity=None, liTypeID=None):
        Agent.register(self, world, parent_Entity, liTypeID, ghost= True)

    def registerChild(self, world, entity, liTypeID=None):
        world.addLink(liTypeID, self.ID, entity.ID)
        
        #entity.loc = self
        
    def delete(self, world):
        """ method to delete the agent from the simulation"""
        world._graph.remNode(self.ID)
        world.removeAgent(self, ghost=True)        