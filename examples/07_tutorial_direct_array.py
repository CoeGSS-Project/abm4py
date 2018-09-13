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

@author: sfuerst
"""

import sys 
import os
import numpy as np
import random
import time

home = os.path.expanduser("~")

sys.path.append('../')

from abm4py import World, Location, Agent #, GhostAgent, World,  h5py, MPI
from abm4py.traits import Mobile
from abm4py import core

N_AGENTS = 10
N_STEPS = 1000

random.seed(1)

# initialization of the world instance, with no 
simNo, outputPath = core.setupSimulationEnvironment()
world = World(simNo,
              outputPath,
              maxNodes = N_AGENTS,
              maxLinks = 0,
              nSteps=N_STEPS)

# register the first AGENT typ and save the numeric type ID as constant
AGENT = world.registerAgentType(AgentClass=Agent,
                                # staticProperties = [('gID', np.int32, 1)],
                                dynamicProperties = [('food', np.int32, 1),
                                                    ('karma', np.int32, 1),
                                                    ('heaven', np.bool_, 1),
                                                    ('diedThisStep', np.bool_, 1)])

# looping over the number of AGENTs set up
for _ in range(N_AGENTS):
    # agent is assigned personal karma and food levels
    karma = random.random() * 20
    food = random.random() * N_STEPS + 1
    
    ##############################################
    # create all agents with properties

    # The init of LIB.Agent requires either the definition of all attributes 
    # that are registered (above) or none.
    agent = Agent(world,
                  food = food,
                  karma = karma,
                  heaven = False,
                  diedThisStep = False)
    ##############################################
    
    # after the agent is created, it needs to register itself to the world
    # in order to get listed within the iterators and other predefined structures
    world.registerAgent(agent)


    

# originalAttrs = np.copy(world._graph.nodes[AGENT])

# def resetAttrs():
#     world._graph.nodes[AGENT]['active'] = True
#     world.setAttrsForType(AGENT, lambda a: originalAttrs)


def testMethod(f, info):
    print("Sum Food Before:" + str(np.sum(world._graph.nodes[AGENT]['food'])))
    start = time.time()
    for _ in range(N_STEPS):
    # for _ in range(500):
        f()
    stop = time.time()
    print(str(info) + ": " + str(stop-start))
    print("Sum Food After:" + str(np.sum(world._graph.nodes[AGENT]['food'])))
    print("-----------------------")
    
    
def doStepDirect():
    for agent in world.getAgents():
        agent.attr['food'] -= 1
        if agent.attr['food'] <= 0:
            # agent.delete(world)
            agent.attr['active'] = False
            if agent.attr['karma'] > 10:
                agent.attr['heaven'] = True


def doStepAttribute():
    for agent in world.getAgents():
        agent.food -= 1
        if agent.food <= 0:
            # agent.delete(world)
            agent.active = False
            if agent.karma > 10:
                agent.heaven = True


def doStepItem():
    for agent in world.getAgents():
        agent['food'] -= 1
        if agent['food'] <= 0:
            agent.delete(world)
            # agent['active'] = False
            if agent['karma'] > 10:
                agent['heaven'] = True
                

def doStepNumpyDirect():
    all = world._graph.nodes[AGENT]
    all['food'] = all['food'] - 1
    all['heaven'] = (all['food'] < 1) & (all['karma'] > 10)
    world.deleteAgentsFilteredType(AGENT, lambda a: a['food'] < 1)
                
def doStepNumpy():
    def heavenOrHell(array):
        def goToHeaven(array):
            array['heaven'] = True
            return array
        world.setAttrsForFilteredArray(array, lambda a: a['karma'] > 10, goToHeaven)
        # array['diedThisStep'] = True
        return array

    def eat(array):
        array['food'] -= 1
        return array

    world.setAttrsForType(AGENT, eat)
    world.setAttrsForFilteredType(AGENT, lambda a: a['food'] <= 0, heavenOrHell)
    # # we must remove all agents which died this step
    # def removeAgents(array):
    #     if len(array) > 0:
    #         for aID in np.nditer(array['gID']):
    #             agent = world.getNode(globID = int(aID))
    #             agent.delete(world)
    #     array['active'] = False
    #     array['gID'] = -1
    #     return array
        
    world.deleteAgentsFilteredType(AGENT, lambda a: a['food'] <= 0)

# def doStepNumpyViaActive():
#     def heavenOrHell(array):
#         def goToHeaven(array):
#             array['heaven'] = True
#             return array
#         world.setAttrsForFilteredArray(array, lambda a: a['karma'] > 10, goToHeaven)
#         array['active'] = False
#         return array

#     def eat(array):
#         array['food'] -= 1
#         return array

#     world.setAttrsForType(AGENT, eat)
#     world.setAttrsForFilteredType(AGENT, lambda a: a['food'] <= 0, heavenOrHell)


# tests = [(doStepNumpy, "numpy"),
#          (doStepAttribute, "via item")]
#testMethod(doStepNumpy, "numpy")
# testMethod(doStepNumpyViaActive, "numpy modify active only")
# testMethod(doStepAttribute, "via attribute")
# testMethod(doStepItem, "via item")
# testMethod(doStepDirect, "direct")
# print(sys.argv[1])
    
# if sys.argv[1] == "numpy":
#     testMethod(doStepNumpy, 'numpy')
# elif sys.argv[1] == "nd":
#     testMethod(doStepNumpyDirect, 'numpy direct')
# else:
#     testMethod(doStepItem, 'simple')

all = world._graph.nodes[AGENT]

def incrFood(row):
    print(row)
    r = row['food'] + 1
    print(r)
    return r

print(all['food'])
t = world.setAttrsForTypeVectorized(AGENT, 'food', np.vectorize(incrFood))
print(all['food'])
