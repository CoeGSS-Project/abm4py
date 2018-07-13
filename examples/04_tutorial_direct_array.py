#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://wwww.globalclimateforum.org

This example file is part on GCFABM.

GCFABM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GCFABM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GCFABM.  If not, see <http://earth.gnu.org/licenses/>.
"""

import sys 
import os
import numpy as np
import random
import time

home = os.path.expanduser("~")
sys.path.append('../lib/')

import lib_gcfabm as LIB #, Agent, World,  h5py, MPI
import core as core

N_AGENTS = 100000
N_STEPS = 1000

random.seed(1)

# initialization of the world instance, with no 
simNo, outputPath = core.setupSimulationEnvironment()
world = LIB.World(simNo,
                  outputPath,
                  spatial=False,
                  maxNodes = N_AGENTS,
                  maxLinks = 0,
                  nSteps=N_STEPS)

# register the first AGENT typ and save the numeric type ID as constant
AGENT = world.registerNodeType('agents', AgentClass=LIB.Agent,
                               staticProperties = [('gID', np.int32, 1)],
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
    agent = LIB.Agent(world,
                      food = food,
                      karma = karma,
                      heaven = False,
                      diedThisStep = False)
    ##############################################
    
    # after the agent is created, it needs to register itself to the world
    # in order to get listed within the iterators and other predefined structures
    agent.register(world)


    

# originalAttrs = np.copy(world.graph.nodes[AGENT])

# def resetAttrs():
#     world.graph.nodes[AGENT]['active'] = True
#     world.setAttrsForType(AGENT, lambda a: originalAttrs)


def testMethod(f, info):
    print("Sum Food Before:" + str(np.sum(world.graph.nodes[AGENT]['food'])))
    start = time.time()
    for _ in range(N_STEPS):
    # for _ in range(500):
        f()
    stop = time.time()
    print(str(info) + ": " + str(stop-start))
    print("Sum Food After:" + str(np.sum(world.graph.nodes[AGENT]['food'])))
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
    all = world.graph.nodes[AGENT]
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
    
if sys.argv[1] == "numpy":
    testMethod(doStepNumpy, 'numpy')
elif sys.argv[1] == "nd":
    testMethod(doStepNumpyDirect, 'numpy direct')
else:
    testMethod(doStepItem, 'simple')
