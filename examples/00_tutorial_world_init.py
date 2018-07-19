#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 23:27:20 2018

@author: geiges
"""

#%% import modules

import sys 
import os

home = os.path.expanduser("~")
sys.path.append('../')

# lib_gcfabm offers the basic interface of the simulation tool.
# It inclueds a basic version of the world class, agent class, location class
# and the respective ghost classes for parallel execution.
from gcfabm import World

# Core comprises the many key components and utility functions that are 
# useful and necessary
from gcfabm import core

#%% Init of the world as the environment of the agents

# core.setupSimulationEnvironment is conveniently generating a simulation
# number or indentifying experiments and the simulations runs and reads or
# set ups a environment of output files
simNo, outputPath = core.setupSimulationEnvironment()

world = World(simNo,            # the simulation number 
              outputPath,       # and the output path organize the output
              maxNodes=1e4,     # optional argument to increase the maximum number of nodes per type
              maxLinks=1e5,     # optional argument to increase the maximum number of links per type
              debug=True,       # put the logging level to debug
              agentOutput=True) # agent attributes are written to file
