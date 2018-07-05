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
sys.path.append('../../lib/')

# lib_gcfabm offers the basic interface of the simulation tool.
# It inclueds a basic version of the world class, agent class, location class
# and the respective ghost classes for parallel execution.
import lib_gcfabm as LIB 

# Core comprises the many key components and utility functions that are 
# useful and necessary
import core as core

#%% Init of the world as the environment of the agents

# core.setupSimulationEnvironment is conveniently generating a simulation
# number or indentifying experiments and the simulations runs and reads or
# set ups a environment of output files
simNo, outputPath = core.setupSimulationEnvironment()

world = LIB.World(simNo,        # the simulation number 
              outputPath,       # and the output path organize the output
              spatial=True,     
              nSteps=10,
              maxNodes=1e4,
              maxLinks=1e5,
              debug=True,
              wAgentOutput=True)