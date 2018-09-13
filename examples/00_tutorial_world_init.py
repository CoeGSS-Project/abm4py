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

@author: ageiges
"""

#%% import modules

import sys 
import os

home = os.path.expanduser("~")
sys.path.append('../')

# lib_abm4py offers the basic interface of the simulation tool.
# It inclueds a basic version of the world class, agent class, location class
# and the respective ghost classes for parallel execution.
from abm4py import World

# Core comprises the many key components and utility functions that are 
# useful and necessary
from abm4py import core

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
