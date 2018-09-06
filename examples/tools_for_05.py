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

import sys 
import os
import numpy as np
import logging as lg
import time
import random

import matplotlib.pyplot as plt
from matplotlib import collections  as mc


class PlotClass():
    
        
    def __init__(self, world, rankIDLayer):
        
        plt.ion()
        self.fig = plt.figure('output')
        self.rankIDLayer = rankIDLayer
        self.extend = world.getParameters()['extend']
        self.globalArray = np.zeros([self.extend, self.extend])
        plt.clf()
        
        
        self.gatherData(world)
        
        if world.isRoot:
            self.hh_area = plt.pcolormesh(self.globalArray, cmap='summer_r')
            
            plt.xlim(0, self.extend)
            plt.ylim(0, self.extend)
            plt.clim(0, np.max(self.globalArray))
            plt.colorbar(self.hh_area)

    def gatherData(self, world):
       
        parts = world.papi.comm.gather(world.getAttrOfAgentType('height', agTypeID=1))
        if world.isRoot:
            for rank, part in enumerate(parts):
                #print(rank)
                #print(part)
                self.globalArray[self.rankIDLayer==rank] = part
    
    def update(self, world):
        self.gatherData(world)
        if world.isRoot:
            self.hh_area.set_array(self.globalArray.flatten())
            plt.clim(np.min(self.globalArray), np.max(self.globalArray))
            plt.draw()
            self.fig.canvas.flush_events()