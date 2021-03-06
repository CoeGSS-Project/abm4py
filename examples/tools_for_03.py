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
    
        
    def __init__(self, world):
        
        plt.ion()
        self.fig = plt.figure('spatial')
        plt.clf()
        plt.subplot(1,2,1)
        extend = world.getParameters()['extend']
        
        grass = np.reshape(world.getAttrOfAgentType('height', agTypeID=1),[extend, extend])
        
        
        self.hh_area = plt.pcolormesh(grass, cmap='summer_r',zorder=-1)
        
        coord = world.getAttrOfAgentType('coord', agTypeID = 2)
        #print(pos.shape)
        self.hh_sheeps = plt.scatter(coord[:,1],coord[:,0], c='w', s = 35, marker='s',zorder=2)
        self.hh_wolfs  = plt.scatter(coord[:,1],coord[:,0], c='k', s = 35, marker='s',zorder=2)
        plt.xlim(0, extend)
        plt.ylim(0, extend)
        plt.clim(0,1)
        plt.colorbar(self.hh_area)
    
        plt.subplot(1,2,2)
        
        self.sheepmax = 0
        from collections import deque
        self.grHeig  = deque([0]*100)
        self.sheeps = deque([0]*100)
        self.wolfs  = deque([0]*100)
        self.timesGrass = plt.plot(self.grHeig)
        self.timeSheeps = plt.plot(self.sheeps)
        self.timesWolfs = plt.plot(self.wolfs)
        
        plt.ylim([0 ,1500])
        plt.legend(['Amount of grass / 10', 'number of sheeps', 'number of wolfs'])
        
    def update(self, world):

        coord = world.getAttrOfAgentType('coord', agTypeID = 2)
        
        self.hh_sheeps.set_offsets(np.c_[coord[:,1],coord[:,0]])
        coord = world.getAttrOfAgentType('coord', agTypeID = 3)
        self.hh_wolfs.set_offsets(np.c_[coord[:,1],coord[:,0]])
        grass = world.getAttrOfAgentType('height', agTypeID=1)
        self.hh_area.set_array(grass)
        plt.draw()
        sumGrassHeight = np.sum(grass/10)
        self.grHeig.popleft()
        self.grHeig.append(sumGrassHeight)
        self.sheeps.popleft()
        self.sheeps.append(world.countAgents(2))
        self.wolfs.popleft()
        self.wolfs.append(world.countAgents(3))

        self.timesGrass[0].set_ydata(self.grHeig)            
        self.timeSheeps[0].set_ydata(self.sheeps)
        self.timesWolfs[0].set_ydata(self.wolfs)
        
        self.fig.canvas.flush_events()
        #self.fig2.canvas.flush_events()
        plt.draw()
