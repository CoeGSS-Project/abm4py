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
        
        

        coord = world.getAttrOfFilteredAgentType('coord', lambda a: a['sick'], 2)
        if len(coord) > 0:
            self.h_sicks = plt.scatter(coord[:,1],coord[:,0], c='r', s = 35, marker='s',zorder=2)
        
        coord = world.getAttrOfFilteredAgentType('coord', lambda a: a['sick'] == False, 2)
        if len(coord) > 0:
            self.h_healths = plt.scatter(coord[:,1],coord[:,0], c='g', s = 35, marker='s',zorder=2)
        

        coord = world.getAttrOfFilteredAgentType('coord', lambda a: a['remainingImmunity'] > 0, 2)
        if len(coord) > 0:
            self.h_immunes = plt.scatter(coord[:,1],coord[:,0], c='k', s = 35, marker='s',zorder=3)
        else:
            self.h_immunes = plt.scatter([],[], c='k', s = 35, marker='s',zorder=3)


        plt.xlim(0, extend)
        plt.ylim(0, extend)
        
    
        plt.subplot(1,2,2)
        
        self.sheepmax = 0
        from collections import deque
        self.sicks = deque([0]*100)
        self.healths  = deque([0]*100)
        self.immunes  = deque([0]*100)
        
        self.timeSicks = plt.plot(self.sicks, c='r')
        self.timeHealths = plt.plot(self.healths, c='g')
        self.timeImmunes = plt.plot(self.immunes, c='k')
        plt.ylim([0, 300])
        plt.legend(['number of sick people', 'number of healthy people', 'number of immune people'])
        
    def update(self, world):

        coord = world.getAttrOfFilteredAgentType('coord', lambda a: a['sick'], 2)
        if len(coord) > 0:
            self.h_sicks.set_offsets(np.c_[coord[:,1],coord[:,0]])

        coord = world.getAttrOfFilteredAgentType('coord', lambda a: a['sick'] == False, 2)
        if len(coord) > 0:
            self.h_healths.set_offsets(np.c_[coord[:,1],coord[:,0]])


        if len(coord) > 0:
            self.h_immunes.set_offsets(np.c_[coord[:,1],coord[:,0]])
        
        plt.draw()
        
        self.sicks.popleft()
        self.sicks.append(world.countAgents(lambda a: a['sick'], 2))
        
        self.healths.popleft()
        self.healths.append(world.countAgents(lambda a: a['sick'] == False, 2))
        
        self.immunes.popleft()


        self.immunes.append(world.countAgents(lambda a: a['remainingImmunity'] > 0, 2))

                    
        self.timeSicks[0].set_ydata(self.sicks)
        self.timeHealths[0].set_ydata(self.healths)
        self.timeImmunes[0].set_ydata(self.immunes)
        
        self.fig.canvas.flush_events()
        #self.fig2.canvas.flush_events()
        plt.draw()
