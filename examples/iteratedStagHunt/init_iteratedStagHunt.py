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

@author: gsteudle
"""
from lib_abm4py import Location, GhostLocation
from class_iteratedStagHunt import Community, Hunter, Village
import numpy as np
import matplotlib.pyplot as plt
import random
import math

#%%

# node types
_inactive = 0
_village  = 1
_hunter   = 2


#%% ################################# Parameters ##########################################

spatial    = True
connRadius = 1.5
nHunters   = 500    
nSteps     = 80
vStag      = 4.
vHare      = 1.

pT = 2*vHare/vStag

outputPath = "./" # core.createOutputDirectory(mpiComm, baseOutputPath, simNo)
    
# Raster Data
#landLayer  = np.asarray([[0, 0, 0, 1], [0,1,1, 0], [0,1,0,0], [1,1,0,0]])
landLayer  = np.asarray([[1,1,0,0,0,0], [1,1,0,0,1,0], [0,0,1,0,0,1], [0,0,0,1,1,1],[1,1,1,0,0,0],[0,1,1,0,0,0]])


#%% ##################################### Initialisation #######################################   
    
community = Community(nSteps, spatial, outputPath, vStag, vHare)
connList= community.computeConnections(connRadius)
community.spatial.initSpatialLayer(landLayer, connList, Village, Location, GhostLocation)


nVillages = len(community.nodeList[_village])

for hu in range(nHunters):
    x,y = random.choice(community.locDict.keys())
    offset = (random.random() - 0.5)/5. 
    pS = max(0, min(1, 2*community.vHare/community.vStag + offset))
    hunter = Hunter(community, pS, agTypeID='hu', xPos=x, yPos=y)
    hunter.connectLocation(community)
    hunter.register(community)   
#community.view()


#%% ##################################### Execution #######################################   
#stagMeans = list()
#hareMeans = list()
nStagHunters = list()
nHareHunters = list()

for i in range(nSteps):
    s, h, nS, nH = community.stagHunt(local=True)
    #stagMeans.append(s)
    #hareMeans.append(h)
    nStagHunters.append(nS)
    nHareHunters.append(nH)


#%% ##################################### Plotting #######################################  

def plotHuntersResults(hunters):
    tEnd = len(hunters[0].results[1])    
    nStagHunters = []
    nHareHunters = []    
    for t in range(tEnd):
        nSH = 0
        nHH = 0        
        for i in range(len(hunters)):    
            prefs = hunters[i].results[1]
            if prefs[t]==1:
                nSH += 1
            elif prefs[t]==0:
                nHH += 1
        nStagHunters.append(nSH)
        nHareHunters.append(nHH)
    return nStagHunters, nHareHunters


nPlots = len(community.nodeList[1])
print(len(community.nodeList[1]))

plt.clf()
"""
plt.subplot(nPlots,1,1)
plt.plot(nHareHunters,'b',linewidth=2)
plt.plot(nStagHunters,'r',linewidth=2)
plt.ylim([-1,nHunters +1])
plt.legend(['# hare hunters','# stag hunters'],loc=0)
"""

for village in community.iterNode(1):
    v = village.ID
    cols = 3
    rows = int((nPlots+1)/cols)
    inhabitants = community.getInhabitants(village)
    nIn = len(inhabitants)
    nStagHunters, nHareHunters = plotHuntersResults(inhabitants)    
    plt.subplot(rows,cols,v+1)
    plt.title('village: '+str(v)+', inhabitants: '+str(nIn))
    plt.plot(nHareHunters,'b',linewidth=2)
    plt.plot(nStagHunters,'r',linewidth=2)
    plt.ylim([-1,len(inhabitants) +1])
    plt.legend(['hare hunters','stag hunters'],loc=0)

plt.subplots_adjust(top=0.96,bottom=0.04,left=0.04,right=0.96,hspace=0.45,wspace=0.1)



