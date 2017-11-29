# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:23:00 2017

@author: gcf
"""
from lib_gcfabm import Location
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


#%% ###################### Parameters ############################

spatial    = True
connRadius = 1.5
nHunters   = 500    
nSteps     = 80
vStag      = 4.
vHare      = 1.

pT = 2*vHare/vStag

    
# Raster Data
#landLayer  = np.asarray([[0, 0, 0, 1], [0,1,1, 0], [0,1,0,0], [1,1,0,0]])
landLayer  = np.asarray([[1,1,0,0,0,0], [1,1,0,0,1,0], [0,0,1,0,0,1], [0,0,0,1,1,1],[1,1,1,0,0,0],[0,1,1,0,0,0]])


#%% ######################### Initialisation ##########################   
    
community = Community(nSteps, spatial, vStag, vHare)
connList= community.computeConnections(connRadius)
community.initSpatialLayerNew(landLayer, connList, Village)


nVillages = len(community.nodeList[_village])

for hu in range(nHunters):
    x,y = random.choice(community.locDict.keys())
    offset = (random.random() - 0.5)/5. 
    pS = max(0, min(1, 2*community.vHare/community.vStag + offset))
    hunter = Hunter(community, pS, nodeType='hu', xPos=x, yPos=y)
    hunter.connectLocation(community)
    hunter.register(community)   
#community.view()


#%% ######################### Execution ##########################   
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


#%% ######################### Plotting ##########################  

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
print len(community.nodeList[1])

plt.clf()
"""
plt.subplot(nPlots,1,1)
plt.plot(nHareHunters,'b',linewidth=2)
plt.plot(nStagHunters,'r',linewidth=2)
plt.ylim([-1,nHunters +1])
plt.legend(['# hare hunters','# stag hunters'],loc=0)
"""

for village in community.iterNode(1):
    v = village.nID
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



