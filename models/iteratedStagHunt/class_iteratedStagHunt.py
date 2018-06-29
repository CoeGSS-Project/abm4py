# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:47:51 2017

@author: gcf
"""

from lib_gcfabm import World, Agent, Location
#import igraph as ig
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copy import copy
import random
import tqdm
import time
import os



#%%

# -------------------------------- Community --------------------------------
class Community(World):
  
    def __init__(self, nSteps, spatial, outputPath, vStag, vHare):
        World.__init__(self, spatial, outputPath)
        self.nSteps     = nSteps
        self.vStag      = vStag
        self.vHare      = vHare
        #self.nVillages  = 0
        self.nHunters   = 0
        self.time       = 0


    def computeConnections(self,radius=1):
        connList = []          
        intRad = int(radius)
        for x in range(-intRad,intRad+1):
            for y in range(-intRad,intRad+1):
                if (x**2 +y**2)**.5 < radius:
                    if x**2 +y**2 > 0:
                        weig  = 1/((x**2 +y**2)**.5)
                    else:
                        weig = 10
                    connList.append((x,y,weig))
        return connList


    def getInhabitants(self,village):
        inhabitants = []
        for hunter in self.iterNode(2):
            if hunter.loc == village:
                inhabitants.append(hunter)
        return inhabitants
        
    
    def chooseHunterPairs(self, hunterIDs):

        huIds = copy(hunterIDs)
        huPairs = list()
        
        for i in range(0,int(len(huIds)/2)):
            first = huIds[0]
            del huIds[0]
            j = random.randint(1,len(huIds))-1
            huPairs.append([first,huIds[j]])            
            del huIds[j]
            
        if len(huIds) > 0:
            last = huIds[0]
            huPairs.append([last,-1])
            
        return huPairs

    """    
    def makeHuntersGraph(self):
        
        huntersGraph = ig.Graph(directed=True)
        huntersGraph.es['weight']=1
        
        for hunter in self.iterNode(2):           
            hID = hunter.nID
            huntersGraph.add_vertices(str(hID))
            nei, wei = hunter.neighboursWithWeights(self)
            for j in range(len(nei)):
                huntersGraph.add_vertex(str(nei[j]))
                huntersGraph[str(hID),str(nei[j])]=wei[j]

        return huntersGraph
        """

    def huntersMeet(self):
        
        destDict = {}
        for village in self.iterNode(1):
            vID = village.nID
            destDict[vID] = []            
        for hunter in self.iterNode(2):
            destID = hunter.chooseHuntingDest()
            destDict[destID].append(hunter.nID)
            
        return destDict

        
    def hunterPairsLocal(self):
        
        destDict = self.huntersMeet()
        hunterPairs = []
        
        for village in self.iterNode(1):
            hunterList = destDict[village.nID]
            localList = self.chooseHunterPairs(hunterList)
            for n in range(len(localList)):
                hunterPairs.append(localList[n])
        
        return hunterPairs
        
        
    def stagHunt(self, local=True):
           
        if local:
            pairs = self.hunterPairsLocal()
        else:
            hunterIDs = self.nodeList[2]
            pairs = self.chooseHunterPairs(hunterIDs)
            
        stagHuntersMean = -1.
        hareHuntersMean = -1.
        nStagHunters = 0
        nHareHunters = 0
        stagHuntersPayoffs = []
        hareHuntersPayoffs = []
        
        for i in range(0,len(pairs)):
            
            hunter1 = self.entDict[pairs[i][0]]
            
            if pairs[i][1] == -1 : 
                payoff1 = self.vHare
                pref1 = hunter1.getValue('huPref')
                hareHuntersPayoffs.append(self.vHare)
                hunter1.results[0].append(payoff1)
                hunter1.results[1].append(pref1)
                
            else:
                hunter2 = self.entDict[pairs[i][1]]           
            
                if hunter1.getValue('huPref') + hunter2.getValue('huPref') == 1:   # one stag hunter, one hare hunter
                    if hunter1.getValue('huPref') == 1:
                        payoff1 = 0
                        pref1 = 1
                        other1 = 0.
                        payoff2 = self.vHare
                        pref2 = 0
                        other2 = 1.
                    else:
                        payoff1 = self.vHare
                        pref1 = 0
                        other1 = 1.
                        payoff2 = 0
                        pref2 = 1
                        other2 = 0.
                    stagHuntersPayoffs.append(0.)
                    hareHuntersPayoffs.append(self.vHare)
                
                elif hunter1.getValue('huPref') + hunter2.getValue('huPref') == 2: # both stag hunters
                    payoff1 = self.vStag/2
                    payoff2 = self.vStag/2
                    pref1 = 1
                    pref2 = 1
                    other1 = 1
                    other2 = 1
                    stagHuntersPayoffs.append(self.vStag/2)
                    stagHuntersPayoffs.append(self.vStag/2)
                
                elif hunter1.getValue('huPref') + hunter2.getValue('huPref') == 0: # both hare hunters                                                                               # both hare hunters
                    payoff1 = self.vHare
                    payoff2 = self.vHare
                    pref1 = 0
                    pref2 = 0
                    other1 = 0.
                    other2 = 0.
                    hareHuntersPayoffs.append(self.vHare)
                    hareHuntersPayoffs.append(self.vHare)
                                
                hunter1.results[0].append(payoff1)
                hunter2.results[0].append(payoff2)
                
                hunter1.results[1].append(pref1)
                hunter2.results[1].append(pref2)
            
                hunter1.setValue('pS', hunter1.updatePs(other1))
                hunter2.setValue('pS', hunter2.updatePs(other2))
            
                if hunter1.getValue('pS')>2*self.vHare/self.vStag:
                    hunter1.setValue('huPref', 1)
                else:
                    hunter1.setValue('huPref', 0)
                
                if hunter2.getValue('pS')>2*self.vHare/self.vStag:
                    hunter2.setValue('huPref', 1)
                else:
                    hunter2.setValue('huPref', 0)           
            
        if len(stagHuntersPayoffs)>0:
            stagHuntersMean = sum(stagHuntersPayoffs)/float(len(stagHuntersPayoffs))
            nStagHunters = len(stagHuntersPayoffs)
        
        if len(hareHuntersPayoffs)>0:
            hareHuntersMean = sum(hareHuntersPayoffs)/float(len(hareHuntersPayoffs))
            nHareHunters = len(hareHuntersPayoffs)
            
        return (stagHuntersMean, hareHuntersMean, nStagHunters, nHareHunters)
            
        



#%%
# -------------------------------- Hunter --------------------------------

class Hunter(Agent):
    
    def __init__(self, community, pS, nodeTypeID = 'hu', xPos = np.nan, yPos = np.nan):
        Agent.__init__(self, community, nodeTypeID, xPos, yPos)
        if pS > 2*community.vHare/community.vStag:
            huPref = 1
        else:
            huPref = 0
        self.setValue('huPref', huPref)
        self.setValue('pS', pS)
        self.results = [[],[]]


    def updatePs(self, observation):
        w = 10.
        pNew = (w*self.getValue('pS') + 1*observation)/(w+1)
        return pNew

        
    def village(self):
        village = self.loc
        return village.nID, village

        
    def chooseHuntingDest(self):        
        myVillage = self.loc
        weights, edges, connectedVs = myVillage.getConnectedVillages()
        cumWeights = np.cumsum(weights)
        r = random.random()        
        i = 0            
        while cumWeights[i]<r:
            i+=1
        destinationID = connectedVs[i]        
        return destinationID

        
    """    
    def weightToOtherHunter(self, hunter, community):
        myVillageID, myVillage = self.village()
        otherVillageID, otherVillage = hunter.villageID()
        weight = 0
        if hunter.nID in self.neighbouringHunters(community):
            wei, edg, con = myVillage.getConnectedVillages()
            idx = con.index(otherVillageID)
            weight = wei[idx]
        return weight
        
        
    def neighbouringHunters(self, community):
        nIds = list()
        #nWeights = list()
        nei, ids = self.getNeigbourhood(3)
        for i in iter(community.nodeList[2]):
            if i in ids and (i is not self.nID):
                nIds.append(i)
        return nIds    
    
                
    def neighboursWithWeights(self, community):
        myVillageID, myVillage = self.village()
        neiIds = list()
        neiWeights = list()
        nei, ids = self.getNeigbourhood(3)
        
        for i in iter(community.nodeList[2]):
            if i in ids and (i is not self.nID):
                neiIds.append(i)                
                neighbour = community.entDict[i]
                otherVillageID, otherVillage = neighbour.village()
                wei, edg, con = myVillage.getConnectedVillages()
                idx = con.index(otherVillageID)
                neiWeights.append(wei[idx])
                
        return neiIds, neiWeights   
        """

      
# -------------------------------- Village --------------------------------

class Village(Location):
    
    def __init__(self, Community,  xPos, yPos):
        Location.__init__(self, Community,  xPos, yPos)
        self.huList = list()


    def getConnectedVillages(self):
        self.weights, self.eIDs = self.getConnProp('weig')
        self.connNodeList = [self.graph.es[x].target for x in self.eIDs ]
        return self.weights, self.eIDs, self.connNodeList
 
            
            
 