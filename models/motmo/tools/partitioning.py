#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 09:21:15 2017

@author: gcf
"""

#%% Splitting of Niedersachsen map for parallel compuing

#%% init
import sys
from os.path import expanduser
home = expanduser("~")
from copy import copy

sys.path.append(home + '/python/modules/')

import numpy as np
import mod_geotiff as gt
import matplotlib.pyplot as plt

# Germany
nClusters = 144
factorSurr = 30
radius = 3.5
nSimulatneously = 5
resourcePath = 'resources_ger/'
resourcePopulation = resourcePath + 'pop_counts_ww_2005_186x219.tiff'

# NBH
#nClusters = 48
#factorSurr = 30
#radius = 3.5
#nSimulatneously = 5
#resourcePath = 'resources_NBH/'
#resourcePopulation = resourcePath + 'pop_counts_ww_2005_62x118.tiff'

population2        = gt.load_array_from_tiff(resourcePopulation)/100

population = np.zeros(list(np.array(population2.shape)+2)) * np.nan
population[1:-1,1:-1] = population2
population[population==0] = 1
#population = np.fliplr(population)
#%%
plt.figure()
plt.imshow(population)    
plt.colorbar()



sumPop = int(np.nansum(population))

positions = np.zeros([sumPop,2])

currPos = 0
cellStart = []
for x in range(population.shape[0]):
    for y in range(population.shape[1]):
        
        if np.isnan (population[x,y]):
            continue
        
        nAgents = int(population[x,y])
        cellStart.append(currPos)
        positions[currPos:currPos+nAgents,:] = np.array([x,y])
        currPos = currPos + nAgents

if False:        
    import numpy as np
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=nClusters, max_iter= 1).fit(positions)
    clusters = kmeans.labels_
    
    clusterMap = population *0
    clusterMap[~ np.isnan(clusterMap)] = clusters[np.asarray(cellStart)]
    
    plt.figure()
    plt.imshow(clusterMap)
    plt.colorbar()
else:
    clusterMap = population *0
    fid = open(resourcePath +"outGraph.txt.part." + str(nClusters),'r')
    x = fid.readlines()
    y = [int(xx) for xx in x]
    yy = np.asarray(y)
    nonNan = np.isnan(population)==False
    clusterMap[nonNan] = yy
    plt.figure()
    plt.imshow(clusterMap)


def calcPopDeviation(population, clusterMap, nClusters):
    nPopPerCluster = np.zeros(nClusters)
    devPerCluster = population*np.nan
    popPerCluster = population*np.nan
    for i in range(nClusters):
        #print np.sum(population[clusterMap==i])
        nPopPerCluster[i] = np.sum(population[clusterMap==i])
        popPerCluster[clusterMap==i] = nPopPerCluster[i] 
    for i in range(nClusters):    
        devPerCluster[clusterMap==i] = (nPopPerCluster[i] - np.mean(nPopPerCluster))  
        
    deviation = np.sum(np.abs((nPopPerCluster - np.mean(nPopPerCluster)))**3)    
    #print 'Deviation: ' + str(deviation)
    return deviation, devPerCluster, popPerCluster
#%%
#plt.figure()
population2        = gt.load_array_from_tiff(resourcePopulation) / 100

population = np.zeros(list(np.array(population2.shape)+2)) * np.nan
population[1:-1,1:-1] = population2
population[population<1] = 1
#population = np.fliplr(population)
from scipy import signal


brush = np.zeros([int(1+ 2*(int(radius))),int(1+ 2*(int(radius)))])
intRad = int(radius)
for x in range(-intRad,intRad+1):
    for y in range(-intRad,intRad+1):
        if (x**2 +y**2)**.5 < radius:
            brush[intRad-x,intRad-y]  = 1

def  calPopSurrounging(population, clusterMap, nClusters, brush, intRad):
    connectedPop = np.zeros(nClusters)
    for i in range(nClusters):
        surrounding = signal.convolve2d(clusterMap==i, brush, mode='same')
        surrounding[clusterMap==i] = 0
        connectedPop[i] = np.nansum(population[surrounding>0])
    popSurrounding = np.sum(connectedPop)
    #print 'Connected polulation: ' + str(popSurrounding)
   
    return popSurrounding, connectedPop


#plt.imshow(devPerCluster)
#plt.colorbar()

useSurround = True

if useSurround:
    score1, devPerCluster, popPerCluster = calcPopDeviation(population,clusterMap, nClusters)
    score2, __ = calPopSurrounging(population, clusterMap, nClusters, brush, intRad)

    oldScore = score1 + score2*factorSurr
else:
    score1, devPerCluster, popPerCluster = calcPopDeviation(population,clusterMap, nClusters)
    oldScore = score1
    
aNCells = np.where(~np.isnan(population))
devPerCluster[aNCells]

clusterMap_old =clusterMap *1
plt.figure()
plt.subplot(1,3,1)
plt.imshow( clusterMap_old)

expectedMean = np.nansum(population) / nClusters
print 'min imbalance: ' + str( (np.nanmin(popPerCluster)) / expectedMean)
print 'max imbalance: ' + str( (np.nanmax(popPerCluster)) / expectedMean)
#%%
iMax = 100000
i = 0
success = 0
successPerDir = [0]*4

dirList = [np.array([-1,0]),np.array([1,0]),np.array([0,-1]),np.array([0,-1])]
brush0 = np.asarray([[0,-1,0],
                     [0,1,0],
                     [0,0,0]])
test = np.asarray([[0,0,0],
                     [0,1.,1.],
                     [0,2.,2.]])
nanIdx = np.isnan(population)


def getxyFromIdx(idx, xList,yList):
    x,y = xList[idx], yList[idx]
    if choice == 0:        
        x2 = x 
        y2 = y +1
    elif choice == 1:        
        x2 = x 
        y2 = y -1
    elif choice == 2:        
        x2 = x +1
        y2 = y 
    elif choice == 3:        
        x2 = x -1
        y2 = y 

    return x,y, x2, y2
oldLabels = [0]*nSimulatneously

allTimeBestScore  = np.inf
while True:
    
    i += 1 
    choice  =np.random.randint(4)
    #brush0[1+xx,1+yy] =-1
    #diff = signal.convolve2d(devPerCluster, brush0, mode='same')
    diff = population *np.nan
   
    if choice == 0:
        diff[:,0:-1] = devPerCluster[:,0:-1]- devPerCluster[:,1:]
    elif choice == 1:
        diff[:,1:] = devPerCluster[:,1:] - devPerCluster[:,0:-1]
    elif choice == 2:
        diff[0:-1,:] = devPerCluster[0:-1,:]- devPerCluster[1:,:]
    elif choice == 3:
        diff[1:,:] = devPerCluster[1:,:] - devPerCluster[0:-1,:]        
    xList, yList = np.where(diff >0)
    diff[nanIdx] = np.nan
    plt.imshow(diff)
    
    diff = np.abs(diff[diff >0])
    if len(diff) < nSimulatneously:
        continue
    
    #plt.colorbar()
    
    weights = diff / np.sum(diff)
    
    idxList = np.random.choice(range(len(diff)), size=nSimulatneously, replace=False,p=weights)

    for ii,idx in enumerate(idxList):
        x,y, x2, y2 =getxyFromIdx(idx, xList,yList)
        #devPerCluster[x,y]
        #devPerCluster[x2,y2]
            
        if not np.isnan(clusterMap[x,y]):
            
            oldLabels[ii] = clusterMap[x,y]
            clusterMap[x,y] = clusterMap[x2,y2]
    
    #check if new score is better
    if useSurround:
        score1, devPerCluster, __ = calcPopDeviation(population,clusterMap, nClusters)
        score2, __ = calPopSurrounging(population, clusterMap, nClusters, brush, intRad)
    
        newScore = score1 + score2*factorSurr
    else:
        score1, devPerCluster, __ = calcPopDeviation(population,clusterMap, nClusters)
        newScore = score1
        
    if newScore >= oldScore or 1 + 10*(1- (newScore / oldScore)) < np.random.rand():
        
        for i,idx in enumerate(idxList):
            x,y, x2, y2 =getxyFromIdx(idx, xList,yList)
            if not np.isnan(clusterMap[x,y]):
                clusterMap[x,y] = oldLabels[i]
    else:
        success +=1
        successPerDir[choice] +=1
        print str(score1) + ',' + str(score2) + '->' + str(newScore)
        oldScore = newScore
        
        if allTimeBestScore > newScore:
            allTimeBestScore = newScore
            bestClusterMap = copy(clusterMap)
    if i > iMax:
        break
            #%%
plt.figure()
plt.subplot(1,3,1)
plt.imshow( clusterMap_old)


plt.figure()
score1, devPerCluster, popPerCluster = calcPopDeviation(population,bestClusterMap, nClusters)
score2, popSurr = calPopSurrounging(population, bestClusterMap, nClusters, brush, intRad)
plt.imshow(popPerCluster)
plt.colorbar()      
plt.figure()
plt.imshow(clusterMap)
plt.figure()
plt.imshow(bestClusterMap)
plt.figure()
plt.imshow(devPerCluster)
plt.imshow(population)
plt.colorbar()   

np.save(resourcePath +'rankMap_nClust' + str(nClusters) + 'new.npy',bestClusterMap[1:-1,1:-1])   
#xx = np.load('rankMap_nClust' + str(nClusters) + '_x_radius' + str(radius) + '.npy')   

expectedMean = np.nansum(population) / nClusters
print 'min imbalance: ' + str( (np.nanmin(popPerCluster)) / expectedMean)
print 'max imbalance: ' + str( (np.nanmax(popPerCluster)) / expectedMean)
