#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 12:18:51 2017

@author: gcf
"""

#%% init
import sys
from os.path import expanduser
home = expanduser("~")
from copy import copy

sys.path.append(home + '/python/modules/')

import numpy as np
import mod_geotiff as gt
import matplotlib.pyplot as plt


nClusters= 60
radius = 1

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
    return deviation, devPerCluster, popPerCluster, nPopPerCluster

from scipy import signal
def  calPopSurrounging(population, clusterMap, nClusters, brush, intRad):
    connectedPop = np.zeros(nClusters)
    for i in range(nClusters):
        surrounding = signal.convolve2d(clusterMap==i, brush, mode='same')
        surrounding[clusterMap==i] = 0
        connectedPop[i] = np.nansum(population[surrounding>0])
    popSurrounding = np.sum(connectedPop)
    #print 'Connected polulation: ' + str(popSurrounding)
   
    return popSurrounding, connectedPop


population2        = gt.load_array_from_tiff('resources_NBH/pop_counts_ww_2005_62x118.tiff')/100

population = np.zeros(list(np.array(population2.shape)+2)) * np.nan
population[1:-1,1:-1] = population2
population[population==0] = 1


brush = np.zeros([int(1+ 2*(int(radius))),int(1+ 2*(int(radius)))])
intRad = int(radius)
for x in range(-intRad,intRad+1):
    for y in range(-intRad,intRad+1):
        if (x**2 +y**2)**.5 < radius:
            brush[intRad-x,intRad-y]  = 1

np.sum(np.isnan(population)==False)
clusterMap = population * np.nan
fid = open("outGraph.txt.part." + str(nClusters),'r')
x = fid.readlines()
y = [int(xx) for xx in x]
yy = np.asarray(y)
nonNan = np.isnan(population)==False
clusterMap[nonNan] = yy
plt.figure()
score1, devPerCluster, popPerCluster, nPopPerCluster = calcPopDeviation(population,clusterMap, nClusters)
score2, popSurr = calPopSurrounging(population, clusterMap, nClusters, brush, intRad)

plt.imshow(popPerCluster)
plt.colorbar()      
plt.figure()
plt.imshow(clusterMap)

# Imbalance
partSizeDesired = np.nansum(population) / nClusters
maxDeviation = np.max(np.abs(nPopPerCluster  - partSizeDesired))

print 'max devation: ', maxDeviation, ' relative: ', maxDeviation / partSizeDesired

print np.nanmin(y)
print np.nanmax(y)
np.save('resources_NBH/rankMap_nClust' + str(nClusters) + '.npy',clusterMap[1:-1,1:-1])   
