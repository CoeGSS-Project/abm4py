#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:32:25 2018

@author: gcf
"""
import sys
sys.path.append('../../../modules')
import numpy as np
import mod_geotiff as gt
import matplotlib.pyplot as plt
import mod_geotiff as gt
#import seaborn as sns
x   = list()
y   = list()
dy  = [0]
for year in range(2005,2019):
    
    #year = 2015
    fileName = '/home/gcf/shared/processed_data/tiff/e_charging_stations_ger/charge_stations_' + str(year) + '_186x219.npy'
    chargeMap = np.load(fileName)
    x.append(year)
    y.append(np.nansum(chargeMap))
    dy.append(np.nansum(chargeMap) - np.sum(dy))
    
plt.plot(x,y)
geoFormat = gt.get_format('/home/gcf/shared/processed_data/tiff/e_charging_stations_ger/charge_stations_' + str(year) + '_186x219.tiff')
coord = geoFormat['rasterOrigin']
delta = geoFormat['s_pixel']
origin = -180,85
lonx = np.round((coord[0] - origin[0] ) / delta[0])
latx = np.round((coord[1] - origin[1] ) / delta[1])


#%%
infraMap = gt.load_array_from_tiff('/home/gcf/model_box/cars/resources_ger/road_km_all_corrected_186x219.tiff')
#infraMap = gt.load_array_from_tiff('/home/gcf/model_box/cars/resources_ger/road_km_all_new_186x219.tiff') / 1000.
popMap = gt.load_array_from_tiff('/home/gcf/model_box/cars/resources_ger/pop_counts_ww_2005_186x219.tiff')
newMap = infraMap *0
#plt.scatter(chargeMap.flatten(), infraMap.flatten(), 2)
idx = ~np.isnan(chargeMap) & ~np.isnan(infraMap)
#np.corrcoef(chargeMap[idx], infraMap[idx])
xIdx, yIdx = np.where(~np.isnan(infraMap))

nonNanidx = ~np.isnan(infraMap)
prop = infraMap[nonNanidx] / np.sum(infraMap[nonNanidx])

plt.figure('input')
plt.clf()
plt.subplot(1,3,1)
plt.imshow(popMap**2.)
plt.clim(0,np.nanpercentile(popMap**2.,99))
plt.colorbar()
plt.title('population')
plt.subplot(1,3,2)
plt.imshow(infraMap**3)
plt.clim(0,np.nanpercentile(infraMap**3,99))
plt.colorbar()
plt.title('road km per cell')
plt.subplot(1,3,3)
plt.imshow(chargeMap)
plt.clim(0,np.nanpercentile(chargeMap,99))
plt.colorbar()
plt.title('charging stations')
    #%%
#prop = popMap[~np.isnan(infraMap)] * infraMap[~np.isnan(infraMap)] 
#prop = prop / np.sum(prop)

factor2 = 2.

factors = list()
absErr = list()
sqrErr = list()
relErr = list()
for factor in np.linspace(.1,4,20):
    newMap = infraMap *0
    for i, year in enumerate(range(2010,2019)):
        
        newStations = dy[i+6]
        
        propMod = (newMap[~np.isnan(newMap)]+1)**factor * (prop**factor2)
        propMod = propMod / np.sum(propMod)
        
        randIdx = np.random.choice(range(len(prop)), int(newStations), p=propMod)
        
        uniqueRandIdx, count = np.unique(randIdx,return_counts=True)
        
        newMap[xIdx[uniqueRandIdx], yIdx[uniqueRandIdx]] += count
    
    factors.append(factor)
    absErr.append(np.nansum(np.abs(newMap - chargeMap) / float(len(xIdx))) )
    sqrErr.append(np.nansum((newMap - chargeMap)**2) / float(len(xIdx))) 
    nonZeroIdx = chargeMap > 0
    relErr.append(np.nansum((newMap[nonZeroIdx] - chargeMap[nonZeroIdx])/ chargeMap[nonZeroIdx]) ) 
plt.figure('error')   
plt.clf() 
fig, ax1 = plt.subplots(num='error')
ax1.plot(factors,absErr, 'b-')
            #ax1.set_xlabel('timeSteps (s)')
            # Make the y-axis label, ticks and tick labels match the line color.
#ax1.set_ylabel(self.columns[0], color='b')
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
ax2.cla()
ax2.plot(factors,sqrErr, 'r-')
#ax2.set_ylabel(self.columns[0], color='r')
ax2.tick_params('y', colors='r')


factor = factors[np.argmin(relErr)]
print 'argmin relative error: ' + str(factor) + ' with error: ' + str(np.min(relErr))
factor = factors[np.argmin(sqrErr)]
print 'argmin squared error: ' + str(factor) + ' with error: ' + str(np.min(sqrErr))
factor = factors[np.argmin(absErr)]
print 'argmin abolute error: ' + str(factor) + ' with error: ' + str(np.min(absErr))
#%
plt.figure('generative map')
plt.clf()
newMap = infraMap *0
for i, year in enumerate(range(2010,2019)):
    
    newStations = dy[i+6]
    
    propMod = (newMap[~np.isnan(newMap)]+1)**factor * (prop ** factor2)
    propMod = propMod / np.sum(propMod)
    
    randIdx = np.random.choice(range(len(prop)), int(newStations), p=propMod)
    
    uniqueRandIdx, count = np.unique(randIdx,return_counts=True)
    
    newMap[xIdx[uniqueRandIdx], yIdx[uniqueRandIdx]] += count  
    
    plt.subplot(3,3,i+1)
    plt.imshow(newMap)
    if np.nanpercentile(newMap,98) > 0:
        plt.clim(0, np.nanpercentile(newMap,98))
    else:
        plt.clim(0, np.nanmax(newMap))
    plt.colorbar()
 
assert np.nansum(chargeMap) ==  np.nansum(newMap)    
plt.figure('real stations')    
plt.clf()
    

plt.subplot(1,2,1)
plt.imshow(chargeMap)
plt.clim(0,np.nanpercentile(chargeMap,98))
plt.colorbar()
plt.title('real')
plt.subplot(1,2,2)
plt.imshow(newMap)
plt.clim(0,np.nanpercentile(newMap,98))
plt.colorbar()
plt.title('generated')

plt.figure('scatter')
plt.clf()
plt.scatter(chargeMap.flatten(), newMap.flatten(), 2)
plt.xlabel('real')
plt.ylabel('generated')
    