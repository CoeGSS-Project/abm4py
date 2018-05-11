#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:32:25 2018

@author: gcf
"""
import sys
sys.path.append('../../../modules/')
import numpy as np
import mod_geotiff as gt
import matplotlib.pyplot as plt
#import mod_geotiff as gt
import seaborn as sns
xData   = list()
yData   = list()
dyData  = [0]
for year in range(2005,2019):
    
    #year = 2015
    fileName = '/home/gcf/shared/processed_data/tiff/e_charging_stations_ger/charge_stations_' + str(year) + '_186x219.npy'
    chargeMap = np.load(fileName)
    xData.append(year)
    yData.append(np.nansum(chargeMap))
    dyData.append(np.nansum(chargeMap) - np.sum(dyData))

xData.append(2020)
yData.append(70000)    
plt.plot(xData,yData)
geoFormat = gt.get_format('/home/gcf/shared/processed_data/tiff/e_charging_stations_ger/charge_stations_' + str(year) + '_186x219.tiff')
coord = geoFormat['rasterOrigin']
delta = geoFormat['s_pixel']
origin = -180,85
lonx = np.round((coord[0] - origin[0] ) / delta[0])
latx = np.round((coord[1] - origin[1] ) / delta[1])
#%%
import numpy as np
import pylab
import seaborn as sns
from scipy.optimize import curve_fit

def sigmoid(x, x0, k):
     y = (1. / (1. + np.exp(-k*(x-x0)))) * 1e6
     return y

DE = 0

xdata = np.array(xData)
ydata = np.array(yData)

popt, pcov = curve_fit(sigmoid, xdata, ydata,p0=[2020, .5])
print popt

xProjection = np.linspace(2005, 2035, 31)
yProjection = sigmoid(xProjection, *popt)

def sigmoid(x, x0, k):
     y = (1. / (1. + np.exp(-k*(x-x0)))) * 2e5
     return y

xdata = np.array(xData)
ydata = np.array(yData)

popt, pcov = curve_fit(sigmoid, xdata, ydata,p0=[2020, .5])
print popt

xProjection2 = np.linspace(2005, 2035, 31)
yProjection2 = sigmoid(xProjection2, *popt)


plt.figure('fit', figsize=(6,3))
plt.clf()

if DE:
    pylab.plot(xdata, ydata, 'o', label='Daten')
    pylab.plot(2020,70000, 'd', label='Regierungsziel')
else:
    pylab.plot(xdata, ydata, 'o', label='data')
    pylab.plot(2020,70000, 'd', label='Germany government goal')


ydata2 = [x for x in yProjection]

for i in np.arange(2020,2036)-2005:
    ydata2[i] = ydata2[15] + (i-15)*24000
if DE:
    pylab.plot(xProjection, ydata2, '--', label='BAU (linear)')
    pylab.plot(xProjection, yProjection, '--', label='Erhoehte Investitionen')
    pylab.plot(xProjection2, yProjection2, '--', label='Verminderte Investitionen')
    #plt.title('Number of electric charging stations')
    pylab.legend(loc='best')
else:
    pylab.plot(xProjection, ydata2, '--', label='BAU (linear)')
    pylab.plot(xProjection, yProjection, '--', label='high investment')
    pylab.plot(xProjection2, yProjection2, '--', label='low investment')
    #plt.title('Number of electric charging stations')
    pylab.legend(loc='best')
pylab.show()
plt.tight_layout()
if DE:    
    plt.savefig('nChargingOverTime_de.png')
else:
    plt.savefig('nChargingOverTime_en.png')
sdf
#if False:
#%%
iEnd = 10
fileName = '/home/gcf/shared/processed_data/tiff/e_charging_stations_ger/charge_stations_' + str(xdata[iEnd]) + '_186x219.npy'
chargeMap = np.load(fileName)

xList, yList = np.where(chargeMap>0)
statX, statY = [], []
for x,y in zip(xList.tolist(), yList.tolist()):
    print x,y 
    statX.extend([x]*int(chargeMap[int(x),int(y)]))
    statY.extend([y]*int(chargeMap[int(x),int(y)]))

rand = np.random.rand(len(statX))    
randIdx = np.argsort(rand)
#random ordered station
statXRand = np.asarray(statX)[randIdx]

statYRand = np.asarray(statY)[randIdx]

plt.plot(xdata[:iEnd], yProjection[:iEnd])

# regernate map
newMap = chargeMap*0
plt.figure()
plt.clf()
for ii in range(iEnd):
    nStations = int(yProjection[ii])
    
    newStat = np.asarray([statXRand[:nStations], statYRand[:nStations]])
    uniquePos, count = np.unique(newStat, axis=1, return_counts=True)
    #uniqueRandIdx, count = np.unique(randIdx,return_counts=True)
        
    newMap[uniquePos[0,:], uniquePos[1,:]] += count   
    if ii > 1:
        print ii
        plt.subplot(3,3,ii-1)
        plt.imshow(newMap)
        plt.clim([0, np.nanpercentile(newMap,98)])
        plt.colorbar()
    fileName = '../resources_ger/charge_stations_' + str(xdata[ii]) + '_186x219.npy'
    np.save(fileName, newMap)

# add okayisch data
for year in range(2015, 2018):
    fileName = '/home/gcf/shared/processed_data/tiff/e_charging_stations_ger/charge_stations_' + str(year) + '_186x219.npy'    
    chargeMap = np.load(fileName)
    if year == 2015:
        plt.subplot(3,3,9)
        plt.imshow(chargeMap)
        plt.clim([0, np.nanpercentile(newMap,98)])
        plt.colorbar()
    fileName = '../resources_ger/charge_stations_' + str(year) + '_186x219.npy'    
    np.save(fileName, chargeMap)
plt.tight_layout()
#%%new fit for simulation
xdata = 1+ (np.array(xData)-2005)*12.
popt, pcov = curve_fit(sigmoid, xdata, ydata,p0=[60., .01])
print popt

xProjection = np.linspace(0, 360, 361)
yProjection = sigmoid(xProjection, *popt)

plt.figure('fit')
plt.clf()
pylab.plot(xdata, ydata, 'o', label='data')
pylab.plot(xProjection, yProjection, label='fit')
pylab.plot(180.,70000., 'x', label='Government goal')
#pylab.ylim(0, 1.05)
plt.title('Number electric charging spots')
pylab.legend(loc='best')
pylab.show()
#%%
infraMap = gt.load_array_from_tiff('/home/gcf/model_box/cars/resources_ger/road_km_per_cell_186x219.tiff')
#infraMap = gt.load_array_from_tiff('/home/gcf/model_box/cars/resources_ger/road_km_europe_186x219.tiff') / 1000.
popMap = gt.load_array_from_tiff('/home/gcf/model_box/cars/resources_ger/pop_counts_ww_2005_186x219.tiff')
newMap = infraMap *0
infraMap[np.isnan(chargeMap)] = np.nan
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
plt.imshow(infraMap)
plt.clim(0,np.nanpercentile(infraMap,99))
plt.colorbar()
plt.title('road km per cell')
plt.subplot(1,3,3)
plt.imshow(chargeMap)
plt.clim(0,np.nanpercentile(chargeMap,99))
plt.colorbar()
plt.title('charging stations')
#%% search for best proxi
from scipy import signal
convMat = np.asarray([[0, 1, 0],[1, 2, 1],[0, 1, 0]])

convMat = np.zeros([5,5])
mid = int(np.floor(convMat.shape[1] / 2))
for x in range(convMat.shape[0]):
    for y in range(convMat.shape[0]):
        if x == mid and y == mid:
            continue
        convMat[x,y] = 1./ ((mid-x)**2 + (mid-y)**2)**.5
convMat[mid,mid] = 2        
popMap[np.isnan(popMap)] = 0
proxi = signal.convolve2d(popMap,convMat,boundary='symm',mode='same') / 14 / 1e2
plt.figure('proxi')
plt.clf()
plt.subplot(1,2,1)
plt.scatter(proxi.flatten(), chargeMap.flatten(),2)
plt.subplot(1,2,2)
plt.imshow(proxi)
plt.clim(0,np.nanpercentile(proxi,100))
plt.colorbar()
plt.title('road km per cell')
idx = ~np.isnan(chargeMap) & ~np.isnan(proxi)
np.corrcoef(proxi[idx], chargeMap[idx])
usageMap = proxi
#%%
#prop = popMap[~np.isnan(infraMap)] * infraMap[~np.isnan(infraMap)] 
#prop = prop / np.sum(prop)


def statiomGrowModel(years, newStation, potentialMap, usageMap, nSubSteps=12, potentialFactor=2.0, hotelingFactor=2.0):
    
    #nSubSteps = 12
    maxVal = 10.
    extend = list(potentialMap.shape) + [len(years)]
    
    
    mapStack = np.zeros(extend)
    currMap = mapStack[:,:,0]*0
    
    nonNanIdx = ~np.isnan(potentialMap)
    for i, year in enumerate(years):
        
        #currMap = mapStack[:,:,i]
        newStations = newStation[i] / nSubSteps
        for substep in range(nSubSteps):
            
            # imition factor
            propImmi = (currMap[nonNanIdx])**hotelingFactor
            propImmi = propImmi / np.nansum(propImmi)
            # propImmi[propImmi> maxVal] = maxVal
            propPot  = potentialMap[nonNanIdx]**potentialFactor
            propPot  = propPot / np.nansum(propPot)
            
            # dampening factor
            dampFac  = (usageMap[nonNanIdx] / (currMap[nonNanIdx]*5.))**4
            dampFac[np.isnan(dampFac)] = 1
            dampFac[dampFac > 1] = 1
            
            #old
#            propability = (propImmi + propPot) * dampFac #* (usageMap[nonNanIdx] / currMap[nonNanIdx]*14.)**2
#          
#            propability = propability / np.sum(propability)
#            
#            randIdx = np.random.choice(range(len(prop)), int(newStations), p=propability)
#            
#            uniqueRandIdx, count = np.unique(randIdx,return_counts=True)
#            
#            currMap[xIdx[uniqueRandIdx], yIdx[uniqueRandIdx]] += count   
            #new
            dampFac  = (usageMap[nonNanIdx] / (currMap[nonNanIdx]*5.))**4
            dampFac[np.isnan(dampFac)] = 1
            dampFac[dampFac > 1] = 1
            
            if np.sum(currMap[nonNanIdx]) > 0:
                propability = propImmi * dampFac
                propability = propability / np.sum(propability)
                randIdx = np.random.choice(range(len(prop)), int(newStations/2), p=propability)
                uniqueRandIdx, count = np.unique(randIdx,return_counts=True)
                currMap[xIdx[uniqueRandIdx], yIdx[uniqueRandIdx]] += count   
            
                propability = propPot * dampFac
                propability = propability / np.sum(propability)
                randIdx = np.random.choice(range(len(prop)), int(newStations/2), p=propability)
                uniqueRandIdx, count = np.unique(randIdx,return_counts=True)
                currMap[xIdx[uniqueRandIdx], yIdx[uniqueRandIdx]] += count
            else:
                propability = propPot 
                propability = propability / np.sum(propability)
                randIdx = np.random.choice(range(len(prop)), int(newStations), p=propability)
                uniqueRandIdx, count = np.unique(randIdx,return_counts=True)
                currMap[xIdx[uniqueRandIdx], yIdx[uniqueRandIdx]] += count
            
#        nDump = np.sum(dampFac < 1)            
#        if nDump > 0:
#            print nDump
            
        mapStack[:,:,i] = currMap
        
    return mapStack


#%%
years = range(2010,2019)
potFactor = 1.5

factors = list()
absErr = list()
sqrErr = list()
relErr = list()
errMap = np.zeros([11,15])
for ii,potFactor in enumerate(np.linspace(.1,2.5,11)):
    for jj, hotFactor in enumerate(np.linspace(.1,2.,15)):
        
        chargMapStack = statiomGrowModel(years, 
                         dyData[6:],
                         infraMap,
                         usageMap,
                         potentialFactor=potFactor, 
                         hotelingFactor=hotFactor)
        newMap = chargMapStack[:,:,-1]
        factors.append((potFactor, hotFactor))
        absErr.append(np.nansum(np.abs(newMap - chargeMap) / float(len(xIdx))) )
        sqrErr.append(np.nansum((newMap - chargeMap)**2) / float(len(xIdx))) 
        nonZeroIdx = chargeMap > 0
        errMap[ii,jj] = np.nansum(((newMap - chargeMap)**2) / float(len(xIdx)))
        relErr.append(np.nansum((newMap[nonZeroIdx] - chargeMap[nonZeroIdx])/ chargeMap[nonZeroIdx]) ) 
plt.figure('error')   
plt.clf()
plt.pcolormesh(errMap)
plt.xticks(range(15), [str(x)[:4] for x in np.linspace(.1,2,15)], rotation=45)
plt.yticks(range(11), [str(x)[:4] for x in np.linspace(.1,2.5,11)], rotation=45)
plt.colorbar() 
#fig, ax1 = plt.subplots(num='error')
#ax1.plot(factors,absErr, 'b-')
#            #ax1.set_xlabel('timeSteps (s)')
#            # Make the y-axis label, ticks and tick labels match the line color.
##ax1.set_ylabel(self.columns[0], color='b')
#ax1.tick_params('y', colors='b')
#ax2 = ax1.twinx()
#ax2.cla()
#ax2.plot(factors,sqrErr, 'r-')
##ax2.set_ylabel(self.columns[0], color='r')
#ax2.tick_params('y', colors='r')


factor = factors[np.argmin(relErr)]
print 'argmin relative error: ' + str(factor) + ' with error: ' + str(np.min(relErr))
factorSqr = factors[np.argmin(sqrErr)]
print 'argmin squared error: ' + str(factorSqr) + ' with error: ' + str(np.min(sqrErr))
factorAbs = factors[np.argmin(absErr)]
print 'argmin abolute error: ' + str(factorAbs) + ' with error: ' + str(np.min(absErr))
#%%



plt.figure('generative map')
plt.clf()

chargMapStack = statiomGrowModel(years, 
                 dyData[6:],
                 infraMap,
                 usageMap,
                 potentialFactor=factorSqr[0], 
                 hotelingFactor=factorSqr[1])
for i, year in enumerate(years):
    
    plt.subplot(3,3,i+1)
    plt.imshow(chargMapStack[:,:,i])
    if np.nanpercentile(chargMapStack[:,:,i],99) > 0:
        plt.clim(0, np.nanpercentile(chargMapStack[:,:,i],99))
    else:
        plt.clim(0, np.nanmax(chargMapStack[:,:,i]))
    plt.colorbar()
#    plt.imshow(newMap)

 
assert np.abs(np.nansum(chargeMap) - np.nansum(newMap)) < 100

plt.figure('real stations')    
plt.clf()

plt.subplot(1,2,1)
plt.imshow(chargeMap)
plt.clim(0,np.nanpercentile(chargeMap,99))
plt.colorbar()
plt.title('real')

plt.subplot(1,2,2)
tmp = chargMapStack[:,:,-1]
tmp[np.isnan(chargeMap)] = np.nan
plt.imshow(tmp)
plt.clim(0,np.nanpercentile(chargeMap,99))
plt.colorbar()
plt.title('generated')

plt.figure('scatter')
plt.clf()
plt.scatter(chargMapStack[:,:,-1].flatten(), chargeMap.flatten(), 2)
plt.xlabel('real')
plt.ylabel('generated')
    

#%% future projections
chargMapStack = statiomGrowModel(xProjection[1:], 
                 np.diff(yProjection),
                 infraMap,
                 usageMap,
                 potentialFactor=factorSqr[0], 
                 hotelingFactor=factorSqr[1])

for i, year in enumerate(xProjection[1:]):
    
    iSub = np.mod(i,9)+1
    if  iSub == 1:
        plt.figure('from year ' + str(year))
        plt.clf()
    plt.subplot(3,3,iSub)
    plt.imshow(chargMapStack[:,:,i])
    if np.nanpercentile(chargMapStack[:,:,i],99) > 0:
        plt.clim(0, np.nanpercentile(chargMapStack[:,:,i],99))
    else:
        plt.clim(0, np.nanmax(chargMapStack[:,:,i]))
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title(str(int(year)))

plt.figure('final 2035')
plt.clf()
plt.imshow(chargMapStack[:,:,-1])
plt.clim(0,np.nanpercentile(chargMapStack[:,:,-1],99))
plt.colorbar()
plt.title('generated charging stations 2035')    


plt.figure('real stations')    
plt.clf()

plt.subplot(1,2,1)
plt.imshow(chargeMap)
plt.clim(0,np.nanpercentile(chargeMap,99))
plt.colorbar()
plt.title('real')

plt.subplot(1,2,2)
tmp = chargMapStack[:,:,12]
tmp[np.isnan(chargeMap)] = np.nan
plt.imshow(tmp)
plt.clim(0,np.nanpercentile(chargeMap,99))
plt.colorbar()
plt.title('generated')
              