#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:12:38 2017

@author: gcf
"""
import sys
sys.path.append('../../../lib/')
sys.path.append('../')
sys.path = ['../../../h5py/build/lib.linux-x86_64-2.7'] + sys.path
sys.path = ['../../../mpi4py/build/lib.linux-x86_64-2.7'] + sys.path

import class_auxiliary as aux
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from bunch import Bunch
import itertools
x = np.linspace(0,4000,500)
sns.color_palette("Paired")
sns.set_color_codes("dark")

loadFile = 1

#%%


if loadFile:
    fileName = '../parameters_all.csv'
    pa = Bunch()
    for item in csv.DictReader(open(fileName)):
        if item['name'][0] != '#':
            pa[item['name']] = aux.convertStr(item['value'])
    fileName = '../parameters_ger.csv'
    for item in csv.DictReader(open(fileName)):
        if item['name'][0] != '#':
            pa[item['name']] = aux.convertStr(item['value'])
    
else:            
    pa.maxConvB = 0.9
    pa.minConvB = 0.2
    pa.sigmaConvB = 25000/10.
    pa.muConvB = 0.
plt.figure(3)
plt.clf()
ax1 = plt.subplot(2,3,1)



palette = itertools.cycle(sns.dark_palette("palegreen",10))
for kappaB in np.linspace(.9,1,10):

    

    y = pa.minConvB + kappaB * (pa.maxConvB - pa.minConvB) * np.exp( - (x - pa.muConvB)**2 / (2 * pa.sigmaConvB**2) )
    plt.scatter(x,y,s=2, c=palette.next())
ax1.set_xlim([0, 4000])
ax2 = ax1.twiny()
ax2.scatter(0,0, s=1,c='k')
ax2.legend(['combustion'])

lim = ax1.get_xlim()
ax2.set_xlim([0, 4000])
cityDict = {4000:'Berlin', 2000: 'Hamburg', 2200:'Hanover',  1200:'Karlsruhe', 1400: 'Desden', 450:'Wolfsburg', 2700:'Stuttgart',3700:'Munich'}
ax2.set_xticks(cityDict.keys())  
ax2.set_xticklabels(cityDict.values(),rotation=45)

plt.ylim([0,1])
ax1 = plt.subplot(2,3,2)

if not loadFile:
    pa.minConvG = .1
    pa.maxConvG = .9
    pa.maxConvGInit = 0.4
    pa.sigmaConvGInit = 20000.
    pa.sigmaConvG = 30000
    pa.muConvGInit = 35000.
    pa.muConvG = 10000

for kappaG in np.linspace(0,1,10):
    y = pa.minConvG + (pa.maxConvGInit-pa.minConvG) *   \
    (1-kappaG)  * np.exp( - (x - pa.muConvGInit)**2 / (pa.sigmaConvGInit**2) ) + \
    (pa.maxConvG-pa.minConvG) * kappaG* (np.exp( - (x - pa.muConvG)**2 / (2 * pa.sigmaConvG**2) ))
    
    plt.scatter(x,y,s=1, c=palette.next())
ax1.set_xlim([0, 4000])
ax2 = ax1.twiny()
ax2.scatter(0,0, s=1,c='k')
ax2.legend(['electric'])    

lim = ax1.get_xlim()
ax2.set_xlim([0, 4000])
#cityDict = {60000:'Berlin', 30000: 'Hamburg', 33000:'Hanover',  18000:'Karlsruhe', 21000: 'Desden', 7000:'Wolfsburg', 40000:'Stuttgart',58000:'Muenchen'}
ax2.set_xticks(cityDict.keys())  
ax2.set_xticklabels(cityDict.values(),rotation=45)
plt.ylim([0,1])
#ax.set_xticks(list(ax.get_xticks()) + extraticks)




ax1 = plt.subplot(2,3,3)
if not loadFile:
    pa.sigmaConvPInit = 20000.
    pa.sigmaConvP = 60000
    pa.muConvP = 4000.
    
    pa.maxConvP = 0.3
    pa.minConvP = 0.1

for kappaP in np.linspace(0,1,10):
    
    y = pa.minConvP + ((pa.maxConvP - pa.minConvP) * (kappaP)) * \
    np.exp( - (x - pa.muConvP)**2 / (2 * ((1-kappaP) * pa.sigmaConvPInit + \
                                     (kappaP * pa.sigmaConvP))**2) )
    plt.scatter(x,y,s=1, c=palette.next())

 
ax2 = ax1.twiny()
ax1.set_xlim([0, 4000])  
print 1
plt.ylim([0,1])
ax2.scatter(0,0, s=1,c='k')
ax2.legend(['public transport'])   
ax2.set_xticks(cityDict.keys())  
ax2.set_xticklabels(cityDict.values(),rotation=45)


ax1 = plt.subplot(2,3,4)
if not loadFile:
    pa.sigmaConvSInit = 3000.
    pa.muConvS = 60000.
    pa.sigmaConvS = 10000
    pa.maxConvS = 0.4
    pa.maxConvSInit = .1
    pa.minConvS = 0.05

for kappaS in np.linspace(0,1,10):
    
    y = (kappaS/10.) + pa.minConvS + (kappaS *(pa.maxConvS - pa.minConvS - (kappaS/10.))  +\
                    ((1-kappaS)* (pa.maxConvSInit - pa.minConvS - (kappaS/10.)))) * \
                    np.exp( - (x - pa.muConvS)**2 / (2 * ((1-kappaS) * \
                    pa.sigmaConvSInit + (kappaS * pa.sigmaConvS))**2) )
    plt.scatter(x,y,s=1, c=palette.next())
plt.legend(['shared mobility'])    
ax1.set_xlim([0, 4000])  
print 1
plt.ylim([0,1])

ax1 = plt.subplot(2,3,5)

if not loadFile:
    pa.sigmaConvNInit = 30000.
    pa.muConvN = 60000.
    pa.sigmaConvN = 30000
    pa.maxConvN = 0.02
    pa.minConvN = 0.00
for kappaN in np.linspace(0,1,10):
    
    y = pa.minConvN + ((pa.maxConvN - pa.minConvN) * \
                       (kappaN)) * np.exp( - (x - pa.muConvN)**2 / (2 * ((1-kappaN) * \
                                          pa.sigmaConvNInit + (kappaN * pa.sigmaConvN))**2) )
    plt.scatter(x,y,s=1, c=palette.next())
plt.legend(['none motorized'])    
ax1.set_xlim([0, 4000])  
print 1
plt.ylim([0,1])
asd
#%%

#x = np.linspace(0,1,100)
#y = x**.2
#plt.scatter(x,y)


#%%
#kappa development
# OICA data via wikipedia

prod  = np.asarray([94976569,90780583,89747430,87507027,84141209,80092840,77629127,66482439,58374162,50046000,48553969,38564516,29419484,16488340,10577426])
years = np.asarray([2016,2015,2014, 2013,2012 ,2011,2010,2005,2000,1995,	1990,1980,1970,1960,1950])
idx = np.argsort(years)

years = years[idx]
prod = prod[idx]

np.trapz(prod,years)

alltimeProduced = 2e8
kappaB = 1
techProgress=[1]
kappaList = list()
growthRate = list()
prodDict = dict()
alltimeProducedList= []
for i,year in enumerate(range(1950,2016)):
    
    if year in years:
        idx = np.where(years ==year)[0][0]
    currProd = prod[idx]
    growthRate.append((currProd) / float(alltimeProduced))
    alltimeProduced +=currProd
    techProgress.append(techProgress[-1]  * (1.+growthRate[-1]))
    kappaList.append(1.- (1. / techProgress[-1]))
    alltimeProducedList.append(alltimeProduced)
    if year == 2005:
        print techProgress[-1]
        print alltimeProduced
        print year 
        print kappaList[-1]

plt.figure(1)    
plt.clf()    
#plt.plot(kappaList)
#plt.plot(growthRate)
plt.plot(range(1950,2016),kappaList)
plt.figure(2)
plt.clf()  
plt.scatter(alltimeProducedList, kappaList)

#%% green  
delta = 1000

alltimeProduced = 1e4
kappaO = 1
techProgress=[1]
kappaList = []
alltimeProducedList = []
for i,year in enumerate(range(2005,2035)):
    delta = delta *1.5
    currProd = delta 
    growthRate.append(1 + (currProd) / float(alltimeProduced))
    alltimeProduced += currProd
    techProgress.append(techProgress[-1]  * (growthRate[-1])**.3)
    kappaList.append(1- (kappaO / techProgress[-1]))
    alltimeProducedList.append(alltimeProduced)
#plt.clf()  
plt.figure(1)  
plt.plot(range(2005,2035),kappaList) 
plt.figure(2)
plt.scatter(alltimeProducedList, kappaList)
plt.xscale('log')
#%% other

alltimeProduced = 1e6
kappaO = 1
techProgress=[3.33]
kappaList = []
alltimeProducedList = []
for i,year in enumerate(range(2005,2035)):
    
    currProd = 50000
    growthRate.append((currProd) / float(alltimeProduced))
    alltimeProduced +=currProd
    techProgress.append(techProgress[-1]  * (1+growthRate[-1]))
    kappaList.append(1- (kappaO / techProgress[-1]))
    alltimeProducedList.append(alltimeProduced)

plt.figure(1)
plt.plot(range(2005,2035),kappaList)  
plt.xticks(np.linspace(1950,2035,18),rotation=45)
plt.xlabel('years')
plt.ylabel('technical progress (1=mature)')
plt.legend(['brown (real sales)','green (est. sales)','other (est. sales)'])
plt.savefig('techProgressOverTime.png')
plt.figure(2)
plt.scatter(alltimeProducedList, kappaList)
plt.xlabel('all time produced (estimate for other)')
plt.ylabel('technical progress (1=mature)')
plt.legend(['brown','green','other'])
plt.savefig('techProgressVSproduced.png')
