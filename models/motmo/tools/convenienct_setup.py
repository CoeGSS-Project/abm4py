#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:12:38 2017

@author: gcf
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
x = np.linspace(0,60000,500)
sns.color_palette("Paired")
sns.set_color_codes("dark")
#%%

maxConvB = 0.9
minConvB = 0.2
sigmaConvB = 25000.
muConvB = 0.
plt.figure(3)
plt.clf()
ax1 = plt.subplot(2,2,1)


for kappaB in np.linspace(.9,1,10):

    

    y = minConvB + kappaB * (maxConvB - minConvB) * np.exp( - (x - muConvB)**2 / (2 * sigmaConvB**2) )
    plt.scatter(x,y,s=2)
ax1.set_xlim([0, 60000])
plt.legend(['combustion'])
ax2 = ax1.twiny()
lim = ax1.get_xlim()
ax2.set_xlim([0, 60000])
cityDict = {60000:'Berlin', 30000: 'Hamburg', 33000:'Hanover',  18000:'Karlsruhe', 21000: 'Desden', 7000:'Wolfsburg', 40000:'Stuttgart',58000:'Muenchen'}
ax2.set_xticks(cityDict.keys())  
ax2.set_xticklabels(cityDict.values(),rotation=45)

plt.ylim([0,1])
ax1 = plt.subplot(2,2,2)

minConvG = .1
maxConvGInit = 0.4
sigmaConvGInit = 20000.
sigmaConvG = 30000
muConvGInit = 35000.
muConvG = 10000

for kappaG in np.linspace(0,1,10):
    y = minConvG + (maxConvGInit-minConvG) *  (1-kappaG)  * np.exp( - (x - muConvGInit)**2 / (sigmaConvGInit**2) ) + (1-minConvG) * kappaG* (np.exp( - (x - muConvG)**2 / (2 * sigmaConvG**2) ))
    plt.scatter(x,y,s=2)
ax1.set_xlim([0, 60000])
plt.legend(['electric'])    
ax2 = ax1.twiny()
lim = ax1.get_xlim()
ax2.set_xlim([0, 60000])
cityDict = {60000:'Berlin', 30000: 'Hamburg', 33000:'Hanover',  18000:'Karlsruhe', 21000: 'Desden', 7000:'Wolfsburg', 40000:'Stuttgart',58000:'Muenchen'}
ax2.set_xticks(cityDict.keys())  
ax2.set_xticklabels(cityDict.values(),rotation=45)
plt.ylim([0,1])
#ax.set_xticks(list(ax.get_xticks()) + extraticks)




ax1 = plt.subplot(2,2,3)
sigmaConvOInit = 20000.
muConvO = 60000.
sigmaConvO = 60000


maxConvO = 0.3
minConvO = 0.05
for kappaO in np.linspace(0,1,10):
    
    y = minConvO + ((maxConvO - minConvO) * (kappaO)) * np.exp( - (x - muConvO)**2 / (2 * ((1-kappaO) * sigmaConvOInit + (kappaO * sigmaConvO))**2) )
    plt.scatter(x,y,s=2)
plt.legend(['other'])    
ax1.set_xlim([0, 60000])  
print 1
plt.ylim([0,1])
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
    techProgress.append(techProgress[-1]  * (1+growthRate[-1]))
    kappaList.append(1- (1 / techProgress[-1]))
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
