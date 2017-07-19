#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:16:37 agStart17

@author: gcf
"""

#%% 
agStart = 20
agEnd   = 80
nRepeat = 2
import time
import igraph as ig
import sys
sys.path.append("/media/sf_shared/python/synEarth/agModel")
import os
import tqdm
from bunch import Bunch
from class_auxiliary  import convertStr
from init_mobilityABM import  initEarth, mobilitySetup, householdSetup, cellTest, generateNetwork, scenarioTestMedium
import matplotlib.pyplot as plt
import seaborn


dirPath = "/media/sf_shared/python/synEarth/agModel"
fileName = '../parameters_med.csv'
parameters = Bunch()
for item in csv.DictReader(open(fileName)):
    parameters[item['name']] = convertStr(item['value'])
print 'Setting loaded:'
print parameters.toDict()

if 'earth' not in locals():
    
    
    parameters = scenarioTestMedium(parameters, dirPath)
    parameters['calibration'] = False
    parameters.showFigures = False
            
    earth = initEarth(parameters)
    
    mobilitySetup(earth, parameters)
    
    householdSetup(earth, parameters)
    
    generateNetwork(earth, parameters)

plt.clf()
plt.subplot(2,2,1)

print '########################'
print 'Testing reading values call'
legStr = []  
plt.subplot(2,2,1)
plt.title('read')
repeatSeq = [1, 5, 25, 100, 250]
print '########################'
j = agStart
agent = earth.entDict[j]
x =agent.getValue('util')
times = []
values = []
ott = time.time()
for nRepeat in repeatSeq:
    tt = time.time()
    for j in range(agStart,agEnd):
        agent = earth.entDict[j]
        for i in range(nRepeat):
            values.append(agent.getValue('income'))
    times.append(time.time() -tt)
print 'getValue()   ' + str(time.time() -ott )  
plt.plot(repeatSeq, times)
legStr.append("getValue()")

times = []
values2 = []
ott = time.time()
for nRepeat in repeatSeq:
    tt = time.time()
    for j in range(agStart,agEnd):
        #agent = earth.entDict[j]
        for i in range(nRepeat):
            values2.append(agent.graph.vs[j]['income'])
    times.append(time.time() -tt )
print 'direct graph access   ' + str(time.time() -ott )  
plt.plot(repeatSeq, times)
legStr.append("direct graph access")
    
times = []
values3 = []
ott = time.time()
for nRepeat in repeatSeq:
    tt = time.time()
    for j in range(agStart,agEnd):
        agent = earth.entDict[j]
        for i in range(nRepeat):
            values3.append(agent.node['income'])
    times.append(time.time() -tt )
print 'access node[]   ' + str(time.time() -ott )   
plt.plot(repeatSeq, times)
legStr.append("access node[]")            

times = []
ott = time.time()
values4 = []
for nRepeat in repeatSeq:
    tt = time.time()
    for j in range(agStart,agEnd):
        agent = earth.entDict[j]
        for i in range(nRepeat):
            values4.append(agent._alt1_getValue('income'))
    times.append(time.time() -tt )
print '_alt1   ' + str(time.time() -ott )   
 
plt.plot(repeatSeq, times)
legStr.append("_alt1")    
plt.legend(legStr)  

assert values == values2
assert values == values3
assert values == values4

print '########################'
print 'Write values call'
legStr = []  
plt.subplot(2,2,2)
plt.title('write')
repeatSeq = [1, 5, 25, 100]
print '########################'

times = []
ott = time.time()
for nRepeat in repeatSeq:
    tt = time.time()
    for j in range(agStart,agEnd):
        agent = earth.entDict[j]
        for i in range(nRepeat):
            x =agent.setValue('util',i)
    times.append(time.time() -tt )
print 'setValue()   ' + str(time.time() -ott )    
plt.plot(repeatSeq, times)
legStr.append("setValue()")    
plt.legend(legStr)  

times = []
ott = time.time()
for nRepeat in repeatSeq:
    tt = time.time()
    for j in range(agStart,agEnd):
        agent.graph.vs[j]
        for i in range(nRepeat):
            agent.node['util'] = i
    times.append(time.time() -tt )
print 'access node[] ' + str(time.time() -ott )  
plt.plot(repeatSeq, times)
legStr.append("access node[]")    
plt.legend(legStr)  

times = []
ott = time.time()
for nRepeat in repeatSeq:
    tt = time.time()
    for j in range(agStart,agEnd):
        agent = earth.entDict[j]
        for i in range(nRepeat):
            agent._alt1_setValue('util',i)
    times.append(time.time() -tt )
print 'old1   ' + str(time.time() -ott )    
plt.plot(repeatSeq, times)
legStr.append("_alt1")    
plt.legend(legStr)  

times = []
ott = time.time()
for nRepeat in repeatSeq:
    tt = time.time()
    for j in range(agStart,agEnd):
        agent.graph.vs[j]
        for i in range(nRepeat):
            agent.node.update_attributes({'util':i, 'income':j})
    times.append(time.time() -tt )
print 'update_attributes() ' + str(time.time() -ott )  
plt.plot(repeatSeq, times)
legStr.append("update_attributes()")    
plt.legend(legStr)  

print '########################'
print 'neigbour node values call'
_hh  = 2
_chh = 3
legStr = []  
plt.subplot(2,2,3)
plt.title('neigbour node values')
repeatSeq = [1, 5, 25, 100]
print '########################'
agent = earth.entDict[67]
#assert agent.getConnNodeValuesNew('util',nodeType=_pers) ==agent.getConnNodeValues('util',nodeType=_pers)
#assert agent.getConnNodeValuesNew('util',nodeType=_pers) ==agent._alt2_getConnNodeValues('util',nodeType=_pers)
#assert agent.getConnNodeValuesNew('util',nodeType=_pers) ==agent._alt1_getConnNodeValues('util',edgeType=_cpp)

times = []
ott = time.time()
for nRepeat in repeatSeq:
    tt = time.time()
    for j in range(agStart,agEnd):
        agent = earth.entDict[j]
        for i in range(nRepeat):
            x,e = agent._alt3_getConnNodeValues('util',nodeType=_pers)
    times.append(time.time() -tt )
        #weig = earth.graph.es[e]['weig']
print 'getConnNodeValuesNew()  ' + str(time.time() -tt )   
plt.plot(repeatSeq, times)
legStr.append("getConnNodeValuesNew()")    
plt.legend(legStr)  


times = []
ott = time.time()
for nRepeat in repeatSeq:
    tt = time.time()
    for j in range(agStart,agEnd):
        agent = earth.entDict[j]
        for i in range(nRepeat):
            x,e = agent.getConnNodeValues('util',nodeType=_pers)
    times.append(time.time() -tt )
        #weig = earth.graph.es[e]['weig']
print 'getConnNodeValues()  ' + str(time.time() -tt )   
plt.plot(repeatSeq, times)
legStr.append("getConnNodeValues()")    
plt.legend(legStr)  

times = []
ott = time.time()
for nRepeat in repeatSeq:
    tt = time.time()
    for j in range(agStart,agEnd):
        agent.graph.vs[j]
        for i in range(nRepeat):
            x = agent._alt2_getConnNodeValues('util',nodeType=_hh)
    times.append(time.time() -tt )
print '_alt2_getConnNodeValues  ' + str(time.time() -tt )    
plt.plot(repeatSeq, times)
legStr.append("_alt2_getConnNodeValues")    
plt.legend(legStr)  

times = []
ott = time.time()
for nRepeat in repeatSeq:
    tt = time.time()
    for j in range(agStart,agEnd):
        agent = earth.entDict[j]
        for i in range(nRepeat):
            x = agent._alt1_getConnNodeValues('util',edgeType=_chh)
    times.append(time.time() -tt )
print '_alt1_getConnNodeValues  ' + str(time.time() -tt )  
plt.plot(repeatSeq, times)
legStr.append("_alt1_getConnNodeValues()")    
plt.legend(legStr)  

if False:
#%% tests
    agent = earth.entDict[61]
    
    tt = time.time()
    neighIDs = agent.graph.neighborhood(agent.nID)        
    time.time() -tt
    tt = time.time()
    neighIDs = [ neig['name'] for neig in agent.node.neighbors()]
    time.time() -tt
    tt = time.time()
    neigbors = earth.graph.vs[neighIDs].select(type=3)
    time.time() -tt
    
    tt = time.time()
    neigbors = earth.graph.vs[neighIDs].select(type=3)
    time.time() -tt
    
    tt = time.time()
    neigbors = earth.graph.vs[neighIDs].select(type_eq=3)
    time.time() -tt
    

#%%


print '########################'
print ' neigbour edge values call'
plt.subplot(2,2,4)
plt.title('neigbour edge value')
repeatSeq = [1, 5, 25, 100]
legStr = []
print '########################'
assert agent.getEdgeValuesFast('weig',_chh)[0] == agent.getEdgeValues('weig',_chh)[0]
assert agent.getEdgeValuesFast('weig',_cpp)[0] == agent._alt1_getEdgeValues('weig',_cpp)[0]

times = []
ott = time.time()
for nRepeat in repeatSeq:
    tt = time.time()
    for j in range(agStart,agEnd):
        agent = earth.entDict[j]
        for i in range(nRepeat):
            x = agent.getEdgeValuesFast('weig',_chh)
    times.append(time.time() -tt )
print 'getEdgeValuesFast  ' + str(time.time() -tt ) 
plt.plot(repeatSeq, times)
legStr.append("getEdgeValuesFast()")    
plt.legend(legStr)  

times = []
ott = time.time()
for nRepeat in repeatSeq:
    tt = time.time()
    for j in range(agStart,agEnd):
        agent = earth.entDict[j]
        for i in range(nRepeat):
            x = agent.getEdgeValues('weig',_chh)
    times.append(time.time() -tt )
print 'getEdgeValues  ' + str(time.time() -tt ) 
plt.plot(repeatSeq, times)
legStr.append("getEdgeValues()")    
plt.legend(legStr)  


times = []
ott = time.time()
for nRepeat in repeatSeq:
    tt = time.time()
    for j in range(agStart,agEnd):
        agent = earth.entDict[j]
        for i in range(nRepeat):
            x = agent._alt4_getEdgeValues('weig',_chh)
    times.append(time.time() -tt )
print 'alt4  ' + str(time.time() -tt ) 
plt.plot(repeatSeq, times)
legStr.append("alt4()")    
plt.legend(legStr)  


times = []
ott = time.time()
for nRepeat in repeatSeq:
    tt = time.time()
    for j in range(agStart,agEnd):
        agent.graph.vs[j]
        for i in range(nRepeat):
            x = agent._alt3_getEdgeValues('weig',_chh)
    times.append(time.time() -tt )
print 'alt3  ' + str(time.time() -tt )
plt.plot(repeatSeq, times)
legStr.append("_alt3")    
plt.legend(legStr)  

times = []
ott = time.time()
for nRepeat in repeatSeq:
    tt = time.time()
    for j in range(agStart,agEnd):
        agent = earth.entDict[j]
        for i in range(nRepeat):
            x = agent._alt2_getEdgeValues('weig',_chh)
    times.append(time.time() -tt )
print '_alt2  ' + str(time.time() -tt )   
plt.plot(repeatSeq, times)
legStr.append("_alt2")    
plt.legend(legStr)  

times = []
ott = time.time()
for nRepeat in repeatSeq:
    tt = time.time()
    for j in range(agStart,agEnd):
        agent = earth.entDict[j]
        for i in range(nRepeat):
            x = agent._alt1_getEdgeValues('weig',_chh)
    times.append(time.time() -tt )
print '_alt1  ' + str(time.time() -tt )  
plt.plot(repeatSeq, times)
legStr.append("_alt1")    
plt.legend(legStr)  

print "get connected neigbours"
plt.title('get connected nodes')
for j in range(agStart,agEnd):
    agent = earth.entDict[j]
    for i in range(nRepeat):
        ag = agent.getConnNodes(_chh,mode="IN")
    
print 'atl1  ' + str(time.time() -tt ) 

tt = time.time()
for j in range(agStart,agEnd):
    agent = earth.entDict[j]
    for i in range(nRepeat):
        ag = agent.getConnNodeIDs("IN", _hh)
print 'alt2  ' + str(time.time() -tt ) 

tt = time.time()
for j in range(agStart,agEnd):
    agent = earth.entDict[j]
    for i in range(nRepeat):
        x,y = agent.getConnNodeValues('name',_hh, mode='in')
print 'alt3  ' + str(time.time() -tt ) 
print "location stuff"

tt = time.time()
loc = earth.entDict[0]
for i in range(nRepeat):
        x = loc.getAgentOfCell(2)
print 'agentofcell  ' + str(time.time() -tt ) 

tt = time.time()
loc = earth.entDict[0]
for i in range(nRepeat):
        x = loc.getConnNodeIDs(nodeType=_hh, mode='in')
print 'enity meth  ' + str(time.time() -tt ) 


#%%
print '########################'
print ' market stuff'
plt.subplot(2,2,4)
plt.title('neigbour edge value')
repeatSeq = [1, 5, 25, 100]
print '########################'
earth.graph.vs.select(earth.nodeDict[_pers])['util'] = np.random.randn(1910)
tt = time.time()
np.mean(earth.graph.vs.select(earth.nodeDict[_pers])['util'])
    
array = np.random.randn(2,1910)
tt = time.time()
mean = np.mean(array[:,1:],axis=0)          
print 'comparions access array: ' + str(time.time() -tt)

tt = time.time()
persons = earth.graph.vs.select(earth.nodeDict[_pers])
tt = time.time()
np.mean(persons['util'])
print 'graph.vs.select:' +str(time.time() -tt)

tt = time.time()
persons = earth.graph.vs[earth.nodeDict[_pers]]
tt = time.time()
np.mean(persons['util'])
print 'indexing of graph' + str(time.time() -tt)


tt = time.time()
persons = earth.graph.vs[earth.nodeDict[_pers]]
np.mean(np.asarray(persons['preferences']),axis=0)
print 'mean of elements of lists: ' + str(time.time() -tt)

persons['preferences3'] = persons['preferences']
tt = time.time()
persons = earth.graph.vs[earth.nodeDict[_pers]]
np.mean(np.asarray(persons['preferences2']),axis=0)
print 'mean of elements of lists: ' + str(time.time() -tt)