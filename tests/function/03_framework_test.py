#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 09:59:15 2018

@author: andreas geiges, gcf
"""

# MINIMAL FUNCTION TEST

import sys
import numpy as np

import abm4py 
core = abm4py.core

outputPath = '.'
simNo= 0


core.configureLogging(outputPath, debug=False)
#core.configureSTD(outputPath) 
    

class Cell(abm4py.Location, abm4py.traits.Parallel):

    def __init__(self, world, **kwAttr):
        abm4py.Location.__init__(self, world, **kwAttr)
        abm4py.traits.Parallel.__init__(self, world, **kwAttr)

class Agent (abm4py.Agent, abm4py.traits.Parallel):

    def __init__(self, world, **kwAttr):
        abm4py.Agent.__init__(self, world, **kwAttr)
        abm4py.traits.Parallel.__init__(self, world, **kwAttr)

import os 
try:
    os.remove(outputPath + '/nodeOutput.hdf5')
except:
    pass


earth = abm4py.World(simNo,
              outputPath,
              nSteps=10,
              agentOutput=0,
              debug=True)

print(earth.isParallel)

print(earth.papi.comm.rank)

#%% global variables and statistic test

earth._graph.glob.registerValue('maxtest' , np.asarray([earth.papi.comm.rank]),'max')
earth._graph.glob.registerStat('meantest', np.random.randint(5,size=3).astype(float),'mean')
earth._graph.glob.registerStat('stdtest', np.random.randint(5,size=2).astype(float),'std')
print(earth._graph.glob.globalValue['maxtest'])
print(earth._graph.glob.globalValue['meantest'])
print('mean of values: ',earth._graph.glob.localValues['meantest'],'-> local mean: ',earth._graph.glob.globalValue['meantest'])
print('std od values:  ',earth._graph.glob.localValues['stdtest'],'-> local std: ',earth._graph.glob.globalValue['stdtest'])

earth._graph.glob.sync()
print('global mean: ', earth._graph.glob.globalValue['maxtest'])
print('global mean: ', earth._graph.glob.globalValue['meantest'])
print('global std: ', earth._graph.glob.globalValue['stdtest'])


mpiRankLayer   = np.asarray([[0, 0, 0, 0, 1],
                          [np.nan, np.nan, np.nan, 1, 1]])
gridMask = (~np.isnan(mpiRankLayer)).astype(int)

#landLayer = np.load('rankMap.npy')

#print connList
CELL    = earth.registerAgentType(AgentClass=Cell, 
                                  GhostAgentClass=abm4py.GhostLocation, 
                                  agTypeStr= 'cell' ,
                                  staticProperties = [('gID', np.int32, 1),
                                                     ('coord', np.int16, 2)],
                                  dynamicProperties = [('value', np.float32, 1),
                                                      ('value2', np.float32, 1)])

AG      = earth.registerAgentType(AgentClass=Agent, 
                                  GhostAgentClass=abm4py.GhostAgent,
                                  agTypeStr = 'agent',
                                  staticProperties   = [('gID', np.int32, 1),
                                                       ('coord', np.int16, 2)],
                                  dynamicProperties  = [('value3', np.float32, 1)])

C_LOLO = earth.registerLinkType('cellCell', CELL, CELL, [('weig', np.float32, 1)])
C_LOAG = earth.registerLinkType('cellAgent', CELL, AG)
C_AGAG = earth.registerLinkType('AgAg', AG, AG, [('weig', np.float32, 1)])

earth.registerGrid(CELL, C_LOLO)   
connList = earth.grid.computeConnectionList(1.5)
earth.grid.init(gridMask, connList, Cell, mpiRankLayer)


for cell in earth.random.shuffleAgentsOfType(CELL):
    cell.attr['value'] = earth.papi.rank
    cell.attr['value2'] = earth.papi.rank+2

    if cell.attr['coord'][0] == 0:
        x,y = cell.attr['coord']
        agent = Agent(earth, value3=np.random.randn(), coord=(x,  y))
        #print 'agent.nID' + str(agent.nID)
        agent.register(earth, cell, C_LOAG)
        #cell.registerEntityAtLocation(earth, agent,_cLocAg)

#earth.queue.dequeueVertices(earth)
#earth.queue.dequeueEdges(earth)
#            if agent.node['nID'] == 10:
#                agent.addLink(8,_cAgAg)

#earth.papi.syncNodes(CELL,['value', 'value2'])
earth.papi.updateGhostAgents([CELL])
print(earth._graph.getPropOfNodeType(CELL, 'names'))
print(str(earth.papi.rank) + ' values' + str(earth._graph.nodes[CELL]['value']))
print(str(earth.papi.rank) + ' values2: ' + str(earth._graph.nodes[CELL]['value2']))

#print earth.papi.ghostNodeRecv
#print earth.papi.ghostNodeSend

print(earth._graph.getPropOfNodeType(AG, 'names'))

print(str(earth.papi.rank) + ' ' + str(earth.getAgentDict()[AG]))

print(str(earth.papi.rank) + ' SendQueue ' + str(earth.papi.ghostNodeQueue))

earth.papi.transferGhostAgents(earth)

cell.getPeerIDs(agTypeID=CELL, mode='out')

print(str(earth.papi.rank) + ' ' + str(earth._graph.nodes[AG].indices()))
print(str(earth.papi.rank) + ' ' + str(earth._graph.nodes[AG]['value3']))

for agent in earth.random.shuffleAgentsOfType(AG):
    agent.attr['value3'] = earth.papi.rank+ agent.nID
    assert agent.attr['value3'] == earth.papi.rank+ agent.nID
#
earth.papi.updateGhostAgents([AG])
#
#earth.io.initAgentFile(earth, [CELL, AG])
#
#earth.io.writeAgentDataToFile(0, [CELL, AG])
#
#print(str(earth.papi.rank) + ' ' + str(earth._graph.nodes[AG]['value3']))

#%% testing agent methods 
nPeers = cell.countPeers(C_LOLO)
writeValues = np.asarray(list(range(nPeers))).astype(np.float32)
for peer, value in zip(cell.getPeers(C_LOLO), writeValues):
    peer.attr['value'] = value
    
readValues = cell.getAttrOfPeers('value', C_LOLO)
assert all(readValues == writeValues)
peerList = cell.getPeerIDs(C_LOLO)
assert all(earth._graph.getNodeSeqAttr('value', peerList,) == writeValues)
print('Peer values write/read successful')

edgeList = cell.getLinkIDs(C_LOLO)
writeValues = np.random.random(len(edgeList[1])).astype(np.float32)
cell.setAttrOfLink('weig',writeValues, C_LOLO)
readValues = cell.getAttrOfLink('weig',C_LOLO)
assert all(readValues == writeValues)
print('Edge values write/read successful')

friendID = earth.getAgentDict()[AG][0]
agent.addLink(friendID, C_AGAG, weig=.51)
assert earth._graph.isConnected(agent.nID, friendID, C_AGAG)
readValue = agent.getAttrOfLink('weig',C_AGAG)
assert readValue[0] == np.float32(0.51)

agent.remLink(friendID, C_AGAG)
assert not(earth._graph.isConnected(agent.nID, friendID, C_AGAG))
print('Adding/removing connection successfull')

value = agent.attr['value3'].copy()
agent.attr['value3'] +=1
assert agent.attr['value3'] == value +1
assert earth.getAttrOfAgents('value3', agent.nID) == value +1
print('Value access and increment sucessful')

#%%
pos = (0,4)
cellID = earth._graph.IDArray[pos]
cell40 = earth.getAgent(cellID)
agentID = cell40.getPeerIDs(liTypeID=C_LOAG, mode='out')
print(cell40)
print(agentID)
print(earth._graph.nodes[2][earth._graph.nodes[2]['active']])
agent = cell40.getPeers(liTypeID=C_LOAG)
print(agent[0].nID)
print(agent[0].gID)

connAgent = earth.getAgent(agentID[0])
assert np.all(cell40.attr['coord'] == connAgent.attr['coord'])
print(cell40)
if earth.papi.rank == 1:
    cell40.attr['value'] = 32.0
    connAgent.attr['value3'] = 43.2
    
earth.papi.updateGhostAgents([CELL])
earth.papi.updateGhostAgents([AG],['value3'])


buff =  earth.papi.all2all(cell40.attr['value'])
print(buff)
assert buff[0] == buff[1]
print('ghost update of cells successful (all attributes) ')

buff =  earth.papi.all2all(connAgent.attr['value3'])
assert buff[0] == buff[1]
print('ghost update of agents successful (specific attribute)')
