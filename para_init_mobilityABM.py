#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://wwww.globalclimateforum.org

MOBILITY INNOVATION MARKET MODEL
-- INIT FILE --

This file is part of GCFABM.

GCFABM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GCFABM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GCFABM.  If not, see <http://earthw.gnu.org/licenses/>.


TODOs

short-term:
    - utility test center
    - realistic physical values
    - test learing niches 
    - I/O on lustre
    - utility potential for omnicent knowledge
    - MCMC evolution of the synthetic population
        - add/remove households and draw acceptance accourding to available statistics
        - what about if no data is available ??? -> extrapolation of statistics
    
    - (done)combine self-experience with community-experience
    - entropy on networks (distribution of weights)
        - entire network
        - per agent
    - save connections + properties to output
    - synthetic poplulations of bremen and hamburg
    - (done) include communication of experiences to ghost agents !!!!!!
    - add price incentive (4000 euro in 2017)
    
long-term:
    - cut connections with near-zero weights
    - replace removed connections with new ones
        - rebuffering of sequences of nodes and edges
    
    - join the synthetic people of niedersachsen, bremen and hamburg
    - jointly create their income
    - place the correct people to the differen regions (different hh distributions)
    - add advertising to experience
    
"""

#%%
#TODO
# random iteration (even pairs of agents)
#from __future__ import division
import matplotlib
matplotlib.use('Agg')
import sys, os
from os.path import expanduser
home = expanduser("~")
#sys.path.append(home + '/python/decorators/')
sys.path.append(home + '/python/modules/')
sys.path.append(home + '/python/agModel/modules/')
#from deco_util import timing_function
import numpy as np
import time
#import mod_geotiff as gt
from para_class_mobilityABM import Person, GhostPerson, Household, GhostHousehold, Reporter, Cell, GhostCell,  Earth, Opinion
from mpi4py import  MPI
import class_auxiliary  as aux #convertStr

import matplotlib.pylab as plt
import seaborn as sns; sns.set()
sns.set_color_codes("dark")
#import matplotlib.pyplot as plt

import pandas as pd
from bunch import Bunch
from copy import copy
import csv
from scipy import signal


overallTime = time.time()
###### Enums ################
#connections
_cll = 1 # loc - loc
_clh = 2 # loc - household
_chh = 3 # household, household
_chp = 4 # household, person
_cpp = 5 # person, person

#nodes
_cell = 1
_hh   = 2
_pers = 3

#time spans
_month = 1
_year  = 2

#%% Scenario definition without calibraton parameters

def scenarioTestSmall(parameterInput, dirPath):
    setup = Bunch()
    
    #general 
    setup.resourcePath = dirPath + '/resources_nie/'
    setup.synPopPath = setup['resourcePath'] + 'hh_niedersachsen.csv'
    setup.progressBar  = True
    setup.allTypeObservations = False
    
    #time
    setup.timeUint         = _month  # unit of time per step
    setup.startDate        = [01,2005]   

    #spatial
    setup.reductionFactor = 50000
    setup.isSpatial       = True
    setup.connRadius      = 1.5     # radíus of cells that get an connection
    setup.landLayer   = np.asarray([[ 1     , 1, 1 , np.nan, np.nan],
                                    [ np.nan, 1, 0 , np.nan, 0     ],
                                    [ np.nan, 1, 0 , 0     , 0     ]])
    if mpiSize == 1:
        setup.landLayer = setup.landLayer*0
    setup.regionIdRaster    = setup.landLayer*1518
    setup.regionIdRaster[0:,0:2] = 6321
    setup.population = (np.isnan(setup.landLayer)==0)* np.random.randint(3,5,setup.landLayer.shape)
    
    #social
    setup.addYourself   = True     # have the agent herself as a friend (have own observation)
    setup.recAgent      = []       # reporter agents that return a diary
    
    #output
    setup.writeOutput   = 1
    setup.writeNPY      = 1
    setup.writeCSV      = 0
    
    #cars and infrastructure
    setup.properties    = ['emmisions','TCO']

    #agents
    setup.randomAgents     = False
    setup.omniscientAgents = False    

    # redefinition of setup parameters used for automatic calibration
    setup.update(parameterInput.toDict())
    
    # calculate dependent parameters
    maxDeviation = (setup.urbanCritical - setup.urbanThreshold)**2
    minCarConvenience = 1 + setup.kappa
    setup.convB =  minCarConvenience / (maxDeviation)
 
    # only for packing with care
    #dummy   = gt.load_array_from_tiff(setup.resourcePath + 'land_layer_62x118.tiff')
    #dummy   = gt.load_array_from_tiff(setup.resourcePath + 'pop_counts_ww_2005_62x118.tiff') / setup.reductionFactor
    #dummy   = gt.load_array_from_tiff(setup.resourcePath + 'subRegionRaster_62x118.tiff')    
    #del dummy
    
    print "Final setting of the parameters"
    print parameterInput
    print "####################################"
    
    return setup


    
def scenarioTestMedium(parameterInput, dirPath):

    
    setup = Bunch()


    #general 
    setup.resourcePath = dirPath + '/resources_nie/'
    setup.synPopPath = setup['resourcePath'] + 'hh_niedersachsen.csv'
    setup.allTypeObservations = False
    
    setup.progressBar  = True
    
    #time
    setup.timeUint         = _month  # unit of time per step
    setup.startDate        = [01,2005]   

        
    #spatial
    setup.reductionFactor = 5000 # only and estimation in comparison to niedersachsen
    setup.isSpatial     = True
    setup.landLayer   = np.asarray([[0, 0, 0, 0, 1, 1, 1, 0, 0], 
                                    [0, 1, 0, 0, 0, 1, 1, 1, 0],
                                    [1, 1, 1, 0, 0, 1, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1, 1, 0, 0],
                                    [1, 1, 1, 1, 0, 0, 0, 0, 0]])
    
    setup.regionIdRaster    = setup.landLayer*1518
    setup.regionIdRaster[3:,0:3] = 6321

    convMat = np.asarray([[0,1,0],[1,0,1],[0,1,0]])
    setup.population = setup.landLayer* signal.convolve2d(setup.landLayer,convMat,boundary='symm',mode='same')
    setup.population = 20*setup.population+ setup.landLayer* np.random.randint(1,10,setup.landLayer.shape)*2
    
    setup.landLayer  = setup.landLayer.astype(float)
    setup.landLayer[setup.landLayer== 0] = np.nan
    setup.landLayer[:,:5] = setup.landLayer[:,:5]*0
    #setup.landLayer[:,:1] = setup.landLayer[:,:1]*0
    #setup.landLayer[:,4:5] = setup.landLayer[:,4:5]*2
    #setup.landLayer[:,6:] = setup.landLayer[:,6:]*3
    

    
    
    #social
    setup.addYourself   = True     # have the agent herself as a friend (have own observation)
    setup.recAgent      = []       # reporter agents that return a diary
    
    #output
    setup.writeOutput   = 1
    setup.writeNPY      = 1
    setup.writeCSV      = 0
    
    #cars and infrastructure
    setup.properties    = ['emmisions','TCO']

    #agents
    setup.randomAgents     = False
    setup.omniscientAgents = False    

    # redefinition of setup parameters used for automatic calibration
    setup.update(parameterInput.toDict())
    
    # calculate dependent parameters
    maxDeviation = (setup.urbanCritical - setup.urbanThreshold)**2
    minCarConvenience = 1 + setup.kappa
    setup.convB =  minCarConvenience / (maxDeviation)
    
    # only for packing with care
    #dummy   = gt.load_array_from_tiff(setup.resourcePath + 'land_layer_62x118.tiff')
    
    #dummy   = gt.load_array_from_tiff(setup.resourcePath + 'pop_counts_ww_2005_62x118.tiff') / setup.reductionFactor
    #dummy   = gt.load_array_from_tiff(setup.resourcePath + 'subRegionRaster_62x118.tiff')    
    
    #del dummy
    
    print "Final setting of the parameters"
    print parameterInput
    print "####################################"
    
    return setup
    
    
def scenarioNiedersachsen(parameterInput, dirPath):
    setup = Bunch()
    
    #general 
    setup.resourcePath = dirPath + '/resources_nie/'
    setup.progressBar  = True
    setup.allTypeObservations = False
    
    setup.synPopPath = setup['resourcePath'] + 'hh_niedersachsen.csv'
    #time
    setup.nSteps           = 340     # number of simulation steps
    setup.timeUint         = _month  # unit of time per step
    setup.startDate        = [01,2005]
    setup.burnIn           = 100
    setup.omniscientBurnIn = 10       # no. of first steps of burn-in phase with omniscient agents, max. =burnIn
    
    #spatial
    setup.isSpatial     = True
    #setup.connRadius    = 3.5      # radíus of cells that get an connection
    setup.reductionFactor = 200.
    
    if hasattr(parameterInput, "reductionFactor"):
        # overwrite the standart parameter
        setup.reductionFactor = parameterInput.reductionFactor
        
    if mpiSize > 1:
        setup.landLayer = np.load(setup.resourcePath + 'rankMap_nClust' + str(mpiSize) + '.npy')
    else:
        setup.landLayer = setup.landLayer * 0
    
    
    print 'max rank:',np.nanmax(setup.landLayer)
    
    setup.population        = gt.load_array_from_tiff(setup.resourcePath + 'pop_counts_ww_2005_62x118.tiff') / setup.reductionFactor
    setup.regionIdRaster    = gt.load_array_from_tiff(setup.resourcePath + 'subRegionRaster_62x118.tiff')
    
    
    if False:
        try:
            #plt.imshow(setup.landLayer)
            plt.imshow(setup.population,cmap='jet')
            plt.clim([0, np.nanpercentile(setup.population,90)])
            plt.colorbar()
        except:
            pass
    setup.landLayer[np.isnan(setup.population)] = np.nan
    nAgents    = np.nansum(setup.population)
    
    #social
    setup.addYourself   = True     # have the agent herself as a friend (have own observation)
    setup.recAgent      = []       # reporter agents that return a diary
    
    #output
    setup.writeOutput   = 1
    setup.writeNPY      = 1
    setup.writeCSV      = 0
    
    #cars and infrastructure
    setup.properties    = ['emmisions','TCO']

    #agents
    setup.randomAgents     = False
    setup.omniscientAgents = False    

    # redefinition of setup parameters used for automatic calibration
    setup.update(parameterInput.toDict())
    
    # calculate dependent parameters
    maxDeviation = (setup.urbanCritical - setup.urbanThreshold)**2
    minCarConvenience = 1 + setup.kappa
    setup.convB =  minCarConvenience / (maxDeviation)
    
    print "Final setting of the parameters"
    print parameterInput
    print "####################################"
    
    #assert np.sum(np.isnan(setup.population[setup.landLayer==1])) == 0
    print 'Running with ' + str(nAgents) + ' agents'
    
    return setup

def scenarioNBH(parameterInput, dirPath):
    setup = Bunch()
    
    #general 
    setup.resourcePath = dirPath + '/resources_NBH/'
    setup.synPopPath = setup['resourcePath'] + 'hh_NBH_1M.csv'
    setup.progressBar  = True
    setup.allTypeObservations = True
    
    #time
    setup.nSteps           = 340     # number of simulation steps
    setup.timeUint         = _month  # unit of time per step
    setup.startDate        = [01,2005]
    setup.burnIn           = 100
    setup.omniscientBurnIn = 10       # no. of first steps of burn-in phase with omniscient agents, max. =burnIn
    
    #spatial
    setup.isSpatial     = True
    #setup.connRadius    = 3.5      # radíus of cells that get an connection
    #setup.reductionFactor = parameterInput['reductionFactor']
    
    if hasattr(parameterInput, "reductionFactor"):
        # overwrite the standart parameter
        setup.reductionFactor = parameterInput.reductionFactor

            
    #
    #setup.landLayer[np.isnan(setup.landLayer)] = 0
    if mpiSize > 1:
        setup.landLayer = np.load(setup.resourcePath + 'rankMap_nClust' + str(mpiSize) + '.npy')
    else:
        
        setup.landLayer=  np.load(setup.resourcePath + 'land_layer_62x118.npy')
        setup.landLayer = setup.landLayer * 0
       
    print 'max rank:',np.nanmax(setup.landLayer)
    
    #setup.population        = gt.load_array_from_tiff(setup.resourcePath + 'pop_counts_ww_2005_62x118.tiff') 
    setup.population = np.load(setup.resourcePath + 'land_layer_62x118.npy')
    #setup.regionIdRaster    = gt.load_array_from_tiff(setup.resourcePath + 'subRegionRaster_62x118.tiff')
    setup.regionIdRaster = np.load(setup.resourcePath + 'subRegionRaster_62x118.npy')
    
    if False:
        try:
            #plt.imshow(setup.landLayer)
            plt.imshow(setup.population,cmap='jet')
            plt.clim([0, np.nanpercentile(setup.population,90)])
            plt.colorbar()
        except:
            pass
    setup.landLayer[np.isnan(setup.population)] = np.nan
    
    
    #social
    setup.addYourself   = True     # have the agent herself as a friend (have own observation)
    setup.recAgent      = []       # reporter agents that return a diary
    
    #output
    setup.writeOutput   = 1
    setup.writeNPY      = 1
    setup.writeCSV      = 0
    
    #cars and infrastructure
    setup.properties    = ['emmisions','TCO']

    #agents
    setup.randomAgents     = False
    setup.omniscientAgents = False    

    # redefinition of setup parameters from file
    setup.update(parameterInput.toDict())
    
    # Correciton of population depend parameter by the reduction factor
    setup['initialGreen'] /= setup['reductionFactor']
    setup['initialBrown'] /= setup['reductionFactor']
    setup['initialOther'] /= setup['reductionFactor']
    
    setup['population']     /= setup['reductionFactor']
    setup['urbanThreshold'] /= setup['reductionFactor']
    setup['urbanCritical']  /= setup['reductionFactor']
    setup['puplicTransBonus']  /= setup['reductionFactor']
    setup.convD /= setup['reductionFactor']
    
    # calculate dependent parameters
    maxDeviation = (setup.urbanCritical - setup.urbanThreshold)**2
    minCarConvenience = 1 + setup.kappa
    setup.convB =  minCarConvenience / (maxDeviation)
    
    
    print "Final setting of the parameters"
    print parameterInput
    print "####################################"
    
    nAgents    = np.nansum(setup.population)
    #assert np.sum(np.isnan(setup.population[setup.landLayer==1])) == 0
    print 'Running with ' + str(nAgents) + ' agents'
    
    return setup
    
def scenarioChina(calibatationInput):
    pass
    
    
###############################################################################
###############################################################################


# Mobility setup setup
def mobilitySetup(earth, parameters):
    import math
    def convienienceBrown(popDensity, paraA, paraB, paraC ,paraD, cell):
         if popDensity<cell.urbanThreshold:
            conv = paraA
         else:
            conv = max(0.2,paraA - paraB*(popDensity - cell.urbanThreshold)**2)          
         return conv
            
    def convienienceGreen(popDensity, paraA, paraB, paraC ,paraD, cell):
        conv = max(0.1, paraA - paraB*(popDensity - cell.urbanThreshold)**2 + cell.kappa  )
        return conv
    
    def convienienceOther(popDensity, paraA, paraB, paraC ,paraD, cell):
        conv = max(0.05 ,paraC/(1+math.exp((-paraD)*(popDensity-cell.urbanThreshold + cell.puplicTransBonus))))
        return conv
    
                         #(emmisions, TCO)         
    earth.initBrand('brown',(440., 200.), convienienceBrown, 'start', earth.para['initialBrown']) # combustion car
    
    earth.initBrand('green',(350., 450.), convienienceGreen, 'start', earth.para['initialGreen']) # green tech car

    earth.initBrand('other',(120., 100.), convienienceOther, 'start', earth.para['initialOther'])  # none or other
            
    earth.para['nMobTypes'] = len(earth.enums['brands'])
    
    return earth
    ##############################################################################

def householdSetup(earth, parameters, calibration=False):
    tt = time.time()
    idx = 0
    nAgents = 0
    nHH     = 0

    boolMask = parameters['landLayer']==earth.mpi.comm.rank
    nAgentsOnProcess = np.sum(parameters.population[boolMask])

    nAgentsPerProcess = earth.mpi.all2all(nAgentsOnProcess)
    
    # calculate start in the agent file (20 overhead for complete households)
    agentStart = int(np.sum(nAgentsPerProcess[:earth.mpi.comm.rank]) + earth.mpi.comm.rank*20)
    agentEnd   = int(np.sum(nAgentsPerProcess[:earth.mpi.comm.rank+1]) + (earth.mpi.comm.rank+1)*20)
    
    print 'Reading agents from ' + str(agentStart) + ' to ' + str(agentEnd)
    
    if earth.debug:
        print 'Vertex count: ',earth.graph.vcount()
        earth.view(str(earth.mpi.rank) + '.png')
    if not parameters.randomAgents:

        hhMat = pd.read_csv(parameters.synPopPath, skiprows = agentStart, nrows= (agentEnd - agentStart)).values
        print 'size of hhMat: ',hhMat.shape

    # find the correct possition in file 
    nPers = hhMat[idx,4] 
    if np.sum(np.diff(hhMat[idx:idx+nPers,4])) !=0:
        
        #new index for start of a complete household
        idx = idx + np.where(np.diff(hhMat[idx:idx+nPers,4]) !=0)[0][0]
        
    opinion =  Opinion(earth)
    nAgentsCell = 0
    for x,y in earth.locDict.keys():
        #print x,y
        nAgentsCell = int(parameters.population[x,y]) + nAgentsCell # subtracting Agents that are places too much in the last cell
        #print nAgentsCell
        loc = earth.entDict[earth.locDict[x,y].nID]
        while True:
             
            #creating persons as agents
            nPers = hhMat[idx,4]    
            #print nPers,'-',nAgents
            ages    = list(hhMat[idx:idx+nPers,12])
            genders = list(hhMat[idx:idx+nPers,13])
            income = hhMat[idx,16]
            income *= parameters.mobIncomeShare
            nKids = np.sum(ages<18)
            
            # creating houshold
            hh = Household(earth, pos= (x, y),
                                  hhSize=nPers,
                                  nKids=nKids,
                                  income=income,
                                  expUtil=0,
                                  util=0,
                                  expenses=0)
            hh.adults = list()
            hh.register(earth, parentEntity=loc, edgeType=_clh)
            #hh.registerAtLocation(earth,x,y,_hh,_clh)
            
            hh.loc.node['population'] += nPers
            
            for iPers in range(nPers):

                nAgentsCell -= 1
                nAgents     += 1
                
                if ages[iPers]< 18:
                    continue    #skip kids
                prefTuple = opinion.getPref(ages[iPers],genders[iPers],nKids,nPers,income,parameters.radicality)
                prefTyp = np.argmax(prefTuple)
                
                pers = Person(earth, preferences =prefTuple,
                                     hhID        = hh.gID,
                                     prefTyp     = prefTyp,
                                     gender     = genders[iPers],
                                     age         = ages[iPers],
                                     util        = 0.,
                                     commUtil    = [50.]*earth.para['nMobTypes'],
                                     selfUtil    = [np.nan]*earth.para['nMobTypes'],
                                     mobType     = 0,
                                     prop        = [0]*len(parameters.properties),
                                     consequences= [0]*len(prefTuple),
                                     lastAction  = 0,
                                     ESSR        = 1,
                                     innovatorDegree = np.random.randn())
                
                pers.register(earth, parentEntity=hh, edgeType=_chp)
                #pers.queueConnection(hh.nID,edgeType=_chp)
                #pers.registerAtLocation(earth, x,y)
                
                # adding reference to the person to household
                hh.adults.append(pers)
            
                earth.nPrefTypes[prefTyp] += 1

    
            idx         += nPers
            nHH         += 1
            
            if nAgentsCell <= 0:
           
                    break
    print 'agents loaded from file'
    earth.queue.dequeueVertices(earth)
    earth.queue.dequeueEdges(earth)
    
    earth.mpi.sendRecvGhostNodes(earth)
    #earth.mpi.comm.Barrier()
    #earth.mpi.recvGhostNodes(earth)
 
    earth.queue.dequeueVertices(earth)
    earth.queue.dequeueEdges(earth)
    #earth.graph.write_graphml('graph' +str(earth.mpi.rank) + '.graphML')
    #earth.view(str(earth.mpi.rank) + '.png')
    
    for ghostCell in earth.iterEntRandom(_cell, ghosts = True, random=False):
        ghostCell.updatePeList(earth.graph)
        ghostCell.updateHHList(earth.graph)
    
    
    earth.mpi.comm.Barrier()
    print str(nAgents) + ' Agents and ' + str(nHH) + ' Housholds created in -- ' + str( time.time() - tt) + ' s'
    return earth




def initEarth(parameters, maxNodes, debug, mpiComm=None):
    tt = time.time()
    
    earth = Earth(parameters,maxNodes=maxNodes, debug = debug, mpiComm=mpiComm)
    

    earth.initMarket(earth,
                     parameters.properties, 
                     parameters.randomCarPropDeviationSTD, 
                     burnIn=parameters.burnIn, 
                     greenInfraMalus=parameters.kappa)
    
    earth.market.mean = np.array([400.,300.])
    earth.market.std = np.array([100.,50.])
    #init location memory
    earth.enums = dict()
    

    
    earth.enums['priorities'] = dict()
    
    earth.enums['priorities'][0] = 'convinience'
    earth.enums['priorities'][1] = 'ecology'
    earth.enums['priorities'][2] = 'money'
    earth.enums['priorities'][3] = 'imitation'
    
    earth.enums['properties'] = dict()
    earth.enums['properties'][1] = 'emissions'
    earth.enums['properties'][2] = 'TCO'
    
    earth.enums['nodeTypes'] = dict()
    earth.enums['nodeTypes'][1] = 'cell'
    earth.enums['nodeTypes'][2] = 'household'
    earth.enums['nodeTypes'][3] = 'pers'
    
    earth.enums['consequences'] = dict()
    earth.enums['consequences'][0] = 'convenience'
    earth.enums['consequences'][1] = 'eco-friendliness'
    earth.enums['consequences'][2] = 'remaining money'
    earth.enums['consequences'][3] = 'innovation'
    
    earth.enums['mobilityTypes'] = dict()
    earth.enums['mobilityTypes'][1] = 'green'
    earth.enums['mobilityTypes'][0] = 'brown'
    earth.enums['mobilityTypes'][2] = 'other'
    
    
    earth.nPref = len(earth.enums['priorities'])
    earth.nPrefTypes = [0]* earth.nPref
    
    print 'Init finished after -- ' + str( time.time() - tt) + ' s'
    return earth 
    
def initTypes(earth, parameters):
    _cell    = earth.registerNodeType('cell' , AgentClass=Cell, GhostAgentClass= GhostCell, 
                                  propertyList = ['type', 
                                                  'gID',
                                                  'pos',
                                                  'population',
                                                  'convenience',
                                                  'regionId',
                                                  'carsInCell',])
    _hh = earth.registerNodeType('hh', AgentClass=Household, GhostAgentClass= GhostHousehold, 
                                  propertyList =  ['type', 
                                                   'gID',
                                                   'pos',
                                                   'hhSize',
                                                   'nKids',
                                                   'income',
                                                   'expUtil',
                                                   'util',
                                                   'expenses'])
    _pers = earth.registerNodeType('pers', AgentClass=Person, GhostAgentClass= GhostPerson, 
                                  propertyList = ['type', 
                                                  'gID',
                                                  'hhID',
                                                  'preferences',
                                                  'prefTyp',
                                                  'gender',
                                                  'age',
                                                  'util',     # current utility
                                                  'commUtil', # comunity utility
                                                  'selfUtil', # own utility at time of action
                                                  'mobType',
                                                  'prop',
                                                  'consequences',
                                                  'lastAction',
                                                  'ESSR',
                                                  'innovatorDegree'])
    
    earth.registerEdgeType('cell-cell',_cell, _cell, ['type','weig'])
    earth.registerEdgeType('cell-hh', _cell, _hh)
    earth.registerEdgeType('hh-hh', _hh,_hh)
    earth.registerEdgeType('hh-pers',_hh,_pers)
    earth.registerEdgeType('pers-pers',_pers,_pers, ['type','weig'])


    
    return _cell, _hh, _pers

def initSpatialLayer(earth, parameters):
    connList= aux.computeConnectionList(parameters.connRadius, ownWeight=1)
    earth.initSpatialLayer(parameters.landLayer, connList, _cell, LocClassObject=Cell, GhstLocClassObject=GhostCell)  
    
    if hasattr(parameters,'regionIdRaster'):
        
        for cell in earth.iterEntRandom(_cell):
            cell.node['regionId'] = parameters.regionIdRaster[cell.node['pos']]
            
    earth.initMemory(parameters.properties + ['utility','label','hhID'], parameters.memoryTime)
                     

    

def cellTest(earth, parameters):   
    #%% cell convenience test
    convArray = np.zeros([earth.market.getNTypes(),len(earth.nodeDict[1])])
    popArray = np.zeros([len(earth.nodeDict[1])])
    for i, cell in enumerate(earth.iterEntRandom(_cell)):
        convAll, population = cell.selfTest(earth)
        convArray[:,i] = convAll
        popArray[i] = population
    
    
    if earth.para['showFigures']:

        plt.figure()
        for i in range(earth.market.getNTypes()):    
            plt.subplot(2,2,i+1)
            plt.scatter(popArray,convArray[i,:], s=2)    
            plt.title('convenience of ' + earth.enums['mobilityTypes'][i])
        plt.show()
    
    
def generateNetwork(earth, parameters):
    # %% Generate Network
    tt = time.time()
    earth.genFriendNetwork(_pers,_cpp)
    print 'Network initialized in -- ' + str( time.time() - tt) + ' s'
    if parameters.scenario == 0:
        earth.view(str(earth.mpi.rank) + '.png')
        
    
def initMobilityTypes(earth, parameters):
    earth.market.initialCarInit()
    earth.market.setInitialStatistics([1000.,2.,500.])
 
def initGlobalRecords(earth, parameters):
    tt = time.time()
    earth.registerRecord('stockBremen', 
                         'total use per mobility type - Bremen', 
                         earth.enums['mobilityTypes'].values(), 
                         style ='plot', 
                         mpiReduce='sum')
    
    calDataDfCV = pd.read_csv(parameters.resourcePath + 'calDataCV.csv',index_col=0, header=1)
    calDataDfEV = pd.read_csv(parameters.resourcePath + 'calDataEV.csv',index_col=0, header=1)
    timeIdxs = list()
    values   = list()
    for column in calDataDfCV.columns[1:]:
        value = [np.nan]*3
        year = int(column)
        timeIdx = 12* (year - parameters['startDate'][1]) + earth.para['burnIn']
        value[0] = (calDataDfCV[column]['re_1518'] ) / parameters['reductionFactor']
        if column in calDataDfEV.columns[1:]:
            value[1] = (calDataDfEV[column]['re_1518'] ) / parameters['reductionFactor']
        
    
        timeIdxs.append(timeIdx)
        values.append(value)
          
    earth.globalRecord['stockBremen'].addCalibrationData(timeIdxs,values)
 
    earth.registerRecord(name='stockNiedersachsen', 
                         title='total use per mobility type - Niedersachsen', 
                         colLables=earth.enums['mobilityTypes'].values(), 
                         style ='plot',
                         mpiReduce='sum')
    
    timeIdxs = list()
    values   = list()
    for column in calDataDfCV.columns[1:]:
        value = [np.nan]*3
        year = int(column)
        timeIdx = 12* (year - parameters['startDate'][1]) + earth.para['burnIn']
        value[0] = ( calDataDfCV[column]['re_6321']) / parameters['reductionFactor']
        if column in calDataDfEV.columns[1:]:
            value[1] = ( calDataDfEV[column]['re_6321']) / parameters['reductionFactor']
        
    
        timeIdxs.append(timeIdx)
        values.append(value)
         
    earth.globalRecord['stockNiedersachsen'].addCalibrationData(timeIdxs,values)

    earth.registerRecord('stockHamburg', 
                         'total use per mobility type - Hamburg', 
                         earth.enums['mobilityTypes'].values(),
                         style ='plot',
                         mpiReduce='sum')
    
    timeIdxs = list()
    values   = list()
    for column in calDataDfCV.columns[1:]:
        value = [np.nan]*3
        year = int(column)
        timeIdx = 12* (year - parameters['startDate'][1]) + earth.para['burnIn']
        value[0] = ( calDataDfCV[column]['re_1520']) / parameters['reductionFactor']
        if column in calDataDfEV.columns[1:]:
            value[1] = ( calDataDfEV[column]['re_1520']) / parameters['reductionFactor']
        
    
        timeIdxs.append(timeIdx)
        values.append(value)
         
    earth.globalRecord['stockHamburg'].addCalibrationData(timeIdxs,values)
    
    earth.registerRecord('growthRate', 'Growth rate of mobitlity types', earth.enums['mobilityTypes'].values(), style ='plot')
    earth.registerRecord('infraKappa', 'Infrastructure kappa', ['Kappa'], style ='plot')
    
    print 'Global records initialized in ' + str( time.time() - tt) + ' s' 
    
def initAgentOutput(earth):
    #%% Init of agent file

    tt = time.time()
    #earth.initAgentFile(typ = _hh)
    #earth.initAgentFile(typ = _pers)
    #earth.initAgentFile(typ = _cell)
    earth.io.initNodeFile(earth, [_cell, _hh, _pers])
    

    print 'Agent file initialized in ' + str( time.time() - tt) + ' s'
    

def calGreenNeigbourhoodShareDist(earth):
    if parameters.showFigures:
        #%%
        import matplotlib.pylab as pl
        
        relarsPerNeigborhood = np.zeros([len(earth.nodeDict[_pers]),3])
        for i, persId in enumerate(earth.nodeDict[_pers]):
            person = earth.entDict[persId]
            x,__ = person.getConnNodeValues('mobType',_pers)
            for mobType in range(3):
                relarsPerNeigborhood[i,mobType] = float(np.sum(np.asarray(x)==mobType))/len(x)
            
        n, bins, patches = pl.hist(relarsPerNeigborhood, 30, normed=0, histtype='bar',
                                label=['brown', 'green', 'other'])
        pl.legend()
        
def plotIncomePerNetwork(earth):        
    #%%
        import matplotlib.pylab as pl
        
        incomeList = np.zeros([len(earth.nodeDict[_pers]),1])
        for i, persId in enumerate(earth.nodeDict[_pers]):
            person = earth.entDict[persId]
            x, friends = person.getConnNodeValues('mobType',_pers)
            incomes = [earth.entDict[friend].hh.node['income'] for friend in friends]
            incomeList[i,0] = np.mean(incomes)
            
        n, bins, patches = pl.hist(incomeList, 20, normed=0, histtype='bar',
                                label=['average imcome '])
        pl.legend()
    #%%
def runModel(earth, parameters):

    #%% Initial actions
    tt = time.time()
    for household in earth.iterEntRandom(_hh):
        
        household.takeAction(earth, household.adults, np.random.randint(0,earth.market.nMobTypes,len(household.adults)))
        #for adult in household.adults:
            #adult.setValue('lastAction', 0)
    print 'Initial actions done'
    for cell in earth.iterEntRandom(_cell):
        cell.step(earth.market.kappa)
    
    print 'Initial market step done'
    
    for household in earth.iterEntRandom(_hh):
        household.calculateConsequences(earth.market)
        household.util = household.evalUtility(earth)
        #household.shareExperience(earth)
    print 'Initial actions randomized in -- ' + str( time.time() - tt) + ' s'
    
    #plotIncomePerNetwork(earth)
    
    
    #%% Simulation 
    earth.time = -1 # hot bugfix to have both models running #TODO Fix later
    print "Starting the simulation:"
    for step in xrange(parameters.nSteps):
        
        earth.step() # looping over all cells
        
        #plt.figure()
        #calGreenNeigbourhoodShareDist(earth)
        #plt.show()

    
        
    
    #%% Finishing the simulation    
    print "Finalizing the simulation (No." + str(earth.simNo) +"):"
    if parameters.writeOutput:
        earth.io.finalizeAgentFile()
    earth.finalize()        

def writeSummary(earth, calRunId, paraDf, parameters):
    errBremen = earth.globalRecord['stockBremen'].evaluateRelativeError()
    errNiedersachsen = earth.globalRecord['stockNiedersachsen'].evaluateRelativeError()
    fid = open('summary.out','w')
    fid.writelines('Calibration run id: ' + str(calRunId) + '\n')
    fid.writelines('Parameters:')
    paraDf.to_csv(fid)
    fid.writelines('Error:')
    fid.writelines('Bremen: ' + str(errBremen))
    fid.writelines('Niedersachsen: ' + str(errNiedersachsen))
    fid.writelines('Gesammt:')
    fid.writelines(str(errBremen + errNiedersachsen))
    fid.close()
    print 'Calibration Run: ' + str(calRunId)
    print paraDf
    print 'The simulation error is: ' + str(errBremen + errNiedersachsen) 


    if parameters.scenario == 2:
        nPeople = np.nansum(parameters.population)

        nCars = float(np.nansum(np.array(earth.graph.vs[earth.nodeDict[_pers]]['mobType'])!=2))
        nGreenCars = float(np.nansum(np.array(earth.graph.vs[earth.nodeDict[_pers]]['mobType'])==1))
        nBrownCars = float(np.nansum(np.array(earth.graph.vs[earth.nodeDict[_pers]]['mobType'])==0))

        print 'Number of agents: ' + str(nPeople)
        print 'Number of agents: ' + str(nCars)
        print 'cars per 1000 people: ' + str(nCars/nPeople*1000.)
        print 'green cars per 1000 people: ' + str(nGreenCars/nPeople*1000.)
        print 'brown cars per 1000 people: ' + str(nBrownCars/nPeople*1000.)


        cellList = earth.graph.vs[earth.nodeDict[_cell]]
        cellListBremen = cellList.select(regionId_eq=1518)
        cellListNieder = cellList.select(regionId_eq=6321)

        carsInBremen = np.asarray(cellListBremen['carsInCell'])
        carsInNieder = np.asarray(cellListNieder['carsInCell'])

        nPeopleBremen = np.nansum(parameters.population[parameters.regionIdRaster==1518])
        nPeopleNieder = np.nansum(parameters.population[parameters.regionIdRaster==6321])

        print 'Bremem - green cars per 1000 people: ' + str(np.sum(carsInBremen[:,1])/np.sum(nPeopleBremen)*1000)
        print 'Bremem - brown cars per 1000 people: ' + str(np.sum(carsInBremen[:,0])/np.sum(nPeopleBremen)*1000)

        print 'Niedersachsen - green cars per 1000 people: ' + str(np.sum(carsInNieder[:,1])/np.sum(nPeopleNieder)*1000)
        print 'Niedersachsen - brown cars per 1000 people: ' + str(np.sum(carsInNieder[:,0])/np.sum(nPeopleNieder)*1000)

        
     
        
        
    elif parameters.scenario == 3:
        
        nPeople = np.nansum(parameters.population)

        nCars = float(np.nansum(np.array(earth.graph.vs[earth.nodeDict[_pers]]['mobType'])!=2))
        nGreenCars = float(np.nansum(np.array(earth.graph.vs[earth.nodeDict[_pers]]['mobType'])==1))
        nBrownCars = float(np.nansum(np.array(earth.graph.vs[earth.nodeDict[_pers]]['mobType'])==0))

        print 'Number of agents: ' + str(nPeople)
        print 'Number of agents: ' + str(nCars)
        print 'cars per 1000 people: ' + str(nCars/nPeople*1000.)
        print 'green cars per 1000 people: ' + str(nGreenCars/nPeople*1000.)
        print 'brown cars per 1000 people: ' + str(nBrownCars/nPeople*1000.)


        cellList = earth.graph.vs[earth.nodeDict[_cell]]
        cellListBremen = cellList.select(regionId_eq=1518)
        cellListNieder = cellList.select(regionId_eq=6321)
        cellListHamb   = cellList.select(regionId_eq=1520)

        carsInBremen = np.asarray(cellListBremen['carsInCell'])
        carsInNieder = np.asarray(cellListNieder['carsInCell'])
        carsInHamb   = np.asarray(cellListHamb['carsInCell'])


        nPeopleBremen = np.nansum(parameters.population[parameters.regionIdRaster==1518])
        nPeopleNieder = np.nansum(parameters.population[parameters.regionIdRaster==6321])
        nPeopleHamb   = np.nansum(parameters.population[parameters.regionIdRaster==1520])


        print 'Bremem  - green cars per 1000 people: ' + str(np.sum(carsInBremen[:,1])/np.sum(nPeopleBremen)*1000)
        print 'Bremem  - brown cars per 1000 people: ' + str(np.sum(carsInBremen[:,0])/np.sum(nPeopleBremen)*1000)
        

        print 'Niedersachsen - green cars per 1000 people: ' + str(np.sum(carsInNieder[:,1])/np.sum(nPeopleNieder)*1000)
        print 'Niedersachsen - brown cars per 1000 people: ' + str(np.sum(carsInNieder[:,0])/np.sum(nPeopleNieder)*1000)
        
        
        print 'Hamburg       - green cars per 1000 people: ' + str(np.sum(carsInNieder[:,1])/np.sum(nPeopleHamb)*1000)
        print 'Hamburg -       brown cars per 1000 people: ' + str(np.sum(carsInHamb[:,0])/np.sum(nPeopleHamb)*1000)  


def onlinePostProcessing(earth):
    # calculate the mean and standart deviation of priorities
    if True:
        df = pd.DataFrame([],columns=['prCon','prEco','prMon','prImi'])
        for agID in earth.nodeDict[3]:
            df.loc[agID] = earth.graph.vs[agID]['preferences']
        
        
        print 'Preferences -average'
        print df.mean()
        print 'Preferences - standart deviation'
        print df.std()
        
        print 'Preferences - standart deviation within friends'
        avgStd= np.zeros([1,4])    
        for agent in earth.iterEntRandom(_hh): 
            friendList = agent.getConnNodeIDs(nodeType=_hh)
            if len(friendList)> 1:
                #print df.ix[friendList].std()
                avgStd += df.ix[friendList].std().values
        nAgents    = np.nansum(parameters.population)         
        print avgStd / nAgents
        prfType = np.argmax(df.values,axis=1)
        #for i, agent in enumerate(earth.iterNode(_hh)):
        #    print agent.prefTyp, prfType[i]
        df['ref'] = prfType

    # calculate the correlation between weights and differences in priorities        
    if False:
        pref = np.zeros([earth.graph.vcount(), 4])
        pref[earth.nodeDict[_pers],:] = np.array(earth.graph.vs[earth.nodeDict[_pers]]['preferences'])
        idx = list()
        for edge in earth.iterEdges(_cpp):
            edge['prefDiff'] = np.sum(np.abs(pref[edge.target, :] - pref[edge.source,:]))
            idx.append(edge.index)
            
            
        plt.figure()
        plt.scatter(np.asarray(earth.graph.es['prefDiff'])[idx],np.asarray(earth.graph.es['weig'])[idx])
        plt.xlabel('difference in preferences')
        plt.ylabel('connections weight')
        
        plt.show()
        x = np.asarray(earth.graph.es['prefDiff'])[idx].astype(float)
        y = np.asarray(earth.graph.es['weig'])[idx].astype(float)
        print np.corrcoef(x,y)


#%%############### Tests ############################################################ 


def prioritiesCalibrationTest():
             
    householdSetup(earth, parameters, calibration=True)
    df = pd.DataFrame([],columns=['prCon','prEco','prMon','prImi'])
    for agID in earth.nodeDict[3]:
        df.loc[agID] = earth.graph.vs[agID]['preferences']

#    df = pd.DataFrame([],columns=['prCon','prEco','prMon','prImi'])
#    for agID in earth.nodeDict[3]:
#        df.loc[agID] = earth.graph.vs[agID]['preferences']

    propMat = np.array(np.matrix(earth.graph.vs[earth.nodeDict[3]]['preferences']))

    return earth 


def setupHouseholdsWithOptimalChoice():

    householdSetup(earth, parameters)            
    initMobilityTypes(earth, parameters)    
    #earth.market.setInitialStatistics([500.0,10.0,200.0])
    for household in earth.iterEntRandom(_hh):    
        household.takeAction(earth, household.adults, np.random.randint(0,earth.market.nMobTypes,len(household.adults)))

    for cell in earth.iterEntRandom(_cell):
        cell.step(earth.market.kappa) 
    
    earth.market.setInitialStatistics([1000.,5.,300.])

    for household in earth.iterEntRandom(_hh):
        household.calculateConsequences(earth.market)
        household.util = household.evalUtility()
        
    for hh in iter(earth.nodeDict[_hh]):
        oldEarth = copy(earth)
        earth.entDict[hh].bestMobilityChoice(oldEarth,forcedTryAll = True)    
    return earth    
     


#%%###################################################################################
########## 
   
######################################################################################

# GLOBAL INIT  MPI
comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()
mpiSize = comm.Get_size()

if mpiRank != 0 and False:
    olog_file  = open('output/log' + str(mpiRank) + '.txt', 'w')
    sys.stdout = olog_file
    elog_file  = open('output/err' + str(mpiRank) + '.txt', 'w')
    sys.stderr = elog_file


if __name__ == '__main__':
#from pycallgraph import PyCallGraph
#from pycallgraph.output import GraphvizOutput
#
#with PyCallGraph(output=GraphvizOutput()):
        

    
    dirPath = os.path.dirname(os.path.realpath(__file__))
    
    
    # loading of standart parameters
    fileName = sys.argv[1]
    parameters = Bunch()
    for item in csv.DictReader(open(fileName)):
        parameters[item['name']] = aux.convertStr(item['value'])
    print 'Setting loaded:'
    print parameters.toDict()
    
    # loading of changing calibration parameters
    if len(sys.argv) > 3:
        # got calibration id
        paraFileName = sys.argv[2]
        colID = int(sys.argv[3])
        parameters['calibration'] = True
        
        calParaDf = pd.read_csv(paraFileName, index_col= 0, header = 0, skiprows = range(1,colID+1), nrows = 1)
        calRunID = calParaDf.index[0]
        for colName in calParaDf.columns:
            print 'Setting "' + colName + '" to value: ' + str(calParaDf[colName][calRunID]) 
            parameters[colName] = aux.convertStr(str(calParaDf[colName][calRunID]))
        

    else:
        parameters['calibration'] = False
        calRunID = -999
        calParaDf = pd.DataFrame()
        # no csv file given
        print "no input of parameters"
        
    showFigures    = 0
    

    if parameters.scenario == 0:
        
        if mpiRank == 0:
            parameters = scenarioTestSmall(parameters, dirPath)
        else: 
            parameters = None
        parameters = comm.bcast(parameters)
        
    elif parameters.scenario == 1:

        if mpiRank == 0:
            parameters = scenarioTestMedium(parameters, dirPath)
        else: 
            parameters = None
        parameters = comm.bcast(parameters)
        
    elif parameters.scenario == 2:
        
        if mpiRank == 0:
            parameters = scenarioNiedersachsen(parameters, dirPath)
        else: 
            parameters = None
        parameters = comm.bcast(parameters)
        
    elif parameters.scenario == 3:
        
        if mpiRank == 0:
            parameters = scenarioNBH(parameters, dirPath)
        else: 
            parameters = None
        parameters = comm.bcast(parameters)
        
        
    if parameters.scenario == 4:
        # test scenario 
        
        parameters.resourcePath = dirPath + '/resources_nie/'
        parameters = scenarioTestMedium(parameters, dirPath)
        parameters.showFigures = showFigures
        #parameters = scenarioNiedersachsen(parameters)
        earth = initEarth(parameters)
        mobilitySetup(earth, parameters)
        #earth = prioritiesCalibrationTest()
        earth = setupHouseholdsWithOptimalChoice()
        
    if parameters.scenario == 5: #graph part
        parameters = scenarioNBH(parameters, dirPath)  
        parameters.landLayer = parameters.landLayer * 0
        parameters.showFigures = showFigures
        earth = initEarth(parameters, maxNodes=1000000, debug =True)
        _cell, _hh, _pers = initTypes(earth,parameters)
        initSpatialLayer(earth, parameters)
        for cell in earth.iterEntRandom(_cell):
            cell.node['population'] = parameters.population[cell.node['pos']]
        earth.view('spatial_graph.png')            
        aux.writeAdjFile(earth.graph,'outGraph.txt')
        
        exit
    else:
        #%% Init 
        parameters.showFigures = showFigures
        
        earth = initEarth(parameters, maxNodes=1000000, debug =False, mpiComm=comm)
        

        
        _cell, _hh, _pers = initTypes(earth,parameters)
        
        initSpatialLayer(earth, parameters)
        
        

        #aux.writeAdjFile(earth.graph,'outGraph.txt')
        #sdf
        mobilitySetup(earth, parameters)
        
        
        #cellTest(earth, parameters)
        householdSetup(earth, parameters)
        
        cellTest(earth, parameters)

        generateNetwork(earth, parameters)
        
        initMobilityTypes(earth, parameters)
        
        initGlobalRecords(earth, parameters)
        #earth.view(str(earth.mpi.rank) + '.png')
        initAgentOutput(earth)
        
        if parameters.scenario == 0:
            earth.view('output/graph.png')
      
        #%% run of the model ################################################
        print '####### Running model with paramertes: #########################'
        import pprint 
        pprint.pprint(parameters.toDict())
        if mpiRank == 0:
            fidPara = open(earth.para['outPath'] + '/parameters.txt','w')
            pprint.pprint(parameters.toDict(), fidPara)
            fidPara.close()
        print '################################################################'
        
        runModel(earth, parameters)
        
        
        print 'Simulation ' + str(earth.simNo) + ' finished after -- ' + str( time.time() - overallTime) + ' s'
        
        
        
        if earth.para['showFigures']:
            onlinePostProcessing(earth)
        
        plt.figure()
        
        allTime = np.zeros(earth.nSteps)
        colorPal =  sns.color_palette("Set2", n_colors=5, desat=.8)
        

        for i,var in enumerate([earth.computeTime, earth.waitTime, earth.syncTime, earth.ioTime]):
            plt.bar(np.arange(earth.nSteps), var, bottom=allTime, color =colorPal[i], width=1)
            allTime += var
        plt.legend(['compute time', 'wait time', 'sync time', 'I/O time'])
        plt.tight_layout()
        plt.savefig(earth.para['outPath'] + '/' + str(mpiRank) + 'times.png')
        
        
        #tmp = earth.mpi.comm.gather(earth.computeTime)
        #if mpiRank == 0:
        #    print tmp
        
        if mpiSize ==1:
            writeSummary(earth, calRunID, calParaDf, parameters)
    
        
   



        
 
