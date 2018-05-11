#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://wwww.globalclimateforum.org

---- MoTMo ----
MOBILITY TRANSFORMATION MODEL
-- INIT FILE --

This file is based on GCFABM.

GCFABM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GCFABM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GCFABM.  If not, see <http://earth.gnu.org/licenses/>.


TODOs

short-term:
    - utility test center
    - realistic physical values
    - test learning niches
    - utility potential for omnicent knowledge
    - MCMC evolution of the synthetic population
        - add/remove households and draw acceptance accourding to available statistics
        - what about if no data is available ??? -> extrapolation of statistics

    - entropy on networks (distribution of weights)
        - entire network
        - per agent
    - save connections + properties to output
    - synthetic poplulations of bremen and hamburg
    - add price incentive (4000 euro in 2017)

long-term:
    - cut connections with near-zero weights
    - replace removed connections with new ones
        - rebuffering of sequences of nodes and edges
    - add advertising to experience

"""
#%%
#TODO
# random iteration (even pairs of agents)
#from __future__ import division

import sys, os
import socket

sys.path.append('../../lib/')
sys.path.append('../../modules/')

dir_path = os.path.dirname(os.path.realpath(__file__))
if socket.gethostname() in ['gcf-VirtualBox', 'ThinkStation-D30']:
    sys.path = ['../../h5py/build/lib.linux-x86_64-2.7'] + sys.path
    sys.path = ['../../mpi4py/build/lib.linux-x86_64-2.7'] + sys.path

else:
    
    import matplotlib
    matplotlib.use('Agg')
    
import mpi4py
mpi4py.rc.threads = False


import csv
import time
#import guppy
from copy import copy
from os.path import expanduser
import pdb
home = expanduser("~")

#from deco_util import timing_function
import numpy as np

#import mod_geotiff as gt # not working in poznan

#from mpi4py import  MPI
#import h5py

from classes_motmo import Person, GhostPerson, Household, GhostHousehold, Reporter, Cell, GhostCell, Earth, Opinion, aux, h5py, MPI
#import class_auxiliary  as aux #convertStr
comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()
mpiSize = comm.Get_size()

import logging as lg
import matplotlib.pylab as plt
import seaborn as sns; sns.set()
sns.set_color_codes("dark")
#import matplotlib.pyplot as plt

import pandas as pd
from bunch import Bunch

from scipy import signal

print 'import done'

overallTime = time.time()
###### Enums ################
#connections
CON_LL = 1 # loc - loc
CON_LH = 2 # loc - household
CON_HH = 3 # household, household
CON_HP = 4 # household, person
CON_PP = 5 # person, person

#nodes
CELL   = 1
HH     = 2
PERS   = 3

#time spans
_month = 1
_year  = 2

#%% Scenario definition without calibraton parameters

def scenarioTestSmall(parameterInput, dirPath):
    setup = Bunch()

    #general
    setup.resourcePath  = dirPath + '/resources/'
    setup.progressBar   = True
    setup.allTypeObservations = False

    #time
    setup.timeUnit         = _month  # unit of time per step
    setup.startDate        = [01,2005]

    a = 60000.
    b = 45000.
    c = 30000.
    d = 25000.
    e = 20000.
    f = 15000.
    g = 10000.
    h = 5000.
    i = 1500.
    
    #spatial
    setup.reductionFactor = 5000
    setup.isSpatial       = True
    setup.spatialRedFactor = 280.
    
    setup.connRadius      = 2.0     # radíus of cells that get an connection

    
    setup.population  = np.asarray([[c, a, b, 0, 0],
                                    [c, b, d, 0, f],
                                    [0, h, i, g, e]])
    
    setup.landLayer  = setup.population.copy().astype(float)
    setup.landLayer[setup.landLayer== 0.] = np.nan
    
    setup.landLayer  = (setup.landLayer*0.) +1.
    setup.landLayer[0,:] = 0.
    
    
    setup.cellSizeMap  = setup.landLayer * 15.
    
    setup.roadKmPerCell   = np.asarray([[1, 5, 3, 0, 0],
                                        [1, 4, 4, 0, 1],
                                        [0, 1, 1, 1, 1]]) / setup.cellSizeMap


    
    
    
    
    setup.regionIdRaster            = ((setup.landLayer*0)+1)*6321
    #setup.regionIdRaster[0:,0:2]    = ((setup.landLayer[0:,0:2]*0)+1) *1519
    if mpiSize == 1:
        setup.landLayer = setup.landLayer*0

    setup.regionIDList = np.unique(setup.regionIdRaster[~np.isnan(setup.regionIdRaster)]).astype(int)
    
    #setup.population = (np.isnan(setup.landLayer)==0)* np.random.randint(3,23,setup.landLayer.shape)
    #setup.population = (np.isnan(setup.landLayer)==0)* 13
    
    #social
    setup.addYourself   = True     # have the agent herself as a friend (have own observation)
    setup.recAgent      = []       # reporter agents that return a diary

    #output
    setup.writeOutput   = 1
    setup.writeNPY      = 1
    setup.writeCSV      = 0

    #cars and infrastructure
    setup.properties    = ['costs', 'emissions']

    #agents
    setup.randomAgents     = False
    setup.omniscientAgents = False

    # redefinition of setup parameters used for automatic calibration
    setup.update(parameterInput.toDict())

    # calculate dependent parameters
    #maxDeviation = (setup.urbanCritical - setup.urbanThreshold)**2
    #minCarConvenience = 1 + setup.kappa
    #setup.convB =  minCarConvenience / (maxDeviation)

    # only for packing with care
    #dummy   = gt.load_array_from_tiff(setup.resourcePath + 'land_layer_62x118.tiff')
    #dummy   = gt.load_array_from_tiff(setup.resourcePath + 'pop_counts_ww_2005_62x118.tiff') / setup.reductionFactor
    #dummy   = gt.load_array_from_tiff(setup.resourcePath + 'subRegionRaster_62x118.tiff')
    #del dummy

    for paName in ['techExpBrown', 'techExpGreen','techExpPublic', 'techExpShared' ,'techExpNone',
                    'population']:
        setup[paName] /= setup['reductionFactor']

    import pprint as pp
    pp.pprint("Final setting of the parameters")
    pp.pprint(setup.toDict())
    lg.info("Final setting of the parameters")
    lg.info(setup.toDict())
    lg.info("####################################")

    return setup



def scenarioTestMedium(parameterInput, dirPath):


    setup = Bunch()


    #general
    setup.resourcePath          = dirPath + '/resources/'
    setup.allTypeObservations   = False

    setup.progressBar = True

    #time
    setup.timeUnit          = _month  # unit of time per step
    setup.startDate         = [01, 2005]


    #spatial
    setup.isSpatial         = True
    setup.spatialRedFactor = 80.
    
    setup.landLayer   = np.asarray([[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1 , 1, 1],
                                    [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1 , 1, 0],
                                    [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1 , 0, 0],
                                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0 , 0, 0],
                                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1 , 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1 , 1, 1]])
    
    setup.cellSizeMap  = setup.landLayer * 15.
    
    setup.roadKmPerCell   = np.asarray([[0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 1 , 1, 0],
                                        [0, 1, 0, 0, 0, 0, 1, 2, 3, 1, 0 , 0, 0],
                                        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0],
                                        [2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0 , 0, 0],
                                        [3, 4, 1, 1, 2, 1, 1, 0, 0, 0, 0 , 0, 1],
                                        [6, 5, 2, 3, 2, 0, 1, 0, 0, 0, 0 , 1, 2]])     / setup.cellSizeMap
    a = 60000.
    b = 45000.
    c = 30000.
    d = 25000.
    e = 20000.
    f = 15000.
    g = 10000.
    h = 5000.
    i = 1500.
    setup.population  = np.asarray([[0, 0, 0, 0, e, d, c, d, h, 0, g, h, i],
                                    [0, c, 0, 0, 0, e, d, d, e, e, f, g, 0],
                                    [b, 0, c, 0, e, e, i, 0, 0, g, f, 0, 0],
                                    [b, c, d, d, e, e, 0, 0, 0, g, i, i, 0],
                                    [a, b, c, c, d, e, f, 0, 0, 0, i, i, i],
                                    [a, a, c, c, d, f, f, 0, 0, 0, i, i, g]])    
    del a, b, c, d, e, f, g, h, i

    #convMat = np.asarray([[0, 1, 0],[1, 0, 1],[0, 1, 0]])
    #setup.population = setup.landLayer* signal.convolve2d(setup.landLayer,convMat,boundary='symm',mode='same')
    #setup.population = 30 * setup.population + setup.landLayer* np.random.randint(0,40,setup.landLayer.shape)*2
    #setup.population = 50 * setup.population 

    setup.landLayer  = setup.landLayer.astype(float)
    setup.landLayer[setup.landLayer== 0] = np.nan

    
    if mpiSize == 1:
        setup.landLayer = setup.landLayer*0

    setup.regionIdRaster    = ((setup.landLayer*0)+1)*6321
    #setup.regionIdRaster[3:, 0:3] = ((setup.landLayer[3:, 0:3]*0)+1) *1519
    setup.regionIDList = np.unique(setup.regionIdRaster[~np.isnan(setup.regionIdRaster)]).astype(int)

    

    setup.landLayer[:, :5] = setup.landLayer[:, :5] * 0
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
    setup.properties    = ['costs', 'emissions']

    #agents
    setup.randomAgents     = False
    setup.omniscientAgents = False

    # redefinition of setup parameters used for automatic calibration
    setup.update(parameterInput.toDict())

    # calculate dependent parameters
    for paName in ['techExpBrown', 'techExpGreen','techExpPublic', 'techExpShared' ,'techExpNone',
                   'population']:
        setup[paName] /= setup['reductionFactor']

    import pprint as pp
    pp.pprint( "Final setting of the parameters")
    pp.pprint(setup.toDict())
    lg.info("Final setting of the parameters")
    lg.info(setup.toDict())
    lg.info("####################################")

    return setup

def scenarioNBH(parameterInput, dirPath):
    setup = Bunch()

    #general
    setup.resourcePath = dirPath + '/resources_NBH/'
    #setup.synPopPath = setup['resourcePath'] + 'hh_NBH_1M.csv'
    setup.progressBar  = True
    setup.allTypeObservations = True

    #time
    setup.nSteps           = 340     # number of simulation steps
    setup.timeUnit         = _month  # unit of time per step
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

        setup.landLayer =  np.load(setup.resourcePath + 'land_layer_62x118.npy')
        setup.landLayer = setup.landLayer * 0

    lg.info('max rank:' +str(np.nanmax(setup.landLayer)))

    #setup.population        = gt.load_array_from_tiff(setup.resourcePath + 'pop_counts_ww_2005_62x118.tiff')
    setup.population = np.load(setup.resourcePath + 'pop_counts_ww_2005_62x118.npy')
    #setup.regionIdRaster    = gt.load_array_from_tiff(setup.resourcePath + 'subRegionRaster_62x118.tiff')
    setup.regionIdRaster = np.load(setup.resourcePath + 'subRegionRaster_62x118.npy')
    # bad bugfix for 4 cells
    setup.regionIdRaster[np.logical_xor(np.isnan(setup.population), np.isnan(setup.regionIdRaster))] = 6321


    setup.chargStat      = np.load(setup.resourcePath + 'charge_stations_62x118.npy')

    setup.cellSizeMap  = np.load(setup.resourcePath + 'cell_area_62x118.npy')

    assert np.sum(np.logical_xor(np.isnan(setup.population), np.isnan(setup.regionIdRaster))) == 0 ##OPTPRODUCTION

    setup.regionIDList = np.unique(setup.regionIdRaster[~np.isnan(setup.regionIdRaster)]).astype(int)

    setup.regionIdRaster[np.isnan(setup.regionIdRaster)] = 0
    setup.regionIdRaster = setup.regionIdRaster.astype(int)



    if False:
        try:
            #plt.imshow(setup.landLayer)
            plt.imshow(setup.population,cmap = 'jet')
            plt.clim([0, np.nanpercentile(setup.population, 90)])
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
    setup.properties    = ['costs', 'emissions']

    #agents
    setup.randomAgents     = False
    setup.omniscientAgents = False

    # redefinition of setup parameters from file
    setup.update(parameterInput.toDict())

    # Correciton of population depend parameter by the reduction factor
    #setup['initialGreen'] /= setup['reductionFactor']
    #setup['initialBrown'] /= setup['reductionFactor']
    #setup['initialOther'] /= setup['reductionFactor']

    #setup['population']     /= setup['reductionFactor']
    #setup['urbanThreshold'] /= setup['reductionFactor']
    #setup['urbanCritical']  /= setup['reductionFactor']
    #setup['publicTransBonus']  /= setup['reductionFactor']
    #setup.convD /= setup['reductionFactor']

    # calculate dependent parameters
    #maxDeviation = (setup.urbanCritical - setup.urbanThreshold)**2
    #minCarConvenience = 1 + setup.kappa
    #setup.convB =  minCarConvenience / (maxDeviation)


    for paName in ['techExpBrown', 'techExpGreen',
                   'techExpOther', 'population']:
        setup[paName] /= setup['reductionFactor']

    import pprint as pp
    pp.pprint( "Final setting of the parameters")
    pp.pprint(setup.toDict())
    lg.info("Final setting of the parameters")
    lg.info(setup.toDict())
    lg.info("####################################")

    return setup

def scenarioGer(parameterInput, dirPath):
    setup = Bunch()

    #general
    setup.resourcePath = dirPath + '/resources_ger/'
    #setup.synPopPath = setup['resourcePath'] + 'hh_NBH_1M.csv'
    setup.progressBar  = True
    setup.allTypeObservations = True

    #time
    setup.nSteps           = 340     # number of simulation steps
    setup.timeUnit         = _month  # unit of time per step
    setup.startDate        = [01, 2005]
    setup.burnIn           = 100
    setup.omniscientBurnIn = 10       # no. of first steps of burn-in phase with omniscient agents, max. =burnIn

    #spatial
    setup.isSpatial     = True
    setup.spatialRedFactor = 1.
    #setup.connRadius    = 3.5      # radíus of cells that get an connection
    #setup.reductionFactor = parameterInput['reductionFactor']

    if hasattr(parameterInput, "reductionFactor"):
        # overwrite the standart parameter
        setup.reductionFactor = parameterInput.reductionFactor


    #setup.landLayer[np.isnan(setup.landLayer)] = 0
    if mpiSize > 1:
        setup.landLayer = np.load(setup.resourcePath + 'partition_map_' + str(mpiSize) + '.npy')
    else:

        setup.landLayer=  np.load(setup.resourcePath + 'land_layer_186x219.npy')
        setup.landLayer[setup.landLayer==0] = np.nan
        setup.landLayer = setup.landLayer * 0

    lg.info('max rank:' + str(np.nanmax(setup.landLayer)))

    #setup.population        = gt.load_array_from_tiff(setup.resourcePath + 'pop_counts_ww_2005_186x219.tiff')
    setup.population = np.load(setup.resourcePath + 'pop_counts_ww_2005_186x219.npy')
    #setup.regionIdRaster    = gt.load_array_from_tiff(setup.resourcePath + 'subRegionRaster_62x118.tiff')
    setup.regionIdRaster = np.load(setup.resourcePath + 'subRegionRaster_186x219.npy')
    # bad bugfix for 4 cells
    #setup.regionIdRaster[np.logical_xor(np.isnan(setup.population), np.isnan(setup.regionIdRaster))] = 6321

    setup.regionIDList  = np.unique(setup.regionIdRaster[~np.isnan(setup.regionIdRaster)]).astype(int)

    setup.cellSizeMap   = np.load(setup.resourcePath + 'cell_area_186x219.npy')

    setup.roadKmPerCell = np.load(setup.resourcePath + 'road_km_all_new_186x219.npy') / setup.cellSizeMap

    
    
    # correction of ID map
    xList, yList = np.where(np.logical_xor(np.isnan(setup.population), np.isnan(setup.regionIdRaster)))

    for x, y in zip(xList,yList):
        reList = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if not np.isnan(setup.regionIdRaster[x+dx,y+dy]):
                    reList.append(setup.regionIdRaster[x+dx,y+dy])
        if len(np.unique(reList)) == 1:
            setup.regionIdRaster[x, y] = np.unique(reList)[0]

    assert np.sum(np.logical_xor(np.isnan(setup.population), np.isnan(setup.regionIdRaster))) == 0 ##OPTPRODUCTION


    setup.regionIdRaster[np.isnan(setup.regionIdRaster)] = 0
    setup.regionIdRaster = setup.regionIdRaster.astype(int)

    if False:
        try:
            #plt.imshow(setup.landLayer)
            plt.imshow(setup.population,cmap='jet')
            plt.clim([0, np.nanpercentile(setup.population,90)])
            plt.colorbar()
        except:
            pass
    print setup.landLayer.shape
    print setup.population.shape    
    setup.landLayer[np.isnan(setup.population)] = np.nan


    #social
    setup.addYourself   = True     # have the agent herself as a friend (have own observation)
    setup.recAgent      = []       # reporter agents that return a diary

    #output
    setup.writeOutput   = 1
    setup.writeNPY      = 1
    setup.writeCSV      = 0

    #cars and infrastructure
    setup.properties    = ['costs', 'emissions']

    #agents
    setup.randomAgents     = False
    setup.omniscientAgents = False

    # redefinition of setup parameters from file
    setup.update(parameterInput.toDict())

    #setup.population = (setup.population ** .5) * 100
    # Correciton of population depend parameter by the reduction factor
    for paName in ['techExpBrown', 'techExpGreen','techExpPublic', 'techExpShared' ,'techExpNone',
                   'population']:
        setup[paName] /= setup['reductionFactor']
    for p in range(0, 105, 5) :
        print 'p' + str(p) + ': ' + str(np.nanpercentile(setup.population[setup.population!=0], p))
    #print 'max population' + str(np.nanmax(setup.population))
    # calculate dependent parameters


    lg.info( "Final setting of the parameters")
    lg.info( parameterInput)
    lg.info( "####################################")

    nAgents = np.nansum(setup.population)
    lg.info('Running with ' + str(nAgents) + ' agents')

    return setup

def scenarioLueneburg(parameterInput, dirPath):
    setup = Bunch()

    #general
    setup.resourcePath = dirPath + '/resources_luen/'
    setup.progressBar  = True
    setup.allTypeObservations = True

    #time
    setup.nSteps           = 340     # number of simulation steps
    setup.timeUnit         = _month  # unit of time per step
    setup.startDate        = [01, 2005]
    setup.burnIn           = 100
    setup.omniscientBurnIn = 10       # no. of first steps of burn-in phase with omniscient agents, max. =burnIn

    #spatial
    setup.isSpatial     = True
    setup.spatialRedFactor = 350.
    
    #setup.connRadius    = 3.5      # radíus of cells that get an connection
    #setup.reductionFactor = parameterInput['reductionFactor']

    if hasattr(parameterInput, "reductionFactor"):
        # overwrite the standart parameter
        setup.reductionFactor = parameterInput.reductionFactor


    #setup.landLayer[np.isnan(setup.landLayer)] = 0
    if mpiSize > 1:
        setup.landLayer = np.load(setup.resourcePath + 'rankMap_nClust' + str(mpiSize) + '.npy')
    else:

        setup.landLayer=  np.load(setup.resourcePath + 'land_layer.npy')
        setup.landLayer[setup.landLayer==0] = np.nan
        setup.landLayer = setup.landLayer * 0

    lg.info('max rank:' + str(np.nanmax(setup.landLayer)))

    #setup.population        = gt.load_array_from_tiff(setup.resourcePath + 'pop_counts_ww_2005_186x219.tiff')
    setup.population = np.load(setup.resourcePath + 'population.npy')
    #setup.regionIdRaster    = gt.load_array_from_tiff(setup.resourcePath + 'subRegionRaster_62x118.tiff')
    setup.regionIdRaster = np.load(setup.resourcePath + 'subRegionRaster.npy')
    # bad bugfix for 4 cells
    #setup.regionIdRaster[np.logical_xor(np.isnan(setup.population), np.isnan(setup.regionIdRaster))] = 6321

    setup.regionIDList = np.unique(setup.regionIdRaster[~np.isnan(setup.regionIdRaster)]).astype(int)

    setup.cellSizeMap  = np.load(setup.resourcePath + 'cell_size.npy')

    setup.roadKmPerCell    = np.load(setup.resourcePath + 'road_km_per_cell.npy') / setup.cellSizeMap

    
    
    # correction of ID map
    xList, yList = np.where(np.logical_xor(np.isnan(setup.population), np.isnan(setup.regionIdRaster)))

#    for x, y in zip(xList,yList):
#        reList = []
#        for dx in [-1, 0, 1]:
#            for dy in [-1, 0, 1]:
#                if not np.isnan(setup.regionIdRaster[x+dx,y+dy]):
#                    reList.append(setup.regionIdRaster[x+dx,y+dy])
#        if len(np.unique(reList)) == 1:
#            setup.regionIdRaster[x, y] = np.unique(reList)[0]
#
#    assert np.sum(np.logical_xor(np.isnan(setup.population), np.isnan(setup.regionIdRaster))) == 0 ##OPTPRODUCTION


    setup.regionIdRaster[np.isnan(setup.regionIdRaster)] = 0
    setup.regionIdRaster = setup.regionIdRaster.astype(int)

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
    setup.properties    = ['costs', 'emissions']

    #agents
    setup.randomAgents     = False
    setup.omniscientAgents = False
    
    publicTranportSetup(setup)

    # redefinition of setup parameters from file
    setup.update(parameterInput.toDict())

    #setup.population = (setup.population ** .5) * 100
    # Correciton of population depend parameter by the reduction factor
    for paName in ['techExpBrown', 
                   'techExpGreen',
                   'techExpPublic', 
                   'techExpShared',
                   'techExpNone']:
        setup[paName] /= setup['reductionFactor'] *  setup.spatialRedFactor
        
    setup['population'] /= setup['reductionFactor']
        
    for p in range(0, 105, 5) :
        print 'p' + str(p) + ': ' + str(np.nanpercentile(setup.population[setup.population!=0], p))
    #print 'max population' + str(np.nanmax(setup.population))
    # calculate dependent parameters


    lg.info( "Final setting of the parameters")
    lg.info( parameterInput)
    lg.info( "####################################")

    nAgents = np.nansum(setup.population)
    lg.info('Running with ' + str(nAgents) + ' agents')

    return setup

def scenarioTest(parameterInput, dirPath):
    
    setup = Bunch()

    #general
    setup.resourcePath = None
    setup.progressBar  = True
    setup.allTypeObservations = True

    #time
    setup.nSteps           = 10     # number of simulation steps
    setup.timeUnit         = _month  # unit of time per step
    setup.startDate        = [01, 2005]
    setup.burnIn           = 0
    setup.omniscientBurnIn = 0       # no. of first steps of burn-in phase with omniscient agents, max. =burnIn

    #spatial
    setup.isSpatial     = True
    setup.spatialRedFactor = 1.
    #setup.connRadius    = 3.5      # radíus of cells that get an connection
    #setup.reductionFactor = parameterInput['reductionFactor']

    setup.landLayer = np.zeros([2, mpiSize])
    setup.landLayer[0,:] = np.asarray(range(mpiSize))
    setup.landLayer[1,:] = np.asarray(range(mpiSize))

    #setup.population        = gt.load_array_from_tiff(setup.resourcePath + 'pop_counts_ww_2005_186x219.tiff')
    setup.population = setup.landLayer * 5
    #setup.regionIdRaster    = gt.load_array_from_tiff(setup.resourcePath + 'subRegionRaster_62x118.tiff')
    setup.regionIdRaster = setup.landLayer * 99
    # bad bugfix for 4 cells
    #setup.regionIdRaster[np.logical_xor(np.isnan(setup.population), np.isnan(setup.regionIdRaster))] = 6321

    setup.regionIDList  = np.unique(setup.regionIdRaster[~np.isnan(setup.regionIdRaster)]).astype(int)

    setup.cellSizeMap   = setup.landLayer * 1.

    setup.roadKmPerCell = setup.landLayer * 2.

    
    
    # correction of ID map
    xList, yList = np.where(np.logical_xor(np.isnan(setup.population), np.isnan(setup.regionIdRaster)))

    setup.randomCarPropDeviationSTD = .05
    setup.connRadius = 2.
    setup.reductionFactor = 2.

    setup.regionIdRaster[np.isnan(setup.regionIdRaster)] = 0
    setup.regionIdRaster = setup.regionIdRaster.astype(int)

    if False:
        try:
            #plt.imshow(setup.landLayer)
            plt.imshow(setup.population,cmap='jet')
            plt.clim([0, np.nanpercentile(setup.population,90)])
            plt.colorbar()
        except:
            pass
    
    #setup.landLayer[np.isnan(setup.population)] = np.nan


    #social
    setup.addYourself   = True     # have the agent herself as a friend (have own observation)
    setup.recAgent      = []       # reporter agents that return a diary

    #output
    setup.writeOutput   = 1
    setup.writeNPY      = 1
    setup.writeCSV      = 0

    #cars and infrastructure
    setup.properties    = ['costs', 'emissions']

    #agents
    setup.randomAgents     = False
    setup.omniscientAgents = False

    # redefinition of setup parameters from file
    setup.update(parameterInput.toDict())


    lg.info( "Final setting of the parameters")
    lg.info( parameterInput)
    lg.info( "####################################")

    nAgents = np.nansum(setup.population)
    lg.info('Running with ' + str(nAgents) + ' agents')

    return setup

    
###############################################################################
###############################################################################

# %% Setup functions

def convolutionMatrix(radius, centerWeight):
    convMat = np.zeros([radius*2+1,radius*2+1])
    for dx in np.arange(-radius, radius+1):
        for dy in np.arange(-radius, radius+1):
            if dx == 0 and dy == 0:
                convMat[radius+dx,radius+dy] = centerWeight
            else:
                convMat[radius+dx,radius+dy] = 1./np.sqrt(dx**2 + dy**2) 
                
    return convMat

def publicTranportSetup(setup):
    """
    computation of the convenience of puplic transport for the Luenburg case
    """
    


    
        
    busMap = np.load(setup.resourcePath + 'nBusStations.npy')
    trainMap = np.load(setup.resourcePath + 'nTrainStations.npy')
    
    #density proxi
    convMat = convolutionMatrix(5,2)
    bus_station_density = signal.convolve2d(busMap,convMat,boundary='symm',mode='same')  #/ sumConv
    bus_station_density[busMap==0] = 0
    
    convMat = convolutionMatrix(20,2)
    train_station_density = signal.convolve2d(trainMap,convMat,boundary='symm',mode='same')
    train_station_density[trainMap==0] = 0
    
    # convolution of the station influence
    convMat = convolutionMatrix(1,2) / 2.
    
    tmp1 = .5 * signal.convolve2d(bus_station_density,convMat,boundary='symm',mode='same')
    tmp2 = 5 *signal.convolve2d(train_station_density,convMat,boundary='symm',mode='same')   
    
    convPup = np.log(1 + tmp1 + tmp2) # one is is used so that log(1) = 0

        
    setup['conveniencePublic'] = convPup / np.max(convPup) * .6

# Mobility setup setup
def mobilitySetup(earth):
    parameters = earth.getParameter()

    # define convenience functions
    def convenienceBrown(density, pa, kappa, cell):
        
        conv = pa['minConvB'] +\
        kappa * (pa['maxConvB'] - pa['minConvB']) * \
        np.exp( - (density - pa['muConvB'])**2 / (2 * pa['sigmaConvB']**2))
        return conv

    def convenienceGreen(density, pa, kappa, cell):
        
        conv = pa['minConvG'] + \
        (pa['maxConvGInit']-pa['minConvG']) * \
        (1 - kappa)  * np.exp( - (density - pa['muConvGInit'])**2 / (2 * pa['sigmaConvGInit']**2)) +  \
        (pa['maxConvG'] - pa['minConvG']) * kappa * (np.exp(-(density - pa['muConvG'])**2 / (2 * pa['sigmaConvB']**2)))
        
        
        return conv


    def conveniencePublicLeuphana(density, pa, kappa, cell):
        
        currKappa   = (1 - kappa) * pa['maxConvGInit'] + kappa * pa['maxConvG']
        return pa['conveniencePublic'][cell._node['pos']] * currKappa


    def conveniencePublic(density, pa, kappa, cell):
        conv = pa['minConvP'] + \
        ((pa['maxConvP'] - pa['minConvP']) * (kappa)) * \
        np.exp(-(density - pa['muConvP'])**2 / (2 * ((1 - kappa) * \
                   pa['sigmaConvPInit'] + (kappa * pa['sigmaConvP']))**2))
        
        return conv
    
    def convenienceShared(density, pa, kappa, cell):
#        conv = pa['minConvS'] + \
#        ((pa['maxConvS'] - pa['minConvS']) * (kappa)) * \
#        np.exp(-(density - pa['muConvS'])**2 / (2 * ((1 - kappa) * \
#                   pa['sigmaConvSInit'] + (kappa * pa['sigmaConvS']))**2))
        
        conv = (kappa/10.) + pa['minConvS'] + (kappa *(pa['maxConvS'] - pa['minConvS'] - (kappa/10.))  +\
                    ((1-kappa)* (pa['maxConvSInit'] - pa['minConvS'] - (kappa/10.)))) * \
                    np.exp( - (density - pa['muConvS'])**2 / (2 * ((1-kappa) * \
                    pa['sigmaConvSInit'] + (kappa * pa['sigmaConvS']))**2) )        
        return conv
        
    def convenienceNone(density, pa, kappa, cell):
        conv = pa['minConvN'] + \
        ((pa['maxConvN'] - pa['minConvN']) * (kappa)) * \
        np.exp(-(density - pa['muConvN'])**2 / (2 * ((1 - kappa) * \
                   pa['sigmaConvNInit'] + (kappa * pa['sigmaConvN']))**2))        
        return conv


    from collections import OrderedDict
    
    # register brown:
    propDict = OrderedDict()
    propDict['costs']     = parameters['initPriceBrown'] #, parameters['initPriceBrown']/10.
    propDict['emissions'] = parameters['initEmBrown'] #, 120. # init, lim
    
    earth.registerGood('brown',                                # name
                    propDict,                                  # (emissions, TCO)
                    convenienceBrown,                          # convenience function
                    'start',                                   # time step of introduction in simulation
                    #parameters['techSlopeBrown'],              # initial technical progress
                    #parameters['techProgBrown'],               # slope of technical progress
                    initExperience = parameters['techExpBrown'], # initial experience
                    priceRed = parameters['priceReductionB'],  # exponent for price reduction through learning by doing
                    emRed    = parameters['emReductionB'],     # exponent for emission reduction through learning by doing
                    emFactor = parameters['emFactorB'],        # factor for emission reduction through learning by doing
                    emLimit  = parameters['emLimitB'],         # emission limit
                    weight   = parameters['weightB'])          # average weight

    # register green:
    propDict = OrderedDict()
    propDict['costs']     = parameters['initPriceGreen']#, parameters['initPriceGreen']/10.
    propDict['emissions'] = parameters['initEmGreen']#, 70. # init, lim    
    earth.registerGood('green',                                #name
                    propDict,                                  # (emissions, TCO)
                    convenienceGreen,                          # convenience function
                    'start',
                    #parameters['techSlopeGreen'],              # initial technical progress
                    #parameters['techProgGreen'],               # slope of technical progress
                    initExperience = parameters['techExpGreen'],                # initial experience
                    priceRed = parameters['priceReductionG'],  # exponent for price reduction through learning by doing
                    emRed    = parameters['emReductionG'],     # exponent for emission reduction through learning by doing
                    emFactor = parameters['emFactorG'],        # factor for emission reduction through learning by doing
                    emLimit  = parameters['emLimitG'],         # emission limit
                    weight   = parameters['weightG'])          # average weight

    # register public:
    propDict = OrderedDict()
    if parameters['scenario'] == 2:
        convFuncPublic = conveniencePublicLeuphana
    else:
        convFuncPublic = conveniencePublic
    propDict['costs']     = parameters['initPricePublic']#, parameters['initPricePublic']/10.
    propDict['emissions'] = parameters['initEmPublic']#, 30. # init, lim   
    earth.registerGood('public',  #name
                    propDict,   #(emissions, TCO)
                    convFuncPublic,
                    'start',
                    initExperience = parameters['techExpPublic'],
                    pt2030  = parameters['pt2030'],          # emissions 2030 (compared to 2012)
                    ptLimit = parameters['ptLimit'])         # emissions limit (compared to 2012)

                    #parameters['techSlopePublic'],           # initial technical progress
                    #parameters['techProgPublic'],            # slope of technical progress
                    #parameters['techExpPublic'])             # initial experience
                    
    # register shared:
    propDict = OrderedDict()
    propDict['costs']     = parameters['initPriceShared']#,  parameters['initPriceShared']/10.
    propDict['emissions'] = parameters['initEmShared']#, 50. # init, lim    
    earth.registerGood('shared', # name
                    propDict,    # (emissions, TCO)
                    convenienceShared,
                    'start',
                    initExperience = parameters['techExpShared'],
                    #parameters['techSlopeShared'],           # initial technical progress
                    #parameters['techProgShared'],            # slope of technical progress
                    #parameters['techExpShared'])             # initial experience
                    weight = parameters['weightS'],           # average weight
                    initMaturity = parameters['initMaturityS']) # initital maturity
    # register none:    
    propDict = OrderedDict()
    propDict['costs']    = parameters['initPriceNone']#,  parameters['initPriceNone']/10.
    propDict['emissions'] = parameters['initEmNone']#, 1.0 # init, lim
    earth.registerGood('none',  #name
                    propDict,   #(emissions, TCO)
                    convenienceNone,
                    'start',
                    initExperience = parameters['techExpNone'],
                    initMaturity = parameters['initMaturityN']) # initital maturity
                    #parameters['techSlopeNone'],           # initial technical progress
                    #parameters['techProgNone'],            # slope of technical progress
                    #parameters['techExpNone'])             # initial experience
    

    earth.para['nMobTypes'] = len(earth.enums['brands'])
    return earth
    ##############################################################################

def householdSetup(earth, calibration=False):
    
    #enumerations for h5File - second dimension
    H5NPERS  = 0
    H5AGE = 1
    H5GENDER = 2
    H5INCOME = 3
    H5HHTYPE = 4
    H5MOBDEM = [5, 6, 7, 8, 9]
    
    
    parameters = earth.getParameter()
    tt = time.time()
    parameters['population'] = np.ceil(parameters['population'])
    nAgents = 0
    nHH     = 0
    overheadAgents = 1000 # additional agents that are loaded 
    tmp = np.unique(parameters['regionIdRaster'])
    tmp = tmp[~np.isnan(tmp)]
    regionIdxList = tmp[tmp>0]
    #print regionIdxList
    #import pdb
    #pdb.set_trace()
    nRegions = np.sum(tmp>0)

    mpi = earth.returnMpiComm()

    boolMask = parameters['landLayer']==mpi.rank
    nAgentsOnProcess = np.zeros(nRegions)
    for i, region in enumerate(regionIdxList):
        boolMask2 = parameters['regionIdRaster']== region
        nAgentsOnProcess[i] = np.sum(parameters['population'][boolMask & boolMask2])

        if nAgentsOnProcess[i] > 0:
            # calculate start in the agent file (20 overhead for complete households)
            nAgentsOnProcess[i] += overheadAgents

    #print mpiRank, nAgentsOnProcess
    nAgentsPerProcess = earth.mpi.all2all(nAgentsOnProcess)
    #print nAgentsPerProcess
    nAgentsOnProcess = np.array(nAgentsPerProcess)
    lg.info('Agents on process:' + str(nAgentsOnProcess))
    hhData = dict()
    currIdx = dict()
    h5Files = dict()

    for i, region in enumerate(regionIdxList):
        # all processes open all region files (not sure if necessary)
        # h5Files[i]      = h5py.File(parameters.resourcePath + 'people' + str(int(region)) + '.hdf5', 'r', driver='mpio', comm=earth.mpi.comm, info=earth.mpi.comm.info)
        h5Files[i]      = h5py.File(parameters['resourcePath'] + 'people' + str(int(region)) + 'new.hdf5', 'r')
    mpi.Barrier()

    for i, region in enumerate(regionIdxList):
#        if i >0:
#            offset= 398700
#        else:

        offset = 0


        agentStart = int(np.sum(nAgentsOnProcess[:mpi.rank,i]))
        agentEnd   = int(np.sum(nAgentsOnProcess[:mpi.rank+1,i]))

        lg.info('Reading agents from ' + str(agentStart) + ' to ' + str(agentEnd) + ' for region ' + str(region))
        lg.debug('Vertex count: ' + str(earth.graph.vcount()))
        
        if earth.debug:
            pass
            #earth.view(str(earth.mpi.rank) + '.png')


        dset = h5Files[i].get('people')
        hhData[i] = dset[offset + agentStart: offset + agentEnd,]
        #print hhData[i].shape
        
        if nAgentsOnProcess[mpi.rank, i] == 0:
            continue

        assert hhData[i].shape[0] >= nAgentsOnProcess[mpi.rank,i] ##OPTPRODUCTION
        
        idx = 0
        # find the correct possition in file
        nPers = int(hhData[i][idx, 0])
        if np.sum(np.diff(hhData[i][idx:idx+nPers, H5NPERS])) !=0:

            #new index for start of a complete household
            idx = idx + np.where(np.diff(hhData[i][idx:idx+nPers, H5NPERS]) != 0)[0][0]
        currIdx[i] = int(idx)


    mpi.Barrier() # all loading done

    for i, region in enumerate(regionIdxList):
        h5Files[i].close()
    lg.info('Loading agents from file done')

    opinion     = Opinion(earth)
    nAgentsCell = 0
    locDict = earth.getLocationDict()
    
    
    for x, y in locDict.keys():
        #print x,y
        nAgentsCell = int(parameters['population'][x, y]) + nAgentsCell # subtracting Agents that are places too much in the last cell
        loc         = earth.getEntity(locDict[x, y].nID)
        region      = parameters['regionIdRaster'][x, y]
        regionIdx   = np.where(regionIdxList == region)[0][0]

        while 1:
            successFlag = False
            nPers   = int(hhData[regionIdx][currIdx[regionIdx], H5NPERS])
            #print nPers,'-',nAgents
            ages    = list(hhData[regionIdx][currIdx[regionIdx]:currIdx[regionIdx]+nPers, H5AGE])
            genders = list(hhData[regionIdx][currIdx[regionIdx]:currIdx[regionIdx]+nPers, H5GENDER])
            
            nAdults = np.sum(np.asarray(ages)>= 18)
            nKids = np.sum(np.asarray(ages) < 18)
            
            if nAdults == 0:
                currIdx[regionIdx]  += nPers
                lg.info('Household without adults skipped')
                continue
                
            if currIdx[regionIdx] + nPers > hhData[regionIdx].shape[0]:
                print 'Region: ' + str(regionIdxList[regionIdx])
                print 'asked size: ' + str(currIdx[regionIdx] + nPers)
                print 'hhDate shape: ' + str(hhData[regionIdx].shape)

            income = hhData[regionIdx][currIdx[regionIdx], H5INCOME]
            hhType = hhData[regionIdx][currIdx[regionIdx], H5HHTYPE]

            # set minimal income
            #income *= (1.- (0.1 * max(3, nKids))) # reduction fo effective income by kids
            income = max(400., income)
            income *= parameters['mobIncomeShare'] 


            nJourneysPerPerson = hhData[regionIdx][currIdx[regionIdx]:currIdx[regionIdx]+nPers, H5MOBDEM]


            # creating houshold
            hh = Household(earth,
                           pos=(x, y),
                           hhSize=nPers,
                           nKids=nKids,
                           income=income,
                           expUtil=0,
                           util=0,
                           expenses=0,
                           hhType=hhType)

            hh.adults = list()
            hh.register(earth, parentEntity=loc, edgeType=CON_LH)
            #hh.registerAtLocation(earth,x,y,HH,CON_LH)

            hh.loc.addValue('population',  nPers)
            
            assert nAdults > 0 ##OPTPRODUCTION
            
            for iPers in range(nPers):

                nAgentsCell -= 1
                nAgents     += 1

                if ages[iPers] < 18:
                    continue    #skip kids
                prefTuple = opinion.getPref(ages[iPers], genders[iPers], nKids, nPers, income, parameters['radicality'])

                
                assert len(nJourneysPerPerson[iPers]) == 5##OPTPRODUCTION
                pers = Person(earth,
                              preferences = np.asarray(prefTuple),
                              hhID        = hh.gID,
                              gender      = genders[iPers],
                              age         = ages[iPers],
                              nJourneys   = nJourneysPerPerson[iPers],
                              util        = 0.,
                              commUtil    = np.asarray([0.5, 0.1, 0.4, 0.3, 0.1]), # [0.5]*parameters['nMobTypes'],
                              selfUtil    = np.asarray([np.nan]*parameters['nMobTypes']),
                              mobType     = 0,
                              prop        = np.asarray([0.]*len(parameters['properties'])),
                              consequences= np.asarray([0.]*len(prefTuple)),
                              lastAction  = 0,
                              hhType      = hhType,
                              emissions   = 0.)
                
                pers.imitation = np.random.randint(parameters['nMobTypes'])
                pers.register(earth, parentEntity=hh, edgeType=CON_HP)
                
                successFlag = True
            
            
            currIdx[regionIdx]  += nPers
            nHH                 += 1
            if not successFlag:
                import pdb
                pdb.set_trace()
            if nAgentsCell <= 0:
                break
    lg.info('All agents initialized')

    if earth.queuing:
        earth.queue.dequeueVertices(earth)
        earth.queue.dequeueEdges(earth)

    earth.mpi.transferGhostNodes(earth)
    #earth.mpi.comm.Barrier()
    #earth.mpi.recvGhostNodes(earth)

    if earth.queuing:
        earth.queue.dequeueVertices(earth)
        earth.queue.dequeueEdges(earth)
    #earth.graph.write_graphml('graph' +str(earth.mpi.rank) + '.graphML')
    #earth.view(str(earth.mpi.rank) + '.png')

#    for ghostCell in earth.iterEntRandom(CELL, ghosts = True, random=False):
#        if mpiRank == 0:
#            print ghostCell.peList
#        ghostCell.hhList = ghostCell.updateAgentList(earth.graph, CON_LH)
#        if mpiRank == 0:
#            print ghostCell.peList
#        asd
#        #ghostCell.peList = ghostCell.updatePeList(earth.graph, CON_)
#        #ghostCell.peList = None
#        print ghostCell.peList
        
    for hh in earth.iterEntRandom(HH, ghosts = False, random=False):
        # caching all adult node in one hh
        hh.setAdultNodeList(earth)
        assert len(hh.adults) == hh.getValue('hhSize') - hh.getValue('nKids')  ##OPTPRODUCTION
        
    earth.mpi.comm.Barrier()
    lg.info(str(nAgents) + ' Agents and ' + str(nHH) +
            ' Housholds created in -- ' + str(time.time() - tt) + ' s')
    
    if mpiRank == 0:
        print'Household setup done'
            
    return earth




def initEarth(simNo,
              outPath,
              parameters,
              maxNodes,
              debug,
              mpiComm=None,
              caching=True,
              queuing=True):
    tt = time.time()

    earth = Earth(simNo,
                  outPath,
                  parameters,
                  maxNodes=maxNodes,
                  debug=debug,
                  mpiComm=mpiComm,
                  caching=caching,
                  queuing=queuing)


    earth.initMarket(earth,
                     parameters.properties,
                     parameters.randomCarPropDeviationSTD,
                     burnIn=parameters.burnIn)

    earth.market.mean = np.array([400., 300.])
    earth.market.std  = np.array([100., 50.])
    
    earth.market.initExogenousExperience(parameters['scenario'])

    
    #init location memory
    earth.enums = dict()



    earth.enums['priorities']    = dict()
    earth.enums['priorities'][0] = 'convinience'
    earth.enums['priorities'][1] = 'ecology'
    earth.enums['priorities'][2] = 'money'
    earth.enums['priorities'][3] = 'imitation'

    earth.enums['properties']    = dict()
    earth.enums['properties'][1] = 'emissions'
    earth.enums['properties'][2] = 'TCO'

    earth.enums['nodeTypes']    = dict()
    earth.enums['nodeTypes'][1] = 'cell'
    earth.enums['nodeTypes'][2] = 'household'
    earth.enums['nodeTypes'][3] = 'pers'

    earth.enums['consequences']    = dict()
    earth.enums['consequences'][0] = 'convenience'
    earth.enums['consequences'][1] = 'eco-friendliness'
    earth.enums['consequences'][2] = 'remaining money'
    earth.enums['consequences'][3] = 'innovation'

    earth.enums['mobilityTypes']    = dict()
    earth.enums['mobilityTypes'][1] = 'green'
    earth.enums['mobilityTypes'][0] = 'brown'
    earth.enums['mobilityTypes'][2] = 'public transport'
    earth.enums['mobilityTypes'][3] = 'shared mobility'
    earth.enums['mobilityTypes'][4] = 'None motorized'

    lg.info('Init finished after -- ' + str( time.time() - tt) + ' s')
    if mpiRank == 0:
        print'Earth init done'
    return earth

def initTypes(earth):
    parameters = earth.getParameter()
    CELL    = earth.registerNodeType('cell' , AgentClass=Cell, GhostAgentClass= GhostCell,
                               staticProperies  = ['type',
                                                   'gID',
                                                   'pos',
                                                   'regionId',
                                                   'popDensity',
                                                   'population'],
                               dynamicProperies = ['convenience',
                                                   'carsInCell',
                                                   'chargStat',
                                                   'emissions',
                                                   'electricConsumption'])


    HH = earth.registerNodeType('hh', AgentClass=Household, GhostAgentClass= GhostHousehold,
                               staticProperies  = ['type',
                                                   'gID',
                                                   'pos',
                                                   'hhSize',
                                                   'nKids',
                                                   'hhType'],
                               dynamicProperies =  ['income',
                                                   'expUtil',
                                                   'util',
                                                   'expenses'])


    PERS = earth.registerNodeType('pers', AgentClass=Person, GhostAgentClass= GhostPerson,
                                staticProperies = ['type',
                                                   'gID',
                                                   'hhID',
                                                   'preferences',
                                                   'gender',
                                                   'nJourneys',
                                                   'hhType'],
                                dynamicProperies = ['age',
                                                   'util',     # current utility
                                                   'commUtil', # comunity utility
                                                   'selfUtil', # own utility at time of action
                                                   'mobType',
                                                   'prop',
                                                   'consequences',
                                                   'lastAction',
                                                   'emissions'])


    earth.registerEdgeType('cell-cell', CELL, CELL, ['type','weig'])
    earth.registerEdgeType('cell-hh', CELL, HH)
    earth.registerEdgeType('hh-hh', HH,HH)
    earth.registerEdgeType('hh-pers', HH, PERS)
    earth.registerEdgeType('pers-pers', PERS, PERS, ['type','weig'])

    if mpiRank == 0:
        print'Initialization of types done'

    return CELL, HH, PERS

def initSpatialLayer(earth):
    parameters = earth.getParameter()
    connList= aux.computeConnectionList(parameters['connRadius'], ownWeight=1.5)
    earth.initSpatialLayer(parameters['landLayer'],
                           connList, CELL,
                           LocClassObject=Cell,
                           GhstLocClassObject=GhostCell)
    
    convMat = np.asarray([[0., 1, 0.],[1., 1., 1.],[0., 1., 0.]])
    tmp = parameters['population']*parameters['reductionFactor']
    tmp[np.isnan(tmp)] = 0
    smoothedPopulation = signal.convolve2d(tmp,convMat,boundary='symm',mode='same')
    tmp = parameters['cellSizeMap']
    tmp[np.isnan(tmp)] = 0
    smoothedCellSize   = signal.convolve2d(tmp,convMat,boundary='symm',mode='same')

    
    popDensity = np.divide(smoothedPopulation, 
                           smoothedCellSize, 
                           out=np.zeros_like(smoothedPopulation), 
                           where=smoothedCellSize!=0)
    popDensity[popDensity>4000.]  = 4000.
    
#    plt.clf()
#    plt.imshow(popDensity)
#    plt.clim([0, np.nanpercentile(popDensity,100)])
#    plt.colorbar()
    
    if 'regionIdRaster' in parameters.keys():

        for cell in earth.iterEntRandom(CELL):
            cell.setValue('regionId', parameters['regionIdRaster'][cell._node['pos']])
            cell.setValue('chargStat', 0)
            cell.setValue('emissions', np.zeros(len(earth.enums['mobilityTypes'])))
            cell.setValue('electricConsumption', 0.)
            cell.cellSize = parameters['cellSizeMap'][cell._node['pos']]
            cell.setValue('popDensity', popDensity[cell._node['pos']])
            
    earth.mpi.updateGhostNodes([CELL],['chargStat'])

    if mpiRank == 0:
        print'Setup of the spatial layer done'

def initInfrastructure(earth):
    # infrastructure
    earth.initChargInfrastructure()
    
    if mpiRank == 0:
        print'Infrastructure setup done'

#%% cell convenience test
def cellTest(earth):

    for good in earth.market.goods.values():
        
        good.emissionFunction(good, earth.market)
        #good.updateMaturity()  
        
    nLocations = len(earth.getLocationDict())
    convArray  = np.zeros([earth.market.getNMobTypes(), nLocations])
    popArray   = np.zeros(nLocations)
    eConvArray = earth.para['landLayer'] * 0
    
    #import tqdm
    #for i, cell in tqdm.tqdm(enumerate(earth.iterEntRandom(CELL))):
    if earth.para['showFigures']:
        for i, cell in enumerate(earth.iterEntRandom(CELL)):        
            #tt = time.time()
            convAll, popDensity = cell.selfTest(earth)
            #cell.setValue('carsInCell',[0,200.,0,0,0])
            convAll[1] = convAll[1] * cell.electricInfrastructure(100.)
            convArray[:, i] = convAll
            popArray[i] = popDensity
            eConvArray[cell.getValue('pos')] = convAll[1]
            #print time.time() - ttclass
        
        
            
        plt.figure('electric infrastructure convenience')
        plt.clf()
        plt.subplot(2,2,1)
        plt.imshow(eConvArray)
        plt.title('el. convenience')
        plt.clim([-.2,np.nanmax(eConvArray)])
        plt.colorbar()
#        plt.subplot(2,2,2)
#        plt.imshow(earth.para['chargStat'])
        plt.clim([-2,10])
        plt.title('number of charging stations')
        plt.colorbar()
        
        plt.subplot(2,2,3)
        plt.imshow(earth.para['landLayer'])
        #plt.clim([-2,10])
        plt.title('processes ID')
        
        plt.subplot(2,2,4)
        plt.imshow(earth.para['population'])
        #plt.clim([-2,10])
        plt.title('population')
        
        
        plt.figure()
        for i in range(earth.market.getNMobTypes()):
            if earth.market.getNMobTypes() > 4:
                plt.subplot(3, int(np.ceil(earth.market.getNMobTypes()/3.)), i+1)
            else:
                plt.subplot(2, 2, i+1)
            plt.scatter(popArray,convArray[i,:], s=2)
            plt.title('convenience of ' + earth.enums['mobilityTypes'][i])
        plt.show()
        adsf
        
# %% Generate Network
def generateNetwork(earth):
    parameters = earth.getParameter()
    
    tt = time.time()
    
    earth.generateSocialNetwork(PERS,CON_PP)
    
    lg.info( 'Social network initialized in -- ' + str( time.time() - tt) + ' s')
    if parameters['scenario'] == 0 and earth.para['showFigures']:
        earth.view(str(earth.mpi.rank) + '.png')
    if mpiRank == 0:
        print'Social network setup done'


def initMobilityTypes(earth):
    earth.market.initialCarInit()
    earth.market.setInitialStatistics([1000.,2.,350., 100.,50.])
    for goodKey in earth.market.goods.keys():##OPTPRODUCTION
        #print earth.market.goods[goodKey].properties.keys() 
        #print earth.market.properties
        assert earth.market.goods[goodKey].properties.keys() == earth.market.properties ##OPTPRODUCTION
    
    if mpiRank == 0:
        print'Setup of mobility types done'


def initGlobalRecords(earth):
    parameters = earth.getParameter()

    calDataDfCV = pd.read_csv(parameters['resourcePath'] + 'calDataCV.csv', index_col=0, header=1)
    calDataDfEV = pd.read_csv(parameters['resourcePath'] + 'calDataEV.csv', index_col=0, header=1)

    for re in parameters['regionIDList']:
        earth.registerRecord('stock_' + str(re),
                         'total use per mobility type -' + str(re),
                         earth.enums['mobilityTypes'].values(),
                         style='plot',
                         mpiReduce='sum')
    
        earth.registerRecord('elDemand_' + str(re),
                         'electric Demand -' + str(re),
                         ['electric_demand'],
                         style='plot',
                         mpiReduce='sum')
        
        earth.registerRecord('emissions_' + str(re),
                         'co2Emissions -' + str(re),
                         earth.enums['mobilityTypes'].values(),
                         style='plot',
                         mpiReduce='sum')

        earth.registerRecord('nChargStations_' + str(re),
                         'Number of charging stations -' + str(re),
                         ['nChargStations'],
                         style='plot',
                         mpiReduce='sum')

        timeIdxs = list()
        values   = list()

        for column in calDataDfCV.columns[1:]:
            value = [np.nan]*earth.para['nMobTypes']
            year = int(column)
            timeIdx = 12* (year - parameters['startDate'][1]) + parameters['burnIn']
            value[0] = (calDataDfCV[column]['re_' + str(re)] ) #/ parameters['reductionFactor']
            if column in calDataDfEV.columns[1:]:
                value[1] = (calDataDfEV[column]['re_' + str(re)] ) #/ parameters['reductionFactor']


            timeIdxs.append(timeIdx)
            values.append(value)

        earth.globalRecord['stock_' + str(re)].addCalibrationData(timeIdxs,values)

    earth.registerRecord('growthRate', 'Growth rate of mobitlity types',
                         earth.enums['mobilityTypes'].values(), style='plot')
    earth.registerRecord('allTimeProduced', 'Overall production of car types',
                         earth.enums['mobilityTypes'].values(), style='plot')
    earth.registerRecord('maturities', 'Technological maturity of mobility types',
                         ['mat_B', 'mat_G', 'mat_P', 'mat_S', 'mat_N'], style='plot')
    earth.registerRecord('globEmmAndPrice', 'Properties',
                         ['meanEmm','stdEmm','meanPrc','stdPrc'], style='plot')

    if mpiRank == 0:
        print'Setup of global records done'

def initAgentOutput(earth):
    #%% Init of agent file
    tt = time.time()
    earth.mpi.comm.Barrier()
    lg.info( 'Waited for Barrier for ' + str( time.time() - tt) + ' s')
    tt = time.time()
    #earth.initAgentFile(typ = HH)
    #earth.initAgentFile(typ = PERS)
    #earth.initAgentFile(typ = CELL)
    earth.io.initNodeFile(earth, [CELL, HH, PERS])


    lg.info( 'Agent file initialized in ' + str( time.time() - tt) + ' s')

    if mpiRank == 0:
        print'Setup of agent output done'


def initCacheArrays(earth):
    maxFriends = earth.para['maxFriends']
    persZero = earth.entDict[earth.nodeDict[PERS][0]]
    
    nUtil = persZero.getValue('commUtil').shape[0]
    Person.cacheCommUtil = np.zeros([maxFriends+1, nUtil])
    Person.cacheUtil     = np.zeros(maxFriends+1)
    Person.cacheMobType  = np.zeros(maxFriends+1, dtype=np.int32)
    Person.cacheWeights  = np.zeros(maxFriends+1)

def initExogeneousExperience(parameters):
    inputFromGlobal         = pd.read_csv(parameters['resourcePath'] + 'inputFromGlobal.csv')
    randomFactor = (5*np.random.randn() + 100.)/100
    parameters['experienceWorldGreen']  = inputFromGlobal['expWorldGreen'].values / 10. * randomFactor
    randomFactor = (5*np.random.randn() + 100.)/100
    parameters['experienceWorldBrown']  = inputFromGlobal['expWorldBrown'].values * randomFactor
    experienceGer                       = inputFromGlobal['expGer'].values
    experienceGerGreen                  = inputFromGlobal['expGerGreen'].values
    parameters['experienceGerGreen']    = experienceGerGreen
    parameters['experienceGerBrown']    = [experienceGer[i]-experienceGerGreen[i] for i in range(len(experienceGer))]
    
    #fake model
#    def func(year):
#        a = -654.47792283470665
#        b = 0.33129306614034293
#        stockGlE = np.exp(b * year+a)
#        return stockGlE
#    
#    years = range(2005, 2036)
#    
#    correctedData = [func(year) for year in years]
#    
#    fig = plt.figure()
#    plt.clf()
#    ax = fig.add_subplot(1, 1, 1)
#    plt.plot(years, parameters['experienceWorldGreen'])
#    plt.plot(years, parameters['experienceWorldGreen']/10)
#    plt.plot(years, correctedData)
#    plt.legend(['old', 'old/10', 'corrected'])
#    ax.set_yscale('log')
    
    return parameters

def readParameterFile(parameters, fileName):
    for item in csv.DictReader(open(fileName)):
        if item['name'][0] != '#':
            parameters[item['name']] = aux.convertStr(item['value'])
    return parameters

def randomizeParameters(parameters):
    #%%
    def randDeviation(percent, minDev=-np.inf, maxDev=np.inf):
        while True:
            dev = np.random.randn() *percent
            if dev < maxDev and dev > minDev:
                break
        return (100. + dev) / 100.
    
    maxFriendsRand  = int( parameters['maxFriends'] * randDeviation(20)) 
    if maxFriendsRand > parameters['minFriends']+1:
        parameters['maxFriends'] = maxFriendsRand
    minFriendsRand  = int( parameters['minFriends'] * randDeviation(5)) 
    if minFriendsRand < parameters['maxFriends']-1:
        parameters['minFriends'] = minFriendsRand
    parameters['mobIncomeShare'] * randDeviation(5)
    parameters['charIncome'] * randDeviation(5)
    parameters['innoPriority'] * randDeviation(5)
    parameters['individualPrio'] * randDeviation(10)
    parameters['priceRedBCorrection'] * randDeviation(3, -3, 3)
    parameters['priceRedGCorrection'] * randDeviation(3, -3, 3)
    
    parameters['hhAcceptFactor'] = 1.0 + (np.random.rand()*5. / 100.) #correct later
    return parameters
# %% Online processing functions

def plot_calGreenNeigbourhoodShareDist(earth):
    if parameters.showFigures:
        nPersons = len(earth.getNodeDict(PERS))
        relarsPerNeigborhood = np.zeros([nPersons,3])
        for i, persId in enumerate(earth.getNodeDict(PERS)):
            person = earth.getEntity(persId)
            x,__ = person.getConnNodeValues('mobType',PERS)
            for mobType in range(3):
                relarsPerNeigborhood[i,mobType] = float(np.sum(np.asarray(x)==mobType))/len(x)

        n, bins, patches = plt.hist(relarsPerNeigborhood, 30, normed=0, histtype='bar',
                                label=['brown', 'green', 'other'])
        plt.legend()

def plot_incomePerNetwork(earth):

    incomeList = np.zeros([len(earth.nodeDict[PERS]),1])
    for i, persId in enumerate(earth.nodeDict[PERS]):
        person = earth.entDict[persId]
        x, friends = person.getConnNodeValues('mobType',PERS)
        incomes = [earth.entDict[friend].hh.getValue('income') for friend in friends]
        incomeList[i,0] = np.mean(incomes)

    n, bins, patches = plt.hist(incomeList, 20, normed=0, histtype='bar',
                            label=['average imcome '])
    plt.legend()

def plot_computingTimes(earth):
    plt.figure()

    allTime = np.zeros(earth.nSteps)
    colorPal =  sns.color_palette("Set2", n_colors=5, desat=.8)


    for i,var in enumerate([earth.computeTime, earth.waitTime, earth.syncTime, earth.ioTime]):
        plt.bar(np.arange(earth.nSteps), var, bottom=allTime, color =colorPal[i], width=1)
        allTime += var
    plt.legend(['compute time', 'wait time', 'sync time', 'I/O time'])
    plt.tight_layout()
    plt.ylim([0, np.percentile(allTime,99)])
    plt.savefig(earth.para['outPath'] + '/' + str(mpiRank) + 'times.png')

    #%%
def runModel(earth, parameters):

    #%% Initial actions
    tt = time.time()
    for household in earth.iterEntRandom(HH):

        household.takeActions(earth, household.adults, np.random.randint(0, earth.market.getNMobTypes(), len(household.adults)))
        for adult in household.adults:
            adult.setValue('lastAction', int(np.random.rand() * float(earth.para['mobNewPeriod'])))

    lg.info('Initial actions done')

    for cell in earth.iterEntRandom(CELL):
        cell.step(earth.para, earth.market.getCurrentMaturity())
     
    
    earth.market.initPrices()
    for good in earth.market.goods.values():
        good.initMaturity()
        good.updateEmissionsAndMaturity(earth.market)
        #good.updateMaturity()
    
    lg.info('Initial market step done')

    for household in earth.iterEntRandom(HH):
        household.calculateConsequences(earth.market)
        household.util = household.evalUtility(earth, actionTaken=True)
        #household.shareExperience(earth)
        
        
    lg.info('Initial actions randomized in -- ' + str( time.time() - tt) + ' s')

    initCacheArrays(earth)
    
    #plotIncomePerNetwork(earth)


    #%% Simulation
    earth.time = -1 # hot bugfix to have both models running #TODO Fix later
    lg.info( "Starting the simulation:")
    for step in xrange(parameters.nSteps):

        earth.step() # looping over all cells

        #plt.figure()
        #plot_calGreenNeigbourhoodShareDist(earth)
        #plt.show()




    #%% Finishing the simulation
    lg.info( "Finalizing the simulation (No." + str(earth.simNo) +"):")
    if parameters.writeOutput:
        earth.io.finalizeAgentFile()
    earth.finalize()

def writeSummary(earth, parameters):

    fid = open(earth.para['outPath'] + '/summary.out','w')

    fid.writelines('Parameters:')

    errorTot = 0
    for re in earth.para['regionIDList']:
        error = earth.globalRecord['stock_' + str(re)].evaluateRelativeError()
        fid.writelines('Error - ' + str(re) + ': ' + str(error) + '\n')
        errorTot += error
    #fid = open('summary.out','w')


    #fid.writelines('Bremen: ' + str(errBremen))
    #fid.writelines('Niedersachsen: ' + str(errNiedersachsen))

    fid.writelines('Total error:' + str(errorTot))
    #fid.writelines(str(errorTot))
    fid.close()
    #lg.info( 'Calibration Run: ' + str(calRunId))
    #lg.info( paraDf)
    lg.info( 'The simulation error is: ' + str(errorTot) )


    if parameters.scenario == 2:
        nPeople = np.nansum(parameters.population)

        nCars      = float(np.nansum(np.array(earth.graph.vs[earth.nodeDict[PERS]]['mobType']) != 2))
        nGreenCars = float(np.nansum(np.array(earth.graph.vs[earth.nodeDict[PERS]]['mobType']) == 1))
        nBrownCars = float(np.nansum(np.array(earth.graph.vs[earth.nodeDict[PERS]]['mobType']) == 0))

        lg.info('Number of agents: ' + str(nPeople))
        lg.info('Number of agents: ' + str(nCars))
        lg.info('cars per 1000 people: ' + str(nCars/nPeople*1000.))
        lg.info('green cars per 1000 people: ' + str(nGreenCars/nPeople*1000.))
        lg.info('brown cars per 1000 people: ' + str(nBrownCars/nPeople*1000.))




    elif parameters.scenario == 3:

        nPeople = np.nansum(parameters.population)

        nCars      = float(np.nansum(np.array(earth.graph.vs[earth.nodeDict[PERS]]['mobType'])!=2))
        nGreenCars = float(np.nansum(np.array(earth.graph.vs[earth.nodeDict[PERS]]['mobType'])==1))
        nBrownCars = float(np.nansum(np.array(earth.graph.vs[earth.nodeDict[PERS]]['mobType'])==0))

        lg.info( 'Number of agents: ' + str(nPeople))
        lg.info( 'Number of agents: ' + str(nCars))
        lg.info( 'cars per 1000 people: ' + str(nCars/nPeople*1000.))
        lg.info( 'green cars per 1000 people: ' + str(nGreenCars/nPeople*1000.))
        lg.info( 'brown cars per 1000 people: ' + str(nBrownCars/nPeople*1000.))


        cellList       = earth.graph.vs[earth.nodeDict[CELL]]
        cellListBremen = cellList.select(regionId_eq=1518)
        cellListNieder = cellList.select(regionId_eq=6321)
        cellListHamb   = cellList.select(regionId_eq=1520)

        carsInBremen = np.asarray(cellListBremen['carsInCell'])
        carsInNieder = np.asarray(cellListNieder['carsInCell'])
        carsInHamb   = np.asarray(cellListHamb['carsInCell'])


        nPeopleBremen = np.nansum(parameters.population[parameters.regionIdRaster==1518])
        nPeopleNieder = np.nansum(parameters.population[parameters.regionIdRaster==6321])
        nPeopleHamb   = np.nansum(parameters.population[parameters.regionIdRaster==1520])

        lg.info('shape: ' + str(carsInBremen.shape))
        lg.info( 'Bremem  - green cars per 1000 people: ' + str(np.sum(carsInBremen[:, 1])/np.sum(nPeopleBremen)*1000))
        lg.info( 'Bremem  - brown cars per 1000 people: ' + str(np.sum(carsInBremen[:, 0])/np.sum(nPeopleBremen)*1000))


        lg.info( 'Niedersachsen - green cars per 1000 people: ' + str(np.sum(carsInNieder[:, 1])/np.sum(nPeopleNieder)*1000))
        lg.info( 'Niedersachsen - brown cars per 1000 people: ' + str(np.sum(carsInNieder[:, 0])/np.sum(nPeopleNieder)*1000))


        lg.info( 'Hamburg       - green cars per 1000 people: ' + str(np.sum(carsInNieder[:, 1])/np.sum(nPeopleHamb)*1000))
        lg.info( 'Hamburg -       brown cars per 1000 people: ' + str(np.sum(carsInHamb[:, 0])/np.sum(nPeopleHamb)*1000))


def onlinePostProcessing(earth):
    # calculate the mean and standart deviation of priorities
    df = pd.DataFrame([],columns=['prCon','prEco','prMon','prImi'])
    for agID in earth.nodeDict[3]:
        df.loc[agID] = earth.graph.vs[agID]['preferences']


    lg.info('Preferences -average')
    lg.info(df.mean())
    lg.info('Preferences - standart deviation')
    lg.info(df.std())

    lg.info( 'Preferences - standart deviation within friends')
    avgStd= np.zeros([1, 4])
    for agent in earth.iterEntRandom(HH):
        friendList = agent.getPeerIDs(edgeType=CON_HH)
        if len(friendList) > 1:
            #print df.ix[friendList].std()
            avgStd += df.ix[friendList].std().values
    nAgents    = np.nansum(parameters.population)
    lg.info(avgStd / nAgents)
    prfType = np.argmax(df.values,axis=1)
    #for i, agent in enumerate(earth.iterNode(HH)):
    #    print agent.prefTyp, prfType[i]
    df['ref'] = prfType

    # calculate the correlation between weights and differences in priorities
    if False:
        pref = np.zeros([earth.graph.vcount(), 4])
        pref[earth.nodeDict[PERS],:] = np.array(earth.graph.vs[earth.nodeDict[PERS]]['preferences'])
        idx = list()
        for edge in earth.iterEdges(CON_PP):
            edge['prefDiff'] = np.sum(np.abs(pref[edge.target, :] - pref[edge.source,:]))
            idx.append(edge.index)


        plt.figure()
        plt.scatter(np.asarray(earth.graph.es['prefDiff'])[idx],np.asarray(earth.graph.es['weig'])[idx])
        plt.xlabel('difference in preferences')
        plt.ylabel('connections weight')

        plt.show()
        x = np.asarray(earth.graph.es['prefDiff'])[idx].astype(float)
        y = np.asarray(earth.graph.es['weig'])[idx].astype(float)
        lg.info( np.corrcoef(x,y))


#%%############### Tests ############################################################


def prioritiesCalibrationTest():

    householdSetup(earth, parameters, calibration=True)
    df = pd.DataFrame([],columns=['prCon','prEco','prMon','prImi'])
    for agID in earth.nodeDict[3]:
        df.loc[agID] = earth.graph.vs[agID]['preferences']

    propMat = np.array(np.matrix(earth.graph.vs[earth.nodeDict[3]]['preferences']))

    return earth


def setupHouseholdsWithOptimalChoice():

    householdSetup(earth, parameters)
    initMobilityTypes(earth, parameters)
    #earth.market.setInitialStatistics([500.0,10.0,200.0])
    for household in earth.iterEntRandom(HH):
        household.takeAction(earth, household.adults, np.random.randint(0,earth.market.nMobTypes,len(household.adults)))

    for cell in earth.iterEntRandom(CELL):
        cell.step(earth.market.kappa)

    earth.market.setInitialStatistics([1000.,5.,300.])

    for household in earth.iterEntRandom(HH):
        household.calculateConsequences(earth.market)
        household.util = household.evalUtility()

    for hh in iter(earth.nodeDict[HH]):
        oldEarth = copy(earth)
        earth.entDict[hh].bestMobilityChoice(oldEarth,forcedTryAll = True)
    return earth



#%% __main__()
######################################################################################

if __name__ == '__main__':


    debug = 1
    showFigures    = 0
    
    simNo, baseOutputPath = aux.getEnvironment(comm, getSimNo=True)
    outputPath = aux.createOutputDirectory(comm, baseOutputPath, simNo)
    
    import logging as lg
    import time
    
    #exit()
    
    if debug:
        lg.basicConfig(filename=outputPath + '/log_R' + str(mpiRank),
                    filemode='w',
                    format='%(levelname)7s %(asctime)s : %(message)s',
                    datefmt='%m/%d/%y-%H:%M:%S',
                    level=lg.DEBUG)
    else:
        lg.basicConfig(filename=outputPath + '/log_R' + str(mpiRank),
                        filemode='w',
                        format='%(levelname)7s %(asctime)s : %(message)s',
                        datefmt='%m/%d/%y-%H:%M:%S',
                        level=lg.INFO)
    
    lg.info('Log file of process '+ str(mpiRank) + ' of ' + str(mpiSize))
    
    # wait for all processes - debug only for poznan to debug segmentation fault
    comm.Barrier()
    if comm.rank == 0:
        print 'log files created'
    
    lg.info('on node: ' + socket.gethostname())
    dirPath = os.path.dirname(os.path.realpath(__file__))
    
    # loading of standart parameters
    if len(sys.argv) > 1:
    
        fileName = sys.argv[1]
        parameters = Bunch()
        # reading of gerneral parameters
        parameters = readParameterFile(parameters, 'parameters_all.csv')
        # reading of scenario-specific parameters
        parameters = readParameterFile(parameters,fileName)
        
        
        lg.info('Setting loaded:')
        
        
        parameters['outPath'] = outputPath
        
        
        scenarioDict = dict()
        
        scenarioDict[0] = scenarioTestSmall
        scenarioDict[1] = scenarioTestMedium
        scenarioDict[2] = scenarioLueneburg
        scenarioDict[3] = scenarioNBH
        scenarioDict[6] = scenarioGer
        
        
        if mpiRank == 0:
            parameters = scenarioDict[parameters.scenario] (parameters, dirPath)
            
            parameters = initExogeneousExperience(parameters)
            parameters = randomizeParameters(parameters)   
            
        else:
            parameters = None
        

         
        parameters = comm.bcast(parameters,root=0)    
            
        if mpiRank == 0:
            print'Parameter exchange done'
        lg.info( 'Parameter exchange done')
        

        #%% Init
        parameters.showFigures = showFigures
        
        earth = initEarth(simNo,
                          outputPath,
                          parameters,
                          maxNodes=1000000,
                          debug =debug,
                          mpiComm=comm,
                          caching=True,
                          queuing=True)
        
        CELL, HH, PERS = initTypes(earth)
        
        initSpatialLayer(earth)
        
        initInfrastructure(earth)
        
        mobilitySetup(earth)
        
        cellTest(earth)
        
        initGlobalRecords(earth)
        
        householdSetup(earth)
        
        generateNetwork(earth)
        
        initMobilityTypes(earth)
        
        initAgentOutput(earth)
        
        cell = earth.entDict[0]
        #cell.setWorld(earth)
        
        if parameters.scenario == 0:
            earth.view('output/graph.png')
        
        #%% run of the model ################################################
        lg.info('####### Running model with paramertes: #########################')
        import pprint
        lg.info(pprint.pformat(parameters.toDict()))
        if mpiRank == 0:
            fidPara = open(earth.para['outPath'] + '/parameters.txt','w')
            pprint.pprint(parameters.toDict(), fidPara)
            fidPara.close()
        lg.info('################################################################')
        
        runModel(earth, parameters)
        
        lg.info('Simulation ' + str(earth.simNo) + ' finished after -- ' + str( time.time() - overallTime) + ' s')
        
        if earth.isRoot:
            print 'Simulation ' + str(earth.simNo) + ' finished after -- ' + str( time.time() - overallTime) + ' s'
        
        if earth.isRoot:
            writeSummary(earth, parameters)
        
        if earth.para['showFigures']:
        
            onlinePostProcessing(earth)
        
        plot_computingTimes(earth)
    
    
    else:
        parameters = Bunch()
        parameters = scenarioTest(parameters, dirPath)
        parameters = comm.bcast(parameters)
        earth = initEarth(simNo,
                      outputPath,
                      parameters,
                      maxNodes=1000000,
                      debug =debug,
                      mpiComm=comm,
                      caching=True,
                      queuing=True)
    
        CELL, HH, PERS = initTypes(earth)
        
        initSpatialLayer(earth)
        
        initInfrastructure(earth)
        
        earth.mpi.comm.Barrier()
        print "test finished"
        exit()
        






