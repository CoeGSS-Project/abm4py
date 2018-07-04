# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://wwww.globalclimateforum.org

---- MoTMo ----
MOBILITY TRANSIOn MODEL
-- Scenario file --

This file is part on GCFABM.

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
"""
import logging as lg
import numpy as np
import matplotlib.pylab as plt
from scipy import signal



import init_motmo as init
from lib import core

mpiSize = core.mpiSize
def convolutionMatrix(radius, centerWeight):
    convMat = np.zeros([radius * 2 + 1, radius * 2 + 1])
    for dx in np.arange(-radius, radius + 1):
        for dy in np.arange(-radius, radius + 1):
            if dx == 0 and dy == 0:
                convMat[radius + dx, radius + dy] = centerWeight
            else:
                convMat[radius + dx, radius + dy] = 1. / np.sqrt(dx ** 2 + dy ** 2)
    return convMat


def publicTransportSetup(setup):
    """
    computation of the convenience of puplic transport for the Luenburg case
    """
    busMap = np.load(setup.resourcePath + 'nBusStations.npy')
    trainMap = np.load(setup.resourcePath + 'nTrainStations.npy')

    # density proxi
    convMat = convolutionMatrix(5, 2)
    bus_station_density = signal.convolve2d(
        busMap, convMat, boundary='symm', mode='same')  # / sumConv
    bus_station_density[busMap == 0] = 0
    convMat = convolutionMatrix(20, 2)
    train_station_density = signal.convolve2d(
        trainMap, convMat, boundary='symm', mode='same')
    train_station_density[trainMap == 0] = 0
    # convolution of the station influence
    convMat = convolutionMatrix(1, 2) / 2.
    tmp1 = .5 * signal.convolve2d(bus_station_density,
                                  convMat, boundary='symm', mode='same')
    tmp2 = 5 * signal.convolve2d(train_station_density,
                                 convMat, boundary='symm', mode='same')
    convPup = np.log(1 + tmp1 + tmp2)  # one is is used so that log(1) = 0
    setup['conveniencePublic'] = convPup / np.max(convPup) * .6


# %% Scenario definition without calibraton parameters
def scenarioTestSmall(parameterInput, dirPath):
    print('in scenarioTestSmall')
    setup = core.AttrDict()

    # general
    setup.resourcePath = dirPath + '/resources/'
    setup.progressBar = True
    setup.allTypeObservations = False

    # time
    setup.timeUnit = init._month  # unit of time per step
    setup.startDate = [1, 2005]

    a = 60000.
    b = 45000.
    c = 30000.
    d = 25000.
    e = 20000.
    f = 15000.
    g = 10000.
    h = 5000.
    i = 1500.

    # spatial
    setup.reductionFactor = 5000
    setup.isSpatial = True
    setup.spatialRedFactor = 280.
    setup.connRadius = 2.0     # radíus of cells that get an connection
    setup.population = np.asarray([[c, a, b, 0, 0],
                                   [c, b, d, 0, f],
                                   [0, h, i, g, e]])
    setup.landLayer = setup.population.copy().astype(float)
    setup.landLayer[setup.landLayer == 0.] = np.nan
    setup.landLayer = (setup.landLayer * 0.) + 1.
    setup.cellSizeMap = setup.landLayer * 15.
    setup.landLayer[0,:] = 0.
    setup.roadKmPerCell = np.asarray([[1, 5, 3, 0, 0],
                                      [1, 4, 4, 0, 1],
                                      [0, 1, 1, 1, 1]]) / setup.cellSizeMap

    setup.roadKmPerCell[np.isnan(setup.roadKmPerCell)] = 0.
    setup.regionIdRaster = ((setup.landLayer * 0) + 1) * 6321
    # setup.regionIdRaster[0:,0:2]    = ((setup.landLayer[0:,0:2]*0)+1) *1519
    if mpiSize == 1:
        setup.landLayer = setup.landLayer * 0
    
        

    setup.regionIDList = np.unique(
        setup.regionIdRaster[~np.isnan(setup.regionIdRaster)]).astype(int)

    # setup.population = (np.isnan(setup.landLayer)==0)* np.random.randint(3,23,setup.landLayer.shape)
    # setup.population = (np.isnan(setup.landLayer)==0)* 13

    # social
    setup.addYourself = True  # have the agent herself as a friend (have own observation)
    setup.recAgent = []       # reporter agents that return a diary

    # output
    setup.writeAgentFile = 1
    setup.writeNPY = 1
    setup.writeCSV = 0

    # cars and infrastructure
    setup.properties    = ['emissions', 'fixedCosts', 'operatingCosts']

    # agents
    setup.randomAgents = False
    setup.omniscientAgents = False

    # redefinition of setup parameters used for automatic calibration
    setup.update(parameterInput.toDict())

    for paName in ['techExpBrown', 'techExpGreen', 'techExpPublic', 'techExpShared', 'techExpNone',
                   'population']:
        setup[paName] /= setup['reductionFactor']

    lg.info("####################################")

    return setup


def scenarioTestMedium(parameterInput, dirPath):
    setup = core.AttrDict()

    # general
    setup.resourcePath = dirPath + '/resources/'
    setup.allTypeObservations = False
    setup.progressBar = True

    # time
    setup.timeUnit = init._month  # unit of time per step
    setup.startDate = [1, 2005]

    # spatial
    setup.isSpatial = True
    setup.spatialRedFactor = 80.
    setup.landLayer = np.asarray([[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1 , 1, 1],
                                  [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1 , 1, 0],
                                  [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1 , 0, 0],
                                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0 , 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1 , 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1 , 1, 1]])

    setup.cellSizeMap = setup.landLayer * 15.

    setup.roadKmPerCell = np.asarray([[0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 1 , 1, 0],
                                      [0, 1, 0, 0, 0, 0, 1, 2, 3, 1, 0 , 0, 0],
                                      [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0],
                                      [2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0 , 0, 0],
                                      [3, 4, 1, 1, 2, 1, 1, 0, 0, 0, 0 , 0, 1],
                                      [6, 5, 2, 3, 2, 0, 1, 0, 0, 0, 0 , 1, 2]]) 
    setup.roadKmPerCell[setup.roadKmPerCell==0] = 1
    setup.roadKmPerCell = setup.landLayer / setup.roadKmPerCell 

    a = 60000.
    b = 45000.
    c = 30000.
    d = 25000.
    e = 20000.
    f = 15000.
    g = 10000.
    h = 5000.
    i = 1500.
    setup.population = np.asarray([[0, 0, 0, 0, e, d, c, d, h, 0, g, h, i],
                                   [0, c, 0, 0, 0, e, d, d, e, e, f, g, 0],
                                   [b, 0, c, 0, e, e, i, 0, 0, g, f, 0, 0],
                                   [b, c, d, d, e, e, 0, 0, 0, g, i, i, 0],
                                   [a, b, c, c, d, e, f, 0, 0, 0, i, i, i],
                                   [a, a, c, c, d, f, f, 0, 0, 0, i, i, g]])
    del a, b, c, d, e, f, g, h, i

    setup.landLayer = setup.landLayer.astype(float)
    setup.landLayer[setup.landLayer == 0] = np.nan

    if mpiSize == 1:
        setup.landLayer = setup.landLayer*0

    setup.regionIdRaster = ((setup.landLayer*0)+1)*6321
    #setup.regionIdRaster[3:, 0:3] = ((setup.landLayer[3:, 0:3]*0)+1) *1519
    setup.regionIDList = np.unique(
        setup.regionIdRaster[~np.isnan(setup.regionIdRaster)]).astype(int)

    setup.landLayer[:, :5] = setup.landLayer[:, :5] * 0

    # social
    setup.addYourself = True     # have the agent herself as a friend (have own observation)
    setup.recAgent = []       # reporter agents that return a diary

    # output
    setup.writeAgentFile = 0
    setup.writeNPY = 1
    setup.writeCSV = 0

    # cars and infrastructure
    setup.properties    = ['emissions', 'fixedCosts', 'operatingCosts']

    # agents
    setup.randomAgents = False
    setup.omniscientAgents = False

    # redefinition of setup parameters used for automatic calibration
    setup.update(parameterInput.toDict())

    # calculate dependent parameters
    for paName in ['techExpBrown', 'techExpGreen', 'techExpPublic', 'techExpShared', 'techExpNone',
                   'population']:
        setup[paName] /= setup['reductionFactor']

    return setup


def scenarioNBH(parameterInput, dirPath):
    setup = core.AttrDict()

    # general
    setup.resourcePath = dirPath + '/resources_NBH/'
    # setup.synPopPath = setup['resourcePath'] + 'hh_NBH_1M.csv'
    setup.progressBar = True
    setup.allTypeObservations = True

    # time
    setup.nSteps = 340     # number of simulation steps
    setup.timeUnit = init._month  # unit of time per step
    setup.startDate = [1, 2005]
    setup.burnIn = 100
    setup.omniscientBurnIn = 10  # no. of first steps of burn-in phase with omniscient agents, max. =burnIn

    # spatial
    setup.isSpatial = True

    if hasattr(parameterInput, "reductionFactor"):
        # overwrite the standart parameter
        setup.reductionFactor = parameterInput.reductionFactor

    # setup.landLayer[np.isnan(setup.landLayer)] = 0
    if mpiSize > 1:
        setup.landLayer = np.load(setup.resourcePath + 'partition_map_' + str(mpiSize) + '.npy')
    else:
        setup.landLayer = np.load(setup.resourcePath + 'land_layer_62x118.npy')
        setup.landLayer = setup.landLayer * 0

    lg.info('max rank:' + str(np.nanmax(setup.landLayer)))

    # setup.population = gt.load_array_from_tiff(setup.resourcePath + 'pop_counts_ww_2005_62x118.tiff')
    setup.population = np.load(setup.resourcePath + 'pop_counts_ww_2005_62x118.npy')
    # setup.regionIdRaster = gt.load_array_from_tiff(setup.resourcePath + 'subRegionRaster_62x118.tiff')
    setup.regionIdRaster = np.load(setup.resourcePath + 'subRegionRaster_62x118.npy')
    # bad bugfix for 4 cells
    setup.regionIdRaster[np.logical_xor(
        np.isnan(setup.population), np.isnan(setup.regionIdRaster))] = 6321

    setup.chargStat = np.load(setup.resourcePath + 'charge_stations_62x118.npy')

    setup.cellSizeMap = np.load(setup.resourcePath + 'cell_area_62x118.npy')

    setup.roadKmPerCell = np.load(
        setup.resourcePath + 'road_km_per_cell_62x118.npy') / setup.cellSizeMap


    assert np.sum(np.logical_xor(np.isnan(setup.population),
                                 np.isnan(setup.regionIdRaster))) == 0  # OPTPRODUCTION
    setup.regionIDList = np.unique(
        setup.regionIdRaster[~np.isnan(setup.regionIdRaster)]).astype(int)

    setup.regionIdRaster[np.isnan(setup.regionIdRaster)] = 0
    setup.regionIdRaster = setup.regionIdRaster.astype(int)

    if False:
        try:
            # plt.imshow(setup.landLayer)
            plt.imshow(setup.population, cmap = 'jet')
            plt.clim([0, np.nanpercentile(setup.population, 90)])
            plt.colorbar()
        except:
            pass
    setup.landLayer[np.isnan(setup.population)] = np.nan


    # social
    setup.addYourself = True     # have the agent herself as a friend (have own observation)
    setup.recAgent = []       # reporter agents that return a diary

    # output
    setup.writeAgentFile = 1
    setup.writeNPY = 1
    setup.writeCSV = 0

    # cars and infrastructure
    setup.properties    = ['emissions', 'fixedCosts', 'operatingCosts']

    # agents
    setup.randomAgents = False
    setup.omniscientAgents = False

    # redefinition of setup parameters from file
    setup.update(parameterInput.toDict())


    for paName in ['techExpBrown', 'techExpGreen', 'techExpPublic', 'techExpShared', 'techExpNone',
                   'population']:
        setup[paName] /= setup['reductionFactor']

    return setup
    return setup

def scenarioGer(parameterInput, dirPath):
    setup = core.AttrDict()

    # general
    setup.resourcePath = dirPath + '/resources_ger/'
    # setup.synPopPath = setup['resourcePath'] + 'hh_NBH_1M.csv'
    setup.progressBar  = True
    setup.allTypeObservations = True

    # time
    setup.nSteps = 340  # number of simulation steps
    setup.timeUnit = init._month  # unit of time per step
    setup.startDate = [1, 2005]
    setup.burnIn = 100
    setup.omniscientBurnIn = 10  # no. of first steps of burn-in phase with omniscient agents, max. =burnIn

    # spatial
    setup.isSpatial = True
    setup.spatialRedFactor = 1.

    if hasattr(parameterInput, "reductionFactor"):
        # overwrite the standart parameter
        setup.reductionFactor = parameterInput.reductionFactor


    # setup.landLayer[np.isnan(setup.landLayer)] = 0
    if mpiSize > 1:
        setup.landLayer = np.load(setup.resourcePath + 'partition_map_' + str(mpiSize) + '.npy')
    else:
        setup.landLayer = np.load(setup.resourcePath + 'land_layer_186x219.npy')
        setup.landLayer[setup.landLayer == 0] = np.nan
        setup.landLayer = setup.landLayer * 0

    lg.info('max rank:' + str(np.nanmax(setup.landLayer)))

    # setup.population        = gt.load_array_from_tiff(setup.resourcePath + 'pop_counts_ww_2005_186x219.tiff')
    setup.population = np.load(setup.resourcePath + 'pop_counts_ww_2005_186x219.npy')
    # setup.regionIdRaster    = gt.load_array_from_tiff(setup.resourcePath + 'subRegionRaster_62x118.tiff')
    setup.regionIdRaster = np.load(setup.resourcePath + 'subRegionRaster_186x219.npy')
    # bad bugfix for 4 cells
    # setup.regionIdRaster[np.logical_xor(np.isnan(setup.population), np.isnan(setup.regionIdRaster))] = 6321

    setup.regionIDList = np.unique(
        setup.regionIdRaster[~np.isnan(setup.regionIdRaster)]).astype(int)

    setup.cellSizeMap = np.load(setup.resourcePath + 'cell_area_186x219.npy')

    setup.roadKmPerCell = np.load(
        setup.resourcePath + 'road_km_all_new_186x219.npy') / setup.cellSizeMap

    # correction of ID map
    xList, yList = np.where(np.logical_xor(
        np.isnan(setup.population), np.isnan(setup.regionIdRaster)))

    for x, y in zip(xList, yList):
        reList = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if not np.isnan(setup.regionIdRaster[x + dx, y + dy]):
                    reList.append(setup.regionIdRaster[x + dx, y + dy])
        if len(np.unique(reList)) == 1:
            setup.regionIdRaster[x, y] = np.unique(reList)[0]

    assert np.sum(np.logical_xor(np.isnan(setup.population),
                                 np.isnan(setup.regionIdRaster))) == 0  # OPTPRODUCTION

    setup.regionIdRaster[np.isnan(setup.regionIdRaster)] = 0
    setup.regionIdRaster = setup.regionIdRaster.astype(int)

    if False:
        try:
            # plt.imshow(setup.landLayer)
            plt.imshow(setup.population, cmap='jet')
            plt.clim([0, np.nanpercentile(setup.population, 90)])
            plt.colorbar()
        except:
            pass
    print(setup.landLayer.shape)
    print(setup.population.shape)    
    setup.landLayer[np.isnan(setup.population)] = np.nan


    # social
    setup.addYourself = True     # have the agent herself as a friend (have own observation)
    setup.recAgent = []       # reporter agents that return a diary

    # output
    setup.writeAgentFile = 1
    setup.writeNPY = 1
    setup.writeCSV = 0

    # cars and infrastructure
    setup.properties    = ['emissions', 'fixedCosts', 'operatingCosts']

    # agents
    setup.randomAgents = False
    setup.omniscientAgents = False

    # redefinition of setup parameters from file
    setup.update(parameterInput.toDict())

    # setup.population = (setup.population ** .5) * 100
    # Correciton of population depend parameter by the reduction factor
    for paName in ['techExpBrown', 'techExpGreen', 'techExpPublic', 'techExpShared', 'techExpNone',
                   'population']:
        setup[paName] /= setup['reductionFactor']
    for p in range(0, 105, 5):
        print('p' + str(p) + ': ' + str(
            np.nanpercentile(setup.population[setup.population != 0], p)))

    # print 'max population' + str(np.nanmax(setup.population))
    # calculate dependent parameters

    nAgents = np.nansum(setup.population)
    lg.info('Running with ' + str(nAgents) + ' agents')

    return setup


def scenarioLueneburg(parameterInput, dirPath):
    setup = core.AttrDict()

    # general
    setup.resourcePath = dirPath + '/resources_luen/'
    setup.progressBar = True
    setup.allTypeObservations = True

    # time
    setup.nSteps = 340     # number of simulation steps
    setup.timeUnit = init._month  # unit of time per step
    setup.startDate = [1, 2005]
    setup.burnIn = 100
    setup.omniscientBurnIn = 10       # no. of first steps of burn-in phase with omniscient agents, max. =burnIn

    # spatial
    setup.isSpatial = True
    setup.spatialRedFactor = 350.

    # setup.connRadius    = 3.5      # radíus of cells that get an connection
    # setup.reductionFactor = parameterInput['reductionFactor']

    if hasattr(parameterInput, "reductionFactor"):
        # overwrite the standart parameter
        setup.reductionFactor = parameterInput.reductionFactor


    # setup.landLayer[np.isnan(setup.landLayer)] = 0
    if mpiSize > 1:
        setup.landLayer = np.load(setup.resourcePath + 'rankMap_nClust' + str(mpiSize) + '.npy')
    else:

        setup.landLayer = np.load(setup.resourcePath + 'land_layer.npy')
        setup.landLayer[setup.landLayer == 0] = np.nan
        setup.landLayer = setup.landLayer * 0

    lg.info('max rank:' + str(np.nanmax(setup.landLayer)))

    # setup.population        = gt.load_array_from_tiff(setup.resourcePath + 'pop_counts_ww_2005_186x219.tiff')
    setup.population = np.load(setup.resourcePath + 'population.npy')
    # setup.regionIdRaster    = gt.load_array_from_tiff(setup.resourcePath + 'subRegionRaster_62x118.tiff')
    setup.regionIdRaster = np.load(setup.resourcePath + 'subRegionRaster.npy')
    # bad bugfix for 4 cells
    # setup.regionIdRaster[np.logical_xor(np.isnan(setup.population), np.isnan(setup.regionIdRaster))] = 6321

    setup.regionIDList = np.unique(
        setup.regionIdRaster[~np.isnan(setup.regionIdRaster)]).astype(int)

    setup.cellSizeMap = np.load(setup.resourcePath + 'cell_size.npy')

    setup.roadKmPerCell = np.load(
        setup.resourcePath + 'road_km_per_cell.npy') / setup.cellSizeMap

    # correction of ID map
    xList, yList = np.where(np.logical_xor(
        np.isnan(setup.population), np.isnan(setup.regionIdRaster)))
    setup.regionIdRaster[np.isnan(setup.regionIdRaster)] = 0
    setup.regionIdRaster = setup.regionIdRaster.astype(int)

    if False:
        try:
            # plt.imshow(setup.landLayer)
            plt.imshow(setup.population, cmap='jet')
            plt.clim([0, np.nanpercentile(setup.population, 90)])
            plt.colorbar()
        except:
            pass
    setup.landLayer[np.isnan(setup.population)] = np.nan


    # social
    setup.addYourself = True     # have the agent herself as a friend (have own observation)
    setup.recAgent = []       # reporter agents that return a diary

    # output
    setup.writeAgentFile = 1
    setup.writeNPY = 1
    setup.writeCSV = 0

    # cars and infrastructure
    setup.properties    = ['emissions', 'fixedCosts', 'operatingCosts']

    # agents
    setup.randomAgents = False
    setup.omniscientAgents = False

    publicTransportSetup(setup)

    # redefinition of setup parameters from file
    setup.update(parameterInput.toDict())

    for paName in ['techExpBrown',
                   'techExpGreen',
                   'techExpPublic',
                   'techExpShared',
                   'techExpNone']:
        setup[paName] /= setup['reductionFactor'] * setup.spatialRedFactor

    setup['population'] /= setup['reductionFactor']

    for p in range(0, 105, 5):
        print('p' + str(p) + ': ' + str(
            np.nanpercentile(setup.population[setup.population != 0], p)))

    # print 'max population' + str(np.nanmax(setup.population))
    # calculate dependent parameters

    nAgents = np.nansum(setup.population)
    lg.info('Running with ' + str(nAgents) + ' agents')

    return setup

def scenarioTest(parameterInput, dirPath):
    setup = core.AttrDict()

    # general
    setup.resourcePath = None
    setup.progressBar = True
    setup.allTypeObservations = True

    # time
    setup.nSteps = 10     # number of simulation steps
    setup.timeUnit = init._month  # unit of time per step
    setup.startDate = [1, 2005]
    setup.burnIn = 0
    # no. of first steps of burn-in phase with omniscient agents, max. =burnIn
    setup.omniscientBurnIn = 0
    # spatial
    setup.isSpatial = True
    setup.spatialRedFactor = 1.
    # setup.connRadius    = 3.5      # radíus of cells that get an connection
    # setup.reductionFactor = parameterInput['reductionFactor']

    setup.landLayer = np.zeros([2, mpiSize])
    setup.landLayer[0, :] = np.asarray(list(range(mpiSize)))
    setup.landLayer[1, :] = np.asarray(list(range(mpiSize)))

    #setup.population        = gt.load_array_from_tiff(setup.resourcePath + 'pop_counts_ww_2005_186x219.tiff')
    setup.population = setup.landLayer * 5
    #setup.regionIdRaster    = gt.load_array_from_tiff(setup.resourcePath + 'subRegionRaster_62x118.tiff')
    setup.regionIdRaster = setup.landLayer * 99
    # bad bugfix for 4 cells
    #setup.regionIdRaster[np.logical_xor(np.isnan(setup.population), np.isnan(setup.regionIdRaster))] = 6321

    setup.regionIDList = np.unique(
        setup.regionIdRaster[~np.isnan(setup.regionIdRaster)]).astype(int)

    setup.cellSizeMap = setup.landLayer * 1.

    setup.roadKmPerCell = setup.landLayer * 2.

    # correction of ID map
    xList, yList = np.where(np.logical_xor(
        np.isnan(setup.population), np.isnan(setup.regionIdRaster)))

    setup.randomCarPropDeviationSTD = .05
    setup.connRadius = 2.
    setup.reductionFactor = 2.

    setup.regionIdRaster[np.isnan(setup.regionIdRaster)] = 0
    setup.regionIdRaster = setup.regionIdRaster.astype(int)

    if False:
        try:
            # plt.imshow(setup.landLayer)
            plt.imshow(setup.population, cmap='jet')
            plt.clim([0, np.nanpercentile(setup.population, 90)])
            plt.colorbar()
        except:
            pass

    # social
    setup.addYourself = True     # have the agent herself as a friend (have own observation)
    setup.recAgent = []       # reporter agents that return a diary

    # output
    setup.writeAgentFile = 1
    setup.writeNPY = 1
    setup.writeCSV = 0

    # cars and infrastructure
    setup.properties    = ['emissions', 'fixedCosts', 'operatingCosts']

    # agents
    setup.randomAgents = False
    setup.omniscientAgents = False

    # redefinition of setup parameters from file
    setup.update(parameterInput.toDict())

    nAgents = np.nansum(setup.population)
    lg.info('Running with ' + str(nAgents) + ' agents')

    return setup

scenarioDict = {
    0: scenarioTestSmall,
    1: scenarioTestMedium,
    2: scenarioLueneburg,
    3: scenarioNBH,
    6: scenarioGer,
}


def create(parameters, dirPath):
    return scenarioDict[parameters.scenario](parameters, dirPath)
