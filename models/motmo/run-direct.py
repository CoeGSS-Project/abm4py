#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import logging as lg
import time
import sys
import os
import socket
import pprint
from bunch import Bunch

import init_motmo as init
from init_motmo import comm, mpiRank, mpiSize
from classes_motmo import aux
import scenarios

debug = 1
showFigures = 0

overallTime = time.time()

simNo, baseOutputPath = aux.getEnvironment(comm, getSimNo=True)
outputPath = aux.createOutputDirectory(comm, baseOutputPath, simNo)

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

lg.info('Log file of process ' + str(mpiRank) + ' of ' + str(mpiSize))

# wait for all processes - debug only for poznan to debug segmentation fault
comm.Barrier()
if comm.rank == 0:
    print 'log files created'

lg.info('on node: ' + socket.gethostname())
dirPath = os.path.dirname(os.path.realpath(__file__))

fileName = sys.argv[1]
parameters = Bunch()
# reading of gerneral parameters
parameters = init.readParameterFile(parameters, 'parameters_all.csv')
# DAKOTA hack
# reading of scenario-specific parameters
parameters = init.readParameterFile(parameters, fileName)

lg.info('Setting loaded:')

parameters['outPath'] = outputPath

if mpiRank == 0:
    parameters = scenarios.create(parameters, dirPath)

    parameters = init.initExogeneousExperience(parameters)
    parameters = init.randomizeParameters(parameters)
else:
    parameters = None

parameters = comm.bcast(parameters, root=0)

if mpiRank == 0:
    print'Parameter exchange done'
lg.info('Parameter exchange done')

# Init
parameters.showFigures = showFigures

earth = init.initEarth(simNo,
                       outputPath,
                       parameters,
                       maxNodes=1000000,
                       debug=debug,
                       mpiComm=comm,
                       caching=True,
                       queuing=True)

if parameters.scenario == 0:
    earth.view('output/graph.png')

#%% run of the model ################################################
lg.info('####### Running model with paramertes: #########################')
lg.info(pprint.pformat(parameters.toDict()))
if mpiRank == 0:
    fidPara = open(earth.para['outPath'] + '/parameters.txt', 'w')
    pprint.pprint(parameters.toDict(), fidPara)
    fidPara.close()
lg.info('################################################################')

init.runModel(earth, parameters)

lg.info('Simulation ' + str(earth.simNo) + ' finished after -- ' + str(time.time() - overallTime) + ' s')

if earth.isRoot:
    print 'Simulation ' + str(earth.simNo) + ' finished after -- ' + str(time.time() - overallTime) + ' s'

if earth.isRoot:
    init.writeSummary(earth, parameters)

if earth.para['showFigures']:
    init.onlinePostProcessing(earth)

init.plot_computingTimes(earth)
