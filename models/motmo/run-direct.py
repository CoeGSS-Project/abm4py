#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import logging as lg
import time
import sys
import os
import socket

import init_motmo as init
import core
import plots

debug = 1
showFigures = 0

comm = core.MPI.COMM_WORLD
mpiRank = comm.Get_rank()

overallTime = time.time()

simNo, baseOutputPath = core.getEnvironment(comm, getSimNo=True)
outputPath = core.createOutputDirectory(comm, baseOutputPath, simNo)
dirPath = os.path.dirname(os.path.realpath(__file__))
fileName = sys.argv[1]

init.initLogger(debug, outputPath)

lg.info('on node: ' + socket.gethostname())

parameters = init.createAndReadParameters(fileName, dirPath)
parameters['outPath'] = outputPath
parameters['showFigures'] = showFigures

global earth
earth = init.initEarth(simNo,
                       outputPath,
                       parameters,
                       maxNodes=1000000,
                       debug=debug,
                       mpiComm=comm)

init.initScenario(earth, parameters)


if parameters.scenario == 0:
    earth.view('output/graph.png')

init.runModel(earth, parameters)

lg.info('Simulation ' + str(earth.simNo) + ' finished after -- ' +
        str(time.time() - overallTime) + ' s')

if earth.isRoot:
    print('Simulation ' + str(earth.simNo) + ' finished after -- ' + str(time.time() - overallTime) + ' s')
    init.writeSummary(earth, parameters)

if showFigures:
    init.onlinePostProcessing(earth)

plots.computingTimes(earth)
