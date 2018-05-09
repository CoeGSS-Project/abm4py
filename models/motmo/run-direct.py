#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import logging as lg
import time
import sys
import os
import socket

import init_motmo as init
from classes_motmo import aux, MPI
import plots

debug = 1
showFigures = 0

comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()

overallTime = time.time()

simNo, baseOutputPath = aux.getEnvironment(comm, getSimNo=True)
outputPath = aux.createOutputDirectory(comm, baseOutputPath, simNo)
dirPath = os.path.dirname(os.path.realpath(__file__))
fileName = sys.argv[1]

init.initLogger(debug, outputPath)

lg.info('on node: ' + socket.gethostname())

parameters = init.createAndReadParameters(fileName, dirPath)
parameters['outPath'] = outputPath
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

init.runModel(earth, parameters)

lg.info('Simulation ' + str(earth.simNo) + ' finished after -- ' +
        str(time.time() - overallTime) + ' s')

if earth.isRoot:
    print 'Simulation ' + str(earth.simNo) + ' finished after -- ' + str(time.time() - overallTime) + ' s'
    init.writeSummary(earth, parameters)

if showFigures:
    init.onlinePostProcessing(earth)

plots.computingTimes(earth)
