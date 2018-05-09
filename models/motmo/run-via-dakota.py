#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import logging as lg
import time
import sys
import os
import socket
import dakota.interfacing as di

import init_motmo as init
from classes_motmo import aux, MPI

comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()

showFigures = 0

overallTime = time.time()

simNo, baseOutputPath = aux.getEnvironment(comm, getSimNo=True)
outputPath = aux.createOutputDirectory(comm, baseOutputPath, simNo)
dirPath = os.path.dirname(os.path.realpath(__file__))

dakotaParams, dakotaResults = di.read_parameters_file(sys.argv[1], sys.argv[2])

init.initLogger(False, outputPath)

lg.info('on node: ' + socket.gethostname())

parameters = init.createAndReadParameters(dakotaParams['scenarioFileName'], dirPath)
parameters['outPath'] = outputPath
parameters.showFigures = showFigures

# TODO loop over all dakota Params
for d in dakotaParams.descriptors:
    parameters[d] = dakotaParams[d]

init.exchangeParameters(parameters)

earth = init.initEarth(simNo,
                       outputPath,
                       parameters,
                       maxNodes=1000000,
                       debug=False,
                       mpiComm=comm,
                       caching=True,
                       queuing=True)

init.runModel(earth, parameters)

lg.info('Simulation ' + str(earth.simNo) + ' finished after -- ' +
        str(time.time() - overallTime) + ' s')

if earth.isRoot:
    print 'Simulation ' + str(earth.simNo) + ' finished after -- ' + str(time.time() - overallTime) + ' s'
    init.writeSummary(earth, parameters)

dakotaResults["relativeError"].function = earth.globalRecord['stock_6321'].evaluateRelativeError()
dakotaResults["absoluteError"].function = earth.globalRecord['stock_6321'].evaluateAbsoluteError()
dakotaResults.write()
