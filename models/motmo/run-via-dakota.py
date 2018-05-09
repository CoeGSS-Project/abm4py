#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import logging as lg
import time
import sys
import os
import socket
import dakota.interfacing as di

# I don't want to polute the model directory with dakota files, but the
# dakota.interfacing library uses relativ file names, so I start with the
# dakota subdirectory, changes to the model diretory after reading the dakota
# input file and switch back to the dakota dir before writing the output file
dakotaParams, dakotaResults = di.read_parameters_file(sys.argv[1], sys.argv[2])
dirPath = os.path.dirname(os.path.realpath(__file__))
os.chdir('..')

import init_motmo as init
from classes_motmo import aux, MPI

comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()

showFigures = 0

overallTime = time.time()

simNo, baseOutputPath = aux.getEnvironment(comm, getSimNo=True)
outputPath = aux.createOutputDirectory(comm, baseOutputPath, simNo)
print(outputPath)
print(dirPath)
init.initLogger(False, outputPath)

lg.info('on node: ' + socket.gethostname())

parameters = init.createAndReadParameters(dakotaParams['scenarioFileName'], dirPath)
parameters['outPath'] = outputPath
parameters.showFigures = showFigures

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
os.chdir('./dakota')
dakotaResults.write()
