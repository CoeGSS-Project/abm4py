#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import logging as lg
import time
import sys
import os
import socket
import dakota.interfacing as di

dakotadir = os.getcwd()
dirPath = os.path.dirname(os.path.realpath(__file__))
os.chdir('..')

import init_motmo as init
from classes_motmo import aux, MPI

comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()

# I don't want to polute the model directory with dakota files, but the
# dakota.interfacing library uses relativ file names, so I start with the
# dakota subdirectory, changes to the model diretory after reading the dakota
# input file and switch back to the dakota dir before writing the output file
if mpiRank == 0:
    os.chdir(dakotadir)
    dakotaParams, dakotaResults = di.read_parameters_file(sys.argv[1], sys.argv[2])
    os.chdir('..')


showFigures = 0

overallTime = time.time()

simNo, baseOutputPath = aux.getEnvironment(comm, getSimNo=True)
outputPath = aux.createOutputDirectory(comm, baseOutputPath, simNo)
print(outputPath)
print(dirPath)
init.initLogger(False, outputPath)

lg.info('on node: ' + socket.gethostname())

parameters = init.createAndReadParameters(dakotaParams['scenarioFileName'], dirPath)

if mpiRank == 0:
    for d in dakotaParams.descriptors:
        parameters[d] = dakotaParams[d]

parameters = init.exchangeParameters(parameters)
parameters['outPath'] = outputPath
parameters.showFigures = showFigures

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

if mpiRank == 0:
    print 'Simulation ' + str(earth.simNo) + ' finished after -- ' + str(time.time() - overallTime) + ' s'
    init.writeSummary(earth, parameters)

    os.chdir('./dakota')
    execfile(dakotaParams['calcResultsScript'])
    # the script executed contains the calcResults function
    calcResults(dakotaResults)
