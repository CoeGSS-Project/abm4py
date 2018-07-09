#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging as lg
import time
import sys
import os
import socket
import dakota.interfacing as di
import random

dakotadir = os.getcwd()
dirPath = os.path.dirname(os.path.realpath(__file__))
os.chdir('..')

import init_motmo as init
from lib import core

comm    = core.comm
mpiRank = core.mpiRank

# I don't want to polute the model directory with dakota files, but the
# dakota.interfacing library uses relativ file names, so I start with the
# dakota subdirectory, changes to the model diretory after reading the dakota
# input file and switch back to the dakota dir before writing the output file
os.chdir(dakotadir)
dakotaParams, dakotaResults = di.read_parameters_file(sys.argv[1], sys.argv[2])
os.chdir('..')

showFigures = 0

overallTime = time.time()

# derive the simulation number from the parameter file name (it's important that
# file_tag attribute is set in the interface section of the dakota input file)
dakotaRunNo = random.randrange(2 ** 63)
print('runNo' + str(dakotaRunNo))
simNo, outputPath = core.setupSimulationEnvironment(comm, dakotaRunNo)
print(outputPath)
core.initLogger(False, outputPath)

lg.info('on node: ' + socket.gethostname())

parameters = init.createAndReadParameters(dakotaParams['scenarioFileName'], dirPath)

if mpiRank == 0:
    for d in dakotaParams.descriptors:
        parameters[d] = dakotaParams[d]
if comm is not None:
    parameters = init.exchangeParameters(parameters)
parameters['outPath'] = outputPath
parameters['showFigures'] = showFigures

earth = init.initEarth(simNo,
                       outputPath,
                       parameters,
                       maxNodes=1000000,
                       maxLinks=5000000,
                       debug=False,
                       mpiComm=comm)

init.initScenario(earth, parameters)

init.runModel(earth, parameters)

lg.info('Simulation ' + str(earth.simNo) + ' finished after -- ' +
        str(time.time() - overallTime) + ' s')

if mpiRank == 0:
    print('Simulation ' + str(earth.simNo) + ' finished after -- ' + str(time.time() - overallTime) + ' s')
    init.writeSummary(earth, parameters)

    os.chdir(dakotadir)
    exec(open(dakotaParams['calcResultsScript']).read())
    # the script executed contains the calcResults function
    calcResults(earth, dakotaResults)
    dakotaResults["runNo"].function = int(dakotaRunNo)

    dakotaResults.write()
