#!/usr/bin/env python3


import logging as lg
import time
import sys
import os
import socket

import init_motmo as init
import core

debug = True
showFigures = False

comm = core.MPI.COMM_WORLD
mpiRank = comm.Get_rank()

overallTime = time.time()

simNo, outputPath = core.setupSimulationEnvironment(comm)

print ('Current simulation number is: ' + str(simNo))
print ('Current ouputPath number is: ' + outputPath)

dirPath = os.path.dirname(os.path.realpath(__file__))
fileName = sys.argv[1]

core.initLogger(debug, outputPath)

lg.info('on node: ' + socket.gethostname())

parameters = init.createAndReadParameters(fileName, dirPath)
parameters = init.exchangeParameters(parameters)
parameters['outPath'] = outputPath
parameters['showFigures'] = showFigures


earth = init.initEarth(simNo,
                       outputPath,
                       parameters,
                       maxNodes=1000000,
                       maxEdges=5000000,
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
    import plots
    init.onlinePostProcessing(earth)

    plots.computingTimes(earth)
