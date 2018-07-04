#!/usr/bin/env python3
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://wwww.globalclimateforum.org

---- MoTMo ----
MOBILITY TRANSIOn MODEL
-- Direct start file --

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
import time
import sys
import os
import socket

import init_motmo as init
from lib import core


debug = True
showFigures = False

comm    = core.comm
mpiRank = core.mpiRank

overallTime = time.time()

simNo, outputPath = core.setupSimulationEnvironment()

print ('Current simulation number is: ' + str(simNo))
print ('Current ouputPath number is: ' + outputPath)

dirPath = os.path.dirname(os.path.realpath(__file__))
fileName = sys.argv[1]

core.initLogger(debug, outputPath)

lg.info('on node: ' + socket.gethostname())

parameters = init.createAndReadParameters(fileName, dirPath)
if comm is not None:
    parameters = init.exchangeParameters(parameters)
parameters['outPath'] = outputPath
parameters['showFigures'] = showFigures


earth = init.initEarth(simNo,
                       outputPath,
                       parameters,
                       maxNodes=1000000,
                       maxLinks=5000000,
                       debug=debug,
                       mpiComm=comm)

init.initScenario(earth, parameters)

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
