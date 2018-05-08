#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import logging as lg
import time

execfile('init_motmo.py')

debug = 1
showFigures = 0

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

lg.info('Log file of process '+ str(mpiRank) + ' of ' + str(mpiSize))

# wait for all processes - debug only for poznan to debug segmentation fault
comm.Barrier()
if comm.rank == 0:
    print 'log files created'

lg.info('on node: ' + socket.gethostname())
dirPath = os.path.dirname(os.path.realpath(__file__))


fileName = sys.argv[1]
parameters = Bunch()
# reading of gerneral parameters
parameters = readParameterFile(parameters, 'parameters_all.csv')
    # DAKOTA hack
# reading of scenario-specific parameters
parameters = readParameterFile(parameters,fileName)

lg.info('Setting loaded:')

parameters['outPath'] = outputPath

scenarioDict = dict()

scenarioDict[0] = scenarioTestSmall
scenarioDict[1] = scenarioTestMedium
scenarioDict[2] = scenarioLueneburg
scenarioDict[3] = scenarioNBH
scenarioDict[6] = scenarioGer

if mpiRank == 0:
    parameters = scenarioDict[parameters.scenario] (parameters, dirPath)

    parameters = initExogeneousExperience(parameters)
    parameters = randomizeParameters(parameters)   

else:
    parameters = None

parameters = comm.bcast(parameters,root=0)    

if mpiRank == 0:
    print'Parameter exchange done'
lg.info( 'Parameter exchange done')


#%% Init
parameters.showFigures = showFigures

earth = initEarth(simNo,
                  outputPath,
                  parameters,
                  maxNodes=1000000,
                  debug =debug,
                  mpiComm=comm,
                  caching=True,
                  queuing=True)

CELL, HH, PERS = initTypes(earth)

initSpatialLayer(earth)

initInfrastructure(earth)

mobilitySetup(earth)

cellTest(earth)

initGlobalRecords(earth)

householdSetup(earth)

generateNetwork(earth)

initMobilityTypes(earth)

initAgentOutput(earth)

cell = earth.entDict[0]
#cell.setWorld(earth)

if parameters.scenario == 0:
    earth.view('output/graph.png')

#%% run of the model ################################################
lg.info('####### Running model with paramertes: #########################')
import pprint
lg.info(pprint.pformat(parameters.toDict()))
if mpiRank == 0:
    fidPara = open(earth.para['outPath'] + '/parameters.txt','w')
    pprint.pprint(parameters.toDict(), fidPara)
    fidPara.close()
lg.info('################################################################')

runModel(earth, parameters)

lg.info('Simulation ' + str(earth.simNo) + ' finished after -- ' + str( time.time() - overallTime) + ' s')

if earth.isRoot:
    print 'Simulation ' + str(earth.simNo) + ' finished after -- ' + str( time.time() - overallTime) + ' s'

if earth.isRoot:
    writeSummary(earth, parameters)

if earth.para['showFigures']:

    onlinePostProcessing(earth)

plot_computingTimes(earth)
