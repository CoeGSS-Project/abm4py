exec(compile(open('init_motmo.py').read(), 'init_motmo.py', 'exec'))

debug = 1 
showFigures = 0

simNo, baseOutputPath = aux.getEnvironment(comm, getSimNo=True)
outputPath = aux.createOutputDirectory(comm, baseOutputPath, simNo)

import logging as lg
import time

#exit()

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
    print('log files created')

lg.info('on node: ' + socket.gethostname())
dirPath = os.path.dirname(os.path.realpath(__file__))

parameters = Bunch()
parameters = scenarioTest(parameters, dirPath)
parameters = comm.bcast(parameters)
earth = initEarth(simNo,
                  outputPath,
                  parameters,
                  maxNodes=1000000,
                  debug=debug,
                  mpiComm=comm,
                  caching=True,
                  queuing=True)

CELL, HH, PERS = initTypes(earth)

initSpatialLayer(earth)

initInfrastructure(earth)

earth.mpi.comm.Barrier()
print("test finished")
exit()
