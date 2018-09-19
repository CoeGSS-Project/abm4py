#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:35:14 2017

@author: gcf
"""

#MINIMAL SETUP
import sys, os
from os.path import expanduser
home = expanduser("~")
#sys.path.append(home + '/python/decorators/')
#sys.path.append(home + '/python/modules/')
#sys.path.append(home + '/python/agModel/modules/')

import socket
dir_path = os.path.dirname(os.path.realpath(__file__))
if socket.gethostname() in ['gcf-VirtualBox', 'ThinkStation-D30']:
    sys.path = [dir_path + '/h5py/build/lib.linux-x86_64-2.7'] + sys.path 
    sys.path = [dir_path + '/mpi4py/build/lib.linux-x86_64-2.7'] + sys.path 
else:
    import matplotlib
    matplotlib.use('Agg')    
#from deco_util import timing_function
import numpy as np
import time
#import mod_geotiff as gt # not working in poznan
from para_class_mobilityABM import Person, GhostPerson, Household, GhostHousehold, Reporter, Cell, GhostCell,  Earth, Opinion
from mpi4py import  MPI
import h5py
import class_auxiliary  as aux #convertStr

import matplotlib.pylab as plt
import seaborn as sns; sns.set()
sns.set_color_codes("dark")
#import matplotlib.pyplot as plt

import pandas as pd
from bunch import Bunch
from copy import copy
import csv
from scipy import signal

######### Enums ########################
#connections
_cll = 1 # loc - loc
_clh = 2 # loc - household
_chh = 3 # household, household
_chp = 4 # household, person
_cpp = 5 # person, person

#nodes
_cell = 1
_hh   = 2
_pers = 3

#time spans
_month = 1
_year  = 2



# Graph library
#ig.Graph(directed=True)
#
### MPI4PY
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()
#
### parallel IO
#h5File      = h5py.File('test.hdf5', 'w', driver='mpio', comm=comm)


def convertStr(string):
    """
    Returns integer, float or string dependent on the input
    """
    if str.isdigit(string):
        return int(string)
    else:
        try:
            return float(string)
        except:
            return string
        
def scenarioTestSmall(parameterInput, dirPath):
    setup = Bunch()
    
    #general 
    setup.resourcePath = dirPath + '/resources_nie/'
    setup.synPopPath = setup['resourcePath'] + 'hh_niedersachsen.csv'
    setup.progressBar  = True
    setup.allTypeObservations = False
    
    #time
    setup.timeUnit         = _month  # unit of time per step
    setup.startDate        = [01,2005]   

    #spatial
    setup.reductionFactor = 50000
    setup.isSpatial       = True
    setup.connRadius      = 1.5     # radíus of cells that get an connection
    setup.landLayer   = np.asarray(range(mpiSize))
    setup.regionIdRaster    = setup.landLayer*0+1
    #setup.regionIdRaster[0:,0:2] = ((setup.landLayer[0:,0:2]*0)+1) *6321
    if mpiSize == 1:
        setup.landLayer = setup.landLayer*0

    setup.population = (np.isnan(setup.landLayer)==0)* np.random.randint(5,10,setup.landLayer.shape)
    
    #social
    setup.addYourself   = True     # have the agent herself as a friend (have own observation)
    setup.recAgent      = []       # reporter agents that return a diary
    
    #output
    setup.writeOutput   = 1
    setup.writeNPY      = 1
    setup.writeCSV      = 0
    
    #cars and infrastructure
    setup.properties    = ['emmisions','TCO']

    #agents
    setup.randomAgents     = False
    setup.omniscientAgents = False    

    # redefinition of setup parameters used for automatic calibration
    setup.update(parameterInput.toDict())
    
    # calculate dependent parameters
    maxDeviation = (setup.urbanCritical - setup.urbanThreshold)**2
    minCarConvenience = 1 + setup.kappa
    setup.convB =  minCarConvenience / (maxDeviation)
 
    # only for packing with care
    #dummy   = gt.load_array_from_tiff(setup.resourcePath + 'land_layer_62x118.tiff')
    #dummy   = gt.load_array_from_tiff(setup.resourcePath + 'pop_counts_ww_2005_62x118.tiff') / setup.reductionFactor
    #dummy   = gt.load_array_from_tiff(setup.resourcePath + 'subRegionRaster_62x118.tiff')    
    #del dummy
    
    lg.info( "Final setting of the parameters")
    lg.info( parameterInput)
    lg.info( "######################################################")
    
    return setup

if __name__ == '__main__':
    # GLOBAL INIT  MPI
    comm = MPI.COMM_WORLD
    mpiRank = comm.Get_rank()
    mpiSize = comm.Get_size()
    

    
#    if mpiRank != 0:
#        olog_file  = open('output/log' + str(mpiRank) + '.txt', 'w')
#        sys.stdout = olog_file
#        elog_file  = open('output/err' + str(mpiRank) + '.txt', 'w')
#        sys.stderr = elog_file

    simNo, baseOutputPath = 0,''
    outputPath = '.'
    
    
    import logging as lg

    lg.basicConfig(filename=outputPath + '/log_R' + str(mpiRank), 
                    filemode='w',
                    format='%(levelname)7s %(asctime)s : %(message)s', 
                    datefmt='%m/%d/%y-%H:%M:%S',
                    level=lg.DEBUG)
    
    lg.info('Log file of process '+ str(mpiRank) + ' of ' + str(mpiSize))
    
    #import subprocess
    #output = subprocess.check_output("cat /proc/loadavg", shell=True)
    
    #lg.info('on node: ' + socket.gethostname() + ' with average load: ' + output)
    lg.info('on node: ' + socket.gethostname())
    dirPath = os.path.dirname(os.path.realpath(__file__))
    # loading of standart parameters
    fileName = sys.argv[1]
    parameters = Bunch()
    for item in csv.DictReader(open(fileName)):
        parameters[item['name']] = convertStr(item['value'])
    lg.info('Setting loaded:')
    
    
    parameters['outPath'] = outputPath
    
    
    parameters = scenarioTestSmall(parameters, dirPath)

    parameters = comm.bcast(parameters)
    
    
    parameters.showFigures = False
    
    earth = Earth(simNo,
                      outputPath,
                      parameters, 
                      maxNodes=1000000, 
                      debug =True, 
                      mpiComm=comm, 
                      caching=True, 
                      queuing=True)