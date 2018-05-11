#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://www.globalclimateforum.org

CAR INNOVATION MARKET MODEL
-- AUXILIARIES CLASS FILE --

This file is part of GCFABM.

GCFABM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GCFABM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GCFABM.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import itertools
import pandas as pd
#import seaborn as sns
import pickle
import time
import mpi4py

mpi4py.rc.threads = False
import sys
sys.path = ['../h5py/build/lib.linux-x86_64-2.7'] + sys.path

sys_excepthook = sys.excepthook
def mpi_excepthook(v, t, tb):
    sys_excepthook(v, t, tb)
    mpi4py.MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook

from mpi4py import MPI
import h5py
import logging as lg
from numba import njit


ALLOWED_MULTI_VALUE_TYPES = (list, tuple, np.ndarray)


def writeAdjFile(graph,fileName):
    graph.to_undirected()
    fid = open(fileName,'w')
    fid.write('% Adjecency file created by gcfABM \n')
    fid.write(str(graph.vcount()) + ' ' + str(graph.ecount()) + ' 010 \n' )
    fid.write('% Adjecencies of verteces \n')
    adjList = graph.get_adjlist()
    popList = graph.vs['population']

    for adjline, popu in zip(adjList, popList):
        fid.write(''.join([str(int(popu*100)) + ' '] + [str(x+1) + ' ' for x in adjline]) + '\n')
    fid.close()


def getEnvironment(comm, getSimNo=True):

    if (comm is None) or (comm.rank == 0) :
        # get simulation number from file
        try:
            fid = open("environment","r")
            simNo = int(fid.readline())
            baseOutputPath = fid.readline().rstrip('\n')
            fid.close()
            if getSimNo:
                fid = open("environment","w")
                fid.writelines(str(simNo+1)+ '\n')
                fid.writelines(baseOutputPath)
                fid.close()
        except:
            print 'ERROR Envirionment file is not set up'
            print ''
            print 'Please create file "environment" which contains the simulation'
            print 'and the baseOutputPath'
            print '#################################################'
            print "0"
            print 'basepath/'
            print '#################################################'
    else:
        simNo    = None
        baseOutputPath = None


    if getSimNo:
        if comm is None:
            return simNo, baseOutputPath
        simNo = comm.bcast(simNo, root=0)
        baseOutputPath = comm.bcast(baseOutputPath, root=0)
        if comm.rank == 0:
            print 'simulation number is: ' + str(simNo)
            print 'base path is: ' + str(baseOutputPath)
        return simNo, baseOutputPath
    else:
        if comm is None:
            return baseOutputPath
        
        baseOutputPath = comm.bcast(baseOutputPath, root=0)
        
        return baseOutputPath

def createOutputDirectory(comm, baseOutputPath, simNo):

        dirPath  = baseOutputPath + 'sim' + str(simNo).zfill(4)

        if comm.rank ==0:
            import os

            if not os.path.isdir(dirPath):
                os.mkdir(dirPath)

        comm.Barrier()
        if comm.rank ==0:
            print 'output directory created'

        return dirPath

@njit
def weightingFunc(x,y):
    return 1./((x**2. +y**2.)**.5)

@njit
def distance(x,y):
    return (x**2. +y**2.)**.5

def computeConnectionList(radius=1, weightingFunc = weightingFunc, ownWeight =2):
    """
    Method for easy computing a connections list of regular grids
    """
    connList = []

    intRad = int(radius)
    for x in range(-intRad,intRad+1):
        for y in range(-intRad,intRad+1):
            if distance(x,y) <= radius:
                if x**2 +y**2 > 0:
                    weig  = weightingFunc(x,y)
                else:
                    weig = ownWeight
                connList.append((x,y,weig))
    return connList

def cartesian2(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

from sklearn.utils.extmath import cartesian
cartesian = cartesian

def cartesian_old(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

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

def saveObj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadObj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


class Writer():

    def __init__(self, world, filename):
        self.fid = open('output/' + filename,'w')
        self.world = world
        self.world.reporter.append(self)

    def write(self,outStr):
        self.fid.write(str(self.world.time) + ' - ' + outStr + '\n')

    def close(self):
        self.fid.close()

class Memory():

    def __init__(self, memeLabels):
        self.freeRows  = list()
        self.columns   = dict()
        for i, label in enumerate(memeLabels):
            self.columns[label] = i
        self.ID2Row    = dict()
        self.memory    = np.zeros([0, len(memeLabels)])
        self.getID     = itertools.count().next

    def addMeme(self, meme):
        """
        add meme to memory
        """
        memeID = self.getID()
        if len(self.freeRows) > 0:
            row = self.freeRows.pop()
            self.memory[row] = meme
            self.ID2Row[memeID] = row
        else:
            row = self.memory.shape[0]
            self.memory = np.vstack(( self.memory, meme))
            self.ID2Row[memeID] = row

        return memeID

    def remMeme(self,memeID):
        """
        remove meme to memory
        """
        row = self.ID2Row[memeID]
        self.memory[row] = np.nan
        self.freeRows.append(row)
        del self.ID2Row[memeID]

    def getMeme(self,memeID,columns):
        """
        used the memeID to identfy the row and returns the meme
        """
        cols = [self.columns[x] for x in columns]
        rows = [self.ID2Row[x]   for x in memeID]
        return self.memory[np.ix_(rows,cols)]

class Globals(dict):
    """ This class manages global variables that are assigned on all processes
    and are synced via mpi. Global variables need to be registered together with
    the aggregation method they ase synced with, .e.g. sum, mean, min, max,...

    
    #TODO
    - enforce the setting (and reading) of global stats
    - implement mean, deviation, std as reduce operators


    """
    


    def __init__(self, world):
        self.world = world
        self.comm  = world.mpi.comm

        # simple reductions
        self.reduceDict = dict()
        
        # MPI operations
        self.operations = dict()
        self.operations['sum']  = MPI.SUM
        self.operations['prod'] = MPI.PROD
        self.operations['min']  = MPI.MIN
        self.operations['max']  = MPI.MAX

        #staticical reductions/aggregations
        self.statsDict       = dict()
        self.localValues     = dict()
        self.nValues         = dict()
        self.updated         = dict()

        # self implemented operations
        statOperations         = dict()
        statOperations['mean'] = np.mean
        statOperations['std']  = np.std
        statOperations['var']  = np.std
        #self.operations['std'] = MPI.Op.Create(np.std)

    #%% simple global reductions
    def registerValue(self, globName, value, reduceType):
        self[globName] = value
        self.localValues[globName] = value
        try:
            self.nValues[globName] = len(value)
        except:
            self.nValues[globName] = 1
        if reduceType not in self.reduceDict.keys():
            self.reduceDict[reduceType] = list()
        self.reduceDict[reduceType].append(globName)
        self.updated[globName] = True

    def syncReductions(self):

        for redType in self.reduceDict.keys():

            op = self.operations[redType]
            #print op
            for globName in self.reduceDict[redType]:

                # enforce that data is updated
                assert  self.updated[globName] is True    ##OPTPRODUCTION
                
                # communication between all proceees
                self[globName] = self.comm.allreduce(self.localValues[globName],op)
                self.updated[globName] = False
                lg.debug('local value of ' + globName + ' : ' + str(self.localValues[globName]))##OPTPRODUCTION
                lg.debug(str(redType) + ' of ' + globName + ' : ' + str(self[globName]))##OPTPRODUCTION

    #%% statistical global reductions/aggregations
    def registerStat(self, globName, values, statType):
        #statfunc = self.statOperations[statType]

        assert statType in ['mean', 'std', 'var']    ##OPTPRODUCTION


        if not isinstance(values, ALLOWED_MULTI_VALUE_TYPES):
            values = [values]
        values = np.asarray(values)


        self.localValues[globName]  = values
        self.nValues[globName]      = len(values)
        if statType == 'mean':
            self[globName]          = np.mean(values)
        elif statType == 'std':
            self[globName]          = np.std(values)
        elif statType == 'var':
            self[globName]          = np.var(values)

        if statType not in self.statsDict.keys():
            self.statsDict[statType] = list()
        self.statsDict[statType].append(globName)
        self.updated[globName] = True
        

    def updateLocalValues(self, globName, values):
        self.localValues[globName]     = values
        self.nValues[globName]         = len(values)
        self.updated[globName]         = True

    def syncStats(self):
        for redType in self.statsDict.keys():
            if redType == 'mean':

                for globName in self.statsDict[redType]:
                    
                    # enforce that data is updated
                    assert  self.updated[globName] is True    ##OPTPRODUCTION
                    
                    # sending data list  of (local mean, size)
                    tmp = [(np.mean(self.localValues[globName]), self.nValues[globName])]* self.comm.size 

                    # communication between all proceees
                    tmp = np.asarray(self.comm.alltoall(tmp))

                    lg.debug('####### Mean of ' + globName + ' #######')       ##OPTPRODUCTION
                    lg.debug('loc mean: ' + str(tmp[:,0]))                     ##OPTPRODUCTION
                    # calculation of global mean
                    globValue = np.sum(np.prod(tmp,axis=1)) # means * size
                    globSize  = np.sum(tmp[:,1])             # sum(size)
                    self[globName] = globValue/ globSize    # glob mean
                    lg.debug('Global mean: ' + str( self[globName] ))   ##OPTPRODUCTION
                    self.updated[globName] = False
                    
            elif redType == 'std':
                for globName in self.statsDict[redType]:

                    # enforce that data is updated
                    assert  self.updated[globName] is True    ##OPTPRODUCTION
                    
                    # local calulation
                    locSTD = [np.std(self.localValues[globName])] * self.comm.size
                    locSTD = np.asarray(self.comm.alltoall(locSTD))
                    lg.debug('####### STD of ' + globName + ' #######')              ##OPTPRODUCTION
                    lg.debug('loc std: ' + str(locSTD))                       ##OPTPRODUCTION

                    # sending data list  of (local mean, size)
                    tmp = [(np.mean(self.localValues[globName]), self.nValues[globName])]* self.comm.size 
                    
                    # communication between all proceees
                    tmp = np.asarray(self.comm.alltoall(tmp))


                    # calculation of the global std
                    locMean = tmp[:,0]
                    
                    lg.debug('loc mean: ' + str(locMean))                     ##OPTPRODUCTION

                    locNVar = tmp[:,1]
                    lg.debug('loc number of var: ' + str(locNVar))            ##OPTPRODUCTION

                    globMean = np.sum(np.prod(tmp,axis=1)) / np.sum(locNVar)  
                    lg.debug('global mean: ' + str( globMean ))               ##OPTPRODUCTION

                    diffSqrMeans = (locMean - globMean)**2

                    deviationOfMeans = np.sum(locNVar * diffSqrMeans)

                    globVariance = (np.sum( locNVar * locSTD**2) + deviationOfMeans) / np.sum(locNVar)

                    self[globName] = np.sqrt(globVariance)
                    lg.debug('Global STD: ' + str( self[globName] ))   ##OPTPRODUCTION
                    self.updated[globName] = False
                    
            elif redType == 'var':
                for globName in self.statsDict[redType]:

                    # enforce that data is updated
                    assert  self.updated[globName] is True    ##OPTPRODUCTION
                    
                    # calculation of local mean
                    locSTD = [np.std(self.localValues[globName])] * self.comm.size
                    locSTD = np.asarray(self.comm.alltoall(locSTD))
                    

                    # out data list  of (local mean, size)
                    tmp = [(np.mean(self.localValues[globName]), self.nValues[globName])]* self.comm.size 
                    tmp = np.asarray(self.comm.alltoall(tmp))

                    locMean = tmp[:,0]
                    #print 'loc mean: ', locMean

                    lg.debug('####### Variance of ' + globName + ' #######')              ##OPTPRODUCTION
                    lg.debug('loc mean: ' + str(locMean))               ##OPTPRODUCTION
                    locNVar = tmp[:,1]
                    #print 'loc number of var: ',locNVar

                    globMean = np.sum(np.prod(tmp,axis=1)) / np.sum(locNVar)
                    #print 'global mean: ', globMean
                    
                    diffSqrMeans = (locMean - globMean)**2
                    lg.debug('global mean: ' + str( globMean )) ##OPTPRODUCTION

                    deviationOfMeans = np.sum(locNVar * diffSqrMeans)

                    globVariance = (np.sum( locNVar * locSTD**2) + deviationOfMeans) / np.sum(locNVar)

                    self[globName] = globVariance
                    lg.debug('Global variance: ' + str( self[globName] ))  ##OPTPRODUCTION
                    self.updated[globName] = False

    def sync(self):

        self.syncStats()
        self.syncReductions()

class GlobalRecord():

    def __init__(self, name, colLables, nSteps, title, style='plot'):
        self.nRec = len(colLables)
        self.columns = colLables
        self.rec = np.zeros([nSteps, self.nRec])*np.nan
        self.name = name
        self.style = style
        self.nSteps = nSteps
        self.title  = title

    def addCalibrationData(self, timeIdxs, values):
        self.calDataDict = dict()
        for idx, value in zip(timeIdxs, values):
            self.calDataDict[idx] = value


    def updateLocalValues(self, timeStep):
        self.glob.updateLocalValues(self.name, self.rec[timeStep,:])

    def gatherGlobalDataToRec(self, timeStep):
        self.rec[timeStep,:] = self.glob[self.name]
        return self.glob[self.name]

    def set(self, timeStep, data):
        self.rec[timeStep,:] = data

    def setIdx(self, timeStep, data, idx):
        self.rec[timeStep,idx] = data

    def add(self, timeStep, data):
        self.rec[timeStep,:] += data

    def addIdx(self, timeStep, data, idx):
        self.rec[timeStep,idx] += data

    def div(self, timeStep, data):
        self.rec[timeStep,:] /= data

    def divIdx(self, timeStep, data, idx):
        self.rec[timeStep,idx] /= data

    def plot(self, path):
        import matplotlib.pyplot as plt
        plt.figure()
        if self.style == 'plot':
            plt.plot(self.rec)
            if hasattr(self,'calDataDict'):
                calData = self.rec*np.nan
                for x,y in self.calDataDict.iteritems():
                    if x <= calData.shape[0]:
                        calData[x,:] = y

                plt.plot(calData,'d')

        elif self.style == 'stackedBar':
            nCars = np.zeros(self.nSteps)
            #colorPal =  sns.color_palette("Set3", n_colors=len(self.columns), desat=.8)
            for i, brand in enumerate(self.columns):
               plt.bar(np.arange(self.nSteps),self.rec[:,i],bottom=nCars, color = colorPal[i], width=1)
               nCars += self.rec[:,i]
        
        elif self.style == 'yyplot':
            fig, ax1 = plt.subplots()
            ax1.plot(self.rec[:,0], 'b-')
            #ax1.set_xlabel('timeSteps (s)')
            # Make the y-axis label, ticks and tick labels match the line color.
            ax1.set_ylabel(self.columns[0], color='b')
            ax1.tick_params('y', colors='b')
            
            ax2 = ax1.twinx()
            ax2.plot(self.rec[:,1], 'r-')
            ax2.set_ylabel(self.columns[0], color='r')
            ax2.tick_params('y', colors='r')
            
            fig.tight_layout()
            plt.show()

        plt.legend(self.columns)
        plt.title(self.title)
        plt.savefig(path +'/' + self.name + '.png')

    def saveCSV(self, path):
        df = pd.DataFrame(self.rec, columns=self.columns)
        df.to_csv(path +'/' + self.name + '.csv')

    def save2Hdf5(self, filePath):
        h5File = h5py.File(filePath, 'a')
        dset = h5File.create_dataset('glob/' + self.name, self.rec.shape, dtype='f8')
        dset[:] = self.rec
        dset.attrs['columns'] = self.columns
        dset.attrs['title']   = self.title
        if hasattr(self,'calDataDict'):
            tmp = np.zeros([len(self.calDataDict), self.rec.shape[1]+1])*np.nan
            for i, key in enumerate(self.calDataDict.keys()):
                tmp[i,:] = [key] + self.calDataDict[key]

            dset = h5File.create_dataset('calData/' + self.name, tmp.shape, dtype='f8')
            dset[:] = tmp
        h5File.close()
    

    def evaluateRelativeError(self):
        if hasattr(self,'calDataDict'):

            err = 0
            for timeIdx ,calValues in self.calDataDict.iteritems():

                for i, calValue in enumerate(calValues):
                   if not np.isnan(calValue):
                       err += np.abs(calValue - self.rec[timeIdx,i]) / calValue
            fid = open('err.csv','w')
            fid.write(str(err))
            fid.close()
            return err

        else:
            return None

class IO():

    class synthInput():
        """ MPI conform loading of synthetic population data"""
        pass # ToDo


    class Record():
        """ 
        This calls manages the translation of different graph attributes to
        the output format as a numpy array. Vectora of values automatically get
        assigned the propper matrix dimensions and indices.

        So far, only integer and float are supported
        """
        def __init__(self, nAgents, agIds, nAgentsGlob, loc2GlobIdx, nodeType, timeStepMag):
            self.ag2FileIdx = agIds
            self.nAgents = nAgents
            self.nAttr = 0
            self.attributeList = list()
            self.attrIdx = dict()
            self.header = list()
            self.timeStep = 0
            self.nAgentsGlob = nAgentsGlob
            self.loc2GlobIdx = loc2GlobIdx
            self.nodeType    = nodeType
            self.timeStepMag = timeStepMag


        def addAttr(self, name, nProp):
            attrIdx = range(self.nAttr,self.nAttr+nProp)
            self.attributeList.append(name)
            self.attrIdx[name] = attrIdx
            self.nAttr += len(attrIdx)
            self.header += [name] * nProp

        def initStorage(self, dtype):
            #print dtype
            self.data = np.zeros([self.nAgents,self.nAttr ], dtype=dtype)

        def addData(self, timeStep, nodeData):
            self.timeStep = timeStep
            self.data = nodeData[self.ag2FileIdx][self.attributeList]

        def writeData(self, h5File, folderName=None):
            #print self.header
            if folderName is None:
                path = '/' + str(self.nodeType)+ '/' + str(self.timeStep).zfill(self.timeStepMag)
                #print 'IO-path: ' + path
                self.dset = h5File.create_dataset(path, (self.nAgentsGlob,), dtype=self.data.dtype)
                self.dset[self.loc2GlobIdx[0]:self.loc2GlobIdx[1],] = self.data
            else:
                path = '/' + str(self.nodeType)+ '/' + folderName
                #print 'IO-path: ' + path
                self.dset = h5File.create_dataset(path, (self.nAgentsGlob,), dtype=self.data.dtype)
                self.dset[self.loc2GlobIdx[0]:self.loc2GlobIdx[1],] = self.data

        

    #%% Init of the IO class
    def __init__(self, world, nSteps, outputPath = ''): # of IO

        self.outputPath  = outputPath
        self._graph      = world.graph
        #self.timeStep   = world.timeStep
        self.h5File      = h5py.File(outputPath + '/nodeOutput.hdf5',
                                     'w',
                                     driver='mpio',
                                     comm=world.mpi.comm)
                                     #libver='latest',
                                     #info = world.mpi.info)
        self.comm        = world.mpi.comm
        self.dynamicData = dict()
        self.staticData  = dict() # only saved once at timestep == 0
        self.timeStepMag = int(np.ceil(np.log10(nSteps)))


    def initNodeFile(self, world, nodeTypes):
        """
        Initializes the internal data structure for later I/O
        """
        lg.info('start init of the node file')

        for nodeType in nodeTypes:
            world.mpi.comm.Barrier()
            tt = time.time()
            lg.info(' NodeType: ' +str(nodeType))
            group = self.h5File.create_group(str(nodeType))

            group.attrs.create('dynamicProps', world.graph.getPropOfNodeType(nodeType, 'dyn')['names'])
            group.attrs.create('staticProps', world.graph.getPropOfNodeType(nodeType, 'sta')['names'])

            lg.info( 'group created in ' + str(time.time()-tt)  + ' seconds'  )
            tt = time.time()

            nAgents = len(world.nodeDict[nodeType])
            self.nAgentsAll = np.empty(1*self.comm.size,dtype=np.int)

            self.nAgentsAll = self.comm.alltoall([nAgents]*self.comm.size)

            lg.info( 'nAgents exchanged in  ' + str(time.time()-tt)  + ' seconds'  )
            tt = time.time()

            lg.info('Number of all agents' + str( self.nAgentsAll ))

            nAgentsGlob = sum(self.nAgentsAll)
            cumSumNAgents = np.zeros(self.comm.size+1).astype(int)
            cumSumNAgents[1:] = np.cumsum(self.nAgentsAll)
            loc2GlobIdx = (cumSumNAgents[self.comm.rank], cumSumNAgents[self.comm.rank+1])

            lg.info( 'loc2GlobIdx exchanged in  ' + str(time.time()-tt)  + ' seconds'  )
            tt = time.time()


            # static data
            staticRec  = self.Record(nAgents, 
                                     world.dataDict[nodeType], 
                                     nAgentsGlob, 
                                     loc2GlobIdx, 
                                     nodeType, 
                                     self.timeStepMag)
            
            attrInfo   = world.graph.getPropOfNodeType(nodeType, 'sta')
            attributes = attrInfo['names']
            sizes      = attrInfo['sizes']
            
            attrDtype = world.graph.getDTypeOfNodeType(nodeType, 'sta')
            
            lg.info('Static record created in  ' + str(time.time()-tt)  + ' seconds')

            for attr, nProp in zip(attributes, sizes):

                #check if first property of first entity is string
                try:
                     
                    entProp = self._graph.getNodeSeqAttr(label=attr, nTypeID=nodeType, dataIDs=staticRec.ag2FileIdx[0])
                except ValueError:

                    raise(BaseException(str(attr) + ' not found on rank' + str(self.comm.rank)))
                if not isinstance(entProp,str):
                    staticRec.addAttr(attr, nProp)

            tt = time.time()
            # allocate storage
            staticRec.initStorage(attrDtype)
            #print attrInfo
            
            self.staticData[nodeType] = staticRec
            lg.info( 'storage allocated in  ' + str(time.time()-tt)  + ' seconds'  )

            # dynamic data
            dynamicRec = self.Record(nAgents, 
                                     world.dataDict[nodeType], 
                                     nAgentsGlob, 
                                     loc2GlobIdx, 
                                     nodeType, 
                                     self.timeStepMag)

            attrInfo   = world.graph.getPropOfNodeType(nodeType, 'dyn')
            attributes = attrInfo['names']
            sizes      = attrInfo['sizes']

            attrDtype = world.graph.getDTypeOfNodeType(nodeType, 'dyn')

            lg.info('Dynamic record created in  ' + str(time.time()-tt)  + ' seconds')


            for attr, nProp in zip(attributes, sizes):
                #check if first property of first entity is string
                entProp = self._graph.getNodeSeqAttr(attr, 
                                                     nTypeID=nodeType,
                                                     dataIDs=staticRec.ag2FileIdx[0])
                if not isinstance(entProp,str):
                    dynamicRec.addAttr(attr, nProp)

            tt = time.time()
            # allocate storage
            dynamicRec.initStorage(attrDtype)
            self.dynamicData[nodeType] = dynamicRec
            
            #lg.info( 'storage allocated in  ' + str(time.time()-tt)  + ' seconds'  )
            
            self.writeDataToFile(0, nodeType, static=True)
            
        lg.info( 'static data written to file in  ' + str(time.time()-tt)  + ' seconds'  )

    def writeDataToFile(self, timeStep, nodeTypes, static=False):
        """
        Transfers data from the graph to record for the I/O
        and writing data to hdf5 file
        """
        if isinstance(nodeTypes,int):
            nodeTypes = [nodeTypes]
        
        for nodeType in nodeTypes:
            if static:
                #for typ in self.staticData.keys():
                self.staticData[nodeType].addData(timeStep, self._graph.nodes[nodeType])
                self.staticData[nodeType].writeData(self.h5File, folderName='static')
            else:
                #for typ in self.dynamicData.keys():
                self.dynamicData[nodeType].addData(timeStep, self._graph.nodes[nodeType])
                self.dynamicData[nodeType].writeData(self.h5File)

               

    def initEdgeFile(self, edgeTypes):
        """
        ToDo
        """
        pass

    def finalizeAgentFile(self):
        """
        finalizing the agent files - closes the file and saves the
        attribute files
        ToDo: include attributes in the agent file
        """

        for nodeType in self.dynamicData.keys():
            group = self.h5File.get('/' + str(nodeType))
            record = self.dynamicData[nodeType]
            for attrKey in record.attrIdx.keys():
                group.attrs.create(attrKey, record.attrIdx[attrKey])

        for nodeType in self.staticData.keys():
            group = self.h5File.get('/' + str(nodeType))
            record = self.staticData[nodeType]
            for attrKey in record.attrIdx.keys():
                group.attrs.create(attrKey, record.attrIdx[attrKey])


        self.h5File.close()
        lg.info( 'Agent file closed')

        for nodeType in self.dynamicData.keys():
            record = self.dynamicData[nodeType]
            saveObj(record.attrIdx, (self.outputPath + '/attributeList_type' + str(nodeType)))

class Mpi():
    """
    MPI communication module that controles all communcation between
    different processes.
    ToDo: change to communication using numpy
    """

    def __init__(self, world, mpiComm=None):

        self.world = world
        if mpiComm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = mpiComm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.info = MPI.Info.Create()
        self.info.Set("romio_ds_write", "disable")
        self.info.Set("romio_ds_read", "disable")

        self.peers    = list()     # list of ranks of all processes that have ghost duplicates of this process

        self.ghostNodeQueue = dict()
        #ID list of ghost nodes that recieve updates from other processes
        self.mpiRecvIDList = dict()           
        #ID list of ghost nodes that send updates from other processes
        self.mpiSendIDList = dict()
        
        self.buffer         = dict()
        self.messageSize    = dict()
        self.sendReqList    = list()

        self.reduceDict = dict()
        world.send = self.comm.send
        world.recv = self.comm.recv

        world.isend = self.comm.isend
        world.irecv = self.comm.irecv

        self._clearBuffer()
        
        self.world.graph.ghostTypeUpdated = dict()
        
        self.world.graph.isParallel =  self.size > 1
            
    #%% Privat functions
    def _clearBuffer(self):
        """
        Method to clear all2all buffer
        """
        self.a2aBuff = []
        for x in range(self.comm.size):
            self.a2aBuff.append([])


    def _add2Buffer(self, mpiPeer, data):
        """
        Method to add data to all2all data to buffer
        """
        self.a2aBuff[mpiPeer].append(data)

    def _all2allSync(self):
        """
        Privat all2all communication method
        """
        recvBuffer = self.comm.alltoall(self.a2aBuff)
        self._clearBuffer()
        return recvBuffer


    def _packData(self, nodeType, mpiPeer, propList, connList=None):
        """
        Privat method to pack all data for MPI transfer
        """
        dataSize = 0
        nNodes = len(self.mpiSendIDList[(nodeType,mpiPeer)])
        
        dataPackage = dict()
        dataPackage['nNodes']  = nNodes
        dataPackage['nTypeID'] = nodeType
        dataPackage['data'] = self.world.graph.getNodeSeqAttr(label=propList, lnIDs=self.mpiSendIDList[(nodeType,mpiPeer)] )
        dataSize += np.prod(dataPackage['data'].shape)
        if connList is not None:
            dataPackage['connectedNodes'] = connList
            dataSize += len(connList)
            
        lg.debug('package size: ' + str(dataSize))
        return dataPackage, dataSize



    def _updateGhostNodeData(self, nodeTypeList= 'dyn', propertyList= 'dyn'):
        """
        Privat method to update the data between processes for existing ghost nodes
        """
        tt = time.time()
        messageSize = 0
        
        for (nodeType, mpiPeer) in self.mpiSendIDList.keys():

            if nodeTypeList == 'all' or nodeType in nodeTypeList:

                if propertyList in ['all', 'dyn', 'sta']:
                    propertyList = self.world.graph.getPropOfNodeType(nodeType, kind=propertyList)['names']
                    
                lg.debug('MPIMPIMPIMPI -  Updating ' + str(propertyList) + ' for nodeType ' + str(nodeType) + 'MPIMPIMPI')
                dataPackage ,packageSize = self._packData(nodeType, mpiPeer, propertyList, connList=None)
                                                    
                messageSize = messageSize + packageSize
                self._add2Buffer(mpiPeer, dataPackage)

        syncPackTime = time.time() -tt

        tt = time.time()
        recvBuffer = self._all2allSync()
        pureSyncTime = time.time() -tt

        tt = time.time()
        
        for mpiPeer in self.peers:
            if len(recvBuffer[mpiPeer]) > 0: # will receive a message


                for dataPackage in recvBuffer[mpiPeer]:
                    nodeType = dataPackage['nTypeID']

                    if propertyList == 'all':
                        propertyList= self.world.graph.nodeProperies[nodeType][:]
                        propertyList.remove('gID')

                    self.world.graph.setNodeSeqAttr(label=propertyList, 
                                                    values=dataPackage['data'],
                                                    lnIDs=self.mpiRecvIDList[(nodeType, mpiPeer)])                        
                    
        syncUnpackTime = time.time() -tt

        lg.info('Sync times - ' +
                ' pack: ' + str(syncPackTime) + ' s , ' +
                ' comm: ' + str(pureSyncTime) + ' s , ' +
                ' unpack: ' + str(syncUnpackTime) + ' s , ')
        return messageSize

    def initCommunicationViaLocations(self, ghostLocationList, locNodeType):
        """
        Method to initialize the communication based on the spatial
        distribution
        """

        tt = time.time()
        # acquire the global IDs for the ghostNodes
        mpiRequest = dict()
        

        
        lg.debug('ID Array: ' + str(self.world.graph.IDArray))##OPTPRODUCTION
        for ghLoc in ghostLocationList:
            owner = ghLoc.mpiOwner
            #print owner
            x,y   = ghLoc.getValue('pos')
            if owner not in mpiRequest:
                mpiRequest[owner]   = (list(), 'gID')
                self.mpiRecvIDList[(locNodeType, owner)] = list()

            mpiRequest[owner][0].append( (x,y) ) # send x,y-pairs for identification
            self.mpiRecvIDList[(locNodeType, owner)].append(ghLoc.nID)
        lg.debug('rank ' + str(self.rank) + ' mpiRecvIDList: ' + str(self.mpiRecvIDList))##OPTPRODUCTION

        for mpiDest in mpiRequest.keys():

            if mpiDest not in self.peers:
                self.peers.append(mpiDest)

                # send request of global IDs
                lg.debug( str(self.rank) + ' asks from ' + str(mpiDest) + ' - ' + str(mpiRequest[mpiDest]))##OPTPRODUCTION
                #self.comm.send(mpiRequest[mpiDest], dest=mpiDest)
                self._add2Buffer(mpiDest, mpiRequest[mpiDest])

        lg.debug( 'requestOut:' + str(self.a2aBuff))##OPTPRODUCTION
        requestIn = self._all2allSync()
        lg.debug( 'requestIn:' +  str(requestIn))##OPTPRODUCTION


        for mpiDest in mpiRequest.keys():

            #self.ghostNodeRecv[locNodeType, mpiDest] = self.world.graph.vs[mpiRecvIDList[mpiDest]] #sequence

            # receive request of global IDs
            lg.debug('receive request of global IDs from:  ' + str(mpiDest))##OPTPRODUCTION
            #incRequest = self.comm.recv(source=mpiDest)
            incRequest = requestIn[mpiDest][0]
            
            #pprint(incRequest)
            lnIDList = [int(self.world.graph.IDArray[xx, yy]) for xx, yy in incRequest[0]]
            #print lnIDList
            lg.debug( str(self.rank) + ' -sendIDlist:' + str(lnIDList))##OPTPRODUCTION
            self.mpiSendIDList[(locNodeType,mpiDest)] = lnIDList
            #self.ghostNodeSend[locNodeType, mpiDest] = self.world.graph.vs[IDList]
            #self.ghostNodeOut[locNodeType, mpiDest] = self.world.graph.vs[iDList]
            
            lg.debug( str(self.rank) + ' - gIDs:' + str(self.world.graph.getNodeSeqAttr('gID', lnIDList)))##OPTPRODUCTION

            for entity in [self.world.entDict[i] for i in lnIDList]:
                entity.mpiPeers.append(mpiDest)

            # send requested global IDs
            lg.debug( str(self.rank) + ' sends to ' + str(mpiDest) + ' - ' + str(self.mpiSendIDList[(locNodeType,mpiDest)]))##OPTPRODUCTION

            x = self.world.graph.getNodeSeqAttr(label=incRequest[1], lnIDs= lnIDList )
            #print 'global IDS' + str(x)
            #print type(x)
            #print x.shape
            self._add2Buffer(mpiDest,self.world.graph.getNodeSeqAttr(label=incRequest[1], lnIDs=lnIDList))

        requestRecv = self._all2allSync()

        for mpiDest in mpiRequest.keys():
            #self.comm.send(self.ghostNodeOut[locNodeType, mpiDest][incRequest[1]], dest=mpiDest)
            #receive requested global IDs
            globIDList = requestRecv[mpiDest][0]
            
            #self.ghostNodeRecv[locNodeType, mpiDest]['gID'] = globIDList
            #print self.mpiRecvIDList[(locNodeType, mpiDest)]
            self.world.graph.setNodeSeqAttr(label=['gID'], values=globIDList, lnIDs=self.mpiRecvIDList[(locNodeType, mpiDest)])
            lg.debug( 'receiving globIDList:' + str(globIDList))##OPTPRODUCTION
            lg.debug( 'localDList:' + str(self.mpiRecvIDList[(locNodeType, mpiDest)]))##OPTPRODUCTION
            for nID, gID in zip(self.mpiRecvIDList[(locNodeType, mpiDest)], globIDList):
                #print nID, gID
                self.world._glob2loc[gID] = nID
                self.world._loc2glob[nID] = gID
            #self.world.mpi.comm.Barrier()
        lg.info( 'Mpi commmunication required: ' + str(time.time()-tt) + ' seconds')

    def transferGhostNodes(self, world):
        """
        Privat method to initially transfer the data between processes and to create
        ghost nodes from the received data
        """

        messageSize = 0
        #%%Packing of data
        for nodeType, mpiPeer in sorted(self.ghostNodeQueue.keys()):

            #get size of send array
            IDsList= self.ghostNodeQueue[(nodeType, mpiPeer)]['nIds']
            connList = self.ghostNodeQueue[(nodeType, mpiPeer)]['conn']

            self.mpiSendIDList[(nodeType,mpiPeer)] = IDsList

            #nodeSeq = world.graph.vs[IDsList]

            # setting up ghost out communication
            #self.ghostNodeSend[nodeType, mpiPeer] = IDsList
            
            propList = world.graph.getPropOfNodeType(nodeType, kind='all')['names']
            #print propList
            dataPackage, packageSize = self._packData( nodeType, mpiPeer,  propList, connList)
            self._add2Buffer(mpiPeer, dataPackage)
            messageSize = messageSize + packageSize
        recvBuffer = self._all2allSync()

        lg.info('approx. MPI message size: ' + str(messageSize * 24. / 1000. ) + ' KB')

        for mpiPeer in self.peers:

            for dataPackage in recvBuffer[mpiPeer]:

        #%% create ghost agents from dataDict

                nNodes   = dataPackage['nNodes']
                nodeType = dataPackage['nTypeID']
               
                IDsList = world.addVertices(nodeType, nNodes)
                # setting up ghostIn communicator
                self.mpiRecvIDList[(nodeType, mpiPeer)] = IDsList


                propList = world.graph.getPropOfNodeType(nodeType, kind='all')['names']
                propList.append('gID')
                     
                self.world.graph.setNodeSeqAttr(label=propList, 
                                                values=dataPackage['data'],
                                                lnIDs=self.mpiRecvIDList[(nodeType, mpiPeer)])                        

                gIDsParents = dataPackage['connectedNodes']

                # creating entities with parentEntities from connList (last part of data package: dataPackage[-1])
                for nID, gID in zip(self.mpiRecvIDList[(nodeType, mpiPeer)], gIDsParents):

                    GhostAgentClass = world.graph.nodeType2Class[nodeType][1]

                    agent = GhostAgentClass(world, mpiPeer, nID=nID)

                    parentEntity = world.entDict[world._glob2loc[gID]]
                    edgeType = world.graph.node2EdgeType[parentEntity.nodeType, nodeType]

                    agent.register(world, parentEntity, edgeType)


        lg.info('################## Ratio of ghost agents ################################################')
        for nodeTypeIdx in world.graph.nodeTypes.keys():
            nodeType = world.graph.nodeTypes[nodeTypeIdx].typeStr
            if len(world.nodeDict[nodeTypeIdx]) > 0:
                nGhostsRatio = float(len(world.ghostNodeDict[nodeTypeIdx])) / float(len(world.nodeDict[nodeTypeIdx]))
                lg.info('Ratio of ghost agents for type "' + nodeType + '" is: ' + str(nGhostsRatio))
        lg.info('#########################################################################################')




    def updateGhostNodes(self, nodeTypeList= 'all', propertyList='all'):
        """
        Method to update ghost node data on all processes
        """
        
        if self.comm.size == 1:
            return None
        tt = time.time()

        if nodeTypeList == 'all':
            nodeTypeList = self.world.graph.nodeTypes
        messageSize = self._updateGhostNodeData(nodeTypeList, propertyList)

        if self.world.timeStep == 0:
            lg.info('Ghost update (of approx size ' +
                 str(messageSize * 24. / 1000. ) + ' KB)' +
                 ' required: ' + str(time.time()-tt) + ' seconds')
        else:                                                           ##OPTPRODUCTION
            lg.debug('Ghost update (of approx size ' +                  ##OPTPRODUCTION
                     str(messageSize * 24. / 1000. ) + ' KB)' +         ##OPTPRODUCTION
                     ' required: ' + str(time.time()-tt) + ' seconds')  ##OPTPRODUCTION
        
        if nodeTypeList == 'all':
            nodeTypeList = self.world.graph.nodeTypes
        
        
        for nodeType in nodeTypeList:
            self.world.graph.ghostTypeUpdated[nodeType] = list()
            if propertyList in ['all', 'dyn', 'sta']:        
                propertyList = self.world.graph.getPropOfNodeType(nodeType, kind=propertyList)['names']
            
            
            for prop in propertyList:
                self.world.graph.ghostTypeUpdated[nodeType].append(prop)
            

    def queueSendGhostNode(self, mpiPeer, nodeType, entity, parentEntity):

        if (nodeType, mpiPeer) not in self.ghostNodeQueue.keys():
            self.ghostNodeQueue[nodeType, mpiPeer] = dict()
            self.ghostNodeQueue[nodeType, mpiPeer]['nIds'] = list()
            self.ghostNodeQueue[nodeType, mpiPeer]['conn'] = list()

        self.ghostNodeQueue[nodeType, mpiPeer]['nIds'].append(entity.nID)
        self.ghostNodeQueue[nodeType, mpiPeer]['conn'].append(parentEntity.gID)



    def all2all(self, value):
        """
        This method is a quick communication implementation that allows +
        sharing data between all processes

        """
        if isinstance(value,int):
            buff = np.empty(1*self.comm.size,dtype=np.int)
            buff = self.comm.alltoall([value]*self.comm.size)
        elif isinstance(value,float):
            buff = np.empty(1*self.comm.size,dtype=np.float)
            buff = self.comm.alltoall([value]*self.comm.size)
        elif isinstance(value,str):
            buff = np.empty(1*self.comm.size,dtype=np.str)
            buff = self.comm.alltoall([value]*self.comm.size)
        else:
            buff = self.comm.alltoall([value]*self.comm.size)

        return buff

class Random():

    def __init__(self, world):
        self.world = world # make world availabel in class random

    def entity(self, nChoice, entType):
        ids = np.random.choice(self.world.nodeDict[entType],nChoice,replace=False)
        return [self.world.entDict[idx] for idx in ids]

if __name__ == '__main__':
    import mpi4py
    mpi4py.rc.threads = False
    comm = MPI.COMM_WORLD
    mpiRank = comm.Get_rank()
    mpiSize = comm.Get_size()
    debug = True
    showFigures    = 1
    
    simNo, baseOutputPath = getEnvironment(comm, getSimNo=True)
    outputPath = createOutputDirectory(comm, baseOutputPath, simNo)
    
    
    
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