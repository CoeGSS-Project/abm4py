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
#import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import pickle

import sys
from numbers import Number
from collections import Set, Mapping, deque

try: # Python 2
    zero_depth_bases = (basestring, Number, xrange, bytearray)
    iteritems = 'iteritems'
except NameError: # Python 3
    zero_depth_bases = (str, bytes, Number, range, bytearray)
    iteritems = 'items'

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
    

def getsize(obj_0):
    """Recursively iterate to sum size of object & members."""
    def inner(obj, _seen_ids = set()):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size
    return inner(obj_0)


def getEnvironment(comm, getSimNo=True):
    
    if comm == None or comm.rank == 0 :
        # get simulation number from file
        try:
            fid = open("environment","r")
            simNo = int(fid.readline())
            basePath = fid.readline().rstrip('\n')
            fid.close()
            if getSimNo:
                fid = open("environment","w")
                fid.writelines(str(simNo+1)+ '\n')
                fid.writelines(basePath)
                fid.close()
        except:
            print 'ERROR Envirionment file is not set up'
            print ''
            print 'Please create file "environment" which contains the simulation'
            print 'and the basePath'
            print '#################################################'
            print "0" 
            print 'basepath/'
            print '#################################################'
    else:
        simNo    = None
        basePath = None

    
    if getSimNo:   
        if comm is None:
            return simNo, basePath
        simNo = comm.bcast(simNo, root=0)
        basePath = comm.bcast(basePath, root=0)
        print 'simulation number is: ' + str(simNo)
        print 'base path is: ' + str(basePath)
        return simNo, basePath
    else:
        if comm is None:
            return basePath
        basePath = comm.bcast(basePath, root=0)
        return basePath
    
def computeConnectionList(radius=1, weightingFunc = lambda x,y : 1/((x**2 +y**2)**.5), ownWeight =2):
    """
    Method for easy computing a connections list of regular grids
    """
    connList = []  
    
    intRad = int(radius)
    for x in range(-intRad,intRad+1):
        for y in range(-intRad,intRad+1):
            if (x**2 +y**2)**.5 <= radius:
                if x**2 +y**2 > 0:
                    weig  = weightingFunc(x,y)
                else:
                    weig = ownWeight
                connList.append((x,y,weig))
    return connList

def cartesian(arrays, out=None):
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
    
    
    
class Record():
    
    def __init__(self, name, colLables, nSteps, title, style='plot'):
        self.nRec = len(colLables)
        self.columns = colLables
        self.rec = np.zeros([nSteps, self.nRec])
        self.name = name
        self.style = style
        self.nSteps = nSteps
        self.title  = title
    
    def addCalibrationData(self, timeIdxs, values):
        self.calDataDict = dict()
        for idx, value in zip(timeIdxs, values):
            self.calDataDict[idx] = value
    
        
    def updateValues(self, timeStep):
        self.glob[self.name] = self.rec[timeStep,:]
        
    def gatherSyncDataToRec(self, timeStep):
        self.rec[timeStep,:] = self.glob[self.name]
        
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
            colorPal =  sns.color_palette("Set3", n_colors=len(self.columns), desat=.8)
            for i, brand in enumerate(self.columns):
               plt.bar(np.arange(self.nSteps),self.rec[:,i],bottom=nCars, color =colorPal[i], width=1)
               nCars += self.rec[:,i]
            
        plt.legend(self.columns)
        plt.title(self.title)
        plt.savefig(path +'/' + self.name + '.png')
        
    def saveCSV(self, path):
        df = pd.DataFrame(self.rec, columns=self.columns)
        df.to_csv(path +'/' + self.name + '.csv')

    def save2Hdf5(self, h5File):
        dset = h5File.create_dataset('glob/' + self.name, self.rec.shape, dtype='f8')
        dset[:] = self.rec
        dset.attrs['columns'] = self.columns   
        
        if hasattr(self,'calDataDict'):
            tmp = np.zeros([len(self.calDataDict), self.rec.shape[1]+1])*np.nan
            for i, key in enumerate(self.calDataDict.keys()):
                tmp[i,:] = [key] + self.calDataDict[key]
            
            dset = h5File.create_dataset('calData/' + self.name, tmp.shape, dtype='f8')
            dset[:] = tmp
            
            
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
        
        
