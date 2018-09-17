#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2018, Global Climate Forun e.V. (GCF)
http://www.globalclimateforum.org

This file is part of ABM4py.

ABM4py is free software: you can redistribute it and/or modify it 
under the terms of the GNU Lesser General Public License as published 
by the Free Software Foundation, version 3 only.

ABM4py is distributed in the hope that it will be useful, 
but WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License 
along with this program. If not, see <http://www.gnu.org/licenses/>. 
GNU Lesser General Public License version 3 (see the file LICENSE).
"""

#%% INIT
__version__ = "0.7.0"

import os
import numpy as np
import itertools
import pandas as pd

try:
    from numba import njit
    NUMBA = True
    
except:
    print('numba import failed')
    NUMBA = False
    
    
import time
import random
from . import misc
from collections import OrderedDict



# Evaluation if mpi4py is installed and workding
try:
    import mpi4py
    mpi4py.rc.threads = False
    from mpi4py import MPI
    comm = mpi4py.MPI.COMM_WORLD
    mpiSize = comm.Get_size()
    mpiRank = comm.Get_rank()
    mpiBarrier = mpi4py.MPI.COMM_WORLD.Barrier
    if mpiSize> 1:
        print('mpi4py import successfull')
       
except:
    
    comm = None
    mpiSize = 1
    mpiRank = 0
    print('mpi4py failed - serial processing')
    def barrier():
        pass
    mpiBarrier = barrier

import h5py
import logging as lg


# This compound typtes are allowed as agent attributes
ALLOWED_MULTI_VALUE_TYPES = (list, tuple, np.ndarray)

#%% FUNCTION DEFINITION

def setupSimulationEnvironment(mpiComm=None, simNo=None):
    """
    Reads an existng or creates a new environment file
    Returns simulation number and outputPath
    """
    if dakota.isActive:
        simNo = dakota.simNo
    if (mpiComm is None) or (mpiComm.rank == 0):
        # get simulation number from file
        try:
            fid = open("environment","r")
            _simNo = int(fid.readline())
            baseOutputPath = fid.readline().rstrip('\n')
            fid.close()
            if simNo is None:
                simNo = _simNo
                fid = open("environment","w")
                fid.writelines(str(simNo+1)+ '\n')
                fid.writelines(baseOutputPath)
                fid.close()
        except:
            print('Environment file is not set up')
            import os
            
            baseOutputPath  = 'output/'
            simNo           = 0
            if not os.path.exists(baseOutputPath):
                os.makedirs(baseOutputPath)
            fid = open("environment","w")
            fid.writelines(str(simNo+1)+ '\n')
            fid.writelines(baseOutputPath)
            fid.close()
            

            print('A new environment was created usning the following template')
            print('#################################################')
            print("0")
            print('output/')
            print('#################################################')
            
    else:
        simNo    = None
        baseOutputPath = None
    
    if (mpiComm is not None) and mpiComm.size > 1:
        # parallel communication
        simNo = mpiComm.bcast(simNo, root=0)
        baseOutputPath = mpiComm.bcast(baseOutputPath, root=0)
    outputPath = createOutputDirectory(mpiComm, baseOutputPath, simNo)
    
    return simNo, outputPath

def createOutputDirectory(mpiComm, baseOutputPath, simNo):
    
        if not os.path.isdir(baseOutputPath):
            os.makedirs(baseOutputPath)
        dirPath  = baseOutputPath + 'sim' + str(simNo).zfill(4)

        if (mpiComm is None) or mpiComm.rank ==0:
            

            if not os.path.isdir(dirPath):
                os.makedirs(dirPath)
        if (mpiComm is not None):
            mpiComm.Barrier()
        if (mpiComm is None) or mpiComm.rank ==0:
            print('output directory created')

        return dirPath

def configureLogging(outputPath, debug=False):
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

def configureSTD(outputPath, out2File=True, err2File=True):
    import sys

    if out2File:    
        log_file   =  open(outputPath + '/out' + str(mpiRank) + '.txt', 'w')
        sys.stdout = log_file    
    
    if out2File:
        err_file   =  open(outputPath + '/err' + str(mpiRank) + '.txt', 'w')
        sys.stderr = err_file

def formatPropertyDefinition(propertyList):
    """
    Checks and completes the property definition for entities and edges
    """
    for iProp in range(len(propertyList)):
        if not isinstance(propertyList[iProp], tuple):
            propertyList[iProp] = (propertyList[iProp], np.float64, 1) 
        else:
            if len(propertyList[iProp]) == 3:
                pass
                
            elif len(propertyList[iProp]) == 2:
                propertyList[iProp] = (propertyList[iProp] + (1,))
                print('Assuming a single number for ' + str(propertyList[iProp][1]))
            elif len(propertyList[iProp]) == 1:
                propertyList[iProp] = (propertyList[iProp] + (np.float64, 1,))
                print('Assuming a single float number for ' + str(propertyList[iProp][1]))
            else:
                raise(BaseException('Property format of ' + str(propertyList[iProp]) + ' not understood'))    
                
        assert isinstance(propertyList[iProp][0],str)
        assert isinstance(propertyList[iProp][1],type)
        assert isinstance(propertyList[iProp][2],int)
        
    return propertyList





#def cartesian2(arrays):
#    arrays = [np.asarray(a) for a in arrays]
#    shape = (len(x) for x in arrays)
#
#    ix = np.indices(shape, dtype=int)
#    ix = ix.reshape(len(arrays), -1).T
#
#    for n, arr in enumerate(arrays):
#        ix[:, n] = arrays[n][ix[:, n]]
#
#    return ix

#from sklearn.utils.extmath import cartesian
#cartesian = cartesian

def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.

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
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out

def readParametersFromFile(paraDict, fileName):
    import csv
    for item in csv.DictReader(open(fileName)):
        if item['name'][0] != '#':
            paraDict[item['name']] = convertStr(item['value'])
    return paraDict

        
def convertStr(string):
    """
    Returns integer, float or string dependent on the input
    """
    if str.isdigit(string):
        return int(string)
    else:
        try:
            return np.float(string)
        except:
            return string

def plotGraph(world, agentTypeID, liTypeID=None, attrLabel=None, ax=None):
    import matplotlib.pyplot as plt
    from matplotlib import collections  as mc
    linesToDraw = list()
    try:
        positions = world.getAttrOfAgentType('coord', agTypeID=agentTypeID)
    except:
        positions = np.asarray([agent.loc.attr['coord'] for agent in world.getAgentsByType(agentTypeID)])
    
    if ax is None:
        plt.ion()
        plt.figure('graph')
        plt.clf()
        ax = plt.subplot(111)
        
    if liTypeID is not None:
        for agent in world.getAgentsByType(agentTypeID):
            
            try:
                pos = agent.attr['coord']
            except:
                pos = agent.loc.attr['coord']
                
            peerDataIDs   = np.asarray(agent.getPeerIDs(liTypeID)) - (world.maxNodes*agentTypeID)
            if len(peerDataIDs)> 0:
                peerPositions = positions[peerDataIDs]
                for peerPos in peerPositions:
                    
                    linesToDraw.append([[pos[0], pos[1]], [peerPos[0], peerPos[1]]])
        lc = mc.LineCollection(linesToDraw, colors='b', lw=.1, zorder=1) 
        ax.add_collection(lc)
        ax.autoscale()
    if attrLabel is not None:
        values = world.getAttrOfAgentType(attrLabel, agTypeID=agentTypeID)
        #values = values / np.max(values)
        #print(values)
        plt.scatter(positions[:,0], positions[:,1], s = 25, c = values,zorder=2)
        plt.colorbar()
    else:
        plt.scatter(positions[:,0], positions[:,1], s = 25, zorder=2)
    
    
    ax.margins(0.1)      
    plt.tight_layout()  
    
    plt.draw()


def _h5File_driver(filePath):

    info = MPI.Info.Create()
    info.Set("romio_ds_write", "disable")
    info.Set("romio_ds_read", "disable")
        
    if mpiSize ==1:
            
        return h5py.File(filePath, 'w')
 
    else:
        return h5py.File(filePath, 'w',
                          driver='mpio',
                          comm=comm,
                          libver='latest',
                          info = info)


def initLogger(debug, outputPath):
    """
    Configuration of the logging library. Input is the debug level as 0 or 1
    and the outputpath
    """
    import logging as lg
    lg.basicConfig(filename=outputPath + '/log_R' + str(mpiRank),
                   filemode='w',
                   format='%(levelname)7s %(asctime)s : %(message)s',
                   datefmt='%m/%d/%y-%H:%M:%S',
                   level=lg.DEBUG if debug else lg.INFO)

    lg.info('Log file of process ' + str(mpiRank) + ' of ' + str(mpiSize))

    # wait for all processes - debug only for poznan to debug segmentation fault
    mpiBarrier()
    if mpiRank == 0:
        print('log files created')
    return lg

#%% CLASS DEFINITION
class Enum():
    """
    This class allows to store global enumerations over mutiple files
    """
    pass
enum = Enum()

class GlobalVariables():
    """
    This class allows to store global variables over mutiple files
    """
    pass
glVar = GlobalVariables()


class Dakota():
    def __init__(self):
        self.isActive = False

    def overwriteParameters(self, simNo, parameterDict):
        self.isActive = True
        self.simNo  = simNo
        self.params = parameterDict
        
    
dakota = Dakota()


class Grid():
    """
    This class is an enhancement of the world class and represent the spatial 
    domain. The grid consists of agents with the trait "GridNode" (see: traits.py) 
    and links  between them resemble the spatial proximity.
    (see: examples/02_tutorial_grid.py)
    """
    from .misc import weightingFunc, distance
    def __init__(self, world, nodeTypeID, linkTypeID):
        
        self.world = world # make world availabel in class random
        self.gNodeTypeID = nodeTypeID
        self.gLinkTypeID = linkTypeID
        self.GridNodeClass = self.world._graph.agTypeID2Class(nodeTypeID)[0]
        self.gridNodeDict  = dict()
    
    
    def registerNode(self, gridNode, x, y):

        self.gridNodeDict[x,y] = gridNode
    
    @staticmethod
    def computeConnectionList(radius=1, weightingFunc = weightingFunc, ownWeight =2, distance_func = distance):
        """
        Method for computing a connections list of regular grids
        (see: examples/02_tutorial_grid.py)
        """

        connList = []   
        intRad = int(radius)
        for x in range(-intRad,intRad+1):
            for y in range(-intRad,intRad+1):
                if distance_func(x,y) <= radius:
                    if x**2 +y**2 > 0:
                        weig  = weightingFunc(x,y)
                    else:
                        weig = ownWeight
                    connList.append((x,y,weig))
        return connList

    def init(self, gridMask, connList, LocClassObject, mpiRankArray=None):
        """
        Auxiliary function to contruct a simple connected layer of spatial locations.
        Use with  the previously generated connection list (see computeConnnectionList)
        """

        agTypeID = self.world._graph.class2NodeType(LocClassObject)
        if self.world.isParallel:
            GhstLocClassObject = self.world._graph.ghostOfAgentClass(LocClassObject)
        
        IDArray = gridMask * np.nan
        xOrg = 0
        yOrg = 0
        xMax = gridMask.shape[0]
        yMax = gridMask.shape[1]
        ghostLocationList = list()
        lg.debug('rank array: ' + str(gridMask)) ##OPTPRODUCTION
        # tuple of idx array of cells that correspond of the spatial input maps 
        if mpiRankArray is not None:
            self.world.cellMapIds = np.where(mpiRankArray == self.world.mpiRank)
        else:
            self.world.cellMapIds = np.where(gridMask)

        # create vertices
        for x in range(xMax):
            for y in range(yMax):

                # only add an vertex if spatial location exist
                if gridMask[x,y] and (not(self.world.isParallel) or (mpiRankArray[x,y] == self.world.mpiRank)):
                    
                    loc = LocClassObject(self.world, coord = [x, y])
                    IDArray[x,y] = loc.nID
                    loc.register(self.world)

        if self.world.isParallel:
            # create ghost location nodes
            for (x,y), loc in list(self.gridNodeDict.items()):
    
                
                for (dx,dy,weight) in connList:
    
                    xDst = x + dx
                    yDst = y + dy
    
                    # check boundaries of the destination node
                    if xDst >= xOrg and xDst < xMax and yDst >= yOrg and yDst < yMax:
    
    
                        if np.isnan(IDArray[xDst,yDst]) and gridMask[xDst,yDst] and mpiRankArray[xDst,yDst] != mpiRank:  # location lives on another process
                            
                            loc = GhstLocClassObject(self.world, mpiOwner=mpiRankArray[xDst,yDst], coord= (xDst, yDst))
                            #print 'rank: ' +  str(mpiRank) + ' '  + str(loc.nID)
                            IDArray[xDst,yDst] = loc.nID
                            
                            #self.world.registerAgent(loc, ghost=True) #so far ghost nodes are not in entDict, nodeDict, entList
                            loc.register(self.world)
                            #self.world.registerLocation(loc, xDst, yDst)
                            ghostLocationList.append(loc)
        
        self.world._graph.IDArray = IDArray
        print('Nodes added: ' + str(self.world.countAgents(agTypeID)))
        self.connectNodes(IDArray, connList, agTypeID, ghostLocationList)


    def connectNodes(self, IDArray, connList, agTypeID, ghostLocationList=None):
        
        xOrg = 0
        yOrg = 0
        xMax = IDArray.shape[0]
        yMax = IDArray.shape[1]

        fullSourceList      = list()
        fullTargetList      = list()
        fullWeightList      = list()
        
        for (x,y), loc in list(self.gridNodeDict.items()):

            srcID = loc.nID
            
            weigList = list()
            destList = list()
            sourceList = list()
            targetList = list()
            for (dx,dy,weight) in connList:

                xDst = x + dx
                yDst = y + dy

                # check boundaries of the destination node
                if xDst >= xOrg and xDst < xMax and yDst >= yOrg and yDst < yMax:

                    trgID = IDArray[xDst,yDst]
                    #assert

                    if not np.isnan(trgID): #and srcID != trgID:
                        destList.append(int(trgID))
                        weigList.append(weight)
                        sourceList.append(int(srcID))
                        targetList.append(int(trgID))

            #normalize weight to sum up to unity
            sumWeig = sum(weigList)
            weig    = np.asarray(weigList) / sumWeig
            fullSourceList.extend(sourceList)
            fullTargetList.extend(targetList)
            fullWeightList.extend(weig)


        # create the links in the._graph
        if 'weig' in self.world._graph.edges[self.gLinkTypeID].dtype.names:
           self.world._graph.addEdges(self.gLinkTypeID, fullSourceList, fullTargetList, weig=fullWeightList)
        else:
            self.world._graph.addEdges(self.gLinkTypeID, fullSourceList, fullTargetList)

        # for parallel execution, share the grid between all neigboring processe
        if self.world.isParallel:   
            
            lg.debug('starting initCommunicationViaLocations')##OPTPRODUCTION
            self.world.papi.initCommunicationViaLocations(ghostLocationList, agTypeID)
            lg.debug('finished initCommunicationViaLocations')##OPTPRODUCTION
            
    def getNodeID(self, x,y):
        #get nID of the location
        nID = self.gridNodeDict[x,y].nID
        return self.world.getAgent(agentID=nID)
    
    def getNodeDict(self):
        return self.gridNodeDict
    
    def getNCloseAgents(self, 
                           agent, 
                           nContacts,
                           agTypeID, 
                           currentContacts = None, 
                           addYourself = True):
        """
        Method to generate a preliminary friend network that accounts for
        proximity in space
        """

        liTypeID = self.world._graph.node2EdgeType[1, agTypeID]
        
        if currentContacts is None:
            isInit=True
        else:
            isInit=False

        if currentContacts is None:
            currentContacts = [agent.nID]
        else:
            currentContacts.append(agent.nID)

        cellConnWeights, cellIds = agent.loc.getConnectedLocation()
        personIdsAll = list()
        nPers = list()
        cellWeigList = list()
              
        for cellWeight, cellIdx in zip(cellConnWeights, cellIds):
            cellWeigList.append(cellWeight)           
            personIds = self.world.getAgents(cellIdx).getPeerIDs(liTypeID)
            personIdsAll.extend(personIds)
            nPers.append(len(personIds))

        # return nothing if too few candidates
        if not isInit and nPers > nContacts:
            lg.info('ID: ' + str(agent.nID) + ' failed to generate friend')
            return [],[],[]

        #setup of spatial weights
        weightData = np.zeros(np.sum(nPers))

        idx = 0
        for nP, we in zip(nPers, cellWeigList):
            weightData[idx:idx+nP] = we
            idx = idx+ nP
        del idx

        #normalizing final row
        weightData /= np.sum(weightData,axis=0)

        if np.sum(weightData>0) < nContacts:
            lg.info( "nID: " + str(agent.nID) + ": Reducting the number of friends at " + str(self.loc.get('pos')))
            nContacts = min(np.sum(weightData>0)-1,nContacts)

        if nContacts < 1:                                                       ##OPTPRODUCTION
            lg.info('ID: ' + str(agent.nID) + ' failed to generate friend')      ##OPTPRODUCTION
            contactList = list()
            sourceList  = list()
            targetList   = list()
        else:
            # adding contacts
            ids = np.random.choice(weightData.shape[0], nContacts, replace=False, p=weightData)
            contactList = [personIdsAll[idx] for idx in ids ]
            targetList  = [personIdsAll[idx] for idx in ids]
            sourceList  = [agent.nID] * nContacts
        
        if isInit and addYourself:
            #add yourself as a friend
            contactList.append(agent.nID)
            sourceList.append(agent.nID)
            targetList.append(agent.nID)
            nContacts +=1

        weigList   = [1./nContacts]*nContacts
        return contactList, (sourceList, targetList), weigList
    

class Writer():
    """
    deprecated
    """
    def __init__(self, world, filename):
        self.fid = open('output/' + filename,'w')
        self.world = world
        self.world.reporter.append(self)

    def write(self,outStr):
        self.fid.write(str(self.world.time) + ' - ' + outStr + '\n')

    def close(self):
        self.fid.close()

class Memory():
    """
    deprecated
    """
    def __init__(self, memeLabels):
        self.freeRows  = list()
        self.columns   = dict()
        for i, label in enumerate(memeLabels):
            self.columns[label] = i
        self.ID2Row    = dict()
        self.memory    = np.zeros([0, len(memeLabels)])
        self.getID     = itertools.count().__next__

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

class Globals():
    """ This class manages global variables that are assigned on all processes
    and are synced via mpi. Global variables need to be registered together with
    the aggregation method they ase synced with, .e.g. sum, mean, min, max,...

    
    #TODO
    - enforce the setting (and reading) of global stats
    - implement mean, deviation, std as reduce operators
    """
    


    def __init__(self, world):
        self.world = world
        self.comm  = comm

        # simple reductions
        self.reduceDict = OrderedDict()
        self.operations         = dict()
        
        if comm is not None:        
        # MPI operations
            
            self.operations['sum']  = MPI.SUM
            self.operations['prod'] = MPI.PROD
            self.operations['min']  = MPI.MIN
            self.operations['max']  = MPI.MAX
        else:
            self.operations['sum']  = np.sum
            self.operations['prod'] = np.prod
            self.operations['min']  = np.min
            self.operations['max']  = np.max
        #staticical reductions/aggregations
        self.statsDict       = OrderedDict()
        self.localValues     = OrderedDict()
        self.globalValue     = OrderedDict()
        self.nValues         = OrderedDict()
        self.updated         = OrderedDict()

        # self implemented operations
        statOperations         = dict()
        statOperations['mean'] = np.mean
        statOperations['std']  = np.std
        statOperations['var']  = np.std
        #self.operations['std'] = MPI.Op.Create(np.std)

    #%% simple global reductions
    def registerValue(self, globName, value, reduceType):
        self.globalValue[globName] = value
        self.localValues[globName] = value
        try:
            self.nValues[globName] = len(value)
        except:
            self.nValues[globName] = 1
        if reduceType not in list(self.reduceDict.keys()):
            self.reduceDict[reduceType] = list()
        self.reduceDict[reduceType].append(globName)
        self.updated[globName] = True

    def syncReductions(self):

        for redType in list(self.reduceDict.keys()):

            op = self.operations[redType]
            #print op
            for globName in self.reduceDict[redType]:

                # enforce that data is updated
                assert  self.updated[globName] is True    ##OPTPRODUCTION
                
                # communication between all proceees
                if comm is None:
                    self.globalValue[globName] = self.localValues[globName]
                else:
                    self.globalValue[globName] = self.comm.allreduce(self.localValues[globName],op)
                self.updated[globName] = False
                lg.debug('local value of ' + globName + ' : ' + str(self.localValues[globName]))##OPTPRODUCTION
                lg.debug(str(redType) + ' of ' + globName + ' : ' + str(self.globalValue[globName]))##OPTPRODUCTION

    #%% statistical global reductions/aggregations
    def registerStat(self, globName, values, statType):

        assert statType in ['mean', 'std', 'var']    ##OPTPRODUCTION


        if not isinstance(values, ALLOWED_MULTI_VALUE_TYPES):
            values = [values]
        values = np.asarray(values)


        self.localValues[globName]  = values
        self.nValues[globName]      = len(values)
        if statType == 'mean':
            self.globalValue[globName]          = np.mean(values)
        elif statType == 'std':
            self.globalValue[globName]          = np.std(values)
        elif statType == 'var':
            self.globalValue[globName]          = np.var(values)

        if statType not in list(self.statsDict.keys()):
            self.statsDict[statType] = list()
        self.statsDict[statType].append(globName)
        self.updated[globName] = True
        

    def updateLocalValues(self, globName, values):
        self.localValues[globName]     = values
        self.nValues[globName]         = len(values)
        self.updated[globName]         = True

    def syncStats(self):
        for redType in list(self.statsDict.keys()):
            if redType == 'mean':

                for globName in self.statsDict[redType]:
                    lg.debug(globName)
                    # enforce that data is updated
                    assert  self.updated[globName] is True    ##OPTPRODUCTION
                    
                    # sending data list  of (local mean, size)
                    
                    
                    if comm is None:
                        self.globalValue[globName] = np.mean(self.localValues[globName])
                    else:    
                        inpComm = [(np.mean(self.localValues[globName]), self.nValues[globName])]* mpiSize 
                        outComm = self.comm.alltoall(inpComm)
                        # communication between all proceees
                        tmp = np.asarray(outComm)
    
                        lg.debug('####### Mean of ' + globName + ' #######')       ##OPTPRODUCTION
                        #lg.debug(outComm)
                        lg.debug('loc mean: ' + str(tmp[:,0]))                     ##OPTPRODUCTION
                        # calculation of global mean
                        globValue = np.sum(np.prod(tmp,axis=1)) # means * size
                        globSize  = np.sum(tmp[:,1])             # sum(size)
                        self.globalValue[globName] = globValue/ globSize    # glob mean
                    lg.debug('Global mean: ' + str( self.globalValue[globName] ))   ##OPTPRODUCTION
                    self.updated[globName] = False
                    
            elif redType == 'std':
                for globName in self.statsDict[redType]:
                    lg.debug(globName)
                    # enforce that data is updated
                    assert  self.updated[globName] is True    ##OPTPRODUCTION
                    
                    # local calulation
                    
                    locSTD = [np.std(self.localValues[globName])] * mpiSize
                    lg.debug(locSTD)
                    if comm is None:
                        self.globalValue[globName] = locSTD[0]
                    else:
                        
                        outComm = self.comm.alltoall(locSTD)
                        lg.debug(outComm)
                        locSTD = np.asarray(outComm)
                        lg.debug('####### STD of ' + globName + ' #######')              ##OPTPRODUCTION
                        lg.debug('loc std: ' + str(locSTD))                       ##OPTPRODUCTION
    
                        # sending data list  of (local mean, size)
                        tmp = [(np.mean(self.localValues[globName]), self.nValues[globName])]* mpiSize 
                        
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
    
                        self.globalValue[globName] = np.sqrt(globVariance)
                    lg.debug('Global STD: ' + str( self.globalValue[globName] ))   ##OPTPRODUCTION
                    self.updated[globName] = False
                    
            elif redType == 'var':
                for globName in self.statsDict[redType]:
                    lg.debug(globName)
                    # enforce that data is updated
                    assert  self.updated[globName] is True    ##OPTPRODUCTION
                    
                    # calculation of local mean
                    
                    if comm is None:
                        self.globalValue[globName] = np.var(self.localValues[globName])
                    else:
                        
                        locSTD = [np.std(self.localValues[globName])] * mpiSize
                        locSTD = np.asarray(self.comm.alltoall(locSTD))
                        
    
                        # out data list  of (local mean, size)
                        tmp = [(np.mean(self.localValues[globName]), self.nValues[globName])]* mpiSize 
                        outComm = self.comm.alltoall(tmp)
                        tmp = np.asarray(outComm)
    
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
    
                        self.globalValue[globName] = globVariance
                        
                        
                    lg.debug('Global variance: ' + str( self.globalValue[globName] ))  ##OPTPRODUCTION
                    self.updated[globName] = False

    def sync(self):

        self.syncStats()
        self.syncReductions()

class GlobalRecord():
    """
    Record to store global variables for each time step. The global record is 
    often connected with a global variable of the same name.
    TODO: Re-think the concept
    """
    def __init__(self, name, colLables, nSteps, title, style='plot'):
        self.nRec = len(colLables)
        self.columns = colLables
        self.rec = np.zeros([nSteps, self.nRec])*np.nan
        self.name = name
        self.style = style
        self.nSteps = nSteps
        self.title  = title

    def addCalibrationData(self, timeIds, values):
        self.calDataDict = dict()
        for idx, value in zip(timeIds, values):
            self.calDataDict[idx] = value
        
        self.calcDataArray = np.asarray(values)
        self.timeIds       = timeIds

    def updateLocalValues(self, timeStep):
        self.glob.updateLocalValues(self.name, self.rec[timeStep,:])

    def gatherGlobalDataToRec(self, timeStep):
        self.rec[timeStep,:] = self.glob.globalValue[self.name]
        return self.glob.globalValue[self.name]

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
                for x,y in self.calDataDict.items():
                    if x <= calData.shape[0]:
                        calData[x,:] = y

                plt.plot(calData,'d')

        elif self.style == 'stackedBar':
            nCars = np.zeros(self.nSteps)
            colorPal =  plt.get_cmap('jet')
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
        dset.attrs['columns'] = [string.encode('utf8') for string in self.columns]
        dset.attrs['title']   = self.title.encode('utf8')
        if hasattr(self,'calDataDict'):
            tmp = np.zeros([len(self.calDataDict), self.rec.shape[1]+1])*np.nan
            for i, key in enumerate(self.calDataDict.keys()):
                tmp[i,:] = [key] + self.calDataDict[key]

            dset = h5File.create_dataset('calData/' + self.name, tmp.shape, dtype='f8')
            dset[:] = tmp
        h5File.close()

    def evaluateAbsoluteError(self):
        if hasattr(self,'calDataDict'):

            err = 0
            for timeIdx ,calValues in self.calDataDict.items():

                for i, calValue in enumerate(calValues):
                   if not np.isnan(calValue):
                       err += np.abs(calValue - self.rec[timeIdx,i]) 
            return err

        else:
            return None    

    def evaluateRelativeError(self):
        if hasattr(self,'calDataDict'):

            err = 0
            for timeIdx ,calValues in self.calDataDict.items():

                for i, calValue in enumerate(calValues):
                   if not np.isnan(calValue):
                       err += np.abs(calValue - self.rec[timeIdx,i]) / calValue
            return err
        else:
            return None

    def evaluateNormalizedError(self):
        if hasattr(self,'calDataDict'):
            
            err = []
            dim = self.calcDataArray.shape
            for iSeq in range(dim[1]):
                
                sequenceMean = np.nanmean(self.calcDataArray[:,iSeq])
                error = np.nansum(np.abs(self.calcDataArray[:,iSeq] - self.rec[self.timeIds,iSeq]) / sequenceMean)
                err.append(error)

            return np.asarray(err)
        else:
            return None

class IO():
    """
    This class helps to write automatically the agents attributes or edge attributes
    of all requrested types.
    """
    
    def __init__(self, world, nSteps, outputPath = '.', inputPath = '.'): # of IO

        self.inputPath  = inputPath
        self.outputPath  = outputPath
        self._graph      = world._graph
        
        if world.agentOutput:
            self.agH5File = _h5File_driver(outputPath + '/nodeOutput.hdf5')
            self.dynamicAgentData = dict()
            self.staticAgentData  = dict() # only saved once at timestep == 0
        if world.linkOutput:
            self.liH5File = _h5File_driver(outputPath + '/linkOutput.hdf5')
            self.dynamicLinkData = dict()
            self.staticLinkData  = dict() # only saved once at timestep == 0
   
        
        self.comm        = comm

        self.timeStepMag = int(np.ceil(np.log10(nSteps)))

    #%% AGENT OUTPUT

    def initAgentFile(self, world, agTypeIDs):
        """
        Initializes the internal data structure for later I/O
        """
        from .misc import Record
        lg.info('Start init of the node file')

        for agTypeID in agTypeIDs:
            mpiBarrier()
            tt = time.time()
            lg.info(' NodeType: ' +str(agTypeID))
            group = self.agH5File.create_group(str(agTypeID))
            
            attrStrings = [string.encode('utf8') for string in world._graph.getPropOfNodeType(agTypeID, 'dyn')['names']]
            group.attrs.create('dynamicProps', attrStrings)
            
            attrStrings = [string.encode('utf8') for string in  world._graph.getPropOfNodeType(agTypeID, 'sta')['names']]
            group.attrs.create('staticProps', attrStrings)

            lg.info( 'group created in ' + str(time.time()-tt)  + ' seconds'  )
            tt = time.time()

            nAgents = world.countAgents(agTypeID)
            self.nAgentsAll = np.empty(1*mpiSize,dtype=np.int)
            
            if world.isParallel:
                self.nAgentsAll = self.comm.alltoall([nAgents]*mpiSize)
            else:
                self.nAgentsAll = [nAgents]

            lg.info( 'nAgents exchanged in  ' + str(time.time()-tt)  + ' seconds'  )
            tt = time.time()

            lg.info('Number of all agents' + str( self.nAgentsAll ))

            nAgentsGlob = sum(self.nAgentsAll)
            cumSumNAgents = np.zeros(mpiSize+1).astype(int)
            cumSumNAgents[1:] = np.cumsum(self.nAgentsAll)
            loc2GlobIdx = (cumSumNAgents[mpiRank], cumSumNAgents[mpiRank+1])

            lg.info( 'loc2GlobIdx exchanged in  ' + str(time.time()-tt)  + ' seconds'  )
            tt = time.time()


            dataIDS = world.getAgentDataIDs(agTypeID)
            # static data
            staticRec  = Record(nAgents, 
                                     dataIDS, 
                                     nAgentsGlob, 
                                     loc2GlobIdx, 
                                     agTypeID, 
                                     self.timeStepMag)
            
            attrInfo   = world._graph.getPropOfNodeType(agTypeID, 'sta')
            attributes = attrInfo['names']
            sizes      = attrInfo['sizes']
            
           
            lg.info('Static node record created in  ' + str(time.time()-tt)  + ' seconds')

            for attr, nProp in zip(attributes, sizes):

                #check if first property of first entity is string
                try:
                     
                    entProp = self._graph.getAttrOfNodesIdx(attribute=attr, nTypeID=agTypeID, dataIDs=staticRec.ag2FileIdx[0])
                except ValueError:

                    raise BaseException
                if not isinstance(entProp,str):
                    staticRec.addAttr(attr, nProp)

            tt = time.time()
            # allocate storage
            #staticRec.initStorage(attrDtype)
            #print attrInfo
            
            self.staticAgentData[agTypeID] = staticRec
            lg.info( 'storage allocated in  ' + str(time.time()-tt)  + ' seconds'  )

            dataIDS = world.getAgentDataIDs(agTypeID)
            # dynamic data
            dynamicRec = Record(nAgents, 
                                     dataIDS, 
                                     nAgentsGlob, 
                                     loc2GlobIdx, 
                                     agTypeID, 
                                     self.timeStepMag)

            attrInfo   = world._graph.getPropOfNodeType(agTypeID, 'dyn')
            attributes = attrInfo['names']
            sizes      = attrInfo['sizes']

            lg.info('Dynamic node record created in  ' + str(time.time()-tt)  + ' seconds')


            for attr, nProp in zip(attributes, sizes):
                #check if first property of first entity is string
                entProp = self._graph.getAttrOfNodesIdx(attr, 
                                                     nTypeID=agTypeID,
                                                     dataIDs=staticRec.ag2FileIdx[0])
                if not isinstance(entProp,str):
                    dynamicRec.addAttr(attr, nProp)

            tt = time.time()
            # allocate storage
            #dynamicRec.initStorage(attrDtype)
            self.dynamicAgentData[agTypeID] = dynamicRec
            
            #lg.info( 'storage allocated in  ' + str(time.time()-tt)  + ' seconds'  )
            
            self.writeAgentDataToFile(0, agTypeID, static=True)
            
        lg.info( 'static data written to file in  ' + str(time.time()-tt)  + ' seconds'  )

    def writeAgentDataToFile(self, timeStep, agTypeIDs, static=False):
        """
        Transfers data from the._graph to record for the I/O
        and writing data to hdf5 file
        """
        if isinstance(agTypeIDs,int):
            agTypeIDs = [agTypeIDs]
        
        for agTypeID in agTypeIDs:
            if static:
                #for type in self.staticAgentData.keys():
                self.staticAgentData[agTypeID].addData(timeStep, self._graph.nodes[agTypeID])
                self.staticAgentData[agTypeID].writeData(self.agH5File, folderName='static')
            else:
                #for type in self.dynamicAgentData.keys():
                self.dynamicAgentData[agTypeID].addData(timeStep, self._graph.nodes[agTypeID])
                self.dynamicAgentData[agTypeID].writeData(self.agH5File)


    def finalizeAgentFile(self):
        """
        Finalizing the node file - closes the file and saves the
        attribute files. This incluses closing the Hdf5 file.
        """


        for agTypeID in list(self.dynamicAgentData.keys()):
            group = self.agH5File.get('/' + str(agTypeID))
            record = self.dynamicAgentData[agTypeID]
            for attrKey in list(record.attrIdx.keys()):
                group.attrs.create(attrKey, record.attrIdx[attrKey])

        for agTypeID in list(self.staticAgentData.keys()):
            group = self.agH5File.get('/' + str(agTypeID))
            record = self.staticAgentData[agTypeID]
            for attrKey in list(record.attrIdx.keys()):
                group.attrs.create(attrKey, record.attrIdx[attrKey])

#        self.comm.Barrier()
#        self.agH5File.flush()
#        self.comm.Barrier()

        self.agH5File.close()
        lg.info( 'Agent file closed')

        for agTypeID in list(self.dynamicAgentData.keys()):
            record = self.dynamicAgentData[agTypeID]
            misc.saveObj(record.attrIdx, (self.outputPath + '/attributeList_type' + str(agTypeID)))


    #%% LINK OUTPUT

    def initLinkFile(self, world, liTypeIDs):
        """
        Initializes the internal data structure for later I/O
        """
        from .misc import Record
        lg.info('Start init of the node file')

        for liTypeID in liTypeIDs:
            mpiBarrier()
            tt = time.time()
            lg.info(' NodeType: ' +str(liTypeID))
            group = self.liH5File.create_group(str(liTypeID))
            
            attrStrings = [string.encode('utf8') for string in world._graph.getPropOfEdgeType(liTypeID, 'dyn')['names']]
            group.attrs.create('dynamicProps', attrStrings)
            
            attrStrings = [string.encode('utf8') for string in  world._graph.getPropOfEdgeType(liTypeID, 'sta')['names']]
            group.attrs.create('staticProps', attrStrings)

            lg.info( 'group created in ' + str(time.time()-tt)  + ' seconds'  )
            tt = time.time()

            nLinks = world.countLinks(liTypeID)
            self.nLinksAll = np.empty(1*mpiSize,dtype=np.int)
            
            if world.isParallel:
                self.nLinksAll = self.comm.alltoall([nLinks]*mpiSize)
            else:
                self.nLinksAll = [nLinks]

            lg.info( 'nAgents exchanged in  ' + str(time.time()-tt)  + ' seconds'  )
            tt = time.time()

            lg.info('Number of all agents' + str( self.nLinksAll ))

            nAgentsGlob = sum(self.nLinksAll)
            cumSumNLinks = np.zeros(mpiSize+1).astype(int)
            cumSumNLinks[1:] = np.cumsum(self.nLinksAll)
            loc2GlobIdx = (cumSumNLinks[mpiRank], cumSumNLinks[mpiRank+1])

            lg.info( 'loc2GlobIdx exchanged in  ' + str(time.time()-tt)  + ' seconds'  )
            tt = time.time()


            dataIDS = np.where(world._graph.edges[liTypeID]['active'])
            # static data
            staticRec  = Record(nLinks, 
                                     dataIDS, 
                                     nAgentsGlob, 
                                     loc2GlobIdx, 
                                     liTypeID, 
                                     self.timeStepMag)
            
            attrInfo   = world._graph.getPropOfEdgeType(liTypeID, 'sta')
            attributes = attrInfo['names']
            sizes      = attrInfo['sizes']
            
           
            lg.info('Static link record created in  ' + str(time.time()-tt)  + ' seconds')

            for attr, nProp in zip(attributes, sizes):

                #check if first property of first entity is string
                try:
                     
                    entProp = self._graph.getAttrOfLinksIdx(attribute=attr, eTypeID=liTypeID, dataIDs=staticRec.ag2FileIdx[0])
                except ValueError:

                    raise BaseException
                if not isinstance(entProp,str):
                    staticRec.addAttr(attr, nProp)

            tt = time.time()
            # allocate storage
            #staticRec.initStorage(attrDtype)
            #print attrInfo
            
            self.staticLinkData[liTypeID] = staticRec
            lg.info( 'storage allocated in  ' + str(time.time()-tt)  + ' seconds'  )

            dataIDS = np.where(world._graph.edges[liTypeID]['active'])[0]
            # dynamic data
            dynamicRec = Record( nLinks, 
                                 dataIDS, 
                                 nAgentsGlob, 
                                 loc2GlobIdx, 
                                 liTypeID, 
                                 self.timeStepMag)

            attrInfo   = world._graph.getPropOfEdgeType(liTypeID, 'dyn')
            attributes = attrInfo['names']
            sizes      = attrInfo['sizes']

            lg.info('Dynamic link record created in  ' + str(time.time()-tt)  + ' seconds')


            for attr, nProp in zip(attributes, sizes):
                #check if first property of first entity is string
                entProp = self._graph.getAttrOfLinksIdx(attr, 
                                                     eTypeID=liTypeID,
                                                     dataIDs=staticRec.ag2FileIdx[0])
                if not isinstance(entProp,str):
                    dynamicRec.addAttr(attr, nProp)

            tt = time.time()
            # allocate storage
            #dynamicRec.initStorage(attrDtype)
            self.dynamicLinkData[liTypeID] = dynamicRec
            
            #lg.info( 'storage allocated in  ' + str(time.time()-tt)  + ' seconds'  )
            
            self.writeLinkDataToFile(0, liTypeID, static=True)
            
        lg.info( 'static data written to file in  ' + str(time.time()-tt)  + ' seconds'  )

    def writeLinkDataToFile(self, timeStep, liTypeIDs, static=False):
        """
        Transfers data from the._graph to record for the I/O
        and writing data to hdf5 file
        """
        if isinstance(liTypeIDs,int):
            liTypeIDs = [liTypeIDs]
        
        for liTypeID in liTypeIDs:
            if static:
                #for type in self.staticAgentData.keys():
                self.staticLinkData[liTypeID].addData(timeStep, self._graph.edges[liTypeID])
                self.staticLinkData[liTypeID].writeData(self.liH5File, folderName='static')
            else:
                #for type in self.dynamicAgentData.keys():
                self.dynamicLinkData[liTypeID].addData(timeStep, self._graph.edges[liTypeID])
                self.dynamicLinkData[liTypeID].writeData(self.liH5File)

               



    def finalizeLinkFile(self):
        """
        Finalizing the link file - closes the file and saves the
        attribute files. This incluses closing the Hdf5 file.
        """

        for liTypeID in list(self.dynamicLinkData.keys()):
            group = self.liH5File.get('/' + str(liTypeID))
            record = self.dynamicLinkData[liTypeID]
            for attrKey in list(record.attrIdx.keys()):
                group.attrs.create(attrKey, record.attrIdx[attrKey])

        for liTypeID in list(self.staticLinkData.keys()):
            group = self.liH5File.get('/' + str(liTypeID))
            record = self.staticLinkData[liTypeID]
            for attrKey in list(record.attrIdx.keys()):
                group.attrs.create(attrKey, record.attrIdx[attrKey])

        self.liH5File.close()
        lg.info( 'Link file closed')

        for liTypeID in list(self.dynamicAgentData.keys()):
            record = self.dynamicAgentData[liTypeID]
            misc.saveObj(record.attrIdx, (self.outputPath + '/attributeList_type' + str(liTypeID)))

    

    def writeAdjFile(self ,fileName, agTypeID, attrForWeight):
        
        adjList, nLinks = self._graph.getAdjList(agTypeID)
        fid = open(fileName,'w')
        fid.write('% Adjecency file created by gcfABM \n')
        fid.write(str(self._graph.nCount(agTypeID)) + ' ' + str(nLinks//2) + ' 010 \n' )
        fid.write('% Adjecencies of verteces \n')
        
        
        weigList = self._graph.getAttrOfNodeType(attrForWeight, agTypeID).tolist()
    
        for adjline, weig in zip(adjList, weigList):
            fid.write(''.join([str(int(weig*100)) + ' '] + [str(x+1) + ' ' for x in adjline]) + '\n')
        fid.close()

class PAPI():
    """
    Parallel Agent Passing Interface
    MPI-based communication module that controles all communcation between
    different processes.
    ToDo: change to communication using numpy
    """

    def __init__(self, world,):

        self.world = world
        self.comm = comm

        self.rank = mpiRank
        self.size = mpiSize
        
        

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
        
        self.world._graph.ghostTypeUpdated = dict()
        
        self.world._graph.isParallel =  self.size > 1
            
    #%% Privat functions
    def _clearBuffer(self):
        """
        Method to clear all2all buffer
        """
        self.a2aBuff = []
        for x in range(mpiSize):
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


    def _packData(self, agTypeID, mpiPeer, propList, connList=None):
        """
        Privat method to pack all data for MPI transfer
        """
        dataSize = 0
        nNodes = len(self.mpiSendIDList[(agTypeID,mpiPeer)])
        
        dataPackage = dict()
        dataPackage['nNodes']  = nNodes
        dataPackage['nTypeID'] = agTypeID

        for prop in propList:
            dataPackage[prop] = self.world._graph.getNodeSeqAttr(attribute=prop, lnIDs=self.mpiSendIDList[(agTypeID,mpiPeer)] )
            dataSize += np.prod(dataPackage[prop].shape)
        
        if connList is not None:
            dataPackage['connectedNodes'] = connList
            dataSize += len(connList)
            
        lg.debug('package size: ' + str(dataSize))
        return dataPackage, dataSize



    def _updateGhostNodeData(self, agTypeIDList= 'dyn', propertyList= 'dyn'):
        """
        Privat method to update the data between processes for existing ghost nodes
        """
        tt = time.time()
        messageSize = 0
        
        for (agTypeID, mpiPeer) in list(self.mpiSendIDList.keys()):

            if agTypeIDList == 'all' or agTypeID in agTypeIDList:

                if propertyList in ['all', 'dyn', 'sta']:
                    propertyList = self.world._graph.getPropOfNodeType(agTypeID, kind=propertyList)['names']
                    
                lg.debug('MPIMPIMPIMPI -  Updating ' + str(propertyList) + ' for agTypeID ' + str(agTypeID) + 'MPIMPIMPI')
                dataPackage ,packageSize = self._packData(agTypeID, mpiPeer, propertyList, connList=None)
                                                    
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
                    agTypeID = dataPackage['nTypeID']

                    if propertyList == 'all':
                        propertyList= self.world._graph.nodeProperies[agTypeID][:]
                        propertyList.remove('gID')

                    for prop in propertyList:
                        self.world._graph.setNodeSeqAttr(attribute=prop, 
                                                    values=dataPackage[prop],
                                                    lnIDs=self.mpiRecvIDList[(agTypeID, mpiPeer)])                        
                    
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
        

        
        lg.debug('ID Array: ' + str(self.world._graph.IDArray))##OPTPRODUCTION
        for ghLoc in ghostLocationList:
            owner = ghLoc.mpiOwner
            x,y   = ghLoc.attr['coord']
            if owner not in mpiRequest:
                mpiRequest[owner]   = (list(), 'gID')
                self.mpiRecvIDList[(locNodeType, owner)] = list()

            mpiRequest[owner][0].append( (x,y) ) # send x,y-pairs for identification
            self.mpiRecvIDList[(locNodeType, owner)].append(ghLoc.nID)
        lg.debug('rank ' + str(self.rank) + ' mpiRecvIDList: ' + str(self.mpiRecvIDList))##OPTPRODUCTION

        for mpiDest in list(mpiRequest.keys()):

            if mpiDest not in self.peers:
                self.peers.append(mpiDest)

                # send request of global IDs
                lg.debug( str(self.rank) + ' asks from ' + str(mpiDest) + ' - ' + str(mpiRequest[mpiDest]))##OPTPRODUCTION
                self._add2Buffer(mpiDest, mpiRequest[mpiDest])

        lg.debug( 'requestOut:' + str(self.a2aBuff))##OPTPRODUCTION
        requestIn = self._all2allSync()
        lg.debug( 'requestIn:' +  str(requestIn))##OPTPRODUCTION


        for mpiDest in list(mpiRequest.keys()):


            # receive request of global IDs
            lg.debug('receive request of global IDs from:  ' + str(mpiDest))##OPTPRODUCTION
            incRequest = requestIn[mpiDest][0]
            
            lnIDList = [int(self.world._graph.IDArray[xx, yy]) for xx, yy in incRequest[0]]
            lg.debug( str(self.rank) + ' -sendIDlist:' + str(lnIDList))##OPTPRODUCTION
            self.mpiSendIDList[(locNodeType,mpiDest)] = lnIDList
 
            lg.debug( str(self.rank) + ' - gIDs:' + str(self.world._graph.getNodeSeqAttr('gID', lnIDList)))##OPTPRODUCTION

            for entity in [self.world.getAgent(agentID=i) for i in lnIDList]:
                entity.mpiGhostRanks.append(mpiDest)

            # send requested global IDs
            lg.debug( str(self.rank) + ' sends to ' + str(mpiDest) + ' - ' + str(self.mpiSendIDList[(locNodeType,mpiDest)]))##OPTPRODUCTION

            x = self.world._graph.getNodeSeqAttr(attribute=incRequest[1], lnIDs= lnIDList )

            self._add2Buffer(mpiDest,self.world._graph.getNodeSeqAttr(attribute=incRequest[1], lnIDs=lnIDList))

        requestRecv = self._all2allSync()

        for mpiDest in list(mpiRequest.keys()):
            #self.comm.send(self.ghostNodeOut[locNodeType, mpiDest][incRequest[1]], dest=mpiDest)
            #receive requested global IDs
            globIDList = requestRecv[mpiDest][0]
            

            self.world._graph.setNodeSeqAttr(attribute='gID', values=globIDList, lnIDs=self.mpiRecvIDList[(locNodeType, mpiDest)])
            
            
            lg.debug( 'receiving globIDList:' + str(globIDList))##OPTPRODUCTION
            lg.debug( 'localDList:' + str(self.mpiRecvIDList[(locNodeType, mpiDest)]))##OPTPRODUCTION
            for nID, gID in zip(self.mpiRecvIDList[(locNodeType, mpiDest)], globIDList):

                self.world.setGlob2Loc(gID, nID)
                self.world.setLoc2Glob(nID, gID)

        lg.info( 'Mpi commmunication required: ' + str(time.time()-tt) + ' seconds')

    def transferGhostAgents(self, world):
        """
        Privat method to initially transfer the data between processes and to create
        ghost nodes from the received data
        """

        messageSize = 0
        #%%Packing of data
        for agTypeID, mpiPeer in sorted(self.ghostNodeQueue.keys()):
            
            #get size of send array
            IDsList= self.ghostNodeQueue[(agTypeID, mpiPeer)]['nIds']
            connList = self.ghostNodeQueue[(agTypeID, mpiPeer)]['conn']

            self.mpiSendIDList[(agTypeID,mpiPeer)] = IDsList

            # setting up ghost out communication
            
            propList = world._graph.getPropOfNodeType(agTypeID, kind='all')['names']
            #print propList
            dataPackage, packageSize = self._packData( agTypeID, mpiPeer,  propList, connList)
            self._add2Buffer(mpiPeer, dataPackage)
            messageSize = messageSize + packageSize
        recvBuffer = self._all2allSync()

        lg.info('approx. MPI message size: ' + str(messageSize * 24. / 1000. ) + ' KB')

        for mpiPeer in self.peers:

            for dataPackage in recvBuffer[mpiPeer]:

        #%% create ghost agents from dataDict

                nNodes   = dataPackage['nNodes']
                agTypeID = dataPackage['nTypeID']
                #
                IDsList = world.addNodes(agTypeID, nNodes)
                # setting up ghostIn communicator
                self.mpiRecvIDList[(agTypeID, mpiPeer)] = IDsList
                
                # setting new local IDsmp
                self.world._graph.setNodeSeqAttr(attribute='ID', 
                                                values=IDsList,
                                                lnIDs=IDsList)
                
                propList = world._graph.getPropOfNodeType(agTypeID, kind='all')['names']
                propList.append('gID')

                for prop in propList:   
                    self.world._graph.setNodeSeqAttr(attribute=prop, 
                                                values=dataPackage[prop],
                                                lnIDs=self.mpiRecvIDList[(agTypeID, mpiPeer)])                        

                gIDsParents = dataPackage['connectedNodes']

                # creating entities with parentEntities from connList (last part of data package: dataPackage[-1])
                for nID, gID in zip(self.mpiRecvIDList[(agTypeID, mpiPeer)], gIDsParents):

                    GhostAgentClass = world._graph.agTypeID2Class(agTypeID)[1]

                    agent = GhostAgentClass(world, mpiPeer, nID=nID)

                    parentEntity = world.getAgent(world.glob2Loc(gID))
                    liTypeID = world._graph.node2EdgeType[parentEntity.agTypeID, agTypeID]

                    agent.register(world, parentEntity, liTypeID)


        lg.info('################## Ratio of ghost agents ################################################')
        for agTypeIDIdx in list(world._graph.agTypeByID.keys()):
            agTypeID = world._graph.agTypeByID[agTypeIDIdx].typeStr
            nAgents = world.countAgents(agTypeIDIdx)
            if nAgents > 0:
                nGhosts = float(world.countAgents(agTypeIDIdx, ghosts=True))
                nGhostsRatio = nGhosts / nAgents
                lg.info('Ratio of ghost agents for type "' + agTypeID + '" is: ' + str(nGhostsRatio))
        lg.info('#########################################################################################')




    def updateGhostAgents(self, agTypeIDList= 'all', propertyList='all'):
        """
        Method to update ghost node data on all processes
        """
        
        if mpiSize == 1:
            return None
        tt = time.time()

        if agTypeIDList == 'all':
            agTypeIDList = self.world._graph.agTypeByID
        messageSize = self._updateGhostNodeData(agTypeIDList, propertyList)

        if self.world.timeStep == 0:
            lg.info('Ghost update (of approx size ' +
                 str(messageSize * 24. / 1000. ) + ' KB)' +
                 ' required: ' + str(time.time()-tt) + ' seconds')
        else:                                                           ##OPTPRODUCTION
            lg.debug('Ghost update (of approx size ' +                  ##OPTPRODUCTION
                     str(messageSize * 24. / 1000. ) + ' KB)' +         ##OPTPRODUCTION
                     ' required: ' + str(time.time()-tt) + ' seconds')  ##OPTPRODUCTION
        
        if agTypeIDList == 'all':
            agTypeIDList = self.world._graph.agTypeByID
        
        
        for agTypeID in agTypeIDList:
            self.world._graph.ghostTypeUpdated[agTypeID] = list()
            if propertyList in ['all', 'dyn', 'sta']:        
                propertyList = self.world._graph.getPropOfNodeType(agTypeID, kind=propertyList)['names']
            
            
            for prop in propertyList:
                self.world._graph.ghostTypeUpdated[agTypeID].append(prop)
            

    def queueSendGhostAgent(self, mpiPeer, agTypeID, entity, parentEntity):

        if (agTypeID, mpiPeer) not in list(self.ghostNodeQueue.keys()):
            self.ghostNodeQueue[agTypeID, mpiPeer] = dict()
            self.ghostNodeQueue[agTypeID, mpiPeer]['nIds'] = list()
            self.ghostNodeQueue[agTypeID, mpiPeer]['conn'] = list()

        self.ghostNodeQueue[agTypeID, mpiPeer]['nIds'].append(entity.nID)
        self.ghostNodeQueue[agTypeID, mpiPeer]['conn'].append(parentEntity.gID)



    def all2all(self, value):
        """
        This method is a quick communication implementation that allows +
        sharing data between all processes

        """
        if isinstance(value,int):
            buff = np.empty(1*mpiSize,dtype=np.int)
            buff = self.comm.alltoall([value]*mpiSize)
        elif isinstance(value,float):
            buff = np.empty(1*mpiSize,dtype=np.float)
            buff = self.comm.alltoall([value]*mpiSize)
        elif isinstance(value,str):
            buff = np.empty(1*mpiSize,dtype=np.str)
            buff = self.comm.alltoall([value]*mpiSize)
        else:
            buff = self.comm.alltoall([value]*mpiSize)

        return buff

class Random():
    
    
    def __init__(self, world, __agentIDsByType, __ghostIDsByType, __agentsByType):
        self.__world = world # make world availabel in class random
        self.__agentIDsByType = __agentIDsByType
        self.__ghostIDsByType = __ghostIDsByType
        self.__agentsByType   = __agentsByType
        

    def nChoiceOfType(self, nChoice, agTypeID):
        return random.sample(self.__agentsByType[agTypeID],nChoice)
    
    
    def locations(self, nChoice):
        return random.sample(self.locDict.items(), nChoice)
    
    def shuffleAgentsOfType(self, agTypeID, ghosts=False):
        # a generator that yields items instead of returning a list
        if isinstance(agTypeID,str):
            agTypeID = self.__world.types.index(agTypeID)

        nodes = self.__world.getAgentIDs(agTypeID=agTypeID, ghosts=ghosts)

        shuffled_list = sorted(nodes, key=lambda x: random.random())
        for agID in shuffled_list:
            yield self.__world.getAgent(agentID=agID)
            

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def toDict(self):
        return dict( (k, v) for k,v in self.items() )


        
if __name__ == '__main__':
    pass
