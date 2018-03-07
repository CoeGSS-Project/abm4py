#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:22:32 2017

processing records
@author: geiges, GCF
"""

import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import os
import sys
import tables as ta
from os.path import expanduser
home = expanduser("~")
import socket

sys.path.append('../../lib/')
sys.path.append('../../modules/')

dir_path = os.path.dirname(os.path.realpath(__file__))
if socket.gethostname() in ['gcf-VirtualBox', 'ThinkStation-D30']:
    sys.path.append('/media/sf_shared/python/modules/biokit')
    sys.path.append('/home/geiges/database/modules/folium/')
    memmap = False
else:
    sys.path.append(home + '/python/modules/folium/')
    memmap = True

#sys.path.append('/home/geiges/database/')
sys.path.append('modules/')
import seaborn as sns; sns.set()
from class_auxiliary import loadObj, getEnvironment

#sns.set_color_codes("dark")
#sns.color_palette("Paired")

percentile = 0.1

plotFunc = list()


#memmap = True

doTry = True

#%% ENUMERATIONS
_ce  = 0
_hh  = 1
_pe  = 2

##%% INIT
plotFunc.append('plotEmissionOverTime')
plotFunc.append('plotElectricDemandOverTime')
plotFunc.append('plot_globalID')
plotFunc.append('plot_globalRecords')
plotFunc.append('plot_emissions')
plotFunc.append('plot_electricConsumption')
plotFunc.append('plot_ChargingStations')
plotFunc.append('plot_GreenConvenienceOverTime')
plotFunc.append('plot_stockAllRegions')
plotFunc.append('scatterConVSPref')
plotFunc.append('plot_averageCarAge')
plotFunc.append('plot_meanESSR')
plotFunc.append('plot_peerBubbleSize')
plotFunc.append('plot_agePerMobType')
plotFunc.append('plot_womanSharePerMobType')
plotFunc.append('plot_expectUtil')
plotFunc.append('plot_selfUtil')
plotFunc.append('plot_carStockBarPlot')
plotFunc.append('plot_carSales')
plotFunc.append('plot_properiesPerMobType')
plotFunc.append('plot_salesProperties')
plotFunc.append('plot_prefPerLabel')
plotFunc.append('plot_utilPerLabel')
plotFunc.append('plot_greenPerIncome')
plotFunc.append('plot_averageIncomePerCell')
plotFunc.append('plot_incomePerLabel')
plotFunc.append('plot_meanPrefPerLabel')
plotFunc.append('plot_meanConsequencePerLabel')
plotFunc.append('plot_convOverTime')
plotFunc.append('plot_cellMovie')
plotFunc.append('plot_carsPerCell')
plotFunc.append('plot_greenCarsPerCell')
plotFunc.append('plot_conveniencePerCell')
plotFunc.append('plot_population')
#plotFunc.append('plot_doFolium')
plotFunc.append('plot_carSharePerHHType')

simNo = sys.argv[1]
#if len(sys.argv) > 2:
#    relID = '#' + sys.argv[2]
#    print 'realization: ' + relID
#else:

path = getEnvironment(None,getSimNo = False) +'sim' + str(simNo).zfill(4) + '/'

try:
    os.remove(path + 'mapData.hdf5')
except:
    pass    
#path = 'poznan_out/sim0795/'

simParas   = loadObj(path + 'simulation_parameters')

nBurnIn       = simParas['burnIn']
withoutBurnIn = False
plotYears     = True         # only applicable in plots without burn-in

print 'omniscient Agents: ' + str(simParas['omniscientAgents'])
print 'burn-in phase: ' + str(nBurnIn)
print 'of which omniscient burn-in: ' + str(simParas['omniscientBurnIn'])



#%% DATA LOADING FILES


def loadMisc(path):
    enums        = loadObj(path + 'enumerations')
    parameters   = loadObj(path + 'simulation_parameters')
    parameters['timeStepMag'] = int(np.ceil(np.log10(parameters['nSteps'])))
    parameters['nPriorities'] = len(enums['priorities'])

    return parameters, enums

def cellDataAsMap(landLayer, posArray, cellData):

    cellArray = landLayer*1.0
    #res[posArray[0],posArray[1]] = data.ce[step,:,propDict.ce['carsInCell'][iBrand]] / data.ce[step,:,propDict.ce['population']]

    cellArray[posArray[:,0],posArray[:,1]] = cellData

    return cellArray




def loadData(path, parameters, data, propDict, filters, nodeType):

    h5file = ta.open_file(path + 'nodeOutput.hdf5', mode = "r")

    def getData(parameters, nodeType,timeStep):

        dataPath = '/' + str(nodeType)+ '/' + str(timeStep).zfill(parameters['timeStepMag'])
        node = h5file.get_node(dataPath)
        array = node.read()
        
        return array
    
    def getStaticData(parameters, nodeType):

        dataPath = '/' + str(nodeType)+ '/static' 
        node = h5file.get_node(dataPath)
        array = node.read()
        
        return array
    
    
   
    def getAttributes(parameters, nodeType):
        
        dataPath = '/' + str(nodeType)
        
        staPropDict = dict()
        properties = h5file.get_node_attr(dataPath, 'staticProps')
        for prop in properties:
            try: 
                staPropDict[prop] = h5file.get_node_attr(dataPath, prop); 
            except: 
                print "error loading: " + prop

        dynPropDict = dict()
        for prop in h5file.get_node_attr(dataPath, 'dynamicProps'):
            dynPropDict[prop] = h5file.get_node_attr(dataPath, prop)
            
            
        return staPropDict,  dynPropDict

    if nodeType == 0:
        filters.ce      = Bunch()
        #propDict.ce     = loadObj(path + 'attributeList_type1')
        
        propDict.ceSta, propDict.ce    = getAttributes(parameters, 1)
        
        filename = path + 'test_ce.dat'
        
        ceStep          = getData(parameters,1,0)
        if memmap:
            writeMemMapData = not os.path.isfile(filename)
            if writeMemMapData:
                print  'creating new ce memmap file'
                data.ce     = np.memmap(filename, dtype='float32', mode='w+', shape=(parameters['nSteps'], ceStep.shape[0], ceStep.shape[1]))
  
                for step in range(parameters['nSteps']):
                    data.ce[step,:,:] = getData(parameters,1,step)
            else:
                print  'loading old ce memmap file'
                data.ce     = np.memmap(filename, dtype='float32', mode='r+', shape=(parameters['nSteps'], ceStep.shape[0], ceStep.shape[1]))
        else:
            data.ce     = np.zeros([parameters['nSteps'], ceStep.shape[0], ceStep.shape[1]])
            for step in range(parameters['nSteps']):
                data.ce[step,:,:] = getData(parameters,1,step)
        data.ceSta = getStaticData(parameters,1)
        

    if nodeType == 1:
        filters.hh      = Bunch()
        #propDict.hh     = loadObj(path + 'attributeList_type2')
        propDict.hhSta, propDict.hh    = getAttributes(parameters, 2)
        hhStep          = getData(parameters,2,0)
#        from tempfile import mkdtemp
#        import os.path as path
        
        
        
        if memmap:
            filename = path + 'test_hh.dat'
            writeMemMapData = not os.path.isfile(filename)
            if writeMemMapData:
                print  'creating new hh memmap file'
                data.hh     = np.memmap(filename, dtype='float32', mode='w+', shape=(parameters['nSteps'], hhStep.shape[0], hhStep.shape[1]))
                for step in range(parameters['nSteps']):
                    data.hh[step,:,:] = getData(parameters,2,step)
            else:
                print  'loading old hh memmap file'
                data.hh     = np.memmap(filename, dtype='float32', mode='r+', shape=(parameters['nSteps'], hhStep.shape[0], hhStep.shape[1]))
        # no memmap file
        else:
            data.hh     = np.zeros([parameters['nSteps'], hhStep.shape[0], hhStep.shape[1]])
            for step in range(parameters['nSteps']):
                data.hh[step,:,:] = getData(parameters,2,step)


        data.hhSta = getStaticData(parameters,2)
        
    if nodeType ==2:
        filters.pe      = Bunch()
        #propDict.pe     = loadObj(path + 'attributeList_type3')
        propDict.peSta, propDict.pe    = getAttributes(parameters, 3)
        peStep          = getData(parameters,3,0)


        if memmap:
            filename = path + 'test_pe.dat'
            writeMemMapData = not os.path.isfile(filename)
            if writeMemMapData:
                print  'creating new memmap file'
                data.pe     = np.memmap(filename, dtype='float32', mode='w+', shape=(parameters['nSteps'], peStep.shape[0], peStep.shape[1]))
                for step in range(parameters['nSteps']):
                    data.pe[step,:,:] = getData(parameters,3,step)
            else:
                print  'loading old memmap file'
                data.pe     = np.memmap(filename, dtype='float32', mode='r+', shape=(parameters['nSteps'], peStep.shape[0], peStep.shape[1]))
        # no memmap file
        else:
            data.pe     = np.zeros([parameters['nSteps'], peStep.shape[0], peStep.shape[1]])
            for step in range(parameters['nSteps']):
                data.pe[step,:,:] = getData(parameters,3,step)
        data.peSta = getStaticData(parameters,3)
#    nSteps, nHhs,  nceProp   = data.ce.shape
#    nSteps, nPers, nPersProp = data.pe.shape
#    nSteps, nHhs,  nHHProp   = data.hh.shape

    return data, propDict, filters



#%% FILTERS

        
def filter_PrefTypes(data, propDict, parameters, enums, filters):

    nSteps, nPers, nPersProp = data.pe.shape
    filters.pe['prefTypeIDs'] = dict()
    for i,prefIdx in enumerate(propDict.peSta['preferences']):
        nPerCut= int(float(nPers) * percentile)
        filters.pe['prefTypeIDs'][i] = np.argsort(data.peSta[:,propDict.peSta['preferences'][i]])[-nPerCut:]

    #del data.peStep, data.hhStep
    #data.pe[0,filters['prefTypeIds'][i],propDict.pe['preferences'][i]]

    return filters


def filter_householdIDsPerMobType(data, propDict, parameters, enums, filters):

    hhglob2datIdx = dict()
    for idx in range(data.hh.shape[1]):
        hhglob2datIdx[data.hhSta[idx,propDict.hhSta['gID'][0]]] = idx
    filters.hh.hhglob2datIdx = hhglob2datIdx

    filters.hh.byMobType = dict()
    for mobKey in enums['brands'].keys():
        filters.hh.byMobType[mobKey] = list()
  
    for ti in range(parameters['nSteps']):
        #print time,
        for mobKey in enums['brands'].keys():
            gIDsofHH = data.peSta[data.pe[ti,:,propDict.pe['mobType'][0]]==mobKey,propDict.peSta['hhID'][0]].astype(int)
            hhIDs = [hhglob2datIdx[gID] for gID in gIDsofHH]
            filters.hh.byMobType[mobKey].append(np.asarray(hhIDs))


    return filters


#%% Plot auxiliary
def labelYears(factor):
    years = (parameters['nSteps'] - nBurnIn) / 12 / factor
    plt.xticks(np.linspace(nBurnIn,parameters['nSteps'],years+1), [str(2005 + year*factor) for year in range(years+1)], rotation=30)


def cellData2Map(cellData, data):
    
    posArray = data.ceSta[:,propDict.ceSta['pos']].astype(int)
    landLayer = np.zeros(np.max(data.ceSta[:,propDict.ceSta['pos']]+1,axis=0).astype(int).tolist())
    
    outMap = np.zeros_like(landLayer) * np.nan
    outMap[posArray[:,0],posArray[:,1]] = cellData
    return outMap

#%% DATA TRANSFER

class CSVWriter():
    
    def __init__(self, fileName, columns):
        global relID
        
        self.fid = open(path + '/' + fileName + '.csv', 'w')
        self.fid.write('step, ' + ', '.join(columns) + '\n')
        
    def addData(self, step, data):
        self.fid.write(str(step) + ', ' + ', '.join([str(x) for x in data]) + '\n')
    
    def close(self):
        self.fid.close()
        
class H5Writer():
    
    def __init__(self, fileName, groupName):
        
        self.h5File  = ta.File(path + '/' + fileName + '.hdf5', 'a')
        try:
            self.h5File.create_group('/',groupName)
        except:
            print "failed to create group"
            
        self.node    = self.h5File.get_node('/' + groupName)
        
        
    def addData(self, step, data):
        #self.shape   = data.shape
#        if step == 134:
#            print 'data'
#            print data
        self.data    = self.h5File.create_array(self.node,'step' + str(step), data)
        #self.data[step] = data
        
        
    def close(self):
        #self.h5File.flush()
        self.h5File.close()



#%% PLOT FUNCTIONS
def plot_globalRecords(data, propDict, parameters, enums, filters):

    reDf = pd.read_csv('resources_ger/regionID_germany.csv',index_col=0)

    import tables as ta

    h5File  = ta.File(path + '/globals.hdf5', 'r')
    glob    = h5File.get_node('/glob/')
    calData = h5File.get_node('/calData/')
    for data in glob._f_iter_nodes():
        plt.figure()
        plt.plot(data.read())

        if 'stock' in data.name and parameters['scenario'] == 7:
            continue

        if 'stock' in data.name:
            plt.title(reDf.ix[int(data.name[6:])]['alpha'])
        else:
            plt.title(data.name)


        if data.name in calData._v_children.keys():
            group = h5File.get_node('/calData/' + data.name)
            cData = group.read()

            plt.gca().set_prop_cycle(None)

            for i in range(1,data.shape[1]):
                plt.plot(cData[:,0], cData[:,i],'o')
                #print cData[:,i]

        if withoutBurnIn:
            plt.xlim([nBurnIn,parameters['nSteps']])
        if parameters['plotYears']:
            years = (parameters['nSteps'] - nBurnIn) / 12
            plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
            if 'stock' in data.name:
                plt.yscale('log')
        plt.tight_layout()
        print "saving file: " + path + data.name
        plt.savefig(path + data.name)


def plotEmissionOverTime(data, propDict, parameters, enums, filters):
    #h5File  = ta.File(path + '/globals.hdf5', 'r')
    #reDf = pd.read_csv('resources_ger/regionID_germany.csv',index_col=0)
    
#    emData = data.ce[:,:,propDict.ce['emissions']]
#    for i,re in enumerate(parameters['regionIDList']):
#        reIdx = data.ceSta[:,propDict.ceSta['regionId'][0]] == re
#        cellData = np.sum(emData[:,reIdx],axis=1)
#        
#        writer = CSVWriter('emission_' + str(re), enums['brands'].values())
#    
#        for ti in range(parameters['nSteps']):
#            stepData = cellData[ti]
#            writer.addData(ti, stepData)
#        writer.close()     
#
#    emData = data.ce[:,:,propDict.ce['emissions']]
##    for i,re in enumerate(parameters['regionIDList']):
#        reIdx = data.ceSta[:,propDict.ceSta['regionId'][0]] == re
#        cellData = np.sum(emData[:,reIdx],axis=1)
        
    writer = CSVWriter('emissions_all', enums['brands'].values())

    for ti in range(parameters['nSteps']):
        stepData = np.zeros(len(enums['brands']))
        for brand in range(0,len(enums['brands'])):
            boolMask = data.pe[ti,:,propDict.pe['mobType'][0]]== brand
        
            emData = data.pe[np.ix_([ti],boolMask,propDict.pe['emissions'])]
            stepData[brand] = np.sum(emData,axis=1)
        writer.addData(ti, stepData)
    writer.close()        

def plotElectricDemandOverTime(data, propDict, parameters, enums, filters):
    #h5File  = ta.File(path + '/globals.hdf5', 'r')
    #reDf = pd.read_csv('resources_ger/regionID_germany.csv',index_col=0)
    
#    emData = data.ce[:,:,propDict.ce['emissions']]
#    for i,re in enumerate(parameters['regionIDList']):
#        reIdx = data.ceSta[:,propDict.ceSta['regionId'][0]] == re
#        cellData = np.sum(emData[:,reIdx],axis=1)
#        
#        writer = CSVWriter('emission_' + str(re), enums['brands'].values())
#    
#        for ti in range(parameters['nSteps']):
#            stepData = cellData[ti]
#            writer.addData(ti, stepData)
#        writer.close()     
#
#    emData = data.ce[:,:,propDict.ce['emissions']]
##    for i,re in enumerate(parameters['regionIDList']):
#        reIdx = data.ceSta[:,propDict.ceSta['regionId'][0]] == re
#        cellData = np.sum(emData[:,reIdx],axis=1)
        
    writer = CSVWriter('electricDemand_all', ['electric Demand'])

    for ti in range(parameters['nSteps']):
        #stepData = np.zeros(1)
        brand = 1

        boolMask = data.pe[ti,:,propDict.pe['mobType'][0]]== brand
        
        emData = data.pe[np.ix_([ti],boolMask,propDict.pe['emissions'])]
        stepData = np.sum(emData) * .6 / 1000  #in /GWh
        writer.addData(ti, [stepData])
    writer.close() 
    
def plot_stockAllRegions(data, propDict, parameters, enums, filters):


    h5File  = ta.File(path + '/globals.hdf5', 'r')
    reDf = pd.read_csv('resources_ger/regionID_germany.csv',index_col=0)
    plt.figure(figsize=(24,12))
    factor = 5
    log = True
    for i,re in enumerate(parameters['regionIDList']):
        plt.subplot(4,4,i+1)
        group = h5File.get_node('/glob/stock_' + str(re))
        data = group.read()
        if i ==0:
            hh = plt.plot(data)
        else:
            plt.plot(data)
        group = h5File.get_node('/calData/stock_' + str(re))
        cData = group.read()
        plt.gca().set_prop_cycle(None)
        for i in range(1,data.shape[1]):
            plt.plot(cData[:,0], cData[:,i],'o')
            #print cData[:,i]
        if log:
            plt.yscale('log')
        if withoutBurnIn:
            plt.xlim([nBurnIn,parameters['nSteps']])
        if parameters['plotYears']:
            years = (parameters['nSteps'] - nBurnIn) / 12 / factor
            plt.xticks(np.linspace(nBurnIn,parameters['nSteps'],years+1), [str(2005 + year*factor) for year in range(years+1)], rotation=30)
        plt.title(' stock ' + reDf.ix[re]['alpha'])

    plt.figlegend(hh,enums['brandTitles'].values(), loc = 'lower center', ncol=3, labelspacing=0. )
    plt.tight_layout()
    plt.subplots_adjust(bottom=.15)
    plt.savefig(path + 'stockAll')


def scatterConVSPref(data, propDict, parameters, enums, filters):
    timeStep = 120
    plt.figure()
    idx = 0
    prefTypeIds = filters.pe['prefTypeIDs']
    for prefType in range(4):
        plt.subplot(2,2,prefType+1)
        boolMask = np.full(data.pe.shape[1], False, dtype=bool)
        boolMask[prefTypeIds[prefType]] = True
        #boolMask = data.pe[timeStep,:,propDict.pe['prefTyp'][0]] == prefType
        x = data.pe[timeStep,boolMask,propDict.pe['consequences'][idx]]
        y = data.peSta[boolMask,propDict.peSta['preferences'][idx]]

        plt.scatter(x,y,s=10)
    plt.xlabel(['comfort','environmental','remainig money','similarity'][idx])
    plt.ylabel(['convenience','ecology','money','innovation'][idx])
    plt.savefig(path + 'scatterConVSPref')


def plot_averageCarAge(data, propDict, parameters, enums, filters):
    res = np.zeros([parameters['nSteps'],3])
    std = np.zeros([parameters['nSteps'],3])
    for time in range(parameters['nSteps']):
        for mobType in enums['mobilityTypes'].keys():
            res[time,mobType] = np.mean(data.pe[time,data.pe[time,:,propDict.pe['mobType'][0]]==mobType,propDict.pe['lastAction'][0]])/12
            std[time,mobType] = np.std(data.pe[time,data.pe[time,:,propDict.pe['mobType'][0]]==mobType,propDict.pe['lastAction'][0]]/12)


    fig = plt.figure()
    plt.fill_between(range(0,parameters['nSteps']), res[:,0]+ std[:,0], res[:,0]- std[:,0], facecolor='blue', interpolate=True, alpha=0.1,)
    plt.fill_between(range(0,parameters['nSteps']), res[:,1]+ std[:,1], res[:,1]- std[:,1], facecolor='green', interpolate=True, alpha=0.1,)
    plt.fill_between(range(0,parameters['nSteps']), res[:,2]+ std[:,2], res[:,2]- std[:,2], facecolor='red', interpolate=True, alpha=0.1,)
    plt.plot(res)
    #plt.title(enums['brands'][carLabel])
    if withoutBurnIn:
        plt.xlim([nBurnIn,parameters['nSteps']])
    if parameters['plotYears']:
        years = (parameters['nSteps'] - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
    plt.legend(['Combution engine', 'Electric engine', 'other mobility types'],loc=0)
    plt.title('Average fleet age per mobility type [years]')
    plt.tight_layout()
    plt.savefig(path + 'fleetAge')

def plot_meanESSR(data, propDict, parameters, enums, filters):
    res = np.zeros([parameters['nSteps'],3])
    for time in range(parameters['nSteps']):
        for mobType in enums['mobilityTypes'].keys():
            res[time,mobType] = np.mean(data.pe[time,data.pe[time,:,propDict.pe['mobType'][0]]==mobType,propDict.pe['ESSR'][0]])

    fig = plt.figure()
    plt.plot(res)
    #plt.title(enums['brands'][carLabel])
    if withoutBurnIn:
        plt.xlim([nBurnIn,parameters['nSteps']])
    if parameters['plotYears']:
        years = (parameters['nSteps'] - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
    plt.legend(['Combution engine', 'Electric engine', 'other mobility types'],loc=0)
    plt.title('Average relative effective sample size')
    plt.tight_layout()
    plt.savefig(path + 'ESSR|mobType')

    res = np.zeros([parameters['nSteps'],3])
    prefTypeIds = filters.pe['prefTypeIDs']
    for time in range(parameters['nSteps']):
        for prefType in range(4):
            res[time,mobType] = np.mean(data.pe[time,prefTypeIds[prefType],propDict.pe['ESSR'][0]])

    fig = plt.figure()
    plt.plot(res)
    #plt.title(enums['brands'][carLabel])
    if withoutBurnIn:
        plt.xlim([nBurnIn,parameters['nSteps']])
    if parameters['plotYears']:
        years = (parameters['nSteps'] - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
    plt.legend(['Combution engine', 'Electric engine', 'other mobility types'],loc=0)
    plt.title('Average relative effective sample size')
    plt.tight_layout()
    plt.savefig(path + 'ESSR|prefType')


def plot_peerBubbleSize(data, propDict, parameters, enums, filters):

    res = np.zeros([parameters['nSteps'],3])
    #std = np.zeros([parameters['nSteps'],3])
    for time in range(parameters['nSteps']):
        for mobType in enums['mobilityTypes'].keys():
            res[time,mobType] = np.mean(data.pe[time,data.pe[time,:,propDict.pe['mobType'][0]]==mobType,propDict.pe['peerBubbleHeterogeneity'][0]])
            #std[time,mobType] = np.std(data.pe[time,data.pe[time,:,propDict.pe['mobType'][0]]==mobType,propDict.pe['age'][0]])
    fig = plt.figure()

    plt.plot(res)
    #plt.title(enums['brands'][carLabel])
    if withoutBurnIn:
        plt.xlim([nBurnIn,parameters['nSteps']])
    if parameters['plotYears']:
        years = (parameters['nSteps'] - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
    plt.legend(['Combution engine', 'Electric engine', 'other mobility types'],loc=0)
    plt.title('Average Bubble size')
    plt.tight_layout()
    plt.savefig(path + 'socialBubbleSize|mobType')

    res = np.zeros([parameters['nSteps'],3])
    #std = np.zeros([parameters['nSteps'],3])
    for time in range(parameters['nSteps']):
        for mobType in enums['mobilityTypes'].keys():
            boolMask = np.full(data.pe.shape[1], False, dtype=bool)
            boolMask[filters.pe['prefTypeIDs'][mobType]] = True
            res[time,mobType] = np.mean(data.pe[time,boolMask,propDict.pe['peerBubbleHeterogeneity'][0]])
            #std[time,mobType] = np.std(data.pe[time,data.pe[time,:,propDict.pe['mobType'][0]]==mobType,propDict.pe['age'][0]])
    fig = plt.figure()

    plt.plot(res)
    #plt.title(enums['brands'][carLabel])
    if withoutBurnIn:
        plt.xlim([nBurnIn,parameters['nSteps']])
    if parameters['plotYears']:
        years = (parameters['nSteps'] - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
    plt.legend(['Combution engine', 'Electric engine', 'other mobility types'],loc=0)
    plt.title('Average Bubble size')
    plt.tight_layout()
    plt.savefig(path + 'socialBubbleSize|prefType')


def plot_agePerMobType(data, propDict, parameters, enums, filters):
    res = np.zeros([parameters['nSteps'],len(enums['mobilityTypes'])])
    #std = np.zeros([parameters['nSteps'],3])
    for time in range(parameters['nSteps']):
        for mobType in enums['mobilityTypes'].keys():
            res[time,mobType] = np.mean(data.pe[time,data.pe[time,:,propDict.pe['mobType'][0]]==mobType,propDict.pe['age'][0]])
            #std[time,mobType] = np.std(data.pe[time,data.pe[time,:,propDict.pe['mobType'][0]]==mobType,propDict.pe['age'][0]])
    fig = plt.figure()

    plt.plot(res)
    #plt.title(enums['brands'][carLabel])
    if withoutBurnIn:
        plt.xlim([nBurnIn,parameters['nSteps']])
    if parameters['plotYears']:
        years = (parameters['nSteps'] - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
    plt.legend(enums['mobilityTypes'].values(),loc=0)
    plt.title('Average age of mobility actors')
    plt.tight_layout()
    plt.savefig(path + 'agePerMobType')


def plot_womanSharePerMobType(data, propDict, parameters, enums, filters):
    res = np.zeros([parameters['nSteps'],len(enums['mobilityTypes'])])
    for ti in range(parameters['nSteps']):
        for mobType in enums['mobilityTypes'].keys():
            res[ti,mobType] = np.mean(data.peSta[data.pe[ti,:,propDict.pe['mobType'][0]]==mobType,propDict.peSta['gender'][0]])-1

    fig = plt.figure()
    plt.plot(res)
    #plt.title(enums['brands'][carLabel])
    if withoutBurnIn:
        plt.xlim([nBurnIn,parameters['nSteps']])
    if parameters['plotYears']:
        years = (parameters['nSteps'] - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
    plt.legend(enums['mobilityTypes'].values(),loc=0)
    plt.title('Share of women')
    plt.tight_layout()
    plt.savefig(path + 'womanShareGreen')



def plot_expectUtil(data, propDict, parameters, enums, filters):
    fig = plt.figure(figsize=(12,8))
    peData = np.asarray(data.pe[:,:,propDict.pe['commUtil']])

    plt.plot(np.mean(peData,axis=1),linewidth=3)
    legStr = list()
    for label in range(len(enums['brands'])):
        legStr.append(enums['brands'][label])
    style = ['-','-', ':','--','-.']
    ledAdd = [' (all)', ' (convenience)', ' (ecology)', ' (money)', ' (immi)']
    newLegStr = []
    newLegStr += [ string + ledAdd[0] for string in  legStr]
    for prefType in range(4):
        plt.gca().set_prop_cycle(None)
        boolMask = np.full(peData.shape[1], False, dtype=bool)
        boolMask[filters.pe['prefTypeIDs'][prefType]] = True
        for mobType in range(peData.shape[2]):
            plt.plot(np.mean(peData[:,boolMask,mobType],axis=1),style[prefType+1])
        newLegStr += [ string + ledAdd[prefType+1] for string in  legStr]
    if withoutBurnIn:
        plt.xlim([nBurnIn,parameters['nSteps']])
    if parameters['plotYears']:
        years = (parameters['nSteps'] - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)

    #plt.title("Expectations for mobility types ")
    plt.legend(newLegStr,loc=0)
    plt.tight_layout()
    plt.savefig(path + 'expectedUtility')

def plot_selfUtil(data, propDict, parameters, enums, filters):
    fig = plt.figure(figsize=(12,8))
    peData = np.asarray(data.pe[:,:,propDict.pe['selfUtil']])

    plt.plot(np.nanmean(peData,axis=1),linewidth=3)
    legStr = list()
    for label in range(len(enums['brands'])):
        legStr.append(enums['brands'][label])
    style = ['-','-', ':','--','-.']
    ledAdd = [' (all)', ' (convenience)', ' (ecology)', ' (money)', ' (immi)']
    newLegStr = []
    newLegStr += [ string + ledAdd[0] for string in  legStr]
    for prefType in range(4):
        plt.gca().set_prop_cycle(None)
        boolMask = np.full(peData.shape[1], False, dtype=bool)
        boolMask[filters.pe['prefTypeIDs'][prefType]] = True
        for mobType in range(peData.shape[2]):
            plt.plot(np.nanmean(peData[:,boolMask,mobType],axis=1),style[prefType+1])
        newLegStr += [ string + ledAdd[prefType+1] for string in  legStr]
    if withoutBurnIn:
        plt.xlim([nBurnIn,parameters['nSteps']])
    if parameters['plotYears']:
        years = (parameters['nSteps'] - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)

    #plt.title("Expectations for mobility types ")
    plt.legend(newLegStr,loc=0)
    plt.tight_layout()
    plt.savefig(path + 'selfUtility')

def plot_carSharePerHHType(data, propDict, parameters, enums, filters):
    #  plot car stock as bar plot dependent on hhType
    print 1
    plt.figure(figsize=[15,12])
    mobMin = parameters['nMobTypes']
    titles = list()
    titles.append('Single HH young')
    titles.append('Single HH medium')
    titles.append('Single HH old')
    
    titles.append('Couple young')
    titles.append('Couple medium')
    titles.append('Couple old')
    
    titles.append('3 Person HH')
    titles.append('HH with Infant')
    titles.append('HH with Teenager')
    
    titles.append('HH with Minor')
    titles.append('Single HH with Child')

#    titles.append('1P_18-30')
#    titles.append('1P_31-60')
#    titles.append('1P_>60')
#    titles.append('2P_j18-30')
#    titles.append('2P_j31-60')
#    titles.append('2P_>j60')
#    titles.append('3+P')
#    titles.append('child<6J')
#    titles.append('child<14J')
#    titles.append('child<18J')
#    titles.append('singelParent')
    for hhType in range(1,12):
        plt.subplot(3,4,hhType)
        carMat = np.zeros([parameters['nSteps'],parameters['nMobTypes']])
        filterIdx = data.peSta[:,propDict.peSta['hhType'][0]] == hhType
        
        
        writer = CSVWriter('carStock_hhType_' + str(hhType), enums['brands'].values())
    
        for ti in range(parameters['nSteps']):
            stepData = np.bincount(data.pe[ti,filterIdx,propDict.pe['mobType'][0]].astype(int),minlength=mobMin).astype(float)
            carMat[ti,:] = stepData
            writer.addData(ti, stepData)
        writer.close()
        
        nCars = np.zeros(parameters['nSteps'])
        colorPal =  sns.color_palette("Set3", n_colors=len(enums['brands'].values()), desat=.8)
        tmp = colorPal[0]
        colorPal[0] = colorPal[1]
        colorPal[1] = tmp
    
        for i, brand in enumerate(enums['brands'].values()):
            plt.bar(np.arange(parameters['nSteps']), carMat[:,i],bottom=nCars, color =colorPal[i], width=1)
            nCars += carMat[:,i]
            
        if withoutBurnIn:
            plt.xlim([nBurnIn,parameters['nSteps']])
        else:
            plt.xlim([0,parameters['nSteps']])
        plt.ylim([0, np.sum(carMat[ti,:])])
        plt.title(titles[hhType-1])
        if parameters['plotYears']:
            labelYears(5)
    #plt.legend(legStr,bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    plt.subplot(3,4,hhType+1)
    
    plt.legend(enums['mobilityTypes'].values(),loc=0)
    plt.tight_layout()
    
    plt.savefig(path + 'carSharePerHHType') 

def plot_carStockBarPlot(data, propDict, parameters, enums, filters):
    #  plot car stock as bar plot
    legStr = list()

    carMat = np.zeros([parameters['nSteps'],parameters['nMobTypes']])
    mobMin = parameters['nMobTypes']
    writer = CSVWriter('carStock_all', enums['brands'].values())
    
    
    for ti in range(parameters['nSteps']):
        
        stepData = np.bincount(data.pe[ti,:,propDict.pe['mobType'][0]].astype(int),minlength=mobMin).astype(float)

        carMat[ti,:]= stepData
        writer.addData(ti, stepData)
    writer.close()
    
    plt.figure()
    enums   = loadObj(path + 'enumerations')
    #df = pd.read_csv(path +  'rec/' + 'carStock.csv', index_col=0)
    #parameters['nSteps'] = agMat.shape[0]
    nCars = np.zeros(parameters['nSteps'])
    colorPal =  sns.color_palette("Set3", n_colors=len(enums['brands'].values()), desat=.8)
    
    tmp = colorPal[0]
    colorPal[0] = colorPal[1]
    colorPal[1] = tmp
    
    for i, brand in enumerate(enums['brands'].values()):
       plt.bar(np.arange(parameters['nSteps']), carMat[:,i],bottom=nCars, color =colorPal[i], width=1)
       nCars += carMat[:,i]
       legStr.append(brand)
#plt.legend(legStr)
    if withoutBurnIn:
        plt.xlim([nBurnIn,parameters['nSteps']])
    if parameters['plotYears']:
        years = (parameters['nSteps'] - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
    plt.subplots_adjust(top=0.96,bottom=0.14,left=0.1,right=0.80,hspace=0.45,wspace=0.1)
    #plt.legend(legStr,bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    plt.legend(enums['mobilityTypes'].values(),loc=0)
    plt.tight_layout()
    plt.xlim([0,parameters['nSteps']])
    plt.ylim([0, np.sum(carMat[ti,:])])
    plt.savefig(path + 'carStock')

def plot_carSales(data, propDict, parameters, enums, filters):
    carSales = np.zeros([parameters['nSteps'],len(enums['brands'])])
    for ti in range(parameters['nSteps']):
        for brand in range(0,len(enums['brands'])):
            #idx = data.pe[ti,:,propDict.pe['predMeth'][0]] == 1
            #carSales[ti,:] = np.bincount(data.pe[ti,idx,propDict.pe['type'][0]].astype(int),minlength=3).astype(float)
            boolMask = data.pe[ti,:,propDict.pe['lastAction'][0]]== 0
            carSales[ti,:] = np.bincount(data.pe[ti,boolMask,propDict.pe['mobType'][0]].astype(int),minlength=len(enums['brands'])).astype(float)


    fig = plt.figure()
    plt.plot(carSales)
    if withoutBurnIn:
        plt.xlim([nBurnIn,parameters['nSteps']])
    if parameters['plotYears']:
        years = (parameters['nSteps'] - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
    plt.legend(enums['mobilityTypes'].values(),loc=0)

    plt.title('sales per mobility Type')
    plt.tight_layout()
    plt.savefig(path + 'salesPerMobType')



def plot_properiesPerMobType(data, propDict, parameters, enums, filters):
    """
    consequences per mobility type
    """
    fig = plt.figure()
    res = np.zeros([parameters['nSteps'],len(enums['brands'])])
    for i in range(2):
        plt.subplot(2,1,i+1)
        for ti in range(parameters['nSteps']):
            for carLabel in range(len(enums['brands'])):
                idx = data.pe[ti,:,propDict.pe['mobType'][0]] == carLabel
                res[ti, carLabel] = np.mean(data.pe[ti,idx,propDict.pe['prop'][i]])
        legStr = list()
        for label in range(len(enums['brands'])):
            legStr.append(enums['brands'][label])

        plt.plot(res)
        if i == 0:
            plt.title('Average costs by mobility type')
        else:
            plt.title('Average emissions by mobility type')
        if withoutBurnIn:
            plt.xlim([nBurnIn,parameters['nSteps']])
        if parameters['plotYears']:
            years = (parameters['nSteps'] - nBurnIn) / 12
            plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
    plt.subplots_adjust(top=0.96,bottom=0.14,left=0.04,right=0.96,hspace=0.45,wspace=0.1)
    plt.legend(enums['mobilityTypes'].values(),loc=0)
    plt.tight_layout()
    plt.savefig(path + 'propertiesPerMobType')
#plt.show()


def plot_salesProperties(data, propDict, parameters, enums, filters):
    plt.figure(figsize=[15,10])

    propList = ['age', 'commUtil','lastAction', 'util']
    for i,prop in enumerate(propList):
        plt.subplot(2,2,i+1)
        res = np.zeros([parameters['nSteps'],len(enums['brands'])])

        for brand in range(0,len(enums['brands'])):
            for ti in range(parameters['nSteps']):
                boolMask = data.pe[ti,:,propDict.pe['lastAction'][0]]== 0
                boolMask2 = data.pe[ti,:,propDict.pe['mobType'][0]]== brand
                if prop in ['lastAction']:
                    res[ti,brand] = np.mean(data.pe[np.ix_([np.max([0,ti-1])],boolMask & boolMask2,propDict.pe[prop]) ],axis=1)
                elif prop in ['commUtil']:
                    res[ti,brand] = np.mean(data.pe[np.ix_([np.max([0,ti-1])],boolMask & boolMask2,[propDict.pe[prop][brand]]) ],axis=1)
                else:
                    res[ti,brand] = np.mean(data.pe[np.ix_([ti],boolMask & boolMask2,propDict.pe[prop]) ],axis=1)

        plt.plot(res)
        plt.legend(enums['mobilityTypes'].values(),loc=0)
        plt.title(prop)
        if withoutBurnIn:
            plt.xlim([nBurnIn,parameters['nSteps']])
        if parameters['plotYears']:
            years = (parameters['nSteps'] - nBurnIn) / 12
            plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
    plt.suptitle('preferences of current sales per time')
    plt.tight_layout()
    plt.savefig(path + 'buyerProperties')




def plot_prefPerLabel(data, propDict, parameters, enums, filters):
    """
    priority types per mobility types
    """
    prefTypePerPerson =  np.argmax(data.peSta[:,propDict.peSta['preferences']],axis=1)
    prefTypes = np.zeros(parameters['nPriorities'])
    for prefTyp in range(0,parameters['nPriorities']):
        prefTypes[prefTyp] = np.sum(prefTypePerPerson == prefTyp)

    res = dict()
    for carLabel in range(len(enums['brands'])):
        res[carLabel] = np.zeros([parameters['nSteps'],parameters['nPriorities']])

    for ti in range(parameters['nSteps']):
        for carLabel in range(len(enums['brands'])):
            idx = data.pe[ti,:, propDict.pe['mobType'][0]] == carLabel
            for prefType in range(parameters['nPriorities']):
                res[carLabel][ti,prefType] = np.sum(prefTypePerPerson[idx] == prefType) / prefTypes[prefType]

    legStr = list()
    fig = plt.figure()
    for prefType in range(parameters['nPriorities']):
        legStr.append(enums['priorities'][prefType])
    for carLabel in range(len(enums['brands'])):
        plt.subplot(2,int(np.ceil(parameters['nMobTypes']/2.)),carLabel+1)
        plt.plot(res[carLabel])
        plt.title(enums['brands'][carLabel])
        if withoutBurnIn:
            plt.xlim([nBurnIn,parameters['nSteps']])
        if parameters['plotYears']:
            years = (parameters['nSteps'] - nBurnIn) / 12
            plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
        plt.legend(legStr,loc=0)

        fig.suptitle('n priority types per mobility types')
    plt.savefig(path + 'prioTypePerMobType')


def plot_utilPerLabel(data, propDict, parameters, enums, filters):
    """
    utiilty per mobility type
    """
    legStr = list()
    for label in range(len(enums['brands'])):
        legStr.append(enums['brands'][label])
    fig = plt.figure(figsize=(12,8))
    res = np.zeros([parameters['nSteps'],len(enums['brands']), 5])
    for ti in range(parameters['nSteps']):

        for carLabel in range(0,len(enums['brands'])):
            boolMask = data.pe[ti,:,propDict.pe['mobType'][0]] == carLabel
            res[ti, carLabel, 0] = np.mean(data.pe[ti,boolMask,propDict.pe['util'][0]])
            for prefType in range(4):
                boolMask2 = np.full(data.pe.shape[1], False, dtype=bool)
                boolMask2[filters.pe['prefTypeIDs'][prefType]] = True
                res[ti, carLabel, prefType+1] = np.mean(data.pe[ti,boolMask & boolMask2,propDict.pe['util'][0]])

    newLegStr= list()


    style = ['-','-', ':','--','-.']
    plt.gca().set_prop_cycle(None)
    plt.plot(res[:,:,0],style[0], linewidth =3)
    newLegStr += [ string + ' (all)' for string in  legStr]
    plt.gca().set_prop_cycle(None)
    plt.plot(res[:,:,1],style[1])
    newLegStr += [ string + ' (convenience)' for string in  legStr]
    plt.gca().set_prop_cycle(None)
    plt.plot(res[:,:,2],style[2])
    newLegStr += [ string + ' (ecology)' for string in  legStr]
    plt.gca().set_prop_cycle(None)
    plt.plot(res[:,:,3],style[3])
    newLegStr += [ string + ' (money)' for string in  legStr]
    plt.gca().set_prop_cycle(None)
    plt.plot(res[:,:,4],style[4])
    newLegStr += [ string + ' (immi)' for string in  legStr]
    plt.legend(newLegStr,loc=0, ncol=5)
    #plt.title('Average utility by mobility type -=conv | ..=eco | --=mon ')
    if withoutBurnIn:
        plt.xlim([nBurnIn,parameters['nSteps']])
    if parameters['plotYears']:
        years = (parameters['nSteps'] - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
    plt.ylim([np.nanpercentile(res,1), np.nanpercentile(res,99)])
    plt.tight_layout()
    plt.savefig(path + 'utilPerMobType2')





def plot_greenPerIncome(data, propDict, parameters, enums, filters):
    res = np.zeros([parameters['nSteps'],2])
    std = np.zeros([parameters['nSteps'],2])
    plt.figure()
    for i,year in enumerate([2005, 2010, 2015, 2020, 2025, 2029]):

        plt.subplot(2,3,i+1)
        timeStep = (nBurnIn + (year-2005) * 12)-1

        #for carLabel in range(len(enums['brands'])):
        if timeStep < data.hh.shape[0]:
            #idx = data.hh[timeStep,:,propDict.hh['type'][0]] == carLabel
            plt.hist(data.hh[timeStep,:,propDict.hh['income'][0]],bins=np.linspace(0,11000,30), color='black')
            plt.title(str(year))
            if len(filters.hh['byMobType'][1][timeStep]) > 0:
                plt.hist(data.hh[timeStep,:,propDict.hh['income'][0]][filters.hh['byMobType'][1][timeStep]],bins=np.linspace(0,11000,30), color='green')

            else:
                pass
        if i < 2:
            plt.xticks([])
    plt.tight_layout()
    plt.savefig(path + 'greenPerIncomeClass')

def plot_globalID(data, propDict, parameters, enums, filters):
    cellData = data.ceSta[:,propDict.ceSta['gID'][0]]
    mapData = cellData2Map(cellData, data)
    h5writer = H5Writer('mapData', 'globalID')
    h5writer.addData(0,mapData)
    h5writer.close()
    
    plt.figure()
    plt.imshow(mapData)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path + 'globalIds')
    
def plot_averageIncomePerCell(data, propDict, parameters, enums, filters):

    step = 0
    income = data.hh[step,:,propDict.hh['income'][0]]
    positions = data.hhSta[:,propDict.hhSta['pos']].astype(int)
    incomeMap  = np.zeros(np.max(data.ceSta[:,propDict.ceSta['pos']]+1,axis=0).astype(int).tolist())
    posArray = data.ceSta[:,propDict.ceSta['pos']].astype(int)
    
    population = np.zeros_like(incomeMap)
    population[posArray[:,0],posArray[:,1]] = data.ce[step,:,propDict.ce['population']]
    
    #population.shape = incomeMap.shape
    #uniquePos, test = np.unique(positions,axis=0, return_inverse=1)
    for i, pos in enumerate(positions):
        incomeMap[pos[0], pos[1]] += income[i]
    #print 1
    incomeMap /= population
    
    incomeMap[population == 0]  = 0
    
    h5writer = H5Writer('mapData', 'averageIncome')
    h5writer.addData(0,incomeMap)
    h5writer.close()    
    
    plt.figure()
    plt.imshow(incomeMap)
    bounds = [np.nanpercentile(incomeMap,2), np.nanpercentile(incomeMap,98)]
    if bounds[0] == bounds[1]:
        print 'equal bounds: ' + str(bounds)
        bounds = [500., np.nanmax(incomeMap)]
    print bounds
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path + 'averageIncome')
    
def plot_incomePerLabel(data, propDict, parameters, enums, filters):
    
    
    

    res = np.zeros([parameters['nSteps'],4])
    std = np.zeros([parameters['nSteps'],4])
    for step in range(parameters['nSteps']):
        for carLabel in range(len(enums['brands'])):

            res[step, 0] = np.mean(data.hh[step,:,propDict.hh['income'][0]])
            std[step, 0] = np.std(data.hh[step,:,propDict.hh['income'][0]])

            brownHH = filters.hh.byMobType[0]
            if len(brownHH[step]) > 0:
                res[step, 1] = np.mean(data.hh[step,:,propDict.hh['income'][0]][brownHH[step]])
                std[step, 1] = .5* np.std(data.hh[step,:,propDict.hh['income'][0]][brownHH[step]])
            else:
                res[step, 1] = np.nan
                std[step, 1] = np.nan

            greenHH = filters.hh.byMobType[1]
            if len(greenHH[step]) > 0:
                res[step, 2] = np.mean(data.hh[step,:,propDict.hh['income'][0]][greenHH[step]])
                std[step, 2] = .5* np.std(data.hh[step,:,propDict.hh['income'][0]][greenHH[step]])
            else:
                res[step, 2] = np.nan
                std[step, 2] = np.nan
            otherHH = filters.hh.byMobType[2]
            if len(otherHH[step]) > 0:
                res[step, 3] = np.mean(data.hh[step,:,propDict.hh['income'][0]][otherHH[step]])
                std[step, 3] = .5* np.std(data.hh[step,:,propDict.hh['income'][0]][otherHH[step]])
            else:
                res[step, 3] = np.nan
                std[step, 3] = np.nan

    legStr = list()
    for label in range(0,len(enums['brands'])):
        legStr.append(enums['brands'][label])
    fig = plt.figure()
    plt.plot(res[:,1:])
    plt.fill_between(range(0,parameters['nSteps']), res[:,1]+ std[:,1], res[:,1]- std[:,1], facecolor='blue', interpolate=True, alpha=0.1,)
    plt.fill_between(range(0,parameters['nSteps']), res[:,2]+ std[:,2], res[:,2]- std[:,2], facecolor='green', interpolate=True, alpha=0.1,)
    plt.fill_between(range(0,parameters['nSteps']), res[:,3]+ std[:,3], res[:,3]- std[:,3], facecolor='red', interpolate=True, alpha=0.1,)

    plt.plot(res[:,1]+ std[:,1],'b--', linewidth = 1)
    plt.plot(res[:,2]+ std[:,2],'g--', linewidth = 1)
    plt.plot(res[:,3]+ std[:,3],'r--', linewidth = 1)
    plt.plot(res[:,1]- std[:,1],'b--', linewidth = 1)

    plt.plot(res[:,2]- std[:,2],'g--', linewidth = 1)

    plt.plot(res[:,3]- std[:,3],'r--', linewidth = 1)
    #ax.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True)

    #plt.plot(res- std,'--')
    #plt.title('Equalized household income')
    if withoutBurnIn:
        plt.xlim([nBurnIn,parameters['nSteps']])
    if parameters['plotYears']:
        years = (parameters['nSteps'] - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
    plt.legend(['Household income using an combution engined car',
                'Household income using an electric car',
                'Household income using other mobility modes',
                'Comb STD', 'Elec STD','other STD'],loc=3)
    plt.tight_layout()
    plt.savefig(path + 'incomePerMobType')



def plot_meanPrefPerLabel(data, propDict, parameters, enums, filters):
    """
    mean priority per car label
    """

    ensembleAverage = np.mean(data.peSta[:,propDict.peSta['preferences']], axis = 0)
    fig = plt.figure()
    res = dict()
    for carLabel in range(len(enums['brands'])):
        res[carLabel] = np.zeros([parameters['nSteps'],parameters['nPriorities']])
    for ti in range(parameters['nSteps']):
        for carLabel in range(len(enums['brands'])):
            idx = np.where(data.pe[ti,:,propDict.pe['mobType'][0]] == carLabel)[0]
            res[carLabel][ti,:] = np.mean(data.peSta[np.ix_(idx,propDict.peSta['preferences'])],axis=0) / ensembleAverage
    legStr = list()
    for prefType in range(parameters['nPriorities']):
        legStr.append(enums['priorities'][prefType])

    h = list()
    for carLabel in range(len(enums['brands'])):
        plt.subplot(2,int(np.ceil(parameters['nMobTypes']/2.)),carLabel+1)
        h.append(plt.plot(res[carLabel]))
        plt.title(enums['brands'][carLabel])
        #plt.legend(legStr,loc=0)
        if withoutBurnIn:
            plt.xlim([nBurnIn,parameters['nSteps']])
        if parameters['plotYears']:
            years = (parameters['nSteps'] - nBurnIn) / 12
            plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
    plt.legend(legStr,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #fig.legend(h, legStr, loc = (1,1,0,0))
    plt.tight_layout()

    fig.suptitle('mean priority per mobility type')
    plt.tight_layout()
    plt.savefig(path + 'meanPriorityPerMobType')
print 1


#enums['consequences'] = {0: 'convenience', 1: 'eco-friendliness', 2: 'remaining money', 3: 'similarity'}

def plot_meanConsequencePerLabel(data, propDict, parameters, enums, filters):
    """
    mean consequences per mobility type
    """
    fig = plt.figure()
    res = dict()
    for carLabel in range(len(enums['brands'])):
        res[carLabel] = np.zeros([parameters['nSteps'],parameters['nPriorities']])
    for ti in range(parameters['nSteps']):
        ensembleAverage = np.mean(data.pe[ti,:,propDict.pe['consequences']], axis = 1)
        for carLabel in range(0,len(enums['brands'])):
            idx = np.where(data.pe[ti,:,propDict.pe['mobType'][0]] == carLabel)[0]
            res[carLabel][ti,:] = np.mean(data.pe[np.ix_([ti],idx,propDict.pe['consequences'])],axis=1) #/ ensembleAverage
    legStr = list()
    for prefType in range(parameters['nPriorities']):
        legStr.append(enums['consequences'][prefType])

    h = list()
    for carLabel in range(0,len(enums['brands'])):
        plt.subplot(2,int(np.ceil(parameters['nMobTypes']/2.)),carLabel+1)
        h.append(plt.plot(res[carLabel]))
        plt.title(enums['brands'][carLabel])
        if withoutBurnIn:
            plt.xlim([nBurnIn,parameters['nSteps']])
        if parameters['plotYears']:
            years = (parameters['nSteps'] - nBurnIn) / 12
            plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
    # plt.legend(legStr,loc=0)
    plt.legend(legStr,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #fig.legend(h, legStr)

    fig.suptitle('mean consequences per mobility type')
    plt.tight_layout()
    plt.savefig(path + 'meanConsequencesPerMobType')


def plot_convOverTime(data, propDict, parameters, enums, filters):

    parameters['nSteps'], nCells, nProp = data.pe.shape
    propDict.ce = loadObj(path + 'attributeList_type1')


    meanCon   = np.zeros([parameters['nSteps'],parameters['nMobTypes']])
    meanEco   = np.zeros([parameters['nSteps'],parameters['nMobTypes']])
    meanPrc   = np.zeros([parameters['nSteps'],parameters['nMobTypes']])
    for ti in range(parameters['nSteps']):
        meanVect = np.mean(data.ce[ti,:,propDict.ce['convenience']],axis=1)
        meanCon[ti,:] = meanVect
#        meanEco[step,:] = meanVect[3:6]
#        meanPrc[step,:] = meanVect[6:]

    fig = plt.figure(figsize=(12,8))
    #plt.subplot(2,2,1)
    plt.plot(meanCon)
    plt.legend(enums['brandTitles'].values())
    plt.title('convenience, mean over cells')
    if parameters['plotYears']:
        years = (parameters['nSteps'] - nBurnIn) / 12
        plt.xticks(np.linspace(nBurnIn,nBurnIn+years*12,years+1), [str(2005 + year) for year in range(years)], rotation=45)
    
    plt.savefig(path + 'convenienceOverTime')

def plot_cellMovie(data, propDict, parameters, enums, filters):
    from matplotlib.colors import ListedColormap
    my_cmap = ListedColormap(sns.color_palette('BuGn_d').as_hex())
    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage

    posArray = data.ceSta[:,propDict.ceSta['pos']].astype(int)

    landLayer = np.zeros(np.max(data.ceSta[:,propDict.ceSta['pos']]+1,axis=0).astype(int).tolist())
    for iCell in range(data.ce.shape[1]):
        x = data.ceSta[iCell,propDict.ceSta['pos'][0]].astype(int)
        y = data.ceSta[iCell,propDict.ceSta['pos'][1]].astype(int)
        landLayer[x,y] = 1
    #plt.pcolormesh(landLayer)
    landLayer = landLayer.astype(bool)

    bounds = dict()
    plotDict = dict()
    tt = 0
    fig = plt.figure()
    plt.clf()

    res = landLayer*1.
    res[res == 0] = np.nan
    for iBrand in range(len(enums['brands'])):
        ceData = data.ce[-1,:,propDict.ce['carsInCell'][iBrand]] / data.ce[-1,:,propDict.ce['population'][0]] * 1000
        ceData[np.isinf(ceData)] = 0
        bounds[iBrand] = [np.nanmin(ceData), np.nanpercentile(ceData,95)]
        print bounds[iBrand]
        res[posArray[0],posArray[1]] = ceData[tt,:,propDict.ce['carsInCell'][iBrand]] / ceData[tt,:,propDict.ce['population'][0]] * 1000
        plt.subplot(2,2,iBrand+1)
        plotDict[iBrand] = plt.imshow(np.flipud(res), cmap=my_cmap)
        plt.colorbar()
        plt.clim(bounds[iBrand])
    plt.tight_layout()

    def make_frame(t):
        #print t
        tt = int(t*15) + nBurnIn
        for iBrand in enums['mobilityTypes'].keys():

            res = landLayer*1.
            #print(type(tt))
            #print tt
            #print data.ce[t,:,propDict.ce['carsInCell'][iBrand]]
            res[posArray[0],posArray[1]] = data.ce[tt,:,propDict.ce['carsInCell'][iBrand]] / data.ce[tt,:,propDict.ce['population'][0]] * 1000

            plotDict[iBrand].set_data(res)
            #plt.clim([0,1])
            #plt.colorbar()
            #plt.clim(bounds[iBrand])
            plt.title(enums['brands'][iBrand] + ' cars per cells')
            #print iBrand
        plt.tight_layout()
        plt.suptitle('tiStep' + str(tt))
        return mplfig_to_npimage(fig)

    timeDur = (parameters['nSteps'] - nBurnIn)/15
    animation = VideoClip(make_frame, duration = timeDur)
    animation.write_gif(path + "svm.gif", fps=15)

    #dsfg

def plot_carsPerCell(data, propDict, parameters, enums, filters):
    #%%
    h5writer = H5Writer('mapData', 'greenCars')
    for step in range(parameters['nSteps']):
        
        cellData = data.ce[step,:,propDict.ce['carsInCell'][1]]
        mapData = cellData2Map(cellData, data)
        h5writer.addData(step,mapData)
    h5writer.close()    
    
    import copy
    fig = plt.figure(figsize=(12,8))
    plt.clf()
    posArray = data.ceSta[:,propDict.ceSta['pos']].astype(int)
    landLayer = np.zeros(np.max(data.ceSta[:,propDict.ceSta['pos']]+1,axis=0).astype(int).tolist())
    for iCell in range(data.ce.shape[1]):
        x = data.ceSta[iCell,propDict.ceSta['pos'][0]].astype(int)
        y = data.ceSta[iCell,propDict.ceSta['pos'][1]].astype(int)
        landLayer[x,y] = 1
    #plt.pcolormesh(landLayer)
    landLayer = landLayer.astype(bool)
    res = landLayer*1.0
    step = parameters['nSteps']-1
    test = landLayer*0
    for iBrand in range(parameters['nMobTypes']):
        res = landLayer*1.0
        res[posArray[:,0],posArray[:,1]] = data.ce[step,:,propDict.ce['carsInCell'][iBrand]] / data.ce[step,:,propDict.ce['population'][0]] *1000.
        #cellData = data.ce[tt,:,propDict.ce['carsInCell'][iBrand]] / data.ce[tt,:,propDict.ce['population'][0]] * 1000
        res[np.isinf(res)] = 0
        res[np.isnan(res)] = 0
        bounds = [0, np.nanpercentile(res,95)]
        if bounds[0] == bounds[1]:
            bounds = [0, np.nanmax(res)]
        print bounds
        test = test + res
        if iBrand == 1:
            arrayData = copy.copy(res)
        #res[landLayer==False] = np.nan
        plt.subplot(2,int(np.ceil(parameters['nMobTypes']/2.)),iBrand+1)
        plt.pcolormesh(np.flipud(res))
        plt.clim(bounds)
        plt.colorbar()

        plt.title(enums['brands'][iBrand] + ' cars per cells')
    plt.tight_layout()
    plt.savefig(path + 'carsPerCell')

def plot_doFolium(data, propDict, parameters, enums, filters):
    bounds = (0,1)
    sys.path.append('/media/sf_shared/python/modules')
    #sys.path.append('/media/sf_shared/python/database')
    import class_map
    import matplotlib
    import folium
    posArray = data.ceSta[:,propDict.ceSta['pos']].astype(int)

    landLayer = np.zeros(np.max(data.ceSta[:,propDict.ceSta['pos']]+1,axis=0).astype(int).tolist())
    landLayer = landLayer.astype(bool)
    res = landLayer*1.0
    foMap = class_map.Map('toner', location = [53.9167-62*0.04166666, 13.9167], zoom_start = 7)
    geoFile = 'resources_NBH/regions.shp'
    import mod_geojson as gjs
    geoData = gjs.extractShapes(geoFile, parameters['regionIDList'].tolist() , 'numID',None)
    foMap.map.choropleth(geo_str=geoData, fill_color=None,fill_opacity=0.00, line_color='green', line_weight=3)
    for year in [2005, 2010, 2015, 2020, 2025, 2030, 2035]:

        step = (nBurnIn + (year-2005) * 12)-1
        if step > parameters['nSteps']:
            break
        # green cars per 1000 people
        peData = data.ce[step,:,propDict.ce['carsInCell'][1]] / data.ce[step,:,propDict.ce['population']]*1000
        arrayData = cellDataAsMap(landLayer,posArray, peData)
        # green cars per 1000 people
        #arrayData = data.ce[step,:,propDict.ce['carsInCell'][1]] / (data.ce[step,:,propDict.ce['carsInCell'][0]] + data.ce[step,:,propDict.ce['carsInCell'][1]])
        arrayData[np.isnan(arrayData)] = 0
        bounds = np.min([bounds[0], np.nanpercentile(arrayData,2)]) , np.max([bounds[1], np.nanpercentile(arrayData,98)])
    for year in  [2005, 2010, 2015, 2020, 2025, 2030, 2035]:
        step = (nBurnIn + (year-2005) * 12)-1
        if step > parameters['nSteps']:
            break
        # green cars per 1000 people
        peData = data.ce[step,:,propDict.ce['carsInCell'][1]] / data.ce[step,:,propDict.ce['population']]*1000

        # green cars per 1000 people
        #data = data.ce[step,:,propDict.ce['carsInCell'][1]] / (data.ce[step,:,propDict.ce['carsInCell'][0]] + data.ce[step,:,propDict.ce['carsInCell'][1]])

        arrayData = cellDataAsMap(landLayer,posArray, peData)
        arrayData[np.isnan(arrayData)] = 0
        cm = matplotlib.cm.get_cmap('YlGn')
        normed_data = (arrayData - bounds[0]) / (bounds[1]- bounds[0])
        #minmax = np.nanpercentile(arrayData,2), np.nanpercentile(arrayData,98)
        colored_data = cm(normed_data)
        if parameters['scenario'] ==3:
            foMap.addImage(colored_data, mercator=False, latMin=53.9167-62.*0.04166666, latMax=53.9167,lonMin=6.625,lonMax=6.625+118.*0.04166666,min_=0,max_=0, name = str(year))
        elif parameters['scenario'] ==7:
            delta = -.12
            foMap.addImage(colored_data, mercator=False, latMin=55.0833+delta-186.*0.04166666, latMax=55.0833+delta,lonMin=5.875,lonMax=5.875+219.*0.04166666,min_=0,max_=0, name = str(year))

    from branca.utilities import color_brewer
    cols = color_brewer('YlGn',6)
    cmap = folium.LinearColormap(cols,vmin=float(bounds[0]),vmax=float(bounds[1]), caption='Electric cars per 1000 people')


    foMap.map.add_child(cmap)
    #foMap.map.add_child(xx)
    foMap.save(path +  'carsPer1000.html')



    bounds = (0,1)
    sys.path.append('/media/sf_shared/python/modules')
    #sys.path.append('/media/sf_shared/python/database')
    import class_map
    import matplotlib
    import folium
    foMap = class_map.Map('toner',location = [53.9167-62*0.04166666, 13.9167], zoom_start = 7)
    geoFile = 'resources_NBH/regions.shp'
    import mod_geojson as gjs
    geoData = gjs.extractShapes(geoFile, parameters['regionIDList'].tolist() , 'numID',None)
    foMap.map.choropleth(geo_str=geoData, fill_color=None,fill_opacity=0.00, line_color='green', line_weight=3)
    for year in [2005, 2010, 2015, 2020, 2025, 2030]:

        step = (nBurnIn + (year-2005) * 12)-1
        if step > parameters['nSteps']:
            break
        # green cars per 1000 people
        #peData = data.ce[step,:,propDict.ce['carsInCell'][1]] / data.ce[step,:,propDict.ce['population']]*1000
        #arrayData = cellDataAsMap(landLayer,posArray, data)
        # green cars per 1000 people
        arrayData = data.ce[step,:,propDict.ce['carsInCell'][1]] / (data.ce[step,:,propDict.ce['carsInCell'][0]] + data.ce[step,:,propDict.ce['carsInCell'][1]])
        arrayData = cellDataAsMap(landLayer,posArray, arrayData)
        arrayData[np.isnan(arrayData)] = 0
        bounds = (0, 1)
    for year in [2005, 2010, 2015, 2020, 2025, 2030]:
        step = (nBurnIn + (year-2005) * 12)-1
        if step > parameters['nSteps']:
            break
        # green cars per 1000 people
        #data = data.ce[step,:,propDict.ce['carsInCell'][1]] / data.ce[step,:,propDict.ce['population']]*1000
        # green cars per 1000 people
        arrayData = data.ce[step,:,propDict.ce['carsInCell'][1]] / (data.ce[step,:,propDict.ce['carsInCell'][0]] + data.ce[step,:,propDict.ce['carsInCell'][1]])
        arrayData = cellDataAsMap(landLayer,posArray, arrayData)
        arrayData[np.isnan(arrayData)] = 0
        cm = matplotlib.cm.get_cmap('YlGn')
        normed_data = (arrayData - bounds[0]) / (bounds[1]- bounds[0])
        #minmax = np.nanpercentile(arrayData,2), np.nanpercentile(arrayData,98)
        colored_data = cm(normed_data)
        if parameters['scenario'] ==3:
            foMap.addImage(colored_data, mercator=False, latMin=53.9167-62.*0.04166666, latMax=53.9167,lonMin=6.625,lonMax=6.625+118.*0.04166666,min_=0,max_=0, name = str(year))
        elif parameters['scenario'] ==7:
            delta = -.12
            foMap.addImage(colored_data, mercator=False, latMin=55.0833+delta-186.*0.04166666, latMax=55.0833+delta,lonMin=5.875,lonMax=5.875+219.*0.04166666,min_=0,max_=0, name = str(year))

    from branca.utilities import color_brewer
    cols = color_brewer('YlGn',6)
    cmap = folium.LinearColormap(cols,vmin=float(bounds[0]),vmax=float(bounds[1]), caption='Electric car share')


    foMap.map.add_child(cmap)
    #foMap.map.add_child(xx)
    foMap.save(path +  'greenCarShare.html')


def plot_greenCarsPerCell(data, propDict, parameters, enums, filters):

    from matplotlib.colors import ListedColormap
    my_cmap = ListedColormap(sns.color_palette('BuGn_d').as_hex())
    years = [2015, 2020, 2025, 2030] + range(2031,2036)
    years = np.arange(2011,2036,3).tolist()
    iBrand = 1
    landLayer = np.zeros(np.max(data.ceSta[:,propDict.ceSta['pos']]+1,axis=0).astype(int).tolist())
    
    for iCell in range(data.ce.shape[1]):
        x = data.ceSta[iCell,propDict.ceSta['pos'][0]].astype(int)
        y = data.ceSta[iCell,propDict.ceSta['pos'][1]].astype(int)
        landLayer[x,y] = 1
    #plt.pcolormesh(landLayer)
    landLayer = landLayer.astype(bool)
    res = landLayer*1.0
    res[res==0] = np.nan
    fig = plt.figure(figsize=(12,8))
    cellData = data.ce[parameters['nSteps']-1,:,propDict.ce['carsInCell'][iBrand]]# / data.ce[parameters['nSteps']-1,:,propDict.ce['population'][0]] * 1000
    bounds = [0, np.nanpercentile(cellData,98)]
    if bounds[0] == bounds[1]:
        bounds = [0, np.nanmax(cellData)]
        print bounds
    posArray = data.ceSta[:,propDict.ceSta['pos']].astype(int)
    
    h5writer = H5Writer('mapData', 'greenCarsPerCell')
    
    for step in range(parameters['nSteps']):
        
        cellData = data.ce[step,:,propDict.ce['carsInCell'][iBrand]]
        if step < 10:
            print 'cellData'
            print cellData   
        mapData = cellData2Map(cellData, data)
#        if step == 134:
#            print 'mapData'
#            print mapData        
        h5writer.addData(step,mapData)
    h5writer.close()

   
    for i, year in enumerate (years):
        tt = (year - 2005)*12 + nBurnIn -1

#        import pdb 
#        pdb.set_trace()
        plt.subplot(3,3,i+1)

        cellData = data.ce[tt,:,propDict.ce['carsInCell'][iBrand]] #/ data.ce[tt,:,propDict.ce['population'][0]] * 1000
        cellData[np.isinf(cellData)] = 0

        print bounds
        res[posArray[:,0],posArray[:,1]] = cellData#.ce[tt,:,propDict.ce['carsInCell'][iBrand]] / data.ce[tt,:,propDict.ce['population'][0]] * 1000

        bounds = [0, np.nanpercentile(res,98)]
        if bounds[0] == bounds[1]:
            bounds = [0, np.nanmax(res)]
            print bounds
        #plt.imshow(res, cmap=my_cmap)
        plt.imshow(res)
        plt.colorbar()
        plt.clim(bounds)
        plt.tight_layout()
#        if year == 2034:
#            plt.title(str(2035))
#        else:
#            plt.title(str(year))
    #plt.suptitle('Electric cars per 1000 people')
    plt.savefig(path + 'greenCarPerCell')

def plot_electricConsumption(data, propDict, parameters, enums, filters):

    
    h5writer = H5Writer('mapData', 'elDemand')
    for step in range(parameters['nSteps']):
        
        cellData = data.ce[step,:,propDict.ce['electricConsumption']] / 1000 #in GWh
        mapData = cellData2Map(cellData, data)
        h5writer.addData(step,mapData)
    h5writer.close()
        
    
    from matplotlib.colors import ListedColormap
    my_cmap = ListedColormap(sns.color_palette('BuGn_d').as_hex())
    years = [2015, 2020, 2025, 2030] + range(2031,2036)

    for i, year in enumerate (years):
        tt = (year - 2005)*12 + nBurnIn -1


        plt.subplot(3,3,i+1)
        cellData = data.ce[tt,:,propDict.ce['electricConsumption']] / 1000 #in GWh
        
        mapData = cellData2Map(cellData, data)
        plt.imshow(mapData)
        plt.colorbar()
        bounds = [0, np.nanpercentile(mapData,98)]
        plt.clim(bounds)
        
        if year == 2034:
            plt.title(str(2035))
        else:
            plt.title(str(year))
    plt.tight_layout()
    plt.savefig(path + 'electricConsumption')
    
def plot_emissions(data, propDict, parameters, enums, filters):
    from matplotlib.colors import ListedColormap
    my_cmap = ListedColormap(sns.color_palette('BuGn_d').as_hex())
    years = [2015, 2020, 2025, 2030] + range(2031,2036)
    fig = plt.figure(figsize=(15,15))
    #landLayer = np.zeros(np.max(data.ceSta[:,propDict.ceSta['pos']]+1,axis=0).astype(int).tolist())
    
    h5writer = H5Writer('mapData', 'emissions')
    for step in range(parameters['nSteps']):
        
        cellData = data.ce[step,:,propDict.ce['emissions']].sum(axis=0)
        mapData = cellData2Map(cellData, data)
        if step < 10:
            print mapData
        h5writer.addData(step,mapData)
    h5writer.close()
        
    for i, year in enumerate (years):
        tt = (year - 2005)*12 + nBurnIn -1


        plt.subplot(3,3,i+1)
        cellData = data.ce[tt,:,propDict.ce['emissions']].sum(axis=0)
        mapData = cellData2Map(cellData, data) / 1000 # in T Co2
        
        plt.imshow(mapData)
        plt.colorbar()
        bounds = [np.nanpercentile(mapData,2), np.nanpercentile(mapData,98)]
        plt.clim(bounds)
        
        if year == 2034:
            plt.title(str(2035))
        else:
            plt.title(str(year))
    plt.tight_layout()
    plt.savefig(path + 'emissions')
    h5writer.close()
def plot_ChargingStations(data, propDict, parameters, enums, filters):

    from matplotlib.colors import ListedColormap
    fig = plt.figure(figsize=(15,15))
    my_cmap = ListedColormap(sns.color_palette('BuGn_d').as_hex())
    years = [2015, 2020, 2025, 2030] + range(2031,2036)
    iBrand = 1
    landLayer = np.zeros(np.max(data.ceSta[:,propDict.ceSta['pos']]+1,axis=0).astype(int).tolist())

    h5writer = H5Writer('mapData', 'chargStations')
    for step in range(parameters['nSteps']):
        
        cellData = data.ce[step,:,propDict.ce['chargStat']]
        mapData = cellData2Map(cellData, data)
        h5writer.addData(step,mapData)
    h5writer.close()
    
    for iCell in range(data.ce.shape[1]):
        x = data.ceSta[iCell,propDict.ceSta['pos'][0]].astype(int)
        y = data.ceSta[iCell,propDict.ceSta['pos'][1]].astype(int)
        landLayer[x,y] = 1
    
    
    #plt.pcolormesh(landLayer)
    landLayer = landLayer.astype(bool)
    res = landLayer*1.0
    res[res==0] = np.nan
    fig = plt.figure(figsize=(12,8))
    cellData = data.ce[parameters['nSteps']-1,:,propDict.ce['chargStat']]
    bounds = [0, np.nanpercentile(cellData,98)]
    if bounds[0] == bounds[1]:
        bounds = [0, np.nanmax(cellData)]
    posArray = data.ceSta[:,propDict.ceSta['pos']].astype(int)
    for i, year in enumerate (years):
        tt = (year - 2005)*12 + nBurnIn -1


        plt.subplot(3,3,i+1)

        cellData = data.ce[tt,:,propDict.ce['chargStat']]
        cellData[np.isinf(cellData)] = np.nan

        bounds = [0, np.nanpercentile(cellData,98)]
        if bounds[0] == bounds[1]:
            bounds = [0, np.nanmax(cellData)]
        res[posArray[:,0],posArray[:,1]] = cellData#.ce[tt,:,propDict.ce['carsInCell'][iBrand]] / data.ce[tt,:,propDict.ce['population'][0]] * 1000
        
        #plt.imshow(res, cmap=my_cmap)
        plt.imshow(res)
        plt.colorbar()
        plt.clim(bounds)
        plt.tight_layout()
        if year == 2034:
            plt.title(str(2035))
        else:
            plt.title(str(year))
    #plt.suptitle('Electric cars per 1000 people')
    plt.savefig(path + 'chargingStations')
    
def plot_GreenConvenienceOverTime(data, propDict, parameters, enums, filters):

    from matplotlib.colors import ListedColormap
    my_cmap = ListedColormap(sns.color_palette('BuGn_d').as_hex())
    years = np.linspace(2005,2035,9).astype(int)
    iBrand = 1
    landLayer = np.zeros(np.max(data.ceSta[:,propDict.ceSta['pos']]+1,axis=0).astype(int).tolist())
    
    for iCell in range(data.ce.shape[1]):
        x = data.ceSta[iCell,propDict.ceSta['pos'][0]].astype(int)
        y = data.ceSta[iCell,propDict.ceSta['pos'][1]].astype(int)
        landLayer[x,y] = 1
    
    #plt.pcolormesh(landLayer)
    landLayer = landLayer.astype(bool)
    res = landLayer*1.0
    res[res==0] = np.nan
    fig = plt.figure(figsize=(12,8))
    cellData = data.ce[parameters['nSteps']-1,:,propDict.ce['convenience'][1]]
    
    bounds = [0, np.nanpercentile(cellData,98)]
    if bounds[0] == bounds[1]:
        bounds = [0, np.nanmax(cellData)]
    posArray = data.ceSta[:,propDict.ceSta['pos']].astype(int)
    for i, year in enumerate (years):
        tt = (year - 2005)*12 + nBurnIn -1


        plt.subplot(3,3,i+1)

        cellData = data.ce[tt,:,propDict.ce['convenience'][1]]
        cellData[np.isinf(cellData)] = 0
        if bounds[0] == bounds[1]:
            bounds = [0, np.nanmax(cellData)]
        else:
            bounds = [0, np.nanpercentile(cellData,98)]
        
        res[posArray[:,0],posArray[:,1]] = cellData#.ce[tt,:,propDict.ce['carsInCell'][iBrand]] / data.ce[tt,:,propDict.ce['population'][0]] * 1000

        #plt.imshow(res, cmap=my_cmap)
        plt.imshow(res)
        plt.colorbar()
        plt.clim(bounds)
        plt.tight_layout()
        if year == 2034:
            plt.title(str(2035))
        else:
            plt.title(str(year))
 
    #plt.suptitle('Electric cars per 1000 people')
    plt.savefig(path + 'greenConvOverTime')
    

def plot_conveniencePerCell(data, propDict, parameters, enums, filters):
    
    #plt.colormap('jet')
#    plt.imshow(simParas['landLayer'])
#    plt.colorbar()
    fig = plt.figure(figsize=(15,10))
#    plt.clf()
#    step = parameters['nSteps']-1
#    posArray = data.ceSta[:,propDict.ceSta['pos']].astype(int)
#    landLayer = np.zeros(np.max(data.ceSta[:,propDict.ceSta['pos']]+1,axis=0).astype(int).tolist())
#    for iCell in range(data.ce.shape[1]):
#        x = data.ceSta[iCell,propDict.ceSta['pos'][0]].astype(int)
#        y = data.ceSta[iCell,propDict.ceSta['pos'][1]].astype(int)
#        landLayer[x,y] = 1
#    landLayer = landLayer.astype(bool)
#    #res = landLayer*1.0
#    step = 1
    #test = landLayer*0
    step = parameters['nSteps']-1
    for iBrand in range(len(enums['brands'])):
        #res = landLayer*1.0
        cellData = data.ce[step,:,propDict.ce['convenience'][iBrand]]
        mapData = cellData2Map(cellData, data)
        #test = test + res
        #res[landLayer==False] = np.nan
        plt.subplot(2,3,iBrand+1)
        plt.imshow(mapData)
        bounds = [np.nanpercentile(mapData,2), np.nanpercentile(mapData,98)]
        plt.clim(bounds)
        
        plt.colorbar()
        plt.title('convenience of ' + enums['brandTitles'][iBrand])
        
    plt.tight_layout()
    plt.savefig(path + 'conveniencePerCell')

def plot_population(data, propDict, parameters, enums, filters):
    plt.figure()

    landLayer = np.zeros(np.max(data.ceSta[:,propDict.ceSta['pos']]+1,axis=0).astype(int).tolist())
    res = landLayer*1.0

    posArray = data.ceSta[:,propDict.ceSta['pos']].astype(int)

    res[posArray[:,0],posArray[:,1]] = data.ce[0,:,propDict.ce['population']][0]

    plt.imshow(res)
    plt.colorbar()
    #plt.clim([0,1])d

    plt.title('population')
    plt.savefig(path + 'population')
    plt.show()


#sys.path.append('/media/sf_shared/python/database')
#import class_map
#import matplotlib
#foMap = class_map.Map()
#cm = matplotlib.cm.get_cmap('YlGn')
#arrayData = res
#normed_data = (arrayData - np.nanpercentile(arrayData,5)) / (np.nanpercentile(arrayData,95) - np.nanpercentile(arrayData,5))
#self.minmax = np.nanpercentile(arrayData,5), np.nanpercentile(arrayData,95)
#colored_data = cm(normed_data)
#foMap.addImage(colored_data, mercator=False, latMin=53.9167-62*0.04166666, latMax=53.9167,lonMin=6.625,lonMax=6.625+118*0.04166666,min_=0,max_=0)
#foMap.view()

if __name__ == "__main__":

    import time
    from bunch import Bunch
    tt = time.time()
    parameters, enums = loadMisc(path)


    data        = Bunch()
    propDict    = Bunch()
    filters     = Bunch()
    if doTry:
        try:
            data, propDict, filters = loadData(path, parameters, data, propDict, filters, nodeType=0)
        except Exception as e:
            print "failed to load cell data"
            print e
            import traceback
            traceback.print_exc()
        
        try:
            data, propDict, filters = loadData(path, parameters, data, propDict, filters, nodeType=1)
        except Exception as e:
            print "failed to load household data"
            print e
            import traceback
            traceback.print_exc()
        
        try:
            data, propDict, filters = loadData(path, parameters, data, propDict, filters, nodeType=2)
        except Exception as e:
            print "failed to load people data"
            print e
            import traceback
            traceback.print_exc()
    
        try:
            filters = filter_PrefTypes(data, propDict, parameters, enums, filters)
        except Exception as e:
            print "failed to run filter_PrefTypes"
            print e
            import traceback
            traceback.print_exc()
        
        try:
            filters = filter_householdIDsPerMobType(data, propDict, parameters, enums, filters)
        except Exception as e:
            print "failed to run filter_householdIDsPerMobType"    
            print e
            import traceback
            traceback.print_exc()
            

    else:
        data, propDict, filters = loadData(path, parameters, data, propDict, filters, nodeType=0)
        data, propDict, filters = loadData(path, parameters, data, propDict, filters, nodeType=1)
        data, propDict, filters = loadData(path, parameters, data, propDict, filters, nodeType=2)
        filters = filter_PrefTypes(data, propDict, parameters, enums, filters)
        filters = filter_householdIDsPerMobType(data, propDict, parameters, enums, filters)
        
    enums['brandTitles'] = dict()
    enums['brandTitles'][0] = 'Combustion engined cars'
    enums['brandTitles'][1] = 'Electric powered cars'
    enums['brandTitles'][2] = 'Puplic transport'
    enums['brandTitles'][3] = 'Car sharing'
    enums['brandTitles'][4] = 'None motorized'
    parameters['plotYears'] = plotYears
    parameters['withoutBurnIn'] = withoutBurnIn
    print 'loading done in ' + str(time.time() - tt) + ' s'

    for funcCall in plotFunc:
        tt = time.time()
        if doTry:
            
            try:
                plt.close('all')
                plt.clf()
                print 'Executing: ' + funcCall + '...',
                locals()[funcCall](data, propDict, parameters, enums, filters)
                
                
            except Exception as e:
    
                print 'failed to plot: ' + funcCall
                print e
                import traceback
                traceback.print_exc()
        else:
            plt.close('all')
            plt.clf()
            print 'Executing: ' + funcCall + '...',
            locals()[funcCall](data, propDict, parameters, enums, filters)
        print ' done in ' + str(time.time() - tt) + ' s'
    print 'All done'
    
#%%
selfUtil = np.asarray(data.pe[:,:,propDict.pe['selfUtil']])
commUtil  = np.asarray(data.pe[:,:,propDict.pe['selfUtil']])
mobType  = np.asarray(data.pe[:,:,propDict.pe['mobType']]) 
consequences  = np.asarray(data.pe[:,:,propDict.pe['consequences']]) 
if False:
    #%%
    
    iPers= 20
    plt.clf()
    plt.subplot(2,2,1)
    plt.plot(mobType[:,iPers,0],'o')
    plt.yticks(range(5),enums['brands'].values())
    #plt.ylabel()
    plt.title('mobType')
    plt.subplot(2,2,2)
    plt.plot(selfUtil[:,iPers,:])
    plt.legend(enums['mobilityTypes'].values(),loc=0)
    plt.title('selfUtil')
    plt.subplot(2,2,3)
    plt.plot(commUtil[:,iPers,:])
    plt.title('commUtil')
    plt.subplot(2,2,4)
    plt.plot(consequences[:,iPers,:])
    plt.title('consequences')
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #plt.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
    plt.legend(enums['consequences'].values(),bbox_to_anchor=(1.00, 1.05))