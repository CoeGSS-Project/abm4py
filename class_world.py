#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://www.globalclimateforum.org

CAR INNOVATION MARKET MODEL
-- WORLD CLASS FILE --

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

from lib_gcfabm import World
from class_auxiliary import Record
import igraph as ig
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import time
import os

###### Enums ################
#connections
_tll = 1 # loc - loc
_tlh = 2 # loc - household
_thh = 3 # household, household
#nodes
_cell = 1
_hh   = 2

class Earth(World):
    
    def __init__(self, nSteps, simNo, spatial):
        World.__init__(self, spatial)
        self.simNo      = simNo
        self.agentRec   = dict()   
        self.globalRec  = dict()
        self.time       = 0
        self.nSteps     = nSteps
        self.reporter   = list()
        self.nAgents    = 0
        self.brandDict  = dict()
        self.brands     = list()
        
        if not os.path.isdir('output'):
            os.mkdir('output')
        
        if not simNo is None:
            self.para['outPath']    = 'output/sim' + str(simNo).zfill(4)
            if not os.path.isdir(self.para['outPath']):
                os.mkdir(self.para['outPath'])
            if not os.path.isdir(self.para['outPath'] + '/rec'):
                os.mkdir(self.para['outPath'] + '/rec')
                
    def registerRecord(self, name, title, colLables, style ='plot')    :
        self.globalRec[name] = Record(name, colLables, self.nSteps, title, style)
        
        
    # init car market    
    def initMarket(self, properties, propRelDev=0.01, time = 0):
        self.market = Market(properties, propRelDev=propRelDev, time=time)
    
    def initMemory(self, memeLabels, memoryTime):
        self.memoryTime = memoryTime
        for location in tqdm.tqdm(self.iterNode(_cell)):
            location.initCellMemory(memoryTime, memeLabels)
    
#    def initObsAtLoc(self,properties):
#        for loc in self.nodeList[1]:
#            #self.agDict[loc].obsMat = np.zeros([0,len(properties)+1])
#            columns = properties + ['utility','label']
#            self.agDict[loc].obsDf = pd.DataFrame(columns = columns)
    
    def addBrand(self, label, propertyTuple, initTimeStep):
        brandID = self.market.addBrand(label, propertyTuple, initTimeStep)
        
        for cell in self.iterNode(_cell):
            cell.traffic[brandID] = 0
        if 'brands' not in self.enums.keys():
            self.enums['brands'] = dict()
        self.enums['brands'][brandID] = label
        
    # init opinion class        
    def initOpinionGen(self,indiRatio = 0.33, ecoIncomeRange=(1000,4000),convIncomeFraction=7000):
        self.og     = OpinionGenerator(indiRatio, ecoIncomeRange, convIncomeFraction)
        # read raster and init surface
        
        # init location nodes
        
        # populate required properties
        
        # populate global variables and lists
             
    def plotTraffic(self,label):
        #import matplotlib.pyplot as plt
        import numpy.ma as ma
        traffic = self.graph.IdArray *0
        if label in self.market.brandLabels.itervalues():
            brandID = self.market.brandLabels.keys()[self.market.brandLabels.values().index(label)]
            
            for cell in self.iterNode(_cell):
                traffic[cell.x,cell.y] += cell.traffic[brandID]
        #Zm = ma.masked_invalid(traffic)
        plt.clf()
        plt.imshow(traffic, cmap='jet',interpolation=None)
        plt.colorbar()
        plt.tight_layout()
        #plt.pcolormesh(Zm.T)
        #plt.pcolormesh(traffic)
        plt.title('traffic of ' + label)
        plt.savefig('output/traffic' + label + str(self.time).zfill(3) + '.png')
        
    def generateHH(self):
        hhSize  = int(np.ceil(np.abs(np.random.randn(1)*2)))
        while True:
            ageList = np.random.randint(1,60,hhSize) 
            if np.sum(ageList>17) > 0: #ensure one adult
                break 
        ageList = ageList.tolist()
        sexList = np.random.randint(1,3,hhSize).tolist()  
        income  = int(np.random.randint(5000))
        nKids   = np.sum(ageList<19)
        #print ageList
        idxAdult = [ n for n,i in enumerate(ageList) if i>17 ]
        idx = np.random.choice(idxAdult)
        prSaf, prEco, prCon, prMon = self.og.getPref(ageList[idx],sexList[idx],nKids,income, self.radicality)
        
        return hhSize, ageList, sexList, income,  nKids, prSaf, prEco, prCon, prMon
    
    def view(self,filename = 'none', vertexProp='none'):
        import matplotlib.cm as cm
        
        
        # Nodes        
        if vertexProp=='none':
            colors = iter(cm.rainbow(np.linspace(0, 1, len(self.types)+1)))   
            colorDictNode = {}
            for i in range(len(self.types)+1):
                hsv =  next(colors)[0:3]
                colorDictNode[i] = hsv.tolist()
            nodeValues = (np.array(self.graph.vs['type']).astype(float)).astype(int).tolist()
        else:
            maxCars = max(self.graph.vs[vertexProp])
            colors = iter(cm.rainbow(np.linspace(0, 1, maxCars+1)))
            colorDictNode = {}
            for i in range(maxCars+1):
                hsv =  next(colors)[0:3]
                colorDictNode[i] = hsv.tolist()
            nodeValues = (np.array(self.graph.vs[vertexProp]).astype(float)).astype(int).tolist()    
        # nodeValues[np.isnan(nodeValues)] = 0
        # Edges            
        colors = iter(cm.rainbow(np.linspace(0, 1, len(self.types)+1)))              
        colorDictEdge = {}  
        for i in range(len(self.types)+1):
            hsv =  next(colors)[0:3]
            colorDictEdge[i] = hsv.tolist()
        
        self.graph.vs["label"] = self.graph.vs["name"]
        edgeValues = (np.array(self.graph.es['type']).astype(float)).astype(int).tolist()
        
        visual_style = {}
        visual_style["vertex_color"] = [colorDictNode[typ] for typ in nodeValues]
        visual_style["vertex_shape"] = list()        
        for vert in self.graph.vs['type']:
            if vert == 0:
                visual_style["vertex_shape"].append('hidden')                
            elif vert == 1:
                    
                visual_style["vertex_shape"].append('rectangle')                
            else:
                visual_style["vertex_shape"].append('circle')     
        visual_style["vertex_size"] = list()  
        for vert in self.graph.vs['type']:
            if vert >= 3:
                visual_style["vertex_size"].append(4)  
            else:
                visual_style["vertex_size"].append(15)  
        visual_style["edge_color"]   = [colorDictEdge[typ] for typ in edgeValues]
        visual_style["edge_arrow_size"]   = [.5]*len(visual_style["edge_color"])
        visual_style["bbox"] = (900, 900)
        if filename  == 'none':
            ig.plot(self.graph,**visual_style)    
        else:
            ig.plot(self.graph, filename, **visual_style)     
    
    def genFriendNetwork(self, nFriendsPerPerson):
        """ 
        Function for the generation of a simple network that regards the 
        distance between the agents and their similarity in their preferences
        """
        tt = time.time()
        edges = list()
        
        for agent, x in tqdm.tqdm(self.iterNodeAndID(_hh)):
            
            frList, connList, weigList = agent.generateFriends(self,nFriendsPerPerson)
            edges += connList
        eStart = self.graph.ecount()
        self.graph.add_edges(edges)
        self.graph.es[eStart:]['type'] = _thh
        self.graph.es[eStart:]['weig'] = weigList
        
        print 'Network created in -- ' + str( time.time() - tt) + ' s'
        tt = time.time()
        

    def step(self):
        """ 
        Method to proceed the next time step
        """
        
        # proceed step
        self.time += 1
        
        # proceed market in time
        self.market.step() # Statistics are computed here
        # The utility of a single agent depends on the whole market
        # if you want a fast car, you are not as happy if everyone
        # else has a fast car same for ecology, etc.
        
        #loop over cells
        for cell in self.iterNode(_cell):
            cell.step()
        # Update observations (remove old ones)
        # Compute the number of cars in each cell

        #self.avgDegree.append(np.mean(self.graph.vs[self.nodeList[2]].degree()))
        self.market.stockbyBrand.loc[len(self.market.stockbyBrand)] = self.market.stockbyBrand.loc[len(self.market.stockbyBrand)-1]

        # get number of new cars
        
        # find which agents will by a new car
        
        #loop over buyers
#        markerlist = ['s', 'o', 'd', 'x', '*', '^','p','d','8','_']
#        colors = sns.color_palette("Set1", n_colors=9, desat=.5)
        # Iterate over households with a progress bar
        for agent in tqdm.tqdm(self.iterNode(_hh)):
            #agent = self.agDict[agID]
            agent.step(self)

                
        # process some records
    
        self.globalRec['avgUtil'].div(self.time, [self.nAgents] + self.nPrefTypes)
        self.globalRec['carStock'].setIdx(self.time, self.market.carsPerLabel, xrange(len(self.market.carsPerLabel)))
        maxSpeedMean = self.market.statistics[0,4]
        maxSpeedSTD  = self.market.statistics[1,4]
        self.globalRec['maxSpeedStat'].set(self.time, [maxSpeedMean, maxSpeedMean+maxSpeedSTD, maxSpeedMean-maxSpeedSTD])



    def writeAgentFile(self):
        for typ in self.agentRec.keys():
            
            self.agentRec[typ].currRecord[:,0] = self.time
            for attr in self.agentRec[typ].attributes:
                if len(self.agentRec[typ].attrIdx[attr]) == 1:
                    self.agentRec[typ].currRecord[:,self.agentRec[typ].attrIdx[attr]] =  np.expand_dims(self.graph.vs[self.agentRec[typ].ag2FileIdx][attr],1)
                    #self.agentRec[typ].record[self.time][:,self.agentRec[typ].attrIdx[attr]] =  np.expand_dims(self.graph.vs[self.agentRec[typ].ag2FileIdx][attr],1)
                else:
                    self.agentRec[typ].currRecord[:,self.agentRec[typ].attrIdx[attr]] = self.graph.vs[self.agentRec[typ].ag2FileIdx][attr]
                    #self.agentRec[typ].record[self.time][:,self.agentRec[typ].attrIdx[attr]] =  self.graph.vs[self.agentRec[typ].ag2FileIdx][attr]
            if self.para['writeNPY']: 
                self.agentRec[typ].recordNPY[self.time] = self.agentRec[typ].currRecord
            if self.para['writeCSV']:
                for record in self.agentRec[typ].currRecord:
                    #print record
                    self.agentRec[typ].writer.writerow(record)
    
    def initAgentFile(self, typ=1):
        from csv import writer
        class Record():
            def __init(self):
                pass
            
        self.agentRec[typ] = Record()
        self.agentRec[typ].ag2FileIdx = self.nodeList[typ]
        nAgents = len(self.nodeList[typ])
        self.agentRec[typ].attributes = list()
        attributes = self.graph.vs.attribute_names()
        self.agentRec[typ].nAttr = 0
        self.agentRec[typ].attrIdx = dict()
        self.agentRec[typ].header = list()
        
        #adding global time
        self.agentRec[typ].attrIdx['time'] = [0]
        self.agentRec[typ].nAttr += 1
        self.agentRec[typ].header += ['time']
        
        for attr in attributes:
            if self.graph.vs[self.agentRec[typ].ag2FileIdx[0]][attr] is not None and not isinstance(self.graph.vs[self.agentRec[typ].ag2FileIdx[0]][attr],str):
                self.agentRec[typ].attributes.append(attr)
                if isinstance(self.graph.vs[self.agentRec[typ].ag2FileIdx[0]][attr],(list,tuple)) :
                    nProp = len(self.graph.vs[self.agentRec[typ].ag2FileIdx[0]][attr])
                    self.agentRec[typ].attrIdx[attr] = range(self.agentRec[typ].nAttr, self.agentRec[typ].nAttr+nProp)
                else:
                    nProp = 1
                    self.agentRec[typ].attrIdx[attr] = [self.agentRec[typ].nAttr]
                self.agentRec[typ].nAttr += nProp
                self.agentRec[typ].header += [attr]*nProp
        
        if self.para['writeNPY']:
            self.agentRec[typ].recordNPY = np.zeros([self.nSteps, nAgents,self.agentRec[typ].nAttr ])
        if self.para['writeCSV']:
            self.agentRec[typ].recordCSV = np.zeros([nAgents,self.agentRec[typ].nAttr ])
            self.agentRec[typ].csvFile   = open(self.para['outPath'] + '/agentFile_type' + str(typ) + '.csv','w')
            self.agentRec[typ].writer = writer(self.agentRec[typ].csvFile, delimiter=',')
            self.agentRec[typ].writer.writerow(self.agentRec[typ].header)
            
        #print self.agentMat.shape            
        self.agentRec[typ].currRecord = np.zeros([nAgents, self.agentRec[typ].nAttr])
    
    def finalize(self):
        
        import pickle
        def saveObj(obj, name ):
            with open( name + '.pkl', 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
        # finishing reporter files
        for writer in self.reporter:        
            writer.close() 
        
        # writing global records to file
        for key in self.globalRec:    
            self.globalRec[key].saveCSV(self.para['outPath'] + '/rec')

        # saving agent files
        for typ in self.agentRec.keys():
            if self.para['writeNPY']:
                np.save(self.para['outPath'] + '/agentFile_type' + str(typ), self.agentRec[typ].recordNPY, allow_pickle=True)
                saveObj(self.agentRec[typ].attrIdx, (self.para['outPath'] + '/attributeList_type' + str(typ)))
            if self.para['writeCSV']:
                self.agentRec[typ].csvFile.close()
            
        # saving enumerations            
        saveObj(self.enums, self.para['outPath'] + '/enumerations')
        
        try:
            # plotting and saving figures
            for key in self.globalRec:
                self.globalRec[key].plot(self.para['outPath'] + '/rec')
        except:
            pass

        
class Market():

    def __init__(self, properties, propRelDev=0.01, time = 0):

        self.time        = time
        self.properties = properties
        self.brandProp  = dict()
        self.nProp      = len(properties)
        self.stock      = np.zeros([0,self.nProp+1]) # first row gives the brandID
        self.owners     = list()
        self.propRelDev = propRelDev # relative deviation of the actual car propeties
        self.obsDict    = dict()
        self.obsDict[self.time] = dict()
        self.freeSlots  = list()
        self.stockbyBrand = pd.DataFrame([])
        self.nBrands     = 0
        self.brandLabels = dict()
        #self.prctValues = [50,80]
        
        self.brandInitDict = dict()
        self.brandsToInit = list()
        
    def initCars(self):
        for label, propertyTuple, _, brandID in  self.brandInitDict[0]:
             self.initBrand(label, propertyTuple, brandID)
    
    def computeStatistics(self):  
        # prct = self.percentiles.keys()
        # for item in self.percentiles.keys():
        #    self.percentiles[item] = np.percentile
        #self.percentiles = np.percentile(self.stock[:,1:],self.prctValues,axis=0)
        #print self.percentiles
        self.statistics      = np.zeros([2,self.nProp])
        self.statistics[0,:] = np.mean(self.stock[:,1:],axis=0)
        self.statistics[1,:] = self.statistics[0,:] + np.std(self.stock[:,1:],axis=0) 
        
    def step(self):
        self.time +=1 
        self.obsDict[self.time] = dict()
        #re-init key for next dict in new timestep
        for key in self.obsDict[self.time-1]:
            self.obsDict[self.time][key] = list()
        self.carsPerLabel = np.bincount(self.stock[:,0].astype(int))    
        
        if self.time in self.brandInitDict.keys():
            
            for label, propertyTuple, _, brandID in  self.brandInitDict[self.time]:
                self.initBrand(label, propertyTuple, brandID)
        
    def addBrand(self, label, propertyTuple, initTimeStep):
        
        brandID = self.nBrands
        self.nBrands +=1 
        self.brandsToInit.append(label)
        if initTimeStep not in self.brandInitDict.keys():
            self.brandInitDict[initTimeStep] = [[label, propertyTuple, initTimeStep , brandID]]
        else:
            self.brandInitDict[initTimeStep].append([label, propertyTuple, initTimeStep, brandID])
       
        return brandID
    
    def initBrand(self, label, propertyTuple, brandID):
        
        #add brand to the market
           
        self.stockbyBrand[brandID] = 0
        if self.stockbyBrand.shape[0] == 0:
            # init first time step row
            self.stockbyBrand.loc[len(self.stockbyBrand)] = 0    
        self.brandProp[brandID]   = propertyTuple
        self.brandLabels[brandID] = label
        #self.buyCar(brandID,0)
        #self.stockbyBrand.loc[self.stockbyBrand.index[-1],brandID] -= 1
        self.obsDict[self.time][brandID] = list()
        
        
    def remBrand(self,label):
        #remove brand from the market
        del self.brandProp[label]
    
    def buyCar(self, brandID, eID):
        # draw the actual car property with a random component
        prop =[float(x * y) for x,y in zip( self.brandProp[brandID], (1 + np.random.randn(self.nProp)*self.propRelDev))]
        
        if len(self.freeSlots) > 0:
            carID = self.freeSlots.pop()
            self.stock[carID] = [brandID] + prop
            self.owners[carID] = (eID, brandID)
        else:
            self.stock = np.vstack(( self.stock, [brandID] + prop))
            self.owners.append((eID, brandID))
            carID = len(self.owners)-1
        #self.stockbyBrand.loc[self.stockbyBrand.index[-1],brandID] += 1
        self.computeStatistics()
        
        return carID, prop
    
    def sellCar(self, carID):
        self.stock[carID] = np.Inf
        self.freeSlots.append(carID)
        label = self.owners[carID][1]
        self.owners[carID] = None
        self.stockbyBrand.loc[self.stockbyBrand.index[-1],label] -= 1
    
    def getPropPercentiles(self, propTuple):
        return np.sum(self.stock[:,1:] < propTuple, axis= 0)/float(self.stock.shape[0])

class OpinionGenerator():
    import numpy as np
    """ 
    Creates preferences for households, given their properties
    """
    def __init__(self, indiRatio = 0.33, ecoIncomeRange=(1000,4000),convIncomeFraction=7000):
        self.indiRatio = indiRatio
        self.ecoIncomeRange = ecoIncomeRange
        self.convIncomeFraction = convIncomeFraction
        
        self.feList = ['saf','eco','con', 'mon']
        self.feDict= dict()
        self.feDict['saf'] = ([0,3], [4])
        self.feDict['eco'] = ([],[0,2,4])
        self.feDict['con'] = ([1,3,4],[])
        self.feDict['mon'] = ([],[2,5])

#        self.feDict['saf'] = ([0], [])
#        self.feDict['eco'] = ([],[2])
#        self.feDict['con'] = ([1],[])
        
    def getPref(self,age,sex,nKids,income, radicality):
        
        #safety
        cs = 0
        if nKids < 0:
            if sex == 2:
                cs += 4
            else:
                cs += 2
        cs += int(float(age)/10)
        if sex == 2:
            cs += 1
        cs = float(cs)**2
        
        #ecology
        ce = 3
        if sex == 2:
            ce +=2
        if income>self.ecoIncomeRange[0] and income<self.ecoIncomeRange[1]:
            rn = np.random.rand(1)
            if rn > 0.9:
                ce += 5
            elif rn > 0.5:
                ce += 3
            else:
                ce +=1
        ce = float(ce)**2
        
        #convinience
        cc = 1
        cc += nKids
        cc += income/self.convIncomeFraction
        if sex == 1:
            cc +=1
        if age > 60:
            cc += 3
        elif age > 50:
            cc += 2
        elif age > 40:
            cc += 1
        cc = float(cc)**2

        
        sumC = cc + cs + ce
        cc /= sumC
        ce /= sumC
        cs /= sumC

        #individual preferences
        cci, cei, csi = np.random.rand(3)
        sumC = cci + cei + csi
        cci /= sumC
        cei /= sumC
        csi /= sumC
        
        csAll = cs* (1-self.indiRatio) + csi*self.indiRatio
        ceAll = ce* (1-self.indiRatio) + cei*self.indiRatio
        ccAll = cc* (1-self.indiRatio) + cci*self.indiRatio
        cmAll = np.random.rand(1) *0# only random component
        
        pref = np.asarray([csAll, ceAll, ccAll, cmAll])
        pref = pref ** radicality
        pref = pref / np.sum(pref)
        return tuple(pref)
    
    def getUtililty(self,prop,pref):
        #print 1
        #safety
        util = 1
        for i,fe in enumerate(self.feList):
            
            xCum = []
            #pos factors
            for x in self.feDict[fe][0]:
                xCum.append(prop[x]*100)
                
            #neg factors
            for x in self.feDict[fe][1]:
                xCum.append((1-prop[x])*100)
            util *= np.mean(xCum)**pref[i]
        return util



    
if __name__ == "__main__":
    market = Market(["wei","ran"])
    
    market.addBrand('g',(2,1))    
    market.addBrand('b',(1,2))    
    market.buyCar('g',1)    
    market.buyCar('b',2) 
    market.buyCar('g',1)
    market.buyCar('b',1)
    
    print market.getPropPercentiles((2,2))

    og = OpinionGenerator(indiRatio= 0.2)
    print 'male young' + str(og.getPref(20,1,0,3001))
    print 'female young' + str(og.getPref(20,2,0,5001))
    print 'male family' + str(og.getPref(25,1,2,5001))
    print 'female family' + str(og.getPref(25,1,4,3501))
    print 'female singe' + str(og.getPref(25,2,0,3501))
    print 'male old' + str(og.getPref(64,1,3,2501))
    print 'female old' + str(og.getPref(64,2,3,2501))
