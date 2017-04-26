#!/usr/bin/env python2
# -*- coding: UTF-8-*-
"""
G R A P H    M O D U L E 

Copyright (c) 2017
Global Climate Forum e.V.
http://www.globalclimateforum.org

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
from __future__ import division
import igraph as ig
import numpy as np
import tqdm

class SynEarth:
    
    def __init__(self,spatial=False):
        self.spatial  = spatial
        self.graph    = ig.Graph(directed=True)
        self.graph.edgeQueue   = (list(),list()) #(nodetuple list, typelist)
        self.graph.vertexQueue = (list(),list()) #(nodelist, typelist)
        self.types    = list()
        self.nodeList = dict()
                
        self.agList   = list()
        self.agDict   = dict()
        
        # init of spatial layer if spatial domain is set
        if spatial:
            self.locDict = dict()

        self.types.append('inactiv')

    def setNodeValues(self,nodeID, prop,value):
        self.graph.vs[nodeID][prop] = value
        
    def getNodeValues(self,nodeID, prop):
        return self.graph.vs[nodeID][prop]
    
    def setEdgeValues(self,edgeIDs, prop, value):
        self.graph.es[edgeIDs][prop] = value
        
    def getEdgeValues(self, edgeIDs, prop):
        return self.graph.es[edgeIDs][prop]    
    
    def registerNode(self, agent, typ):
        self.agList.append(agent)
        self.agDict[agent.nID] = agent
        
        if not(typ in self.types):
            self.types.append(typ)
            self.nodeList[typ]= list()
            self.nodeList[typ].append(agent.nID)
        else:
            self.nodeList[typ].append(agent.nID)
    
    def iterNode(self,nodeType):
        nodeList = self.nodeList[nodeType]
        return  iter([self.agList[i] for i in nodeList])
    
    def iterEdges(self, edgeType):
        for i in range(self.graph.ecount()):
            if self.graph.es[i]['type'] == edgeType:
                yield self.graph.es[i]
    
    def iterNodeAndID(self,nodeType):
        nodeList = self.nodeList[nodeType]
        return  iter([(self.agList[i], i) for i in nodeList]) 
    
    def registerLocation(self, location):
        
        self.locDict[location.x,location.y] = location
        
    def getNode(self,nodeID):
        return self.agDict[nodeID]
    
    def nodeDegreeHist(self,nodeType,nbars=20):
        import matplotlib.pyplot as plt
        plt.hist(self.graph.vs[self.nodeList[nodeType]].degree().nbars)
        
 
    def computeConnectionList(self,radius=1):
        connList = []  
        
        intRad = int(radius)
        for x in range(-intRad,intRad+1):
            for y in range(-intRad,intRad+1):
                if (x**2 +y**2)**.5 < radius:
                    if x**2 +y**2 > 0:
                        weig  = 1/((x**2 +y**2)**.5)
                    else:
                        weig = 2
                    connList.append((x,y,weig))
        return connList
    
    def initSpatialLayerNew(self, nodeArray, connList, LocObject):
        self.graph.IdArray = nodeArray * np.nan
        self.graph.IdArray[nodeArray == 1] = xrange(np.sum(nodeArray))
        IDArray = self.graph.IdArray
        # spatial extend
        xOrg = 0
        yOrg = 0
        xMax = nodeArray.shape[0]
        yMax = nodeArray.shape[1]

        # create vertices 
        id = 0
        #self.spatialNodeList = []
        for x in range(nodeArray.shape[0]):
            for y in range(nodeArray.shape[1]):

                # only add an vertex if spatial location exist                    
                if nodeArray[x,y] == 1:
                    # add vertex with type sp = spatial
                    loc = LocObject(self,x,y)
                    self.registerLocation(loc)
                    self.registerNode(loc,1)
                    self.locDict[(x,y)] = loc
                    id +=1
        
        for (x,y), loc in tqdm.tqdm(self.locDict.items()):
            
            srcID = loc.nID
            #print loc.nID
            
            weigList = list()
            destList = list()
            connectionList = list()
            
            for (dx,dy,weight) in connList:
                
                xDst = x + dx
                yDst = y + dy
                
                # check boundaries of thedestination node
                if xDst >= xOrg and xDst < xMax and yDst >= yOrg and yDst < yMax:                        
                    
                    trgID = IDArray[xDst,yDst]
                    if not np.isnan(trgID):
                        destList.append(int(trgID))
                        weigList.append(weight)
                        connectionList.append((int(srcID),int(trgID)))
                    
                    # check if destination location exist
                    #if (xDst,yDst) in self.locDict:
                    #    destList.append(self.locDict[xDst,yDst].nID)
                    #    weigList.append(weight)
                        #edgeList.append(eID)  
                        
                        #print 'connected to= ' +str(IdDst)
                        #eID = self.graph.ecount()
                        #self.graph.add_edge(int(IdSrc),int(IdDst))         
                        #eID = self.graph.get_eid(int(IdSrc),int(IdDst))
                        
                        #print "connecting " + str(self.graph.es[eID].source) + " with " + str(self.graph.es[eID].target)
                              
                
            #normalize weight to sum up to unity                    
            sumWeig = sum(weigList)
            weig    = np.asarray(weigList) / sumWeig
            eStart = self.graph.ecount()
            self.graph.add_edges(connectionList)
            self.graph.es[eStart:]['type'] = 1
            self.graph.es[eStart:]['weig'] = weig
#            for destID,weig in zip(destList, weigList):
#                self.graph.add_edge(int(srcID),int(destID), weig=weig/sumWeig, type=1)   
                        

            #for eID,weig in zip(destList, edgeList,weigList):
            #    self.graph.es[eID]['weig'] = weig/sumWeig
            #    self.graph.es[eID]['type'] = 1   
                    
                        
    def updateSpatialLayer(self,propName,array):
        
        for nId in self.nodeList['lo']:
            [x,y] = self.graph.vs[nId]['pos']
            self.graph.vs[nId][propName] = array[x,y]
    
    def getCellAgents(self,x,y,agType=0):
        print 'old version - do not use'
        spNodeID = synEarth.graph.vs.select(pos=[x,y],type=0).indices[0]
        eIDSeq = synEarth.graph.es.select(_source=spNodeID,type=1).indices
        cellAgents= []            
        for edge in synEarth.graph.es[eIDSeq]:
            cellAgents.append(edge.target)
        return cellAgents
#            return synEarth.graph.vs.select(pos=[x,y],type=agType).indices

################ ENTITY CLASS #########################################    
# general ABM entity class for objects connected with the graph

class Entity():
    
    def __init__(self, Earth, nodeType):
        self.graph= Earth.graph
        self.nID = len(self.graph.vs)
        for i, liTyp in enumerate(Earth.types):
            if liTyp == nodeType :
                break
        self.graph.add_vertex(self.nID, type=i)
        self.type= nodeType      
        #self.node = self.graph.vs[self.nID]
        #Earth.registerNode(self,nodeType)
        #for i, liTyp in enumerate(Earth.types):
        #    if liTyp == nodeType :
        #        break
        #self.graph.vs[self.nID]['type']= i      
        
        #self.graph.vs[self.nID]['type']= 0
    
    def getNeigbourhood(self, order):
        
        neigIDList = self.graph.neighborhood(self.nID, order)
        neigbours = []
        for neigID in neigIDList:
            neigbours.append(self.graph.vs[neigID])
        return neigbours
        
        
    def getAllNeighNodes(self):
        neigIDList = self.graph.neighbors(self.nID)
        neigbours = []
        for neigID in neigIDList:
            neigbours.append(self.graph.vs[neigID])
        return neigbours
          
    def getOutNeighNodes(self, edgeType=None):
        neigbours = []
        eList = self.graph.incident(self.nID,mode="OUT")

        if edgeType is not None:
            for edge in eList:
                if self.graph.es[edge]['type'] == edgeType:
                    neigbours.append(self.graph.es[edge].target)  
                    
        else:
            for edge in eList:
                neigbours.append(self.graph.es[edge].target)     

        return neigbours
    
    def getInNeighNodes(self, edgeType=None):
        neigbours = [] 
        eList = self.graph.incident(self.nID,mode="OUT")

        if edgeType is not None:
            for edge in eList:
                if self.graph.es[edge]['type'] == edgeType:
                    neigbours.append(self.graph.es[edge].source)     
        else:
            for edge in eList:
                neigbours.append(self.graph.es[edge].source)     

        return neigbours
    
    def queueConnection(self, friendID,edgeType=0):
        if not self.graph.are_connected(self.nID,friendID) and (self.nID,friendID) not in self.graph.edgeQueue[0]:
            self.graph.edgeQueue[0].append((self.nID,friendID))
            self.graph.edgeQueue[1].append(edgeType)
    
    def addConnection(self, friendID,edgeType=0):
        if not self.graph.are_connected(self.nID,friendID):
            self.graph.add_edge(self.nID,friendID, type=edgeType)         
            #eID = self.graph.get_eid(self.nID,friendID)
            #self.graph.es[eID]['type'] = edgeType
            
    def remConnection(self, fiendID,edgeType=0):
        eID = self.graph.get_eid(self.nID,friendID)
        self.graph.delete_edges(eID)

    def setValue(self,prop,value):
        self.graph.vs[self.nID][prop] = value
        
    def getValue(self,prop):
        return self.graph.vs[self.nID][prop]
    
    def addValue(self,prop,value):
        self.graph.vs[self.nID][prop] += value

    def delete(self,Earth):
        nID = self.nID
        
        #self.graph.delete_vertices(nID) # not really possible at the current igraph lib
        # Thus, the node is set to inactive and removed from the iterator lists
        # This is due to the library, but the problem is general and pose a challenge.
        Earth.graph.vs[nID]['type'] = 0
        Earth.nodeList[self.type].remove(nID)
        #remove edges        
        eIDSeq = self.graph.es.select(_target=self.nID).indices        
        self.graph.delete_edges(eIDSeq)
        eIDSeq = self.graph.es.select(_source=self.nID).indices        
        self.graph.delete_edges(eIDSeq)

    def getConnProp(self,label, edgeType = 1,mode ='OUT'):
        #esSeq = self.graph.es.select(_source=self.nID,type=conTyp).indices
        #return self.graph.es[esSeq][label]
        eList = self.graph.incident(self.nID,mode=mode) 
        if edgeType is not None:
            edges = list()
            for edge in eList:
                 if self.graph.es[edge]['type'] == edgeType:
                      edges.append(edge)
            return self.graph.es[edges][label], edges
        else:
            return self.graph.es[eList][label], eList
    
################ LOCATION CLASS #########################################      
class Location(Entity):

    def __init__(self, world,  xPos, yPos):
        nodeType = 'lo'
        Entity.__init__(self,world,nodeType)
        self.x = xPos
        self.y = yPos
        self.graph.vs[self.nID]['pos']= [xPos,yPos]
        
    def register(self,world):
        world.registerLocation(self)
        self.world = world
        
    def getAgentOfCell(self,edgeType):
        #agList= []            
#        esSeq = self.graph.es.select(_target=self.nID,type=edgeType).indices
#        return map(self.getSource,esSeq)
        #return [self.graph.es[x].source for x in esSeq]
#        for edge in esSeq:
#            agList.append(self.graph.es[edge].source)
#        return agList
        agList= []    
        eList = self.graph.incident(self.nID,mode="IN")
        for es in eList:
            if self.graph.es[es]['type'] == edgeType:
                agList.append(self.graph.es[es].source)
        return agList       
        #return map(self.getSource,eList)
    def getSource(self,eID):
        
            return self.graph.es[eID].source
        
    
################ AGENT CLASS #########################################
class Agent(Entity):
        
    def __init__(self,Earth, nodeType = 'ag', xPos = np.nan, yPos = np.nan):
        Entity.__init__(self,Earth,nodeType)

        if not(np.isnan(xPos)) and not(np.isnan(xPos)):
            self.x = xPos
            self.y = yPos
    
    def register(self, world, nodeType):
        world.registerNode(self,nodeType)
        
        
    def connectGeoNode(self, world):
        # connect agent to its spatial location
        geoNodeID = int(self.graph.IdArray[int(self.x),int(self.y)])
        self.graph.add_edge(self.nID,geoNodeID, type=2)         
        #eID = self.graph.get_eid(self.nID,geoNodeID)
        #self.graph.es[eID]['type'] = 2
        self.loc = world.agDict[geoNodeID]

        
#    def getNeighNodeValues(self, prop,edge_Type=0):
#        eIDSeq = self.graph.es.select(_target=self.nID,type=edge_Type).indices
#        values = []            
#        for edge in self.graph.es[eIDSeq]:
#            values.append(self.graph.vs[edge.source][prop])
#        # TODO replace by 
#        #eList = self.graph.incident(self.nID,mode="IN")
#        #for es in eList:
#        #    if self.graph.es[es]['type'] == edgeType:
#        return values
    
    def getNeighNodeValues(self, prop, edgeType=0):
        neigbours = [] 
        edges     = []
        eList = self.graph.incident(self.nID,mode="OUT")

        if edgeType is not None:
            for edge in eList:
                if self.graph.es[edge]['type'] == edgeType:
                    neigbours.append(self.graph.es[edge].target)   
                    edges.append(edge)
        else:
            for edge in eList:
                neigbours.append(self.graph.es[edge].target)     
                edges.append(edge)
        return self.graph.vs[neigbours][prop], edges
    
    def getEnvValue(self,prop):
        # TODO replace by 
        #eList = self.graph.incident(self.nID,mode="IN")
        #for es in eList:
        #    if self.graph.es[es]['type'] == edgeType:
        eIDSeq = self.graph.es.select(_target=self.nID,type=_locAgLink).indices[0]       
        return self.graph.vs[synEarth.graph.es[eIDSeq].source][prop]
    

    
########################################################################################
#  END OF CLASS DESCRIPTION
########################################################################################


#%% ########### Enum types #######################################

if __name__ == '__main__':
    
    # node types
    _inactive = 0
    _location = 1
    _agent    = 2
    _car      = 3
    _grCar    = 4
    
    # edge types
    _inactive    = 0
    _locLocLink  = 1
    _locAgLink   = 2
    _agAgLink    = 3
    _agCarLink   = 4
    _agGrCarLink = 5
    
    def plot(self,filename = 'none', vertexProp='none'):
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
            
            #self.graph.vs["label"] = self.graph.vs["nCars"]
            edgeValues = (np.array(self.graph.es['type']).astype(float)).astype(int).tolist()
            
            visual_style = {}
            visual_style["vertex_color"] = [colorDictNode[typ] for typ in nodeValues]
            visual_style["vertex_shape"] = list()        
            for vert in synEarth.graph.vs['type']:
                if vert == 0:
                    visual_style["vertex_shape"].append('hidden')                
                elif vert == 1:
                        
                    visual_style["vertex_shape"].append('rectangle')                
                else:
                    visual_style["vertex_shape"].append('circle')     
            visual_style["vertex_size"] = list()  
            for vert in synEarth.graph.vs['type']:
                if vert >= 3:
                    visual_style["vertex_size"].append(6)  
                else:
                    visual_style["vertex_size"].append(20)  
            visual_style["edge_color"]   = [colorDictEdge[typ] for typ in edgeValues]
            if filename  == 'none':
                ig.plot(synEarth.graph,**visual_style)    
            else:
                ig.plot(synEarth.graph, filename, **visual_style)    
    #%% #################### USER DEFINED FUNCTIONS ##################        
    def carAgeStep():
        for car in synEarth.iterNode('car'):
            print car.nID
            if car.getValue('age') > 5 and np.random.random(1) > .5:
                # car dies
                print 'deleting car: '+ str(car.nID)            
                car.delete(synEarth)
            else:
                print car.nID
                car.setValue('age',car.getValue('age')+1)
    
    def computeCars():
    
        # for the agent nodes
        for agent in synEarth.iterNode('ag'):
            #print agent.nID
            nBrownCars = 0
            nGreenCars = 0
            neigList = agent.getAllNeighNodes()
            for neigbour in neigList:
                if neigbour['type'] == _car:
                    nBrownCars +=1
                elif neigbour['type'] == _grCar:
                    nGreenCars += 1
                    
            agent.setValue('nCars',nBrownCars + nGreenCars)
            agent.setValue('nGreenCars',nGreenCars)
        
        # for spatial nodes
        for location in synEarth.iterNode('lo'):
            nCars = 0
            #reach all agents of the nodes and their cars => neighboorhood order of 2
            neigList = location.getNeigbourhood(order = 2) 
            for neigbour in neigList:
                if neigbour['type'] == _car or neigbour['type'] == _grCar:
                    nCars +=1
            location.setValue('nCars', nCars)
                    
    # step of the environment
    def environmentStep(deltaCars):
        for location in synEarth.iterNode('lo'):
            location.setValue('deltaCars',deltaCars[location.x,location.y])
    
    
    
    def simulationStep(deltaCars):
        
        
        for loc in synEarth.iterNode('lo'):
            
            print 'location is ' + str(loc.x) + ' x '+ str(loc.y)
            agList = loc.getAgentOfCell(edgeType=_locAgLink)
    
            while loc.getValue('deltaCars') > 0:
    
                # choose one agent from the cell that buys a car
                agID = np.random.choice(agList)
                agent = synEarth.agDict[agID]
                gdp = agent.getEnvValue('gdp')
                print 'gdp: ' + str(),
                ownedCars = agent.getValue('nCars')
                print'own cars: ' + str(ownedCars),
                #ownedGreenCars = agent.getValue('nGreenCars')
                
                sumGreenCars = np.sum(agent.getNeighNodeValues('nGreenCars',edge_Type=_agAgLink))
                print 'Sum Green Cars: ' + str(sumGreenCars),
                sumCars = np.sum(agent.getNeighNodeValues('nCars',edge_Type=_agAgLink))
                print 'SumCars: ' + str(sumCars)     
                print agent.getNeighNodeValues('nCars',edge_Type=_agAgLink)
                
                green_pressure = sumGreenCars / sumCars
                if green_pressure > 0:
                    probGreenCar = eta * gdp + (1 - (eta * gdp)) * green_pressure**kappa 
                else:
                    probGreenCar = eta * gdp
                print 'probability of buying a green car: ' +str(probGreenCar)
                
                if np.random.random(1) < probGreenCar:
                    # buy a green car
                    #agent.setValue('nGreenCars',ownedGreenCars+1)
                    car = Agent(synEarth,'greenCar', x, y)
                    carList.append(car)
                    car.addConnection(agID,edgeType=_agGrCarLink)
                    car.setValue('age',0)
                else:
                    # buy a brown car
                    #agent.setValue('nCars',ownedCars+1)
                    car = Agent(synEarth,'car', x, y)
                    carList.append(car)
                    car.addConnection(agID,edgeType=_agCarLink)
                    car.setValue('age',0)
                loc.setValue('deltaCars',loc.getValue('deltaCars')-1)
    
    
    ##################### END USER DEFINED FUNCTIONS ##################
    
            
    #%% ###################### Parameters: ############################
    flgSpatial = True
    connRadius = 1.5
    nAgents   = 30    
    
    eta = 0.5
    kappa = 0.1
    
    
    # Raster Data
    landLayer   = np.asarray([[0, 0, 1, 1], [0,1,1, 0],[1,1,0,0], [0,0,1,0]])
    nCarsArray  = np.asarray([[0, 0, 10,5], [0,15,8, 0],[17,3,0,0], [0,0,0,0]])
    gdpArray    = np.asarray([[0, 0, .9,1.1], [0,.8,.9, 0],[.7,1.3,0,0], [0,0,0,0]])
    deltaCars = np.zeros(nCarsArray.shape)
     
    #%% ######################### INITIALIZATION ##########################   
    synEarth = SynEarth(spatial=flgSpatial)
    connList= synEarth.computeConnectionList(connRadius)
    synEarth.initSpatialLayerNew(landLayer, connList)
    
    # init agents
    for ag in range(nAgents):
        while True:
            (x,y) = (np.random.random(1)[0]*(landLayer.shape[0]),np.random.random(1)[0]*(landLayer.shape[0]))
            if landLayer[int(x),int(y)] == 1:
                agent = Agent(synEarth,'ag', x, y)
                agent.connectGeoNode()
                         
                break
    #init attributes of agents
    synEarth.graph.vs['nCars'] = 0
    synEarth.graph.vs['nGreenCars'] = 0  
    
    # add social connections
    for ag in synEarth.iterNode('ag'):
    
        nFriends = np.random.randint(3)+1
        
        for socLink in range(nFriends):
            
            while True:
                friendID = np.random.choice(synEarth.nodeList['ag'])
                if friendID != ag.nID:
                    ag.addConnection(friendID,edgeType=_agAgLink)
                    break
    
    
    # load car data for nodes
    synEarth.updateSpatialLayer('nCars',nCarsArray)
    synEarth.updateSpatialLayer('gdp', gdpArray)
    
    
    carList = list()
    # distribute cars
    indices = np.where(nCarsArray > 0 )
    for x,y in zip(*indices):
        loc = synEarth.locDict[(x,y)]
        agList= loc.getAgentOfCell(edgeType=2)
        for iCar in range(nCarsArray[x,y]):    
            agID = np.random.choice(agList)
            
            assert  synEarth.graph.vs[agID]['type'] == _agent
            agent = synEarth.agDict[agID]
            #agent.setValue('nCars',agent.getValue('nCars')+1)
            car = Agent(synEarth,'car', x, y)
            carList.append(car)
            car.addConnection(agID,edgeType=_agCarLink)
            car.setValue('age',np.random.randint(1,4))
    
    #%%########################### Simulation step    ######################################
    
    # step to age the cars
    computeCars()  
    for step in range(10):
        #generating environment input    
        deltaCars[landLayer==1] = np.random.randint(1,3,np.sum(landLayer))
        environmentStep(deltaCars)
        simulationStep(deltaCars)   
        #carAgeStep()
        computeCars()
        synEarth.plot('out' + str(step) + '.png')