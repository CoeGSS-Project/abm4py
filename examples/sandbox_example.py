#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://wwww.globalclimateforum.org

SAND BOX EXAMPLE
-- INIT FILE --

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
along with GCFABM.  If not, see <http://earthw.gnu.org/licenses/>.
"""

from lib_gcfabm import World, Agent
import numpy as np


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
            for vert in earth.graph.vs['type']:
                if vert == 0:
                    visual_style["vertex_shape"].append('hidden')                
                elif vert == 1:
                        
                    visual_style["vertex_shape"].append('rectangle')                
                else:
                    visual_style["vertex_shape"].append('circle')     
            visual_style["vertex_size"] = list()  
            for vert in earth.graph.vs['type']:
                if vert >= 3:
                    visual_style["vertex_size"].append(6)  
                else:
                    visual_style["vertex_size"].append(20)  
            visual_style["edge_color"]   = [colorDictEdge[typ] for typ in edgeValues]
            if filename  == 'none':
                ig.plot(earth.graph,**visual_style)    
            else:
                ig.plot(earth.graph, filename, **visual_style)    
    #%% #################### USER DEFINED FUNCTIONS ##################        
    def carAgeStep():
        for car in earth.iterNode('car'):
            print car.nID
            if car.getValue('age') > 5 and np.random.random(1) > .5:
                # car dies
                print 'deleting car: '+ str(car.nID)            
                car.delete(earth)
            else:
                print car.nID
                car.setValue('age',car.getValue('age')+1)
    
    def computeCars():
    
        # for the agent nodes
        for agent in earth.iterNodes(_agent):
            #print agent.nID
            nBrownCars = 0
            nGreenCars = 0
            neigList = agent.getConnAgents(_agent)
            for neigbour in neigList:
                if neigbour['type'] == _car:
                    nBrownCars +=1
                elif neigbour['type'] == _grCar:
                    nGreenCars += 1
                    
            agent.setValue('nCars',nBrownCars + nGreenCars)
            agent.setValue('nGreenCars',nGreenCars)
        
        # for spatial nodes
        for location in earth.iterNodes(_location):
            nCars = 0
            #reach all agents of the nodes and their cars => neighboorhood order of 2
            neigList, neigIDList = location.getNeigbourhood(order = 2) 
            for neigbour in neigList:
                if neigbour['type'] == _car or neigbour['type'] == _grCar:
                    nCars +=1
            location.setValue('nCars', nCars)
                    
    # step of the environment
    def environmentStep(deltaCars):
        for location in earth.iterNodes(_location):
            location.setValue('deltaCars',deltaCars[location.x,location.y])
    
    
    
    def simulationStep(deltaCars):
        
        
        for loc in earth.iterNodes(_location):
            
            print 'location is ' + str(loc.x) + ' x '+ str(loc.y)
            agList = loc.getAgentOfCell(agTypeID=_agent)
    
            while loc.getValue('deltaCars') > 0:
    
                # choose one agent from the cell that buys a car
                agID = np.random.choice(agList)
                agent = earth.entDict[agID]
                gdp = agent.getEnvValue('gdp')
                print 'gdp: ' + str(),
                ownedCars = agent.getValue('nCars')
                print'own cars: ' + str(ownedCars),
                #ownedGreenCars = agent.getValue('nGreenCars')
                tmp, __ = agent.getConnNodeValues('nGreenCars',agTypeID=_agent)
                sumGreenCars = np.sum(tmp)
                print 'Sum Green Cars: ' + str(sumGreenCars),
                tmp, __ = agent.getConnNodeValues('nCars',agTypeID=_agent)
                sumCars = np.sum(tmp)
                print 'SumCars: ' + str(sumCars)     
                print agent.getConnNodeValues('nCars',agTypeID=_agent)
                
                green_pressure = sumGreenCars / sumCars
                if green_pressure > 0:
                    probGreenCar = eta * gdp + (1 - (eta * gdp)) * green_pressure**kappa 
                else:
                    probGreenCar = eta * gdp
                print 'probability of buying a green car: ' +str(probGreenCar)
                
                if np.random.random(1) < probGreenCar:
                    # buy a green car
                    #agent.setValue('nGreenCars',ownedGreenCars+1)
                    car = Agent(earth,'greenCar', x, y)
                    carList.append(car)
                    car.addLink(agID,liTypeID=_agGrCarLink)
                    car.setValue('age',0)
                else:
                    # buy a brown car
                    #agent.setValue('nCars',ownedCars+1)
                    car = Agent(earth,'car', x, y)
                    carList.append(car)
                    car.addLink(agID,liTypeID=_agCarLink)
                    car.setValue('age',0)
                loc.setValue('deltaCars',loc.getValue('deltaCars')-1)
    
    
    ##################### END USER DEFINED FUNCTIONS ##################
    
            
    #%% ###################### Parameters: ############################
    flgSpatial = True
    connRadius = 1.5
    nAgents   = 30    
    nSteps    = 10
    eta = 0.5
    kappa = 0.1
    
    
    # Raster Data
    landLayer   = np.asarray([[0, 0, 1, 1], [0,1,1, 0],[1,1,0,0], [0,0,1,0]])
    nCarsArray  = np.asarray([[0, 0, 10,5], [0,15,8, 0],[17,3,0,0], [0,0,0,0]])
    gdpArray    = np.asarray([[0, 0, .9,1.1], [0,.8,.9, 0],[.7,1.3,0,0], [0,0,0,0]])
    deltaCars = np.zeros(nCarsArray.shape)
     
    #%% ######################### INITIALIZATION ##########################   
    earth = World(spatial=flgSpatial)
    _inactive = 0
    _location = 1
    _agent    = 2
    _car      = 3
    _grCar    = 4
    
    # edge types
    earth.registerLinkType('locLocLink')
    earth.registerLinkType('locAgLink')
    earth.registerLinkType('agAgLink')
    earth.registerLinkType('agCarLink')
    earth.registerLinkType('agGrCarLink')
    connList= earth.computeConnectionList(connRadius)
    earth.initSpatialLayerNew(landLayer, connList)
    
    # init agents
    for ag in range(nAgents):
        while True:
            (x,y) = (np.random.random(1)[0]*(landLayer.shape[0]),np.random.random(1)[0]*(landLayer.shape[0]))
            if landLayer[int(x),int(y)] == 1:
                agent = Agent(earth,'ag', x, y)
                agent.connectLocation(earth)
                agent.register(earth)         
                break
    #init attributes of agents
    earth.graph.vs['nCars'] = 0
    earth.graph.vs['nGreenCars'] = 0  
    
    # add social connections
    for ag in earth.iterNodes(_agent):
    
        nFriends = np.random.randint(3)+1
        
        for socLink in range(nFriends):
            
            while True:
                friendID = np.random.choice(earth.nodeList[_agent])
                if friendID != ag.nID:
                    ag.addLink(friendID,liTypeID=_agAgLink)
                    break
    
    
    # load car data for nodes
    earth.updateSpatialLayer('nCars',nCarsArray)
    earth.updateSpatialLayer('gdp', gdpArray)
    
    #earth.view()
    carList = list()
    # distribute cars
    indices = np.where(nCarsArray > 0 )
    for x,y in zip(*indices):
        loc = earth.locDict[(x,y)]
        agList= loc.getAgentOfCell(agTypeID= _agent)
        for iCar in range(nCarsArray[x,y]):    
            agID = np.random.choice(agList)
            
            assert  earth.graph.vs[agID]['type'] == _agent
            agent = earth.entDict[agID]
            #agent.setValue('nCars',agent.getValue('nCars')+1)
            car = Agent(earth,'car', x, y)
            carList.append(car)
            car.addLink(agID,liTypeID=_agCarLink)
            car.setValue('age',np.random.randint(1,4))
    
    #%%########################### Simulation step    ######################################
    
    # step to age the cars
    computeCars()  
    for step in range(nSteps):
        #generating environment input    
        deltaCars[landLayer==1] = np.random.randint(1,3,np.sum(landLayer))
        environmentStep(deltaCars)
        simulationStep(deltaCars)   
        #carAgeStep()
        computeCars()
        earth.view('out' + str(step) + '.png')