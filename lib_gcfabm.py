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


################ ENTITY CLASS #########################################    
# general ABM entity class for objects connected with the graph

class Entity():
    
    def __init__(self, world, nodeStr):
        self.graph= world.graph
        self.nID = len(self.graph.vs)
        nodeType = world.getNodeType(nodeStr)
        self.graph.add_vertex(self.nID, type=nodeType)
        self.type= nodeType      

    
    def getNeigbourhood(self, order):
        
        neigIDList = self.graph.neighborhood(self.nID, order)
        neigbours = []
        for neigID in neigIDList:
            neigbours.append(self.graph.vs[neigID])
        return neigbours, neigIDList
        
        
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
    
    def addConnection(self, friendID, edgeType=0):
        if not self.graph.are_connected(self.nID,friendID):
            self.graph.add_edge(self.nID,friendID, type=edgeType)         
            #eID = self.graph.get_eid(self.nID,friendID)
            #self.graph.es[eID]['type'] = edgeType
            
    def remConnection(self, friendID,edgeType=0):
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
        self.graph.vs[self.nID]['pos']= (xPos,yPos)
        
    def register(self,world):
        world.registerLocation(self)
        self.world = world
        
    def getAgentOfCell(self,edgeType):
        #entList= []            
#        esSeq = self.graph.es.select(_target=self.nID,type=edgeType).indices
#        return map(self.getSource,esSeq)
        #return [self.graph.es[x].source for x in esSeq]
#        for edge in esSeq:
#            entList.append(self.graph.es[edge].source)
#        return entList
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
        
    def __init__(self, world, nodeType = 'ag', xPos = np.nan, yPos = np.nan):
        Entity.__init__(self, world, nodeType)

        if not(np.isnan(xPos)) and not(np.isnan(xPos)):
            self.x = xPos
            self.y = yPos
    
    def register(self, world):
        world.registerNode(self, self.type)
        
        
    def connectLocation(self, world):
        # connect agent to its spatial location
        locationID = int(self.graph.IdArray[int(self.x),int(self.y)])
        self.graph.add_edge(self.nID,locationID, type=2)         
        #eID = self.graph.get_eid(self.nID,geoNodeID)
        #self.graph.es[eID]['type'] = 2
        self.loc = world.entDict[locationID]

        
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
#        print 1
#        eIDSeq = self.graph.es.select(_target=self.nID,type=_locAgLink).indices[0]       
#        return self.graph.vs[synEarth.graph.es[eIDSeq].source][prop]
    
        return self.graph.vs[self.loc.nID][prop]



################ WORLD CLASS #########################################
    
class World:
    
    def __init__(self,spatial=False):
        self.spatial  = spatial
        self.graph    = ig.Graph(directed=True)
        self.graph.edgeQueue   = (list(),list()) #(nodetuple list, typelist)
        self.graph.vertexQueue = (list(),list()) #(nodelist, typelist)
        self.types    = list()
        self.nodeList = dict()
                
        self.entList   = list()
        self.entDict   = dict()
        
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
        self.entList.append(agent)
        self.entDict[agent.nID] = agent
        self.nodeList[typ].append(agent.nID)
    
    def iterNode(self,nodeType):
        nodeList = self.nodeList[nodeType]
        return  iter([self.entList[i] for i in nodeList])
    
    def iterEdges(self, edgeType):
        for i in range(self.graph.ecount()):
            if self.graph.es[i]['type'] == edgeType:
                yield self.graph.es[i]
    
    def iterNodeAndID(self,nodeType):
        nodeList = self.nodeList[nodeType]
        return  iter([(self.entList[i], i) for i in nodeList]) 
    
    def registerLocation(self, location):
        
        self.locDict[location.x,location.y] = location
        
    def getNode(self,nodeID):
        return self.entDict[nodeID]
    
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
    
    def initSpatialLayerNew(self, nodeArray, connList, LocClassObject=Location):
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
                    loc = LocClassObject(self,x,y)
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
        nodetype = self.getNodeType('lo')
        for nId in self.nodeList[nodetype]:
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
    
    def getNodeType(self, typeStr):
        if typeStr not in self.types:
            self.registerType(typeStr)
                
        for iType, liTyp in enumerate(self.types):
            if liTyp == typeStr :
                break
        
        return iType
    
    def registerType(self, typeStr):
        iType = len(self.types)
        self.types.append(typeStr)
        self.nodeList[iType]= list()
    
    def dequeueEdges(self):
        eStart = self.graph.ecount()
        self.graph.add_edges(self.graph.edgeQueue[0])
        self.graph.es[eStart:]['type'] = self.graph.edgeQueue[1]
    
    
    def view(self,filename = 'none', vertexProp='none'):
        """
        Very basic visualization method. Only works for small graphs, since layouting large graph
        takes time forever. Not tested very well
        """
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
            
    
########################################################################################
#  END OF CLASS DESCRIPTION
########################################################################################

