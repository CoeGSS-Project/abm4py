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
        self.type = nodeType     
        self.node = self.graph.vs[self.nID]
        self.edges = dict()
            
    
    def getNeigbourhood(self, order):
        
        neigIDList = self.graph.neighborhood(self.nID, order)
        neigbours = []
        for neigID in neigIDList:
            neigbours.append(self.graph.vs[neigID])
        return neigbours, neigIDList
        
    
    def queueConnection(self, friendID,edgeType=0):
        if not self.graph.are_connected(self.nID,friendID) and (self.nID,friendID) not in self.graph.edgeQueue[0]:
            self.graph.edgeQueue[0].append((self.nID,friendID))
            self.graph.edgeQueue[1].append(edgeType)
    
    def addConnection(self, friendID, edgeType=0):
        if not self.graph.are_connected(self.nID,friendID):
            self.graph.add_edge(self.nID,friendID, type=edgeType)         
            self.updateEdges()
            
    def remConnection(self, friendID,edgeType=0):
        eID = self.graph.get_eid(self.nID,friendID)
        self.graph.delete_edges(eID)
        self.updateEdges()

    def updateEdges(self):
            
        for typ in self.graph.edgeTypes:
            self.edges[typ] = self.graph.es[self.graph.incident(self.nID,'out')].select(type=typ)
        self.edgesAll = self.graph.es[self.graph.incident(self.nID,'out')]
    
    def _old1_setValue(self,prop,value):
        self.graph.vs[self.nID][prop] = value
        
    def _old1_getValue(self,prop):
        return self.graph.vs[self.nID][prop]
    
    def setValue(self,prop,value):
        self.node[prop] = value
        
    def getValue(self,prop):
        return self.node[prop]
    
    def addValue(self,prop,value):
        self.node[prop] += value

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

    def getEdgeValues(self, prop, edgeType = None,mode ='OUT'):
        #esSeq = self.graph.es.select(_source=self.nID,type=conTyp).indices
        #return self.graph.es[esSeq][label]
        eList = self.graph.incident(self.nID,mode=mode) 
        if edgeType is not None:
            edges = list()
            for edge in eList:
                 if self.graph.es[edge]['type'] == edgeType:
                      edges.append(edge)
            return self.graph.es[edges][prop], edges
        else:
            return self.graph.es[eList][prop], eList
        
    def getEdges(self, edgeType=0):
        edges = self.edges[edgeType]
        return edges
    
    def getEdgeValuesFast(self, prop, edgeType=0):
        edges = self.edges[edgeType]
        return edges[prop], edges

    def _old3_getEdgeValues(self, prop, edgeType=0, mode="OUT"):
        if edgeType is not None:
            edges = self.graph.es[self.graph.incident(self.nID,mode)]
        else:
            edges = self.graph.es[self.graph.incident(self.nID,mode)].select(type=edgeType)
        values = edges[prop]
        return values, edges
    
    def _old2_getEdgeValues(self, prop, edgeType=0):
        values = [] 
        edges     = []
        
        if edgeType is not None:
            for edge in self.edgesAll:
                if edge['type'] == edgeType:
                    values.append(edge[prop])   
                    edges.append(edge)
        else:
            for edge in self.edgesAll:
                if edge['type'] == edgeType:
                    values.append(edge[prop])   
                    edges.append(edge)
        return values, edges

    def _old1_getEdgeValues(self, prop, edgeType=0, mode="OUT"):
        values = [] 
        edges     = []
        eList = self.graph.incident(self.nID,mode)

        if edgeType is not None:
            for edge in eList:
                if self.graph.es[edge]['type'] == edgeType:
                    values.append(self.graph.es[edge][prop])   
                    edges.append(edge)
        else:
            for edge in eList:
                values.append(self.graph.es[edge][prop])        
                edges.append(edge)
        return values, edges
    
    def getConnNodes(self, nodeType=0, mode='out'):
        if mode is None:
            neigbours = self.node.neighbors()
        else:
            neigbours = self.node.neighbors(mode)
    
        return [x for x in neigbours if x['type'] == nodeType]

    def getConnNodeIDs(self, nodeType=0, mode='out'):
        if mode is None:
            neigbours = self.node.neighbors()
        else:
            neigbours = self.node.neighbors(mode)
    
        return [x['name'] for x in neigbours if x['type'] == nodeType]
        
    def getConnNodeValues(self, prop, nodeType=0, mode='out'):
        nodeList = self.node.neighbors(mode)
        neighbourIDs     = list()
        values          = list()

        for node in nodeList:
            if node['type'] == nodeType:
                neighbourIDs.append(node['name'])   
                values.append(node[prop])
        
        return values, neighbourIDs
    
    def _old2_getConnNodeValues(self, prop, nodeType=None, mode="OUT"):
        neighIDs = self.graph.neighborhood(self.nID)
        
           
        if nodeType is not None:
            neigbors = self.graph.vs[neighIDs].select(type=nodeType)
        else:
            neigbors = self.graph.vs[neighIDs]
            
        values = neigbors[prop]
        
        return values, neighIDs
    

    
    def _old1_getConnNodeValues(self, prop, edgeType=None, mode="OUT"):
        neigbours = [] 
        edges     = []
        eList = self.graph.incident(self.nID,mode)

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
        return self.getConnNodeIDs( nodeType=2, mode='in')
        
        
    
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
        self.updateEdges()  

    def getEdges(self, edgeType=0):
        edges = self.edges[edgeType]
        return edges
    
    def getEdgeValues(self, prop, edgeType=0):
        edges = self.edges[edgeType]
        return edges[prop], edges

    def _old3_getEdgeValues(self, prop, edgeType=0, mode="OUT"):
        if edgeType is not None:
            edges = self.graph.es[self.graph.incident(self.nID,mode)]
        else:
            edges = self.graph.es[self.graph.incident(self.nID,mode)].select(type=edgeType)
        values = edges[prop]
        return values, edges
    
    def _old2_getEdgeValues(self, prop, edgeType=0):
        values = [] 
        edges     = []
        
        if edgeType is not None:
            for edge in self.edgesAll:
                if edge['type'] == edgeType:
                    values.append(edge[prop])   
                    edges.append(edge)
        else:
            for edge in self.edgesAll:
                if edge['type'] == edgeType:
                    values.append(edge[prop])   
                    edges.append(edge)
        return values, edges

    def _old1_getEdgeValues(self, prop, edgeType=0, mode="OUT"):
        values = [] 
        edges     = []
        eList = self.graph.incident(self.nID,mode)

        if edgeType is not None:
            for edge in eList:
                if self.graph.es[edge]['type'] == edgeType:
                    values.append(self.graph.es[edge][prop])   
                    edges.append(edge)
        else:
            for edge in eList:
                values.append(self.graph.es[edge][prop])        
                edges.append(edge)
        return values, edges
    
    def getConnNodes(self, nodeType=0, mode='out'):
        if mode is None:
            neigbours = self.node.neighbors()
        else:
            neigbours = self.node.neighbors(mode)
    
        return [x for x in neigbours if x['type'] == nodeType]

    def getConnNodeIDs(self, nodeType=0, mode='out'):
        if mode is None:
            neigbours = self.node.neighbors()
        else:
            neigbours = self.node.neighbors(mode)
    
        return [x['name'] for x in neigbours if x['type'] == nodeType]
        
    def getConnNodeValues(self, prop, nodeType=0, mode='out'):
        nodeList = self.node.neighbors(mode)
        neighbourIDs     = list()
        values          = list()

        for node in nodeList:
            if node['type'] == nodeType:
                neighbourIDs.append(node['name'])   
                values.append(node[prop])
        
        return values, neighbourIDs
    
    def _old2_getConnNodeValues(self, prop, nodeType=None, mode="OUT"):
        neighIDs = self.graph.neighborhood(self.nID)
        
           
        if nodeType is not None:
            neigbors = self.graph.vs[neighIDs].select(type=nodeType)
        else:
            neigbors = self.graph.vs[neighIDs]
            
        values = neigbors[prop]
        
        return values, neighIDs
    

    
    def _old1_getConnNodeValues(self, prop, edgeType=None, mode="OUT"):
        neigbours = [] 
        edges     = []
        eList = self.graph.incident(self.nID,mode)

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
        
        # dict of types
        self.graph.nodeTypes    = dict()
        self.graph.edgeTypes    = dict()
        
        # inactive is used to virtually remove nodes
        self.registerNodeType('inactiv')
        self.registerEdgeType('inactiv')
        # init of spatial layer if spatial domain is set
        if spatial:
            self.locDict = dict()

        
        self.para     = dict()
        
        
    def setParameters(self, parameterDict):
        for key in parameterDict.keys():
            self.para[key] = parameterDict[key]
            
            
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
    
    
    def iterNodes(self,nodeType):
        if isinstance(nodeType,str):
            nodeType = self.types.index(nodeType)
        nodeList = self.nodeList[nodeType]
        return  [self.entList[i] for i in nodeList]
    
    def randomIterNodes(self,nodeType):
        if isinstance(nodeType,str):
            nodeType = self.types.index(nodeType)
        nodeList = self.nodeList[nodeType]
        shuffled_list = sorted(nodeList, key=lambda x: np.random.random())
        return [self.entList[i] for i in shuffled_list]
    
    def iterNodeAndID(self,nodeType):
        if isinstance(nodeType,str):
            nodeType = self.types.index(nodeType)
        nodeList = self.nodeList[nodeType]
        return  [(self.entList[i], i) for i in nodeList]

    def randomIterNodeAndID(self,nodeType):
        if isinstance(nodeType,str):
            nodeType = self.types.index(nodeType)
        nodeList = self.nodeList[nodeType]
        shuffled_list = sorted(nodeList, key=lambda x: np.random.random())
        return  [(self.entList[i], i) for i in shuffled_list]

    def iterEdges(self, edgeType):
        for i in range(self.graph.ecount()):
            if self.graph.es[i]['type'] == edgeType:
                yield self.graph.es[i]
    
    
    def registerLocation(self, location):
        
        self.locDict[location.x,location.y] = location
        
    def getNode(self,nodeID):
        return self.entDict[nodeID]
    
    def nodeDegreeHist(self,nodeType,nbars=20):
        import matplotlib.pyplot as plt
        plt.hist(self.graph.vs[self.nodeList[nodeType]].degree().nbars)
        
 
    def computeConnectionList(self,radius=1, weightingFunc = lambda x,y : 1/((x**2 +y**2)**.5), ownWeight =2):
        """
        Method for easy computing a connections list of regular grids
        """
        connList = []  
        
        intRad = int(radius)
        for x in range(-intRad,intRad+1):
            for y in range(-intRad,intRad+1):
                if (x**2 +y**2)**.5 < radius:
                    if x**2 +y**2 > 0:
                        weig  = weightingFunc(x,y)
                    else:
                        weig = ownWeight
                    connList.append((x,y,weig))
        return connList
    
    def initSpatialLayerNew(self, nodeArray, connList, LocClassObject=Location):
        """
        Auiliary function to contruct a simple connected layer of spatial locations.
        Use with  the previously generated connection list (see computeConnnectionList)
        
        """
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
        spNodeID = self.graph.vs.select(pos=[x,y],type=0).indices[0]
        eIDSeq = self.graph.es.select(_source=spNodeID,type=1).indices
        cellAgents= []            
        for edge in self.graph.es[eIDSeq]:
            cellAgents.append(edge.target)
        return cellAgents
#            return synEarth.graph.vs.select(pos=[x,y],type=agType).indices
    
    def getNodeType(self, typeStr):
        if typeStr not in self.graph.nodeTypes.values():
            self.registerNodeType(typeStr)
                
        for iType, liTyp in enumerate(self.graph.nodeTypes.iteritems()):
            if liTyp == typeStr :
                break
        
        return iType
    
    def registerNodeType(self, typeStr):
        iType = len(self.graph.nodeTypes)
        self.graph.nodeTypes[iType] = typeStr
        self.nodeList[iType]= list()
    
    def registerEdgeType(self, typeStr):
        iType = len(self.graph.edgeTypes)
        self.graph.edgeTypes[iType] = typeStr
        
    def dequeueEdges(self):
        eStart = self.graph.ecount()
        self.graph.add_edges(self.graph.edgeQueue[0])
        self.graph.es[eStart:]['type'] = self.graph.edgeQueue[1]
        
        for node in self.entList:
            node.updateEdges()

    def view(self,filename = 'none', vertexProp='none'):
        import matplotlib.cm as cm
        
        
        # Nodes        
        if vertexProp=='none':
            colors = iter(cm.rainbow(np.linspace(0, 1, len(self.graph.nodeTypes)+1)))   
            colorDictNode = {}
            for i in range(len(self.graph.nodeTypes)+1):
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
        colors = iter(cm.rainbow(np.linspace(0, 1, len(self.graph.edgeTypes))))              
        colorDictEdge = {}  
        for i in range(len(self.graph.edgeTypes)):
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
    # %% Agent file ###
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
    
    def finalizeAgentFile(self):
        # saving agent files
        from class_auxiliary import saveObj
        for typ in self.agentRec.keys():
            if self.para['writeNPY']:
                np.save(self.para['outPath'] + '/agentFile_type' + str(typ), self.agentRec[typ].recordNPY, allow_pickle=True)
                saveObj(self.agentRec[typ].attrIdx, (self.para['outPath'] + '/attributeList_type' + str(typ)))
            if self.para['writeCSV']:
                self.agentRec[typ].csvFile.close()
########################################################################################
#  END OF CLASS DESCRIPTION
########################################################################################

