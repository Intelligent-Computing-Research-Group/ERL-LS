##
# @file graphExtractor.py
# @author Keren Zhu
# @date 11/16/2019
# @brief The functions and classes for processing the graph


import numpy as np

from numpy import linalg as LA
import numpy as np
import dgl
import torch
import numba as nb

# @nb.jit()

def symmetricLaplacian(abc):
    numNodes = abc.numNodes()
    L = np.zeros((numNodes, numNodes))
    print("numNodes", numNodes)
    for nodeIdx in range(numNodes):
        aigNode = abc.aigNode(nodeIdx)
        degree = float(aigNode.numFanouts())
        if (aigNode.hasFanin0()):
            degree += 1.0
            fanin = aigNode.fanin0()
            L[nodeIdx][fanin] = -1.0
            L[fanin][nodeIdx] = -1.0
        if (aigNode.hasFanin1()):
            degree += 1.0
            fanin = aigNode.fanin1()
            L[nodeIdx][fanin] = -1.0
            L[fanin][nodeIdx] = -1.0
        L[nodeIdx][nodeIdx] = degree
    return L

def symmetricLapalacianEigenValues(abc):
    L = symmetricLaplacian(abc)
    print("L", L)
    eigVals = np.real(LA.eigvals(L))
    print("eigVals", eigVals)
    return eigVals

def extract_dgl_graph(abc):
    numNodes = abc.numNodes()
    G = dgl.DGLGraph()
    G.add_nodes(numNodes)
    features = torch.zeros(numNodes, 6)
    for nodeIdx in range(numNodes):
        aigNode = abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()
        features[nodeIdx][nodeType] = 1.0
        if (aigNode.hasFanin0()):
            fanin = aigNode.fanin0()
            G.add_edge(fanin, nodeIdx)
        if (aigNode.hasFanin1()):
            fanin = aigNode.fanin1()
            G.add_edge(fanin, nodeIdx)
    G.ndata['feat'] = torch.tensor(features)
    return G

def extract_dgl_graph_mig(mtl):
    numNodes = mtl.numNodes()
    #print("numNodes in mig:", numNodes)
    G = dgl.DGLGraph()
    G.add_nodes(numNodes)
    features = torch.zeros(numNodes, 8)
    for nodeIdx in range(numNodes):
        migNode = mtl.migNode(nodeIdx)
        #aigNode = mtl.aigNode(nodeIdx)
        nodeType = migNode.nodeType()
        features[nodeIdx][nodeType] = 1.0
        if (migNode.hasFanin0()):
            fanin = migNode.fanin0()
            G.add_edge(fanin, nodeIdx)
        if (migNode.hasFanin1()):
            fanin = migNode.fanin1()
            G.add_edge(fanin, nodeIdx)
    G.ndata['feat'] = torch.tensor(features)
    return G

def extract_dgl_graph_xmg(mtl):
    numNodes = mtl.xmg_numNodes()
    # print("numNodes in xmg:", numNodes)
    G = dgl.DGLGraph()
    G.add_nodes(numNodes)
    features = torch.zeros(numNodes, 10)
    # print("features",features)
    for nodeIdx in range(numNodes):
        xmgNode = mtl.xmgNode(nodeIdx)
        nodeType = xmgNode.nodeType()
        features[nodeIdx][nodeType] = 1.0
        if (xmgNode.hasFanin0()):
            fanin = xmgNode.fanin0()
            G.add_edge(fanin, nodeIdx)
        if (xmgNode.hasFanin1()):
            fanin = xmgNode.fanin1()
            G.add_edge(fanin, nodeIdx)
        if (xmgNode.hasFanin2()):
            fanin = xmgNode.fanin2()
            G.add_edge(fanin, nodeIdx)
    G.ndata['feat'] = torch.tensor(features)
    return G
