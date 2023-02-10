from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T
from pyquaternion import Quaternion
import numpy as np
import random
np.random.seed(0)
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import os.path as osp
import msgpack
from torch_geometric.utils.loop import add_self_loops
from torch_geometric.data import Dataset
from math import radians
import utils
import torch.nn as nn
from pose_utils import class_PGO, Quaternion_add_noise

device = "cuda" if torch.cuda.is_available() else "cpu"
def read_data(filename):

    with open(filename, "rb") as data_file:
        byte_data = data_file.read()
    data_wrong = msgpack.unpackb(byte_data)
    nodes = []
    edges = []
    edge_attributes = []
    for key in list(data_wrong["nodes"].keys()):
        data_wrong["nodes"][int(key)] = data_wrong["nodes"].pop(key)
    for key in range(max(data_wrong["nodes"].keys()) + 1):
        if key not in data_wrong["nodes"].keys():
            nodes.append([[0]*7])
        else:
            nodes.append([data_wrong['nodes'][key]['rotation']+ data_wrong['nodes'][key]['translation']])
            for edge in data_wrong['nodes'][key]['edges']:
                edges.append([key, edge])
    # EDGE Attributes
    for node1, node2 in edges:
        for item1, item2 in zip(nodes[node1], nodes[node2]):
            edge_attributes.append(item1 + item2)
    edge_index = torch.tensor(edges, dtype=torch.long)
    test = torch.tensor(edge_attributes, dtype=torch.float)
    test = test.reshape(-1, len(edge_attributes))
    
    

    x = torch.tensor(nodes, dtype=torch.float)
    x = x.reshape(len(nodes), -1)

    filename = filename.replace('before', 'after')
    with open(filename, "rb") as data_file:
        byte_data = data_file.read()
    data_true = msgpack.unpackb(byte_data)
    nodes = []
    for key in list(data_true["nodes"].keys()):
        data_true["nodes"][int(key)] = data_true["nodes"].pop(key)
    for key in range(max(data_true["nodes"].keys()) + 1):
        if key not in data_true["nodes"].keys():
            nodes.append([[0]*7])
        else:
            nodes.append([data_true['nodes'][key]['rotation']+ data_true['nodes'][key]['translation']])
    
    y = torch.tensor(nodes, dtype=torch.float)
    y = y.reshape(len(nodes), -1)
    loss = nn.MSELoss()
    print("The loss for raw data is: ", loss(x, y).item())

    '''
    temp = class_PGO(x, edges)
    print("The loss for raw data is after PGO: ", loss(torch.tensor(temp, dtype=torch.float), y).item())
    '''

    temp = Data(x=x, y = y, edge_index=edge_index.t().contiguous(), edge_attr= test).to(device)
    assert temp.edge_index.max() < temp.num_nodes
    return temp, edges


def read_data_edge_attributes(filename):

    with open(filename, "rb") as data_file:
        byte_data = data_file.read()
    data_wrong = msgpack.unpackb(byte_data)
    nodes = []
    edges = []
    edge_attributes = []
    for key in list(data_wrong["nodes"].keys()):
        data_wrong["nodes"][int(key)] = data_wrong["nodes"].pop(key)
    for key in range(max(data_wrong["nodes"].keys()) + 1):
        if key not in data_wrong["nodes"].keys():
            nodes.append([0]*7)
        else:
            q = Quaternion(data_wrong['nodes'][key]['rotation']).normalised.elements
            nodes.append(q.tolist()+ data_wrong['nodes'][key]['translation'])
            for edge, edge_attribute in zip(data_wrong['nodes'][key]['edges'], data_wrong['nodes'][key]['edges_attributes']):
                edges.append([key, edge])
                #edge_attributes.append(edge_attribute)
    # EDGE Attributes
    for [node1, node2] in edges:
        temp = nodes[node1]
        q1 = Quaternion(nodes[node1][0:4])
        q2 = Quaternion(nodes[node2][0:4])
        relative_q = q1.inverse*q2
        #relative_q = Quaternion_add_noise(relative_q)
        Ri = q1.rotation_matrix
        ti = np.array(nodes[node1][4:])
        tj = np.array(nodes[node2][4:])
        relative_t = np.dot(np.linalg.inv(Ri), (tj-ti).T)
        # Add translation noise
        
        for i in range(len(relative_t)):
            relative_t[i] = random.gauss(relative_t[i], 0.05)
        
        edge_attributes.append(relative_q.elements.tolist() + relative_t.tolist())
                        

    edge_index = torch.tensor(edges, dtype=torch.long)
    test = torch.tensor(edge_attributes, dtype=torch.float)
    
    noised_nodes = []
    for node in nodes:
        q = Quaternion(node[0:4])
        t = node[4:]
        #q = Quaternion_add_noise(q)
        for i in range(len(t)):
            t[i] = random.gauss(t[i], 0.5)
        noised_nodes.append(q.elements.tolist() + t)


    x = torch.tensor(noised_nodes, dtype=torch.float)
    x = x.reshape(len(noised_nodes), -1)

    y = torch.tensor(nodes, dtype=torch.float)
    y = y.reshape(len(nodes), -1)

    loss = nn.MSELoss()
    print("The loss for raw data is: ", loss(x, y).item())

    transform = T.Compose([T.remove_isolated_nodes.RemoveIsolatedNodes()])
    graph = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr= test.t()).to(device)
    graph = transform(graph)
    assert graph.edge_index.max() < graph.num_nodes
    
    graph_temp = Data(x = y, edge_index=edge_index.t().contiguous()).to(device)
    graph_temp = transform(graph)
    graph.y = graph_temp.x
    
    temp = class_PGO(graph.x.tolist(), graph.edge_index.t().tolist(), edges_attributes= graph.edge_attr.t().tolist())
    print("The loss for raw data is after PGO: ", loss(torch.tensor(temp, dtype=torch.float).to(device), graph.y).item())

    return graph, edges




def read_data_noisy(filename):

    with open(filename, "rb") as data_file:
        byte_data = data_file.read()
    data_wrong = msgpack.unpackb(byte_data)
    nodes = []
    edges = []
    edge_attributes = []
    for key in list(data_wrong["nodes"].keys()):
        data_wrong["nodes"][int(key)] = data_wrong["nodes"].pop(key)
    for key in range(max(data_wrong["nodes"].keys()) + 1):
        if key not in data_wrong["nodes"].keys():
            nodes.append([[0]*7])
        else:
            nodes.append([data_wrong['nodes'][key]['rotation']+ data_wrong['nodes'][key]['translation']])
            for edge in data_wrong['nodes'][key]['edges']:
                edges.append([key, edge])
    # EDGE Attributes
    for node1, node2 in edges:
        for item1, item2 in zip(nodes[node1], nodes[node2]):
            edge_attributes.append(item1 + item2)
    edge_index = torch.tensor(edges, dtype=torch.long)
    test = torch.tensor(edge_attributes, dtype=torch.float)
    test = test.reshape(-1, len(edge_attributes))
    test = test + (0.1**0.5)*torch.randn(test.shape)
    

    x = torch.tensor(nodes, dtype=torch.float)
    x = x.reshape(len(nodes), -1)
    x = x + (0.1**0.5)*torch.randn(x.shape)

    filename = filename.replace('before', 'after')
    with open(filename, "rb") as data_file:
        byte_data = data_file.read()
    data_true = msgpack.unpackb(byte_data)
    nodes = []
    for key in list(data_true["nodes"].keys()):
        data_true["nodes"][int(key)] = data_true["nodes"].pop(key)
    for key in range(max(data_true["nodes"].keys()) + 1):
        if key not in data_true["nodes"].keys():
            nodes.append([[0]*7])
        else:
            nodes.append([data_true['nodes'][key]['rotation']+ data_true['nodes'][key]['translation']])
    
    y = torch.tensor(nodes, dtype=torch.float)
    y = y.reshape(len(nodes), -1)
    loss = nn.MSELoss()
    print("The loss for raw data is: ", loss(x, y).item())

    '''
    temp = class_PGO(x, edges)
    print("The loss for raw data is after PGO: ", loss(torch.tensor(temp, dtype=torch.float), y).item())
    '''

    temp = Data(x=x, y = y, edge_index=edge_index.t().contiguous(), edge_attr= test).to(device)
    assert temp.edge_index.max() < temp.num_nodes
    return temp, edges


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['data_posegraph/data/wrong_loops/graph_data_before_91_646.msg', 
        'data_posegraph/data/wrong_loops/graph_data_before_229_823.msg', 
        'data_posegraph/data/wrong_loops/graph_data_before_245_562.msg',
        'data_posegraph/data/wrong_loops/graph_data_before_276_910.msg']

    @property
    def processed_file_names(self):
        return []

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        for filename in self.raw_file_names():
            data, 

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])