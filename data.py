from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T

import numpy as np
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
import utils
import torch.nn as nn
from classic_pgo import class_PGO

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