
import torch

from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.utils import dropout_edge, to_dense_adj, from_scipy_sparse_matrix
import copy
from utils import set_requires_grad, EMA, update_moving_average, init_weights, Namespace, parse_args
from classic_pgo import class_PGO
from data import read_data
from torch_geometric.loader import DataLoader
from torch.nn.functional import normalize
import time
from datetime import datetime
from tensorboardX import SummaryWriter
device = "cuda" if torch.cuda.is_available() else "cpu"

def currentTime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def training(args, layer):
    # Read the training data
    filename = ['data_posegraph/data/wrong_loops/graph_data_before_91_646.msg', 
        'data_posegraph/data/wrong_loops/graph_data_before_229_823.msg', 
        'data_posegraph/data/wrong_loops/graph_data_before_245_562.msg',
        'data_posegraph/data/wrong_loops/graph_data_before_276_910.msg']
    data_list = []
    edge_list = []
    for item in filename:
        data, edge = read_data(item)
        data_list.append(data)
        edge_list.append(edge)
    
    batch_size = 1
    loader = DataLoader(data_list, batch_size=batch_size)

    filename = ['data_posegraph/data/correct_loops/graph_data_before_224_722.msg']
    test_data_list = []
    test_edge_list = []
    for item in filename:
        data, edge = read_data(item)
        test_data_list.append(data)
        test_edge_list.append(edge)
    
    batch_size = 1
    test_loader = DataLoader(test_data_list, batch_size=batch_size)

    model = Self_PGO(layer, args).to(device)
    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay= 1e-5)
    writer = SummaryWriter('./log_teacher_edge_drop')

    # Start training
    print("Training Start!")
    model.train()
    for epoch in range(args.epochs):
        for idx, batch_data in enumerate(loader):
            batch_data.to(device)
            _, loss = model(x= batch_data.x, edge_index = batch_data.edge_index, edges = edge_list[idx], edge_weight = batch_data.edge_attr, epoch = epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_moving_average()
            st = '[{}][Epoch {}/{}] Loss: {:.4f}'.format(currentTime(), epoch, args.epochs, loss.item())
            print(st)
            writer.add_scalar("Loss", loss.item(), epoch)
        
        if (epoch) % 5 == 0:
            MSE_error = 0
            for idx, batch_data in enumerate(test_loader):
                data, _ = model(x= batch_data.x, edge_index = batch_data.edge_index, edges = test_edge_list[idx], edge_weight = batch_data.edge_attr)
                loss = nn.MSELoss()
                MSE_error += loss(data, batch_data.y)
            print("The current epoch is: ", epoch, " and evaluate loss: ", MSE_error.item())
            writer.add_scalar("Evaluate Loss", MSE_error.item(), epoch)
    writer.close()
    print("\nTraining Done!")



class Edgedropping(nn.Module):
    def __init__(self, tau = 0.05):
        super(Edgedropping, self).__init__()
        in_channels = 14
        middle_channels = 7
        out_channels = 1
        self.tau = tau
        self.a = -0.1
        self.b = 1.1
        self.random = torch.rand(1)
        self.random = torch.log(self.random).to(device) - torch.log(1 - self.random).to(device)
        self.mlp_weight = Linear(14, in_channels)

        self.mlp = Seq(Linear(in_channels, middle_channels),
                       ReLU(),
                       Linear(middle_channels, out_channels))
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, edge_index, edge_weight = None):
        temp = edge_weight
        edge_weight = self.mlp(edge_weight.reshape(-1,14))
        edge_weight = self.sigmoid((self.random + edge_weight)/self.tau)
        edge_weight = edge_weight.reshape(-1,)
        v_min, v_max = edge_weight.min(), edge_weight.max()
        edge_weight = (edge_weight - v_min)/(v_max - v_min)*(self.b - self.a) + self.a

        edge_mask = edge_weight > 0
        edge_weight = temp[:, edge_mask]
        edge_index = edge_index[:, edge_mask]
        return edge_index, edge_weight
    


class Encoder(nn.Module):

    def __init__(self, layer_config, dropout=None, project=False, **kwargs):
        super().__init__()
        self.stacked_gnn = nn.ModuleList([GCNConv(layer_config[i - 1], layer_config[i]) for i in range(1, len(layer_config))])
        self.stacked_prelus = nn.ModuleList([nn.PReLU() for _ in range(1, len(layer_config))])
        self.edge_drop = nn.ModuleList([Edgedropping() for _ in range(1, len(layer_config))])


    def forward(self, x, edge_index, edge_weight=None):
        for i, gnn in enumerate(self.stacked_gnn):
          #  edge_index, _ = dropout_edge(edge_index, p = 0.1)
            edge_index, edge_weight = self.edge_drop[i](x, edge_index, edge_weight)
            x = gnn(x, edge_index, edge_weight=None)
            x = self.stacked_prelus[i](x)

        return x


class Self_PGO(torch.nn.Module):
    def __init__(self, layer_config, args, **kwargs):
        super().__init__()
        self.student_encoder = Encoder(layer_config=layer_config, dropout=args.dropout, **kwargs)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(args.mad, args.epochs)


        rep_dim = layer_config[-1]
        self.student_predictor = nn.Sequential(nn.Linear(rep_dim, args.pred_hid), nn.BatchNorm1d(args.pred_hid), nn.PReLU(), nn.Linear(args.pred_hid, rep_dim))
        self.student_predictor.apply(init_weights)


    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def forward(self, x, edge_index, edges, edge_weight=None, epoch=None):
        time1 = time.perf_counter()
        student = self.student_encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
    
        pred = self.student_predictor(student)
        time2 = time.perf_counter()
        print("The inference time: ", time2 - time1)
        
        if epoch is not None and epoch >= 100:
            with torch.no_grad():
                teacher = self.teacher_encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        else:
            teacher = x
        
        
        target = class_PGO(teacher, edges)

        loss = nn.MSELoss()

        return pred, loss(pred, torch.tensor(target, dtype=torch.float).to(device))




if __name__ == '__main__':
    args = Namespace( embedder='AFGRL',
                            dataset='wikics',
                            task='node',
                            layers='[1024]',
                            pred_hid=2048,
                            lr=0.001,
                            topk = 8,
                            epochs = 100
                            )
    args, unknown = parse_args()
    layer = [7]*5
    training(args, layer)
