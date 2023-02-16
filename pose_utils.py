import numpy as np
from pyquaternion import Quaternion
import g2o
import time
import torch
import random
from math import radians
import theseus as th
from typing import List, Optional, Tuple, Union, cast
import theseus.utils.examples as theg
import torch.nn as nn
class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(id)
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_edge(self, vertices, measurement, 
            information=np.identity(6),
            robust_kernel=None):

        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        if measurement is None:
            measurement = (
                edge.vertex(0).estimate().inverse() * 
                edge.vertex(1).estimate())
            '''
            measurement = (
                edge.vertex(0).estimate().inverse() * 
                edge.vertex(1).estimate())
            '''
        else:
            rot = g2o.Quaternion(np.array(measurement[3:], dtype=np.float64))
            t = g2o.Isometry3d(rot, measurement[0:3])
            measurement = t
            

        edge.set_measurement(measurement) 
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id).estimate()


def class_PGO(node_features, edges, edges_attributes = None):
    optimizer = PoseGraphOptimization()
    node_features = node_features
    for i, item in enumerate(node_features):
        quat = g2o.Quaternion(np.array(item[3:], dtype=np.float64))
        t = g2o.Isometry3d(quat, item[0:3])
        optimizer.add_vertex(id = i, pose = t)
    
    for item in edges:
        optimizer.add_edge(item, None)
    '''
    for item, relative_measurement in zip(edges, edges_attributes):
        if edges_attributes is None:
            optimizer.add_edge(item, None)
        else:
            optimizer.add_edge(item, relative_measurement)
    '''

    time1 = time.perf_counter()
    optimizer.optimize()
    time2 = time.perf_counter()
    print("The optimizer time is:", time2 - time1)
    optimizer.save('result.g2o')
    result_dict = []
    for i in range(len(node_features)):
        temp = rotationMatrixToQuaternion1(optimizer.get_pose(i).R)
        temp  = Quaternion(temp).normalised
        t = optimizer.get_pose(i).t
        result_dict.append(list(t) + temp.elements.tolist() )
    #might cause problems need to define a loss function
    return result_dict

def Quaternion_add_noise(q):
    noised_quat = Quaternion(axis = [1, 0, 0], angle = radians(5)).normalised
    noised_q = noised_quat*q
    return noised_q.normalised



def rotationMatrixToQuaternion1(m):
    #q0 = qw
    t = np.matrix.trace(m)
    q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if(t > 0):
        t = np.sqrt(t + 1)
        q[3] = 0.5 * t
        t = 0.5/t
        q[0] = (m[2,1] - m[1,2]) * t
        q[1] = (m[0,2] - m[2,0]) * t
        q[2] = (m[1,0] - m[0,1]) * t

    else:
        i = 0
        if (m[1,1] > m[0,0]):
            i = 1
        if (m[2,2] > m[i,i]):
            i = 2
        j = (i+1)%3
        k = (j+1)%3

        t = np.sqrt(m[i,i] - m[j,j] - m[k,k] + 1)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (m[k,j] - m[j,k]) * t
        q[j] = (m[j,i] + m[i,j]) * t
        q[k] = (m[k,i] + m[i,k]) * t

    
    temp = q[3]
    q[3] = q[2]
    q[2] = q[1]
    q[1] = q[0]
    q[0] = temp
    
    return q


def normalize_quaternion(quaternion: torch.Tensor, eps: float = 1e-12):

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}.".format(quaternion.shape)
        )
    return torch.nn.functional.normalize(quaternion, p=2, dim=-1, eps=eps)




def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(quaternion.shape)
        )

    # Normalize the input quaternion
    #quaternion = normalize_quaternion(quaternion)

    x, y, z, w = torch.chunk(quaternion, chunks=4, dim=-1)

    # Compute the actual conversion
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    one = torch.tensor(1.0)

    matrix = torch.stack(
        [
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ],
        dim=-1,
    ).view(-1, 3, 3)

    if len(quaternion.shape) == 1:
        matrix = torch.squeeze(matrix, dim=0)
    return matrix


class PoseGraphEdge:
    def __init__(
        self,
        i: int,
        j: int,
        relative_pose: Union[th.SE2, th.SE3],
        weight: Optional[th.DiagonalCostWeight] = None,
    ):
        self.i = i
        self.j = j
        self.relative_pose = relative_pose
        self.weight = weight

    def to(self, *args, **kwargs):
        self.weight.to(*args, **kwargs)
        self.relative_pose.to(*args, **kwargs)




def differentiable_PGO(node_features, edges, edges_attributes = None):
    vectice_features = torch.tensor(node_features)
    '''
    trans = vectice_features[:,4:]
    rots = vectice_features[:,0:4]
    vectice_features = torch.cat((trans, rots), dim=1)
    '''
    vectices = []
    for i in range(vectice_features.shape[0]):
        x_y_z_quat = vectice_features[i, :].to(torch.float64)
        vectices.append(th.SE3(x_y_z_quat, name="VERTEX_SE3__{}".format(i)))

    weight = th.Variable(torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]).to(torch.float64))
    objective = th.Objective()
    objective.to(torch.float64)
    for i, item in enumerate(edges):
        Rij = torch.tensor(edges_attributes[i]).to(torch.float64)
        '''
        Rij_quat = Rij[:4]
        Rij_xyz = Rij[4:]
        Rij = torch.cat((Rij_xyz, Rij_quat))
        '''
        relative_pose = th.SE3(x_y_z_quaternion=Rij, name="EDGE_SE3__{}".format(i))
        cost_func = th.Between(
            vectices[item[0]],
            vectices[item[1]],
            relative_pose,
            th.DiagonalCostWeight(weight),
        )
        objective.add(cost_func)
    
    
    pose_prior = th.Difference(
        var=vectices[0],
        cost_weight=th.ScaleCostWeight(torch.tensor(1e-6, dtype=torch.float64)),
        target=vectices[0].copy(new_name=vectices[0].name + "PRIOR"),
    )

    objective.add(pose_prior)
    

    optimizer = th.GaussNewton(
        objective,
        max_iterations=10,
        step_size=0.5,
        linearization_cls=th.SparseLinearization,
        linear_solver_cls=th.CholmodSparseSolver,
        vectorize=True,
    )

    inputs = {var.name: var.tensor for var in vectices}
    theseus_optim = th.TheseusLayer(optimizer)
    theseus_optim.forward(inputs)

    

    loss = nn.MSELoss()
    updated_node_features = []
    for key in inputs:
        rot = inputs[key][:, :, :3].reshape(3,3).numpy()
        t = inputs[key][:,:,3]
        t = np.reshape(t, (3,))
        q2 = Quaternion(matrix = rot, atol=1e-05, rtol=1e-05).normalised
        updated_node_features.append(t.tolist() + q2.elements.tolist())
  

    print("The MSE between noised data after Diff-LM and true data is: ", 
          loss(torch.tensor(node_features), torch.tensor(updated_node_features)).item())


    return None
    




if __name__ == '__main__':
    q1 = np.array([0.1, 0.2, 0.3, 0.4])
    q1 = Quaternion(q1).normalised
    R = q1.rotation_matrix
    q2 = Quaternion(matrix=R)
    print(q1, q2)
    


