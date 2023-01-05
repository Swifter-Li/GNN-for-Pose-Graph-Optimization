import numpy as np
import g2o
import time

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

        edge.set_measurement(measurement)  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id).estimate()


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


def class_PGO(node_features, edges):
    optimizer = PoseGraphOptimization()
    node_features = node_features.tolist()
    for i, item in enumerate(node_features):
        quat = g2o.Quaternion(np.array(item[0:4], dtype=np.float64))
        t = g2o.Isometry3d(quat, item[4:7])
        optimizer.add_vertex(id = i, pose = t)
    
    for item in edges:
        optimizer.add_edge(item, None)
    
    time1 = time.perf_counter()
    optimizer.optimize()
    time2 = time.perf_counter()
    print("The optimizer time is:", time2 - time1)
    result_dict = []
    for i in range(len(node_features)):
        temp = rotationMatrixToQuaternion1(optimizer.get_pose(i).R)
        t = optimizer.get_pose(i).t
        result_dict.append(list(temp) + list(t))
    #might cause problems need to define a loss function
    print("the error is: ", np.sum((np.array(node_features) - np.array(result_dict))**2))
    return result_dict





if __name__ == '__main__':
    a = 1


