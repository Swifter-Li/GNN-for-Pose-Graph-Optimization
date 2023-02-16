# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pathlib

import hydra
import torch
from scipy.io import savemat

import theseus as th
import theseus.utils.examples as theg

from typing import List, Optional, Tuple, Union, cast

import numpy as np
# To run this example, you will need the cube datasets available at
# https://dl.fbaipublicfiles.com/theseus/pose_graph_data.tar.gz
#
# The steps below should let you run the example.
# From the root project folder do:
#   mkdir datasets
#   cd datasets
#   cp your/path/pose_graph_data.tar.gz .
#   tar -xzvf pose_graph_data.tar.gz
#   cd ..
#   python examples/pose_graph_benchmark.py

# Logger
log = logging.getLogger(__name__)


DATASET_DIR = pathlib.Path.cwd() / "datasets" / "pose_graph"

# This function reads a file in g2o formate and returns the number of of poses, initial
# values and edges.
# g2O format: https://github.com/RainerKuemmerle/g2o/wiki/File-format-slam-3d

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


def read_3D_g2o_file(
    path: str, dtype: Optional[torch.dtype] = None
) -> Tuple[int, List[th.SE3], List[PoseGraphEdge]]:
    with open(path, "r") as file:
        lines = file.readlines()

        num_vertices = 0
        verts = dict()
        edges: List[PoseGraphEdge] = []

        for line in lines:
            tokens = line.split()

            if tokens[0] == "EDGE_SE3:QUAT":
                i = int(tokens[1])
                j = int(tokens[2])

                n = len(edges)

                x_y_z_quat = torch.from_numpy(
                    np.array([tokens[3:10]], dtype=np.float64)
                ).to(dtype)
                x_y_z_quat[:, 3:] /= torch.norm(x_y_z_quat[:, 3:], dim=1)
                x_y_z_quat[:, 3:] = x_y_z_quat[:, [6, 3, 4, 5]]
                relative_pose = th.SE3(
                    x_y_z_quaternion=x_y_z_quat, name="EDGE_SE3__{}".format(n)
                )

                sel = [0, 6, 11, 15, 18, 20]
                weight = th.Variable(
                    torch.from_numpy(np.array(tokens[10:], dtype=np.float64)[sel])
                    .to(dtype)
                    .sqrt()
                    .view(1, -1)
                )

                edges.append(
                    PoseGraphEdge(
                        i,
                        j,
                        relative_pose,
                        th.DiagonalCostWeight(np.identity(6,dtype=np.float64), name="EDGE_WEIGHT__{}".format(n)),
                    )
                )

                num_vertices = max(num_vertices, i)
                num_vertices = max(num_vertices, j)
            elif tokens[0] == "VERTEX_SE3:QUAT":
                i = int(tokens[1])

                x_y_z_quat = torch.from_numpy(
                    np.array([tokens[2:]], dtype=np.float64)
                ).to(dtype)
                x_y_z_quat[:, 3:] /= torch.norm(x_y_z_quat[:, 3:], dim=1)
                x_y_z_quat[:, 3:] = x_y_z_quat[:, [6, 3, 4, 5]]
                verts[i] = x_y_z_quat

                num_vertices = max(num_vertices, i)

        num_vertices += 1

        if len(verts) > 0:
            vertices = [
                th.SE3(x_y_z_quaternion=x_y_z_quat, name="VERTEX_SE3__{}".format(i))
                for i, x_y_z_quat in sorted(verts.items())
            ]
        else:
            vertices = []

        return (num_vertices, vertices, edges)


@hydra.main(config_path="../configs/pose_graph", config_name="pose_graph_benchmark")
def main(cfg):
    dataset_name = cfg.dataset
    file_path = f"{DATASET_DIR}/{dataset_name}_init.g2o"
    dtype = eval(f"torch.{cfg.dtype}")

    _, verts, edges = read_3D_g2o_file(file_path, dtype=torch.float64)
    d = 3

    objective = th.Objective(torch.float64)

    for edge in edges:
        cost_func = th.Between(
            verts[edge.i],
            verts[edge.j],
            edge.relative_pose,
            edge.weight,
        )
        objective.add(cost_func)

    pose_prior = th.Difference(
        var=verts[0],
        cost_weight=th.ScaleCostWeight(torch.tensor(1e-6, dtype=torch.float64)),
        target=verts[0].copy(new_name=verts[0].name + "PRIOR"),
    )
    objective.add(pose_prior)

    objective.to(dtype)
    optimizer = th.GaussNewton(
        objective,
        max_iterations=10,
        step_size=1,
        linearization_cls=th.SparseLinearization,
        linear_solver_cls=th.CholmodSparseSolver,
        vectorize=True,
    )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    inputs = {var.name: var.tensor for var in verts}
    optimizer.objective.update(inputs)

    start_event.record()
    torch.cuda.reset_peak_memory_stats()
    optimizer.optimize(verbose=True)
    end_event.record()

    torch.cuda.synchronize()
    forward_time = start_event.elapsed_time(end_event)
    forward_mem = torch.cuda.max_memory_allocated() / 1048576
    log.info(f"Forward pass took {forward_time} ms.")
    log.info(f"Forward pass used {forward_mem} MBs.")

    results = {}
    results["objective"] = objective.error_metric().detach().cpu().numpy().sum()
    results["R"] = torch.cat(
        [pose.tensor[:, :, :d].detach().cpu() for pose in verts]
    ).numpy()
    results["t"] = torch.cat(
        [pose.tensor[:, :, d].detach().cpu() for pose in verts]
    ).numpy()

    savemat(dataset_name + ".mat", results)


if __name__ == "__main__":
    main()
