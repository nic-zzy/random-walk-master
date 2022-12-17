# -*- coding: utf-8 -*-
"""
@Time    : 2022-12-06 9:55
@Author  : zzy
@File    : model.py

"""
import random

from torch_sparse import SparseTensor


class DeepWalk:
    def __init__(self, data):
        self.adj_t = SparseTensor(
            row=data.edge_index[0],
            col=data.edge_index[1],
            sparse_sizes=(data.num_nodes, data.num_nodes)
        )
        self.rowptr = self.adj_t.storage.rowptr().numpy()
        self.col = self.adj_t.storage.col().numpy()

    def _get_neighbors(self, node):
        return self.col[self.rowptr[node]:self.rowptr[node+1]]

    def _walk_from_start(self, start, walk_length):
        walk = [start]

        for i in range(walk_length - 1):
            cur = walk[-1]
            neighbors = self._get_neighbors(cur)
            if len(neighbors) == 0:
                break

            next_node = random.choice(neighbors)
            walk.append(next_node)

        return walk

    def generate_walks(self, nodes, walk_per_node=50, walk_length=5):
        walks = []

        for cnt in range(walk_per_node):
            random.shuffle(nodes)
            for node in nodes:
                walk = self._walk_from_start(node, walk_length)
                walks.append(walk)

        return walks