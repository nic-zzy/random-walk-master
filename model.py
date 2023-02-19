# -*- coding: utf-8 -*-
"""
@Time    : 2023-02-19 14:56
@Author  : zzy
@File    : model.py

"""


import random
import numpy as np

from torch_sparse import SparseTensor


class DeepWalk:
    def __init__(self, data, walk_per_node=50, walk_len=5):
        self.walk_len = walk_len
        self.walk_per_node = walk_per_node
        self.adj_t = SparseTensor(
            row=data.edge_index[0],
            col=data.edge_index[1],
            sparse_sizes=(data.num_nodes, data.num_nodes)
        )
        self.rowptr = self.adj_t.storage.rowptr().numpy()
        self.col = self.adj_t.storage.col().numpy()

    def _get_neighbors(self, node):
        return self.col[self.rowptr[node]:self.rowptr[node+1]]

    def _walk_from_start(self, start):
        walk = [start]

        for i in range(self.walk_len - 1):
            cur = walk[-1]
            neighbors = self._get_neighbors(cur)
            if len(neighbors) == 0:
                break

            next_node = random.choice(neighbors)
            walk.append(next_node)

        return walk

    def generate_walks(self, nodes):
        walks = []

        for cnt in range(self.walk_per_node):
            random.shuffle(nodes)
            for node in nodes:
                walk = self._walk_from_start(node)
                walks.append(walk)

        return walks


class Node2Vec:
    def __init__(self, data, walk_per_node=50, walk_len=5, p=4, q=0.5):
        self.p, self.q = p, q
        self.walk_len = walk_len
        self.walk_per_node = walk_per_node
        self.adj_t = SparseTensor(
            row=data.edge_index[0],
            col=data.edge_index[1],
            sparse_sizes=(data.num_nodes, data.num_nodes)
        )
        self.rowptr = self.adj_t.storage.rowptr().numpy()
        self.col = self.adj_t.storage.col().numpy()

    def _get_neighbors(self, node):
        return self.col[self.rowptr[node]:self.rowptr[node+1]]

    def _is_neighbors(self, node_a, node_b):
        node_a_neighbors = self._get_neighbors(node_a)
        if node_b <= node_a_neighbors[-1] and np.searchsorted(node_a_neighbors, node_b, side='right'):
            return True
        return False

    def _sample_neighbors_uniformly(self, node):
        neighbors = self._get_neighbors(node)

        if neighbors.shape[0] == 0:
            return node
        return np.random.choice(neighbors)

    def _walk_from_start(self, start):
        walk = [start, self._sample_neighbors_uniformly(start)]

        max_prob = max(1 / self.p, 1, 1 / self.q)
        probs = [1 / self.p / max_prob, 1 / max_prob, 1/ self.q / max_prob]

        for i in range(self.walk_len - 2):
            while True:
                next_node = self._sample_neighbors_uniformly(walk[-1])
                r = np.random.rand()
                if next_node == walk[i-2]:
                    if r < probs[0]:
                        break
                elif self._is_neighbors(walk[-2], next_node):
                    if r < probs[1]:
                        break
                elif r < probs[2]:
                    break
            walk.append(next_node)

        return walk

    def generate_walks(self, nodes):
        walks = []

        for cnt in range(self.walk_per_node):
            random.shuffle(nodes)
            for node in nodes:
                walk = self._walk_from_start(node)
                walks.append(walk)

        return walks
