# -*- coding: utf-8 -*-
"""
@Time    : 2022-12-05 16:03
@Author  : zzy
@File    : model.py

"""
import math
import random
from collections import deque

from torch_sparse import SparseTensor


class JUST:
    def __init__(self, data):
        self.data = data.cpu()
        self.adj_t = SparseTensor(
            row=data.edge_index[0],
            col=data.edge_index[1],
            sparse_sizes=(data.num_nodes, data.num_nodes)
        )
        self.rowptr = self.adj_t.storage.rowptr().numpy()
        self.col = self.adj_t.storage.col().numpy()
        self.num_nodes = data.num_nodes

    def _get_neighbors(self, node):
        return self.col[self.rowptr[node]:self.rowptr[node+1]]

    def _walk_from_start(self, start, alpha, m, walk_length):
        walk = [start]
        ho_len = 1
        node_type = self.data.node_type
        qlist = deque([])

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_neighbors = self._get_neighbors(cur)

            qlist.append(node_type[cur])
            if len(qlist) >= m:
                qlist.popleft()

            ho_type = node_type[cur]
            he_type = set([node_type[n] for n in cur_neighbors if node_type[n] != ho_type and node_type[n] not in qlist])

            he_prob = 1 - math.pow(alpha, ho_len)
            r = random.uniform(0, 1)

            next_node_options = []
            if r <= he_prob:
                for ht in he_type:
                    next_node_options.extend([e for e in cur_neighbors if node_type[e] == ht])
                if not next_node_options:
                    next_node_options = [e for e in cur_neighbors if node_type[e] == ho_type]
            else:
                next_node_options = [e for e in cur_neighbors if node_type[e] == ho_type]
                if not next_node_options:
                    for ht in he_type:
                        next_node_options.extend([e for e in cur_neighbors if node_type[e] == ht])

            if not next_node_options:
                break

            next_node = random.choice(next_node_options)
            walk.append(next_node)

            if node_type[next_node] == node_type[cur]:
                ho_len += 1
            else:
                ho_len = 1
        print(walk)
        return walk

    def generate_walks(self, alpha=0.5, m=1, walk_per_node=50, walk_length=5):
        walks = []

        for node in range(self.num_nodes):
            for cnt in range(walk_per_node):
                walk = self._walk_from_start(node, alpha, m, walk_length)
                walks.append(walk)

        return walks


