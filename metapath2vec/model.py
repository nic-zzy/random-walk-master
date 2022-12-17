# -*- coding: utf-8 -*-
"""
@Time    : 2022-12-05 9:50
@Author  : zzy
@File    : model.py

"""
import re
import random

import numpy as np
from torch_sparse import SparseTensor


class Metapath2Vec:
    def __init__(self, data, node_type_name):
        self.data = data.cpu()
        self.adj_t = SparseTensor(
            row=data.edge_index[0],
            col=data.edge_index[1],
            sparse_sizes=(data.num_nodes, data.num_nodes)
        )
        self.rowptr = self.adj_t.storage.rowptr().numpy()
        self.col = self.adj_t.storage.col().numpy()
        self.node_type = data.node_type
        self.node_type_name = node_type_name

    def _get_neighbors(self, node):
        return self.col[self.rowptr[node]:self.rowptr[node + 1]]

    def _match_metapath(self, walk, metapaths):

        node_types = [self.node_type_name[self.node_type[node].item()] for node in walk]

        for metapath in metapaths:
            metapath_node_type, encoded_metapath = np.unique(metapath, return_inverse=True)

            # skip empty metapaths
            if len(encoded_metapath) == 0:
                continue

            # skip diff is not empty
            diff = list(set(node_types).difference(set(metapath_node_type)))
            if len(diff) != 0:
                continue

            encoded_walk = ''.join([str(np.where(metapath_node_type == x)[0][0]) for x in node_types])
            metapath_pattern = '(?=(' + ''.join([str(nt) for nt in encoded_metapath]) + '))'

            n_matched = len(re.findall(metapath_pattern, encoded_walk))
            n_expected = (len(encoded_walk) - 1) / (len(encoded_metapath) - 1)

            if n_matched > 0 and n_matched == n_expected:
                return True

        return False

    def _walk_from_start(self, start, metapaths, walk_length, max_iters):
        iters = 0
        walk = None

        while iters < max_iters:
            t_walk = [start]
            while len(t_walk) < walk_length:
                cur = t_walk[-1]
                neighbors = self._get_neighbors(cur)
                if len(neighbors) == 0:
                    break
                next_node = np.random.choice(neighbors)
                t_walk.append(next_node)

            if (metapaths is None) or self._match_metapath(t_walk, metapaths):
                walk = t_walk
                break

            iters += 1
        return walk

    def generate_walks(self, nodes, metapaths, walk_per_node=50, walk_length=5, max_iters=50):
        walks = []

        for cnt in range(walk_per_node):
            random.shuffle(nodes)

            for node in nodes:
                walk = self._walk_from_start(node, metapaths, walk_length, max_iters)
                if walk:
                    walks.extend(walk)

        # remove duplicate walks
        walks = list(set(map(tuple, walks)))

        return walks
