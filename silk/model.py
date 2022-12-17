# -*- coding: utf-8 -*-
"""
@Time    : 2022-12-06 10:35
@Author  : zzy
@File    : model.py

"""
import math
import random

from collections import deque, Counter
from torch_sparse import SparseTensor
from torch_geometric.utils import degree


class SILK:
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
        self.node_type = data.node_type

    def _get_neighbors(self, node):
        return self.col[self.rowptr[node]:self.rowptr[node + 1]]

    def _walk_from_start(self, start, alpha, ntype, guidance_matrix, m, walk_length):
        walk = [start]
        his_types = deque([])

        while len(walk) < walk_length:
            cur = walk[-1]
            candidates = []

            neighbors = self._get_neighbors(cur)
            if self.node_type[cur] == ntype:
                # 寻找历史列表出现最多次的类型
                count_dic = Counter(his_types).most_common()
                arr_same = [e[0] for e in count_dic if e[1] == count_dic[0][1]]

                # 寻找 a_same 并更新 length
                length = 0
                a_same = -1
                for e in reversed(his_types):
                    if e in arr_same:
                        length += 1
                        if a_same == -1:
                            a_same = e

                probability = 1 - math.pow(alpha, length)
                r = random.uniform(0, 1)

                select_type = list(set([self.node_type[e] for e in neighbors if self.node_type[e] != ntype]))
                if r < probability:
                    # condition one
                    ret = [e for e in select_type if e not in arr_same]
                    if len(ret) == 0:
                        # condition two
                            ret = [e for e in select_type if e != a_same]
                            if len(ret) == 0:
                                # condition three
                                ret = [a_same]

                    select_type = ret

                for t, _ in enumerate(sorted(guidance_matrix[cur], reverse=True)):
                    if t in select_type:
                        candidates.extend([e for e in neighbors if self.node_type[e] == t])
                        break

                node_degree = degree(self.data.edge_index[0], num_nodes=self.num_nodes)
                weights = []
                for candidate in candidates:
                    weights.append((candidate, node_degree[candidate]))
                weights = sorted(weights, key=lambda x: x[1], reverse=True)
                candidates = []
                for i in range(int(len(weights) / 3) + 1):
                    candidates.extend([weights[i][0]])

            else:
                candidates.extend(neighbors)

            if candidates:
                next_node = random.choice(candidates)
                walk.append(next_node)
                his_types.append(self.node_type[cur])
                if len(his_types) > m:
                    his_types.popleft()
            else:
                break

        return walk

    def generate_walks(self, ntype, guidance_matrix, m=1, alpha=0.5, walk_per_node=50, walk_length=5):
        walks, sequences, paths = [], [], []
        nodes = [n for n in range(self.num_nodes)]

        for cnt in range(walk_per_node):
            random.shuffle(nodes)

            for node in nodes:
                walk = self._walk_from_start(node, alpha, ntype, guidance_matrix, m, walk_length)
                walks.append(walk)

                walkn, pathn = [], []
                for n in walk:
                    if self.node_type[n] == ntype:
                        walkn.append(n)
                    else:
                        pathn.append(self.node_type[n])

                sequences.append(walkn)
                paths.append(pathn)

        return walks, sequences, paths

    def generate_pairs(self, walks, paths, walk_per_node, window_size):
        pairs = []
        skip_window = window_size // 2

        for index in range(walk_per_node):
            for i in range(len(walks[index])):
                for j in range(1, skip_window + 1):
                    if i - j >= 0:
                        path = paths[index]
                        words = path[i - j:1]
                        t = max(words, key=words.count)
                        pairs.append((walks[index][i], walks[index][i - j], t))
                    if i + j < len(walks[index]):
                        path = paths[index]
                        words = path[i:i + j]
                        t = max(words, key=words.count)
                        pairs.append((walks[index][i], walks[index][i + j], t))
        return pairs