# -*- coding: utf-8 -*-
"""
@Time    : 2022-12-08 9:21
@Author  : zzy
@File    : main.py

"""
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from torch_geometric.datasets import DBLP

from metapath2vec.model import Metapath2Vec

if __name__ == '__main__':
    # Step 1: load data
    he_data = DBLP(root='../data/DBLP')[0]
    ho_data = he_data.to_homogeneous()
    node_type_name = he_data.metadata()[0]
    metapaths = [['author', 'paper', 'author'],
                 ['author', 'paper', 'conference', 'paper', 'author'],
                 ['term', 'paper', 'term'],
                 ['paper', 'conference', 'paper']]

    # Step 2: load model
    walker = Metapath2Vec(ho_data, node_type_name)
    model = Word2Vec(
        vector_size=6, window=2,
        sg=1, hs=0, alpha=0.03,
        min_alpha=0.0007, seed=14
    )

    # Step 3: train model
    walks = walker.generate_walks(nodes=[n for n in range(ho_data.num_nodes)], metapaths=metapaths,
                                  walk_per_node=50, walk_length=5, max_iters=50)
    print(len(walks))
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=10, report_delay=1)
    # Step 4: eval model

    # Step 5: save model
    model.save("../save/Metapath2Vec_Sg.model")


