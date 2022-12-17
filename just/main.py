# -*- coding: utf-8 -*-
"""
@Time    : 2022-12-05 15:43
@Author  : zzy
@File    : main.py

"""
from gensim.models import Word2Vec
from torch_geometric.datasets import DBLP

from just.model import JUST

if __name__ == '__main__':
    # Step 1: load data
    he_data = DBLP(root='../data/DBLP')[0]
    ho_data = he_data.to_homogeneous()

    # Step 2: load model
    walker = JUST(ho_data)
    model = Word2Vec(
        vector_size=6, window=2,
        sg=1, hs=0, alpha=0.03,
        min_alpha=0.0007, seed=14
    )

    # Step 3: train model
    walks = walker.generate_walks(alpha=0.5, m=1, walk_per_node=1, walk_length=5)
    print(len(walks))
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=10, report_delay=1)

    # Step 4: eval model

    # Step 5: save model
    model.save("../save/JUST_Sg.model")
