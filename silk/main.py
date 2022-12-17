# -*- coding: utf-8 -*-
"""
@Time    : 2022-12-06 16:28
@Author  : zzy
@File    : main.py

"""
import time
import numpy as np

from gensim.models import Word2Vec
from torch_geometric.datasets import DBLP

from silk.model import SILK


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


if __name__ == '__main__':
    # Step 1: load data
    he_data = DBLP(root='../data/DBLP')[0]
    ho_data = he_data.to_homogeneous()

    # Step 2: load model
    walker = SILK(ho_data)
    model = Word2Vec(
        vector_size=6, window=2,
        sg=1, hs=0, alpha=0.03,
        min_alpha=0.0007, seed=14
    )
    guidance_matrix = dict()
    num_type = len(he_data.node_types)
    for node in range(ho_data.num_nodes):
        guidance_matrix[node] = [1.0/(num_type-1)] * num_type
        guidance_matrix[node][ho_data.node_type[node]] = 0

    # Step 3: train model
    t1 = time.time()
    epochs = 5
    ntype = 0
    for epoch in range(epochs):
        print('epoch num:', epoch)
        walks, sequences, paths = \
            walker.generate_walks(ntype, guidance_matrix, m=1, alpha=0.5, walk_per_node=1, walk_length=5)
        pairs = walker.generate_pairs(walks=sequences, paths=paths, walk_per_node=1, window_size=2)
        print(sequences)
        print(pairs)

        model.build_vocab(walks)
        model.train(walks, total_examples=model.corpus_count, epochs=5, report_delay=1)
        vectors = model.wv

        new_guidance_matrix = dict()
        matrix_dis = 0
        for node in range(ho_data.num_nodes):
            dis = {e: -1 for e in range(num_type)}
            neis = {e: [] for e in range(num_type)}

            for pair in pairs:
                if pair[0] == node:
                    neis[pair[2]].extend([vectors[pair[1]]])

            for t, nei in neis.items():
                if len(nei) != 0:
                    vec = [np.mean(nei, axis=0)]
                    dis[t] = np.dot(vec, vectors[node]) / (np.linalg.norm(vec) * np.linalg.norm(vectors[node]))

            all_dic = []
            for _, e in dis.items():
                all_dic.append(float(e))
            new_guidance_matrix[node] = list(softmax(all_dic))

        if new_guidance_matrix == guidance_matrix:
            break
        else:
            guidance_matrix = new_guidance_matrix

        print(guidance_matrix)
    t2 = time.time()
    print('time:', t2 - t1)

    # Step 4: eval model

    # Step 5: save model
    model.save("../save/SILK_Sg.model")