# -*- coding: utf-8 -*-
"""
@Time    : 2022-12-06 10:06
@Author  : zzy
@File    : main.py

"""
from gensim.models import Word2Vec
from torch_geometric.datasets import DBLP
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier


from deepwalk.model import DeepWalk

if __name__ == '__main__':
    # Step 1: load data
    he_data = DBLP(root='../data/DBLP')[0]
    ho_data = he_data.to_homogeneous()

    # Step 2: load model
    walker = DeepWalk(ho_data)
    model = Word2Vec(
        vector_size=6, window=2,
        sg=1, hs=0, alpha=0.03,
        min_alpha=0.0007, seed=14
    )
    classifier = GradientBoostingClassifier(random_state=21)

    # Step 3: train model
    walks = walker.generate_walks(nodes=[i for i in range(ho_data.num_nodes)], walk_per_node=1, walk_length=5)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=10, report_delay=1)

    start_idx, cnt = {}, 0
    for nt in he_data.metadata()[0]:
        start_idx[nt] = cnt
        cnt += he_data[nt].num_nodes

    train_x = [model.wv[idx+start_idx['author']] for idx, e in enumerate(he_data['author'].train_mask) if e]
    train_y = [he_data['author'].y[idx] for idx, e in enumerate(he_data['author'].train_mask) if e]

    test_x = [model.wv[idx + start_idx['author']] for idx, e in enumerate(he_data['author'].test_mask) if e]
    y_true = [he_data['author'].y[idx] for idx, e in enumerate(he_data['author'].test_mask) if e]

    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(test_x)

    # Step 4: eval model
    print(classification_report(y_true, y_pred, digits=4))

    # Step 5: save model
    save_flag = False
    if save_flag:
        model.save("../save/DeepWalk_Sg.model")