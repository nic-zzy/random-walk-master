# -*- coding: utf-8 -*-
"""
@Time    : 2023-02-19 15:05
@Author  : zzy
@File    : main.py

"""


from gensim.models import Word2Vec
from torch_geometric.datasets import DBLP
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier

from model import Node2Vec, DeepWalk
from utils import load_split_data, set_seed


def init_setting():
    set_seed(seed=14)
    data_name, walker_name = "DBLP", "Node2Vec"

    save_flag = False
    return data_name, walker_name, save_flag


def load_data(data_name):
    data = DBLP(root='data/' + data_name)[0]
    return data.to_homogeneous()


def load_walker(walker_name, data):
    walker = None
    if walker_name == "DeepWalk":
        walker = DeepWalk(data, walk_per_node=1, walk_len=5)
    elif walker_name == "Node2Vec":
        walker = Node2Vec(data, walk_per_node=1, walk_len=5, p=4, q=0.5)

    return walker


if __name__ == '__main__':
    data_name, walker_name, save_flag = init_setting()

    # Step 1: load data
    data = load_data(data_name)

    # Step 2: load model
    walker = load_walker(walker_name, data)
    model = Word2Vec(
        vector_size=6, window=2,
        sg=1, hs=0, alpha=0.03,
        min_alpha=0.0007, seed=14
    )
    classifier = GradientBoostingClassifier(random_state=21)

    # Step 3: train model
    walks = walker.generate_walks(nodes=[i for i in range(data.num_nodes)])

    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=10, report_delay=1)

    train_x, train_y = load_split_data(model, data, 'train_mask')
    classifier.fit(train_x, train_y)

    # Step 4: eval model
    test_x, y_true = load_split_data(model, data, 'test_mask')
    y_pred = classifier.predict(test_x)

    print(classification_report(y_true, y_pred, digits=4))

    # Step 5: save model
    if save_flag:
        model.save("save/" + walker_name + "_Sg.model")