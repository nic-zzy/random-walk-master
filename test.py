# -*- coding: utf-8 -*-
"""
@Time    : 2022-12-05 11:37
@Author  : zzy
@File    : test.py

"""
from collections import deque

from torch_geometric.datasets import DBLP

if __name__ == '__main__':
    # dataset = DBLP(root='data/DBLP')
    # print(dataset[0])

    from collections import Counter
    arr = deque([])
    count_dic = Counter(arr).most_common()
    arr_same = [e[0] for e in count_dic if e[1] == count_dic[0][1]]
    # print(Counter(arr).most_common())
    # print(max(set(arr), key=arr.count))
    print(arr_same)

    for e in reversed(arr):
        print(e)
    print(arr)


# https://www.aminer.cn/billboard/aminernetwork