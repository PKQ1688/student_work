#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/5/14 00:13
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/5/14 00:13
# @File         : node2embed.py
import math
# from collections import defaultdict
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gensim.models import Word2Vec


def LoadGraphFromTxtFile(graph_txt_file):
    matrix = np.loadtxt(graph_txt_file, dtype=int)
    graph = nx.from_numpy_array(matrix)
    # print(graph[1][0]['weight'])
    return graph


def PlotGraph(graphs):
    fig = plt.figure(figsize=(20, 20))
    n_graphs = len(graphs)
    rows = 2
    cols = math.ceil(n_graphs / rows)
    for i in range(n_graphs):
        ax = fig.add_subplot(rows, cols, i + 1)
        nx.draw(graphs[i])
    plt.show()


def SaveGraph(graph, save_file):
    As = nx.adjacency_matrix(graph)
    # 转化成二维数组形式的矩阵
    A = As.todense()
    np.savetxt(save_file, A, fmt='%d', delimiter=' ')


def conduct_random_walks(G, num_walks=10, walk_length=40):
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        node = random.choice(nodes)
        walk = [node]

        for _ in range(walk_length - 1):
            neighbors = list(G.neighbors(node))
            if not neighbors:
                break
            node = random.choice(neighbors)
            walk.append(node)
        walks.append(walk)

    return walks


def cosine_similarity(vector_a, vector_b):
    """
    计算两个向量的余弦相似度。

    参数:
    vector_a, vector_b: 两个numpy数组，表示要比较的向量。

    返回:
    相似度值，范围在[-1, 1]之间。
    """
    # 确保输入是numpy数组
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    # 计算点积
    dot_product = np.dot(vector_a, vector_b)

    # 计算各自的模长（欧几里得范数）
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    # 避免除以零错误
    if norm_a == 0 or norm_b == 0:
        return 0

    # 计算余弦相似度
    similarity = dot_product / (norm_a * norm_b)

    return similarity


if __name__ == '__main__':
    graph = LoadGraphFromTxtFile('COCO-graph.txt')
    # 执行随机游走]
    walks = conduct_random_walks(graph)
    # 将随机游走转换为Word2Vec模型的句子形式
    sentences = [list(map(str, walk)) for walk in walks]
    # 使用gensim的Word2Vec训练模型
    model = Word2Vec(sentences, vector_size=128, window=5, min_count=0, sg=1, workers=4)

    # print(graph.nodes)
    print(model.wv['1'])
    # PlotGraph([graph])
    res = cosine_similarity(model.wv['1'], model.wv['2'])
    print(res)
