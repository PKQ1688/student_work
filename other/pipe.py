#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/4/7 14:46
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/4/7 14:46
# @File         : pipe.py
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# import matplotlib.pyplot as plt
col_name = json.load(open("col_name.json"))
# print(col_name)

col_name_list = [item["name"] for item in col_name]
# type_list = [item['type'] for item in col_name]
column_types = {
    item["name"]: pd.Int64Dtype() if item["type"] == "bigint" else item["type"]
    for item in col_name
}


# print(col_name_list)
# print(column_types)


# col_name = []
# 统计某一个文件的数据
def get_data_info(
        file_path="example_data/part-00000-156cad27-9faf-4385-9a6c-05a5b0ee5d28-c000.csv",
):
    data_csv = pd.read_csv(
        file_path, sep=",", header=None, names=col_name_list, dtype=column_types
    )
    # file_name = file_path.split("/")[-1]
    # print(data_csv.head())
    res_json = {}
    # 获取列数
    # column_count = data_csv.shape[1]
    cnt = data_csv.shape[0]

    # print("DataFrame共有", cnt, "行")
    # print("DataFrame共有", column_count, "列")

    df_min = data_csv.min(numeric_only=False).to_dict()
    df_max = data_csv.max(numeric_only=False).to_dict()
    df_nunique = data_csv.nunique(dropna=True).to_dict()

    # print("最小值统计：", df_min)
    # print("最大值统计：", df_max)
    # print("非重复项数量统计：", df_nunique)

    res_json["cnt"] = cnt
    # res_json["column_count"] = column_count
    res_json["df_min"] = df_min
    res_json["df_max"] = df_max
    res_json["cntd"] = df_nunique

    # return {file_name: res_json}
    return res_json


def handle_sql(sql_text):
    # conditions = re.findall(r'\bWHERE\b(.*)', sql_text)[0]
    where_clause_start = sql_text.lower().find("where")
    where_clause_fh = sql_text.find(";", where_clause_start)
    where_clause_group_by = sql_text.find("GROUP BY", where_clause_start)

    where_clause_end = (
        min(where_clause_fh, where_clause_group_by)
        if where_clause_group_by != -1
        else where_clause_fh
    )

    conditions = sql_text[where_clause_start:where_clause_end]

    conditions = re.split(r"\bAND\b", conditions)

    conditions_list = []
    flag = -1
    for index, condition in enumerate(conditions):
        condition = condition.replace("WHERE", "")
        if index == flag:
            continue
        if "BETWEEN" in condition:
            condition += " AND " + conditions[index + 1]
            flag = index + 1
        conditions_list.append(condition.strip())
    return conditions_list


def calculate_sql_file(sql_text, file_path):
    conditions = handle_sql(sql_text)

    file_json = get_data_info(file_path)
    # print(file_json)
    print(conditions)
    res_score = dict()

    for condition in conditions:
        if ">=" in condition or ">" in condition:
            if ">" in condition and ">=" not in condition:
                condition = condition.replace(">", ">=")
            content = condition.split(">=")
            # print(content)
            column_name = content[0].strip().lower()
            if column_name not in ["p_brand", "p_category", "p_mfgr"]:
                value = content[1].strip()
            else:
                value = content[1].strip().replace("'", "").split("#")[1]
            score = (
                    file_json["cnt"]
                    * (int(value) - file_json["df_min"][column_name])
                    / (file_json["df_max"][column_name] - file_json["df_min"][column_name])
            )
            print("> score:", score)
            res_score[column_name] = score

        elif "<=" in condition or "<" in condition:
            if "<" in condition and "<=" not in condition:
                condition = condition.replace("<", "<=")
            content = condition.split("<=")
            # print(content)
            column_name = content[0].strip().lower()
            if column_name not in ["p_brand", "p_category", "p_mfgr"]:
                value = content[1].strip()
            else:
                value = content[1].strip().replace("'", "").split("#")[1]
            score = (
                    file_json["cnt"]
                    * (file_json["df_max"][column_name] - int(value))
                    / (file_json["df_max"][column_name] - file_json["df_min"][column_name])
            )
            print("< score:", score)
            if column_name in res_score:
                score += res_score[column_name]
                res_score[column_name] = file_json["cnt"] - score
            else:
                res_score[column_name] = score
        elif "BETWEEN" in condition:
            content = condition.split("BETWEEN")
            # print(content)
            column_name = content[0].strip().lower()
            value = content[1].strip().split("AND")
            value = [int(i) for i in value]
            score = (
                    file_json["cnt"]
                    * (value[1] - value[0])
                    / (file_json["df_max"][column_name] - file_json["df_min"][column_name])
            )
            print("between score:", score)
            if column_name in res_score:
                res_score[column_name] += score
            else:
                res_score[column_name] = score
        elif "=" in condition:
            # condition = condition.replace("=", "==")
            content = condition.split("=")
            column_name = content[0].strip().lower()
            value = content[1].strip()

            # print(column_name)
            # print(value)

            if (value >= file_json["df_min"][column_name]) and (
                    value <= file_json["df_max"][column_name]
            ):
                score = file_json["cnt"] / file_json["cntd"][column_name]
            else:
                score = 0

            print("string =  score:", score)
            if column_name not in res_score:
                res_score[column_name] = score
            else:
                res_score[column_name] += score
        elif "IN" in condition:
            content = condition.split("IN")
            # print(content)
            column_name = content[0].strip().lower()
            score = 0
            for value in (
                    content[1].strip().replace("(", "").replace(")", "").split(",")
            ):
                # print(value)
                if (value >= file_json["df_min"][column_name]) and (
                        value <= file_json["df_max"][column_name]
                ):
                    score += file_json["cnt"] / file_json["cntd"][column_name]
            print("IN score:", score)
            res_score[column_name] = score

    print(res_score)
    # min_value = min(res_score.values())
    mean_value = np.mean(list(res_score.values()))
    # print(min_value)
    return mean_value
    # conditions = [condition.split() for condition in conditions]


def z_score_normalization(matrix, eps=1e-8):
    # Step 1: Compute the mean of each feature (column)
    means = matrix.mean(axis=1, keepdims=True)

    # Step 2: Compute the standard deviation of each feature (column), avoiding division by zero
    stds = matrix.std(axis=1, ddof=1, keepdims=True)
    stds[stds == 0] = eps

    # Step 3: Perform Z-score normalization
    normalized_matrix = (matrix - means) / stds

    return normalized_matrix


def softmax(matrix):
    # Step 1: Compute the exponentials
    matrix = z_score_normalization(matrix)
    print("matrix:", matrix)
    exp_matrix = np.exp(matrix)

    # Step 2: Compute the sum of exponentials for each row
    row_sums = exp_matrix.sum(axis=1, keepdims=True)

    # Step 3: Normalize by dividing each row by its corresponding sum
    normalized_matrix = exp_matrix / row_sums

    return normalized_matrix


# 构建查询语句和数据文件的矩阵
def build_sql_file_matrix(sql_list, file_list):
    res_matrix = []
    for sql_text in sql_list:
        file_res_list = []
        for file_path in file_list:
            # calculate_sql_file(sql_text, file_path)
            print(file_path)
            file_res_list.append(calculate_sql_file(sql_text, file_path))
        res_matrix.append(file_res_list)

    res_matrix = np.array(res_matrix)
    res_matrix = softmax(res_matrix)
    print("resmarix:", res_matrix)
    return res_matrix


# 使用svd++算法来进行矩阵分解
def svd_plus_plus(R, lambda_, num_iterations, learning_rate):
    # 初始化用户/物品数量、隐向量维度
    num_users, num_items = R.shape
    latent_dim = 50  # 可调整的隐向量维度

    # 初始化用户隐向量、物品隐向量、物品特征向量、用户/物品偏置向量
    P_u = np.zeros(num_users)
    Q_i = np.zeros(num_items)
    P = np.random.normal(size=(num_users, latent_dim))
    Q = np.random.normal(size=(num_items, latent_dim))
    Y = np.random.normal(size=(num_items, latent_dim))

    for _ in range(num_iterations):
        for u in range(num_users):
            for i in range(num_items):
                if R[u][i] > 0:  # 用户u对物品i有评分/行为记录
                    error = R[u][i] - (P[u].dot(Q[i]) + P_u[u] + Q_i[i])
                    P[u] += learning_rate * (error * Q[i] - lambda_ * P[u])
                    Q[i] += learning_rate * (error * P[u] - lambda_ * Q[i])
                    Y[i] += learning_rate * (error * P[u] - lambda_ * Y[i])

                    # 更新用户/物品偏置向量（此处简化处理，实际应用可能需要更复杂的策略）
                    P_u[u] += learning_rate * error
                    Q_i[i] += learning_rate * error

    return Q + Y  # 返回每个物品的向量表示（隐向量与物品特征向量之和）


# 构建层次聚类树
def build_hierarchical_clustering_tree(item_vectors, n_clusters=3):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(item_vectors)

    cos_sim_matrix = cosine_similarity(data_scaled)
    distance_matrix = 1 - cos_sim_matrix  # 将相似度转换为距离

    distance_matrix = pdist(data_scaled, metric='euclidean')
    distance_matrix = squareform(distance_matrix)

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')

    labels = model.fit_predict(distance_matrix)
    # print(labels)
    return labels


def calculate_hilbert_index(x, y, z):  # 假设x, y, z是三个维度的值
    # 这里应使用适当的库或算法计算Hilbert索引
    x = int(x)
    y = int(y.split("#")[1])
    z = int(z.split("#")[1])

    hilbert_index_value = hilbert(x=np.array([x, y, z]))

    instantaneous_phase = np.unwrap(np.angle(hilbert_index_value))
    instantaneous_phase = np.mean(instantaneous_phase)
    # print(hilbert_index_value)
    return instantaneous_phase


def build_hilbert_curve(df):
    # 假设df是你的DataFrame，有三列'x', 'y', 'z'
    df['hilbert_index'] = df.apply(
        lambda row: calculate_hilbert_index(row['o_orderdate'], row['p_brand'], row['p_mfgr']), axis=1)
    df_sorted = df.sort_values(by='hilbert_index')

    return df_sorted


def main_pipe(sql_list, file_list):
    res_matrix = build_sql_file_matrix(sql_list, file_list)
    # svd_plus_plus(res_matrix)

    # Instantiate and train SVD++
    item_vectors = svd_plus_plus(res_matrix, lambda_=0.01, num_iterations=10, learning_rate=0.001)
    print(len(item_vectors))

    n_clusters = 3
    labels = build_hierarchical_clustering_tree(item_vectors, n_clusters=n_clusters)
    # print(labels)

    file_hb_dict = {i: [] for i in range(n_clusters)}
    for i in range(n_clusters):
        for j in range(len(file_list)):
            if labels[j] == i:
                file_hb_dict[i].append(file_list[j])

    print(file_hb_dict)

    # 初始化一个空的DataFrame来存储所有数据
    for i, csv_files_list in file_hb_dict.items():
        df_concat = pd.DataFrame()

        # 遍历列表并读取每个CSV文件
        for file_name in csv_files_list:
            # 读取单个CSV文件并将其追加到df_concat中
            df_concat = pd.concat(
                [pd.read_csv(file, sep=',', header=None, names=col_name_list, dtype=column_types) for file in
                 csv_files_list], ignore_index=True)

        df_concat = build_hilbert_curve(df_concat)
        df_concat.to_csv("output_{}.csv".format(i), index=False)
    # return item_vectors
    # df = pd.read_csv(
    #     "example_data/part-00000-156cad27-9faf-4385-9a6c-05a5b0ee5d28-c000.csv", sep=",", header=None,
    #     names=col_name_list, dtype=column_types
    # )
    # df = build_hilbert_curve(df)
    # print(df)


if __name__ == "__main__":
    file_path_list = os.listdir("example_data")
    file_path_list = ["example_data/" + file_path for file_path in file_path_list]
    # pprint(get_data_info())

    sql_text_ = """SELECT SUM(O_EXTENDEDPRICE * O_DISCOUNT) AS revenue FROM ssbflat WHERE O_ORDERDATE >= 19940101 AND O_ORDERDATE <= 19940131 AND O_DISCOUNT BETWEEN 4 AND 6 AND O_DISCOUNT BETWEEN 26 AND 35;"""
    sql_text_2 = """SELECT (O_ORDERDATE DIV 10000) AS YEAR,S_NATION, P_CATEGOR Y, SUM(LO_REVENUE - LO_SUPPLYCOST) AS profit FROM ssbflat WHERE C_REGION = 'AMERICA' AND S_REGION = 'AMERICA' AND O_ORDERDATE >= 19970101 AND O_ORDERDATE <= 19981231 AND P_MFGR IN ('MFGR#1', 'MFGR#2') GROUP BY YEAR, S_NATION, P_CATEGORY ORDER BY YEAR ASC, S_NATION ASC, P_CATEGORY ASC;"""
    # print(handle_sql(sql_text_))
    # calculate_sql_file(sql_text_, file_name)

    main_pipe([sql_text_, sql_text_2], file_path_list)
