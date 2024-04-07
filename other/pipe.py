#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/4/7 14:46
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/4/7 14:46
# @File         : pipe.py
import json
import re

import pandas as pd

col_name = json.load(open("col_name.json"))
# print(col_name)

col_name_list = [item['name'] for item in col_name]
# type_list = [item['type'] for item in col_name]
column_types = {item["name"]: pd.Int64Dtype() if item["type"] == "bigint" else item["type"] for item in col_name}


# print(col_name_list)
# print(column_types)


# col_name = []
# 统计某一个文件的数据
def get_data_info(file_path="example_data/part-00000-156cad27-9faf-4385-9a6c-05a5b0ee5d28-c000.csv"):
    data_csv = pd.read_csv(file_path, sep=",", header=None, names=col_name_list, dtype=column_types)
    file_name = file_path.split("/")[-1]
    # print(data_csv.head())
    res_json = {}
    # 获取列数
    column_count = data_csv.shape[1]
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

    where_clause_end = min(where_clause_fh, where_clause_group_by) if where_clause_group_by != -1 else where_clause_fh

    conditions = sql_text[where_clause_start:where_clause_end]

    conditions = re.split(r'\bAND\b', conditions)

    conditions_list = []
    flag = -1
    for index, condition in enumerate(conditions):
        condition = condition.replace('WHERE', '')
        if index == flag:
            continue
        if 'BETWEEN' in condition:
            condition += ' AND ' + conditions[index + 1]
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
            score = file_json["cnt"] * (int(value) - file_json["df_min"][column_name]) / (
                    file_json["df_max"][column_name] - file_json["df_min"][column_name])
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
            score = file_json["cnt"] * (file_json["df_max"][column_name] - int(value)) / (
                    file_json["df_max"][column_name] - file_json["df_min"][column_name])
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
            score = file_json["cnt"] * (value[1] - value[0]) / (
                    file_json["df_max"][column_name] - file_json["df_min"][column_name])
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

            if (value >= file_json["df_min"][column_name]) and (value <= file_json["df_max"][column_name]):
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
            for value in content[1].strip().replace("(", "").replace(")", "").split(","):
                # print(value)
                if (value >= file_json["df_min"][column_name]) and (value <= file_json["df_max"][column_name]):
                    score += file_json["cnt"] / file_json["cntd"][column_name]
            print("IN score:", score)
            res_score[column_name] = score

    print(res_score)
    min_value = min(res_score.values())
    print(min_value)
    return min_value
    # conditions = [condition.split() for condition in conditions]


if __name__ == '__main__':
    file_name = "example_data/part-00000-156cad27-9faf-4385-9a6c-05a5b0ee5d28-c000.csv"
    # pprint(get_data_info())

    sql_text_ = """SELECT SUM(O_EXTENDEDPRICE * O_DISCOUNT) AS revenue FROM ssbflat WHERE O_ORDERDATE >= 19940101 AND O_ORDERDATE <= 19940131 AND O_DISCOUNT BETWEEN 4 AND 6 AND O_DISCOUNT BETWEEN 26 AND 35 AND P_CATEGORY = 'MFGR#12' AND S_REGION = 'AMERICA' AND S_CITY IN ('UNITED KI1', 'UNITED KI5') GROUP BY C_CITY, S_CITY, YEAR ORDER BY YEAR ASC, revenue DESC;"""

    # print(handle_sql(sql_text_))
    calculate_sql_file(sql_text_, file_name)
