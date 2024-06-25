#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/6/1 16:44
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/6/1 16:44
# @File         : data_analysis.py
import ast
import json

import pandas as pd

# 示例字符串

# 将字符串转换为列表


data_df = pd.read_csv("res_v3.csv")

topic_df = pd.read_csv("主题词与子主题.csv")
# print(topic_df)

topic_count = topic_df["主题词"].values
topic_count_json = {topic: 0 for topic in topic_count}
topic_count_json["其他"] = 0

# topic_json = {}
# for index, row in topic_df.iterrows():
#     print(row)
#     new = row["子主题"].replace("\n", ";").replace(" ", "").replace("；", ";")
#     sub_topic_list = new.split(";")
#     print(sub_topic_list)
#     for sub_topic in sub_topic_list:
#         topic_json[sub_topic] = row['主题词']
#
# with open('topic.json', 'w') as f:
#     json.dump(topic_json, f, ensure_ascii=False, indent=4)


with open("topic.json", "r") as f:
    topic_json = json.load(f)


# print(topic_json)


def get_year(years_list=None):
    all_text = []
    for index, row in data_df.iterrows():
        # print(row)
        # print(row['发表年份'])

        if years_list is not None and str(row["发表年份"]) not in years_list:
            continue
        if pd.isna(row["主题词"]):
            continue

        # import pdb
        # pdb.set_trace()

        list_from_string = ast.literal_eval(row["主题词"])
        for item in list_from_string:
            if item in topic_json.keys():
                topic_count_json[topic_json[item]] += 1
            else:
                topic_count_json["其他"] += 1
        # all_text.extend(list_from_string)
        # break

    # print(all_text)
    # word_counts = Counter(all_text)
    # print(word_counts)
    # print(topic_count_json)
    return topic_count_json
    # df = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency'])
    # return df
    # print(df)
    # 保存 DataFrame 到 CSV 文件
    # df.to_csv('word_counts.csv', index=False)


if __name__ == "__main__":
    # year_2014 = []
    # year_2015 = []
    # year_2016 = []
    # year_2017 = []
    # year_2018 = []
    # year_2019 = []
    # year_2020 = []
    # year_2021 = []
    # year_2022 = []
    # year_2023 = []
    year_all_list = [
        ["2010", "2011", "2012", "2013", "2014"],
        ["2015"],
        ["2016"],
        ["2017"],
        ["2018"],
        ["2019"],
        ["2020"],
        ["2021"],
        ["2022"],
        ["2023", "2024"],
    ]
    res_list = []
    for year_list in year_all_list:
        topic_json_year_json = get_year(year_list)
        # print(topic_json_year_json)
        res_list.append(topic_json_year_json.copy())
        # print("=========")
    res_df = pd.DataFrame(res_list)
    print(res_df)
    res_df["年份"] = ["2010-2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023-2024"]
    res_df["篇数"] = [57, 35, 36, 79, 113, 206, 291, 301, 232, 478]
    # res_df = res_df[["年份", "人工智能", "大数据", "区块链", "云计算", "物联网", "其他", "篇数"]]
    for index, row in res_df.iterrows():
        for key in topic_count_json.keys():
            res_df.loc[index, key] = int(row[key] * 100 / row["篇数"])
    res_df.to_csv("topic_count.csv", index=False, encoding="utf-8")
