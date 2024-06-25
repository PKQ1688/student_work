#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/6/2 14:27
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/6/2 14:27
# @File         : plot_image.py
import random

import pandas as pd
from matplotlib.font_manager import FontProperties
from pyecharts import options as opts
# import numpy as np
from pyecharts.charts import Line
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot


# 设置中文字体
font_path = '微软雅黑.ttf'
font_prop = FontProperties(fname=font_path)


def random_color():
    return '#%02X%02X%02X' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


res_df = pd.read_csv("topic_count.csv")
# print(res_df)
topic = [
    "信息化与系统建设",
    "质量和安全管理",
    "财务管理与运营分析",
    "患者服务与模式创新",
    "医疗技术与创新",
    "后勤与供应链管理",
    "数据管理和分析",
    "人工智能和机器学习",
    "管理与政策法规",
    "医疗教育与人才培养",
]
x_data = ["2010-2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023-2024.5"]


for i in range(len(topic)):

    line = (
        Line()
        .add_xaxis(x_data))

    topic_word = topic[i]
    print(topic_word)
    y_data = res_df[topic_word].values.tolist()

    # print(x_data)
    # print(y_data)

    line.add_yaxis(f"{topic_word}", y_data, is_smooth=True, color=random_color())

    # line.render("smooth_line_chart.html")

    line.set_global_opts(
        title_opts=opts.TitleOpts(title=f"{topic_word}主题强度研究趋势"),
        xaxis_opts=opts.AxisOpts(
            name="年份",
            axislabel_opts={"interval": 0},
            splitline_opts={"is_show": True},
        ),
        yaxis_opts=opts.AxisOpts(name="主题强度"))

    make_snapshot(snapshot, line.render(), f"{topic_word}.png")
