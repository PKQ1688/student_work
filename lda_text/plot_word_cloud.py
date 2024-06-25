#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/6/2 22:00
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/6/2 22:00
# @File         : plot_word_cloud.py
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties

font_path = '微软雅黑.ttf'
font_prop = FontProperties(fname=font_path)

# 示例词语及其词频字典
word_frequency_df = pd.read_csv("关键词情况.csv")
word_frequency = word_frequency_df.set_index("Word")["Frequency"].to_dict()
print(word_frequency)

# 创建WordCloud实例
wordcloud = WordCloud(
    font_path="微软雅黑.ttf", width=800, height=400, background_color="white"
).generate_from_frequencies(word_frequency)

# 显示词云图
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")  # 不显示坐标轴
plt.title("主题词词频词云图",fontproperties=font_prop)
# plt.show()

plt.savefig('主题词词频词云图.png', dpi=1200)
