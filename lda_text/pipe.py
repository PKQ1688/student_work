#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/5/28 15:38
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/5/28 15:38
# @File         : pipe.py
import os
import re
from typing import List

import fitz  # PyMuPDF
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import string
import gensim
import jieba
from gensim import corpora


# 加载中文停用词
def load_stopwords(filepath_list: List[str]):
    stopwords = []
    for filepath in filepath_list:
        with open(filepath, "r", encoding="utf-8") as file:
            stopwords.extend(file.read().splitlines())
    return stopwords


# 第一步：读取和解析PDF文档
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text


# 第二步：预处理文本数据
def preprocess_text(text, stop_words):
    jieba.load_userdict("userdict(2).txt")
    tokens = jieba.cut(text)
    tokens = [word for word in tokens if word not in stop_words and word.strip()]
    return tokens


# 第三步：创建词袋模型
def create_bow(tokens):
    dictionary = corpora.Dictionary([tokens])
    bow = [dictionary.doc2bow(tokens)]
    return dictionary, bow


# 第四步：应用LDA模型
def apply_lda(dictionary, bow, num_topics=5):
    lda_model = gensim.models.ldamodel.LdaModel(
        bow, num_topics=num_topics, id2word=dictionary, passes=15, alpha="auto", eta="auto"
    )

    return lda_model


# 主函数
def main(pdf_path_list, stopwords_path_list):
    text = ""
    for pdf in pdf_path_list:
        text += extract_text_from_pdf(pdf)

    text.replace("ai", "人工智能")
    text = ''.join(re.findall(r'[\u4e00-\u9fa5]+', text))  # 仅保留中文
    # text = extract_text_from_pdf(pdf_path)
    # print(text)
    # exit()
    stop_words = load_stopwords(stopwords_path_list)
    tokens = preprocess_text(text, stop_words)
    dictionary, bow = create_bow(tokens)
    lda_model = apply_lda(dictionary, bow, num_topics=10)
    # topics = lda_model.print_topics(num_words=10)

    topics = lda_model.print_topics(num_words=30)
    for topic in topics:
        print(topic)
        perplexity = lda_model.log_perplexity(bow)
        print("perplexity:", perplexity)
        # 可视化主题词云
        # wordcloud = WordCloud(font_path="微软雅黑.ttf").generate_from_frequencies(
        #     dict(lda_model.show_topic(topic_terms[0], topn=30))
        # )
        # plt.imshow(wordcloud, interpolation="bilinear")
        # plt.axis("off")
        # plt.show()


# 示例用法
# pdf_path = "智慧医院新增数据/智慧医院2024（新）/5G专网下的多生命体征参数采集研究.pdf"
# stopwords_path = "hit_stopwords.txt"  # 你需要一个中文停用词表文件
# main(pdf_path, stopwords_path)


def get_all_pdfs(directory):
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


# pdf_list = get_all_pdfs(".")
# for i in pdf_list:
#     shutil.copy(i, "use_pdf")
# print(pdf_list)
# pdf_name_list = [pdf.split("/")[-1] for pdf in pdf_list]
# print(pdf_name_list)
# print(len(set(pdf_name_list)))

stopwords_path_list = [
    "hit_stopwords.txt",
    "baidu_stopwords.txt",
    "cn_stopwords.txt",
    "scu_stopwords.txt",
    "user_stopwords.txt"
]  # 你需要一个中文停用词表文件

pdf_list = os.listdir("use_pdf")
pdf_list = [os.path.join("use_pdf", pdf) for pdf in pdf_list]
print(len(pdf_list))
main(pdf_list, stopwords_path_list)
