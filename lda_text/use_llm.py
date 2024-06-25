#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/5/31 20:21
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/5/31 20:21
# @File         : use_llm.py
import json
import os
import re

import fitz  # PyMuPDF
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    api_key="sk-e5f2a94e37e74614ae88e1102aea5ca2",
    # api_key="sk-8d2417d8a574435d82b3df006720b5e9",
    base_url="https://api.deepseek.com"
)


def extract_text_from_pdf(pdf_name):
    doc = fitz.open(os.path.join('use_pdf', pdf_name))
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text


def get_llm_res(input_text: str):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个文档内容分析的助手"},
            {"role": "user", "content": input_text},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content


prompt = "提取一下这篇文章的发表年份，并整理出这篇文章正文中的主题词，只需要给出对应的主题词，不要有额外的东西。使用json格式返回。文章内容如下:"


def handle_pdf(pdf, clean=False):
    text = extract_text_from_pdf(pdf)
    # print(text)

    if clean:
        text.replace("ai", "人工智能")
        text = ''.join(re.findall(r'[\u4e00-\u9fa5]+', text))  # 仅保留中文

    prompt_text = prompt + text
    llm_out = get_llm_res(prompt_text)
    # print(llm_out)
    llm_out = llm_out.replace("```json\n", "").replace("```", "")
    return llm_out


def main():
    pdf_list = os.listdir("use_pdf")
    # pdf_list = [os.path.join("use_pdf", pdf) for pdf in pdf_list]

    title_list = []
    year = []
    topic_list = []

    for pdf in tqdm(pdf_list, total=len(pdf_list)):
        try:
            llm_out = handle_pdf(pdf, clean=True)

            python_dict = json.loads(llm_out)

        except:
            print(pdf)
            python_dict = {"发表年份": "", "主题词": ""}

        title_list.append(pdf)
        year.append(python_dict.get('发表年份', ""))
        topic_list.append(python_dict.get('主题词', ""))

    res_df = pd.DataFrame({"文章": title_list, "发表年份": year, "主题词": topic_list})

    res_df.to_csv("res.csv", index=False, encoding="utf-8")


def handle_no_data():
    res_df = pd.read_csv("res_v2.csv")
    # print(res_df)
    for index, row in res_df.iterrows():
        if pd.isna(row["发表年份"]) or pd.isna(row["主题词"]):
            try:
                llm_out = handle_pdf(row["文章"])
                python_dict = json.loads(llm_out)
                res_df.loc[index, "发表年份"] = python_dict.get('发表年份', "")
                res_df.loc[index, "主题词"] = python_dict.get('主题词', "")
            # print(llm_out)
            # break
            except Exception as e:
                print(e)
                print(row["文章"])
    res_df.to_csv("res_v3.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    # _prompt = "现在你是一个男科医生，请根据患者的问题给出建议。"
    # _message = "现在18岁了，最近半年，发觉，性生活总是提不起劲，同时，每次才开始就已经射了，请问：男孩早泄究竟是什么因素引发的。"
    # llm_out = get_llm_res(_prompt, _message)
    # print(llm_out)
    main()
    # handle_no_data()
    # res = handle_pdf(pdf="5G“打工人”与医疗信息化共成长.pdf")
    # print(res)
