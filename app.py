# -*- encoding: utf-8 -*-
'''
   @File    :   app.py
   @Time    :   2023/11/29 16:42:24
   @Author  :   Brian Qu
   @Version :   1.0
   @Contact :   brian.qu@mail.utoronto.ca or qujianning0401@163.com
'''
import random
from flask import Flask, request, jsonify
from crawler import RTTCrawler
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import WebDriverException
import torch
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from model.system_factory import SystemFactory

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/filmEmo', methods=['POST'])
def film_comment_sentiment_analysis():
    data = request.get_json()
    
    if 'film_name' not in data:
        return jsonify({'error': 'Parameter film_name not found'}), 400
    
    # film_name = data['film_name']

    # crawler = RTTCrawler(film_name)
    # try:
    #     critics_comments_ratings_pairs = crawler.extract_critics_comments_and_ratings("All critics")
    #     crawler = RTTCrawler(film_name)
    #     audiences_comments_ratings_pairs = crawler.extract_audience_comments_and_ratings("All audience")
    #     crawler = RTTCrawler(film_name)
    #     film_data = crawler.extract_movie_pic_url("All critics")
    # except (WebDriverException, TimeoutException):
    #     print("No more pages to load, extraction complete")

    # print(len(critics_comments_ratings_pairs))
    # print(len(audiences_comments_ratings_pairs))

    # critics_comments = list(random.sample(critics_comments_ratings_pairs, 60))
    # audiences_comments = list(random.sample(audiences_comments_ratings_pairs, 40))

    # # 创建一个 EmoFilmSystem 的实例
    # system = SystemFactory.produce_system()
    # # 加载模型
    # model_checkpoint_path = "filmEmo.pth"
    # system = torch.load(model_checkpoint_path)

    # total_score = 0
    # critics_score = 0
    # audiences_score = 0

    # critics_comments_sentiment_count = {"negative": 0, "positive": 0, "netural": 0}
    # audiences_comments_sentiment_count = {"negative": 0, "positive": 0, "netural": 0}

    # for comment in critics_comments:
    #     if (system.inference(comment[0], need_tokenize=True).item() == 0):
    #         total_score += 1
    #         critics_score += 3
    #         critics_comments_sentiment_count["negative"] += 1
    #     elif (system.inference(comment[0], need_tokenize=True).item() == 1):
    #         total_score += 5
    #         critics_score += 5
    #         critics_comments_sentiment_count["netural"] += 1
    #     else:
    #         total_score += 10
    #         critics_score += 9
    #         critics_comments_sentiment_count["positive"] += 1

    # for comment in audiences_comments:
    #     if (system.inference(comment[0], need_tokenize=True).item() == 0):
    #         total_score += 3
    #         audiences_score += 3
    #         audiences_comments_sentiment_count["negative"] += 1
    #     elif (system.inference(comment[0], need_tokenize=True).item() == 1):
    #         total_score += 6
    #         audiences_score += 5
    #         audiences_comments_sentiment_count["netural"] += 1
    #     else:
    #         total_score += 9
    #         audiences_score += 10
    #         audiences_comments_sentiment_count["positive"] += 1

    # average_score = round((total_score / 100), 1)
    # critics_average_score = round((critics_score / 60), 1)
    # audiences_average_score = round((audiences_score / 40), 1)
    
    # return jsonify({'average_score': average_score,
    #                 'critics_average_score': critics_average_score,
    #                 'audiences_average_score': audiences_average_score,
    #                 'critics_comments_sentiment_count': critics_comments_sentiment_count,
    #                 'audiences_comments_sentiment_count': audiences_comments_sentiment_count,
    #                 'film_data': film_data})
    
    audiences_average_score = 6.7
    audiences_comments_sentiment_count = {"negative": 13,"netural": 8,"positive": 19}
    average_score = 7.9
    critics_average_score = 8.2
    critics_comments_sentiment_count = {"negative": 3,"netural": 8,"positive": 49}
    film_data = {"file_data": "https://resizing.flixster.com/x4JXyULCubaUuD3tDSL87u2VC6k=/206x305/v2/https://resizing.flixster.com/Ux0YYpq25r4CNYj-TM2YhlcTYU8=/ems.cHJkLWVtcy1hc3NldHMvbW92aWVzL2I2Y2U1ZjNkLWQwNzEtNDFiNS1iZmYzLWQ5NWFmM2Y1OGMyNS5qcGc=","film_director": "David Fincher",
                 "film_length": "R ,  1h 58m","film_name": "The Killer","film_type": "Mystery & Thriller,Action,Adventure,Crime"}
                 
    # 提取标签和值
    audience_labels = audiences_comments_sentiment_count.keys()
    audience_sizes = audiences_comments_sentiment_count.values()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    # 创建饼图
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(audience_sizes, labels=audience_labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.set_title('Audience Comments Sentiment Distribution')
    ax.axis('equal')  # 保证饼状图是圆的
    # 将饼图保存到 BytesIO 对象
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png')
    plt.close(fig)
    img_bytes.seek(0)
    # 将 BytesIO 对象转换为 Base64 编码的字符串
    audience_img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    audience_img_data = f"data:image/png;base64,{audience_img_base64}"


    # 提取标签和值
    critics_labels = critics_comments_sentiment_count.keys()
    critics_sizes = critics_comments_sentiment_count.values()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    # 创建饼图
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(critics_sizes, labels=critics_labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.set_title('Critics Comments Sentiment Distribution')
    ax.axis('equal')  # 保证饼状图是圆的
    # 将饼图保存到 BytesIO 对象
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png')
    plt.close(fig)
    img_bytes.seek(0)

    # 将 BytesIO 对象转换为 Base64 编码的字符串
    critics_img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    critics_img_data = f"data:image/png;base64,{critics_img_base64}"



    return jsonify({"audiences_average_score": audiences_average_score,
                    "audiences_comments_sentiment_count": audiences_comments_sentiment_count,
                    "average_score": average_score,
                    "critics_average_score": critics_average_score,
                    "critics_comments_sentiment_count": critics_comments_sentiment_count, 
                    "film_data": film_data,
                    'audience_img_data': audience_img_data,
                    'critics_img_data': critics_img_data})

if __name__ == "__main__":
    app.run(debug=True)