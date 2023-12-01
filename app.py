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

from model.system_factory import SystemFactory

app = Flask(__name__)

@app.route('/filmEmo', methods=['POST'])
def film_comment_sentiment_analysis():
    data = request.get_json()
    
    if 'film_name' not in data:
        return jsonify({'error': 'Parameter film_name not found'}), 400
    
    film_name = data['film_name']

    crawler = RTTCrawler(film_name)
    try:
        critics_comments_ratings_pairs = crawler.extract_critics_comments_and_ratings("All critics")
        crawler = RTTCrawler(film_name)
        audiences_comments_ratings_pairs = crawler.extract_audience_comments_and_ratings("All audience")
    except (WebDriverException, TimeoutException):
        print("No more pages to load, extraction complete")

    print(len(critics_comments_ratings_pairs))
    print(len(audiences_comments_ratings_pairs))

    critics_comments = list(random.sample(critics_comments_ratings_pairs, 60))
    audiences_comments = list(random.sample(audiences_comments_ratings_pairs, 40))

    # 创建一个 EmoFilmSystem 的实例
    system = SystemFactory.produce_system()
    # 加载模型
    model_checkpoint_path = "filmEmo.pth"
    system = torch.load(model_checkpoint_path)

    res_critics = []
    res_audiences = []
    total_score = 0

    for comment in critics_comments:
        res_critics.append(system.inference(comment[0], need_tokenize=True).item())
        if (system.inference(comment[0], need_tokenize=True).item() == 0):
            total_score += 1.5
        elif (system.inference(comment[0], need_tokenize=True).item() == 1):
            total_score += 5.5
        else:
            total_score += 10

    for comment in audiences_comments:
        res_audiences.append(system.inference(comment[0], need_tokenize=True).item())
        if (system.inference(comment[0], need_tokenize=True).item() == 0):
            total_score += 3
        elif (system.inference(comment[0], need_tokenize=True).item() == 1):
            total_score += 6
        else:
            total_score += 8

    average_score = total_score / 100    
    
    return jsonify({'average_score': average_score})

if __name__ == "__main__":
    app.run(debug=True)