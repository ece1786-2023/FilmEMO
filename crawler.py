# -*- encoding: utf-8 -*-
'''
@File    :   crawler.py
@Time    :   2023/11/12 22:59:46
@Author  :   Brian Qu
@Version :   1.0
@Contact :   brian.qu@mail.utoronto.ca or qujianning0401@163.com
'''
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options


class RTTCrawler(object):
    def __init__(self, movie_name):
        self.movie_name = movie_name
        self.base_url = "https://www.rottentomatoes.com/m/"
        self.all_critics_tail_url = "/reviews"
        self.all_audience_tail_url = "/reviews?type=user"
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=chrome_options)

    def construct_url(self, reviewType):
        if reviewType == "All critics":
            url = f"{self.base_url}{self.movie_name}{self.all_critics_tail_url}"
        elif reviewType == "All audience":
            url = f"{self.base_url}{self.movie_name}{self.all_audience_tail_url}"
        return url
    
    def extract_critics_comments_and_ratings(self):
        url = self.construct_url("All critics")
        self.driver.get(url)
        comments = []
        ratings = []

        page_source = self.driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")
        reviews_blocks = soup.find_all('div', {'class': "review-row"})
        for block in reviews_blocks:
            comment = block.find('p', {'class': 'review-text'}).get_text(strip=True)
            comments.append(comment)
            rating = block.find('score-icon-critic-deprecated').get('state')
            ratings.append(rating)
        
        self.driver.quit()
        return comments, ratings
    
movie_name = "the_killer_2023"
scraper = RTTCrawler(movie_name)
comments, ratings = scraper.extract_critics_comments_and_ratings()
print(len(comments))
print(len(ratings))