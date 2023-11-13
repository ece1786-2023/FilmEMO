# -*- encoding: utf-8 -*-
'''
@File    :   crawler.py
@Time    :   2023/11/12 22:59:46
@Author  :   Brian Qu
@Version :   1.0
@Contact :   brian.qu@mail.utoronto.ca or qujianning0401@163.com
'''
import requests
from bs4 import BeautifulSoup
import re

class RTTCrawler(object):
    def __init__(self, movie_name):
        # use movie_name to construct movie url
        self.url = self._construct_url(movie_name)

    def construct_url(self, movie_name):
        # replace spaces with '_', and construct url
        movie_name = re.sub(r'\s+', '_', movie_name)
        return f"https://www.rottentomatoes.com/m/{movie_name}"
    
    def fetch_page(self):
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page: {e}")
            return None
        
    def extract_reviews(self, soup, review_type):
        reviews = []