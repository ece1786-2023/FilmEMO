# -*- encoding: utf-8 -*-
'''
@File    :   crawler.py
@Time    :   2023/11/12 22:59:46
@Author  :   Brian Qu
@Version :   1.0
@Contact :   brian.qu@mail.utoronto.ca or qujianning0401@163.com
'''
from time import sleep
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options


class RTTCrawler(object):
    def __init__(self, movie_name: str) -> None:
        '''
           @Description: Initialize the RTTCrawler, using headless Chrome to run selenium
           @Param movie_name: string
           @Return: None
           @Author: Brian Qu
           @Time: 2023/11/14 10:36:05
        '''
        # Initialize the instance
        self.movie_name = movie_name
        self.base_url = "https://www.rottentomatoes.com/m/"
        self.all_critics_tail_url = "/reviews"
        self.all_audience_tail_url = "/reviews?type=user"
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=chrome_options)
        # self.driver = webdriver.Chrome(options=chrome_options)

    def construct_url(self, reviewType: str) -> str:
        '''
           @Description: Use review type to construct url of different comments groups
           @Param reviewType: string
           @Return url: string
           @Author: Brian Qu
           @Time: 2023/11/14 11:00:15
        '''
        # Use reviewType to get different comments group website
        if reviewType == "All critics":
            url = f"{self.base_url}{self.movie_name}{self.all_critics_tail_url}"
        elif reviewType == "All audience":
            url = f"{self.base_url}{self.movie_name}{self.all_audience_tail_url}"
        return url
    
    def extract_critics_comments_and_ratings(self, reviewType: str) -> set:
        '''
           @Description: Use to extract critics' comments and ratings
           @Param reviewType: string
           @Return comments_ratings_pairs: set
           @Author: Brian Qu
           @Time: 2023/11/14 11:05:01
        '''
        url = self.construct_url(reviewType)
        self.driver.get(url)
        comments_ratings_pairs = set()

        try:
            while True:
                WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'review-row')))
                page_source = self.driver.page_source
                soup = BeautifulSoup(page_source, "html.parser")
                reviews_blocks = soup.find_all('div', {'class': "review-row"})

                for block in reviews_blocks:
                    comment = block.find('p', {'class': 'review-text'}).get_text(strip=True)
                    rating = block.find('score-icon-critic-deprecated').get('state')
                    comments_ratings_pairs.add((comment, rating))

                load_more_button = WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="reviews"]/div[1]/rt-button[2]')))
                self.driver.execute_script("arguments[0].click();", load_more_button)
        except TimeoutException:
            print("No more pages to load, extraction complete")
        
        self.driver.quit()

        return comments_ratings_pairs
    
    def extract_audience_comments_and_ratings(self, reviewType: str) -> set:
        '''
           @Description: Use to extract audiences' comments and ratings
           @Param reviewType: string 
           @Return comments_ratings_pairs: set
           @Author: Brian Qu
           @Time: 2023/11/14 15:50:13
        '''
        url = self.construct_url(reviewType)
        self.driver.get(url)
        comments_ratings_pairs = set()

        try:
            while True:
                page_source = self.driver.page_source
                soup = BeautifulSoup(page_source, "html.parser")
                reviews_blocks = soup.find_all('div', {'class': 'audience-review-row'})

                for block in reviews_blocks:
                    comment = block.find('p', {'slot': 'content'}).get_text(strip=True)
                    rating = 0
                    stars = block.find_all('span')
                    for star in stars:
                        if star.get('class')[0] == 'star-display__filled':
                            rating += 1
                        elif star.get('class')[0] == 'star-display__half':
                            rating += 0.5
                    comments_ratings_pairs.add((comment, rating))
                
                load_more_button = WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="reviews"]/div[1]/rt-button[2]')))
                self.driver.execute_script("arguments[0].click();", load_more_button)
        except TimeoutException:
            print("No more pages to load, extraction complete")

        self.driver.quit()

        return comments_ratings_pairs
    
crawler = RTTCrawler("the_killer_2023")
pairs = crawler.extract_audience_comments_and_ratings("All audience")
print(len(pairs))
pairs = list(pairs)
print(pairs[0])