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
import re
from selenium.common.exceptions import WebDriverException
import os
import csv


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
        self.movie_name = movie_name.lower().replace(' ', "_")
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
        try:
            self.driver.get(url)
        except WebDriverException:
            print(f"Failed to load URL: {url}")
            return []
        comments_ratings_pairs = set()

        try:
            while True:
                WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'review-row')))
                page_source = self.driver.page_source
                soup = BeautifulSoup(page_source, "html.parser")
                reviews_blocks = soup.find_all('div', {'class': "review-row"})

                for block in reviews_blocks:
                    comment = block.find('p', {'class': 'review-text'}).get_text(strip=True).replace('\n', '')
                    rating = block.find('score-icon-critic-deprecated').get('state')
                    comments_ratings_pairs.add((comment, rating))

                load_more_button = WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="reviews"]/div[1]/rt-button[2]')))
                self.driver.execute_script("arguments[0].click();", load_more_button)
        except (WebDriverException, TimeoutException):
            print("No more pages to load, extraction complete")
        
        self.driver.quit()

        return list(comments_ratings_pairs)
    
    def extract_audience_comments_and_ratings(self, reviewType: str) -> set:
        '''
           @Description: Use to extract audiences' comments and ratings
           @Param reviewType: string 
           @Return comments_ratings_pairs: set
           @Author: Brian Qu
           @Time: 2023/11/14 15:50:13
        '''
        url = self.construct_url(reviewType)
        try:
            self.driver.get(url)
        except WebDriverException:
            print(f"Failed to load URL: {url}")
            return []
        comments_ratings_pairs = set()

        try:
            while True:
                page_source = self.driver.page_source
                soup = BeautifulSoup(page_source, "html.parser")
                reviews_blocks = soup.find_all('div', {'class': 'audience-review-row'})

                for block in reviews_blocks:
                    comment = block.find('p', {'slot': 'content'}).get_text(strip=True).replace('\n', '')
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
        except (WebDriverException, TimeoutException):
            print("No more pages to load, extraction complete")

        self.driver.quit()

        return list(comments_ratings_pairs)
    
    def get_popular_movies(self, movieCount: int) -> list[str]:
        '''
           @Description: Extract top movies from Rotten Tomatoes
           @Param movieCount: int
           @Return movieList: list[str]  
           @Author: Brian Qu
           @Time: 2023/11/14 18:29:51
        '''
        popular_movie_url = "https://www.rottentomatoes.com/browse/movies_at_home/sort:popular?page=6"
        self.driver.get(popular_movie_url)
        movieList = []
        try:
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, "html.parser")
            movieNames = soup.find_all('span', {'class': 'p--small'})

            for movie in movieNames[:movieCount]:
                movie_name = movie.get_text(strip=True)
                movie_name = re.sub(r'\W+', '_', movie_name)
                movieList.append(movie_name)
        except TimeoutException:
            print("Extracted all needed movies")

        self.driver.quit()

        return movieList

    def write_comments_to_file(self, comments_ratings_pairs: list, reviewType: str) -> None:
        '''
            @Description: Write comments and ratings to a specified file.
            @Param comments_ratings_pairs: list
            @Param reviewType: string
            @Return: None
            @Author: Brian Qu
            @Time: 2023/11/14 16:20:00
        '''
        txt_path = "Comments_Ratings_Files/" + self.movie_name + "-" + reviewType + ".txt"
        with open(txt_path, 'w', encoding='utf-8') as file:
            for comment, rating in comments_ratings_pairs:
                file.write(f"Rating: {rating}\nComment: {comment}\n\n")

    def read_txt_file(self, file_path: str) -> list:
        '''
           @Description: read critics comments and ratings from txt
           @Param file_path: string
           @Return: list
           @Author: Brian Qu
           @Time: 2023/11/16 13:27:06
        '''
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        entries = content.split('\n\n')
        return [tuple(entry.split('\n')) for entry in entries if entry]
    
    def write_critics_to_csv(self, data: list, csv_file_path: str) -> None:
        '''
           @Description: write critics data to scsv file
           @Param data: list
           @Param csv_file_path: string
           @Return: None
           @Author: Brian Qu
           @Time: 2023/11/16 13:30:09
        '''
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for entry in data:
                for rating, comment in entry:
                    if rating.split(": ")[1] == "fresh":
                        writer.writerow([1, comment.split(": ")[1]])
                    else:
                        writer.writerow([-1, comment.split(": ")[1]])

    def write_audience_to_csv(self, data: list, csv_file_path: str) -> None:
        '''
           @Description: write audience data to scsv file
           @Param data: list
           @Param csv_file_path: string
           @Return: None
           @Author: Brian Qu
           @Time: 2023/11/16 13:30:09
        '''
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for entry in data:
                for rating, comment in entry:
                    if 0 <= float(rating.split(": ")[1]) <= 1.5:
                        writer.writerow([-1, comment.split(": ")[1]])
                    elif 1.5 < float(rating.split(": ")[1]) <= 3.5:
                        writer.writerow([0, comment.split(": ")[1]])
                    else:
                        writer.writerow([1, comment.split(": ")[1]])

    def process_folder(self, folder_path: str, csv_file_path: str):
        '''
           @Description: iterate all txt files and write the content to csv file
           @Param folder_path: str 
           @Param csv_file_path: str 
           @Return: None
           @Author: Brian Qu
           @Time: 2023/11/16 13:48:23
        '''
        all_data = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('All critics.txt'):
                file_path = os.path.join(folder_path, file_name)
                all_data.append(self.read_txt_file(file_path))
        self.write_critics_to_csv(all_data, csv_file_path)

        all_data = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('All audience.txt'):
                file_path = os.path.join(folder_path, file_name)
                all_data.append(self.read_txt_file(file_path))
        self.write_audience_to_csv(all_data, csv_file_path)

def main():
    crawler = RTTCrawler("")
    crawler.process_folder("Comments_Ratings_Files/", "ratings_comments_pairs.csv")

if __name__ == "__main__":
    main()