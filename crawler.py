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
        self.url = self.construct_url(movie_name)

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
        review_container = soup.find("section", class_=f"{review_type}-reviews")
        if review_container:
            for review in review_container.find_all("div", class_="review_table_row"):
                review_text = review.find("div", class_="the_review").text.strip()
                review_score = review.find("div", class_="review_icon").find("span", class_="rating").text.strip()
                reviews.append({"text": review_text, "score": review_score})
        return reviews
    
    def scrape_reviews(self):
        page_content = self.fetch_page()
        if page_content:
            soup = BeautifulSoup(page_content, 'html.parser')
            movie_title = soup.find("h1", class_="title").text.strip()
            critical_reviews = self.extract_reviews(soup, "mop")
            audience_reviews = self.extract_reviews(soup, "audience")

            return {
                "movie_name": movie_title,
                "critical_reviews": critical_reviews,
                "audience_reviews": audience_reviews
            }
        else:
            return "Error in scrape_reviews"
        
def main():
    movie_name = "the_killer_2023"
    scraper = RTTCrawler(movie_name)
    movie_reviews = scraper.scrape_reviews()

    if movie_reviews:
        print(f"Movie Title: {movie_reviews['movie_name']}")
        print("Critical Reviews:")
        count = 0
        for review in movie_reviews['critical_reviews']:
            print(f"Score: {review['score']}")
            print(f"Review: {review['text']}")
            print("\n")
            count += 1
            if count == 5:
                break
        count = 0
        print("Audience Reviews:")
        for review in movie_reviews['audience_reviews']:
            print(f"评分: {review['score']}")
            print(f"内容: {review['text']}")
            print("\n")
            count += 1
            if count == 5:
                break

if __name__ == "__main__":
    main()