import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords

class MovieReviewDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {"text": self.data.iloc[idx]["comment"], "label":self.data.iloc[idx]["rating"]}


# This factory is for loading the stored data from the csv_file to the 
class DataFactory:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.process_data = self.preprocess()
    
    # return a pd df
    def loading_dataset(self):
        return pd.read_csv(self.csv_file_path)
    
    def _preprocess_text(self, text):
        text = text.lower()  
        text = re.sub(r'\W', ' ', text)  
        text = re.sub(r'\s+', ' ', text)
        text = re.sub("["
                    u"\U0001F600-\U0001F64F"  # emoticons
                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    "]+", '', text, flags=re.UNICODE)  # delete emoji
        words = text.split()
        words = [word for word in words if word not in stopwords.words('english')]
        text = ' '.join(words)
        return text

    
    def preprocess(self):
        # loading raw dataset
        data = self.loading_dataset()
        # remove unwanted string
        # data["comment"] = data["comment"].apply(self._preprocess_text)
        data["rating"] += 1
        # data = data.apply(self.tokenize, axis=1)
        return data
    
    def train_test_split(self, train_ratio, n_data=5000):
        train_data, test_data = train_test_split(self.process_data.sample(n=min(n_data, len(self.process_data)), random_state=1),train_size=train_ratio, random_state=3, shuffle=True)
        train_data = MovieReviewDataset(train_data)
        test_data = MovieReviewDataset(test_data)
        return train_data, test_data