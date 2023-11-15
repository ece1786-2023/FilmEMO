from classifier import Classifier
from extractor import GPT2Extractor
from transformers import AutoTokenizer
from emofilmsystem import EmoFilmSystem
import torch.nn as nn


class SystemFactory:
    @staticmethod
    def ExtractorFactory(pre_train_model_name):
        if pre_train_model_name == "gpt2":
            return GPT2Extractor()
        else:
            raise Exception(f'no available pre_train_model {pre_train_model_name}')
    
    @staticmethod
    def ClassifierFactory(number_features, number_classes, hidden_units):
        return Classifier(number_features, number_classes, hidden_units)
    
    @staticmethod
    def produce_system(pre_train_model_name="gpt2", classifier_config={"number_classes":3, "hidden_units":[64, 32]}, Lossfn=nn.CrossEntropyLoss):
        tokenizer = AutoTokenizer.from_pretrained(pre_train_model_name)
        feature_extractor = SystemFactory.ExtractorFactory(pre_train_model_name)
        classifier_config["number_features"] = feature_extractor.get_output_shape()[-1]
        classifier = SystemFactory.ClassifierFactory(**classifier_config)
        return EmoFilmSystem(tokenizer=tokenizer, feature_extractor=feature_extractor, classifier=classifier, Lossfn=Lossfn)

# emofilmsystem = SystemFactory.produce_system()
# print(emofilmsystem.inference("test model"))