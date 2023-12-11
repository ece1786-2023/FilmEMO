from model.classifier import Classifier
from model.extractor import GPT2Extractor
from transformers import AutoTokenizer
from model.emofilmsystem import EmoFilmSystem
import torch.nn as nn


class SystemFactory:
    @staticmethod
    def ExtractorFactory(pre_train_model_name, train_extractor):
        if pre_train_model_name in ["gpt2-large", "gpt2-medium"]:
            return GPT2Extractor(pre_train_model_name, train_extractor)
        else:
            raise Exception(f'no available pre_train_model {pre_train_model_name}')
    
    @staticmethod
    def ClassifierFactory(number_features, number_classes, hidden_units):
        return Classifier(number_features, number_classes, hidden_units)
    
    @staticmethod
    def produce_system(extractor_config={"pre_train_model_name":"gpt2", "train_extractor":True}, classifier_config={"number_classes":3, "hidden_units":[64, 32]}, Lossfn=nn.CrossEntropyLoss, max_length=60):
        tokenizer = AutoTokenizer.from_pretrained(extractor_config["pre_train_model_name"])
        tokenizer.pad_token = tokenizer.eos_token
        feature_extractor = SystemFactory.ExtractorFactory(**extractor_config)
        classifier_config["number_features"] = feature_extractor.get_output_shape()[-1]
        classifier = SystemFactory.ClassifierFactory(**classifier_config)
        return EmoFilmSystem(tokenizer=tokenizer, feature_extractor=feature_extractor, classifier=classifier, Lossfn=Lossfn, max_length=max_length)