import numpy as np
import torch
from utils import generate_data, check_data, word_ids_to_sentence
import torchtext
from torchtext import data
import torch.nn as nn
import torch.nn.functional as F
from train_eval import train, test

from torch.autograd import Variable as V
from train_eval import test_sentence,load_model

from model import RNNModel


class Config(object):
    def __init__(self):
        self.model_name = "lm_model"
        self.data_ori =  "/mnt/data3/wuchunsheng/data_all/data_mine/lm_data/"
        self.train_path = "train_0.csv"
        self.valid_path = "train_0.csv"
        self.test_path = "test_100.csv"
        self.sen_max_length = 150
        # self.embedding_path = "need_bertembedding"
        self.embedding_path = "bert_embedding"
        self.embedding_dim = 768
        self.vocab_maxsize = 4000
        self.vocab_minfreq = 10
        self.save_path = "lm_ckpt"

        self.batch_size = 64
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.hidden_size = 200
        self.nlayers = 1
        self.dropout = 0.5
        self.epoch = 2

        self.train_len = 0
        self.test_len = 0
        self.valid_len = 0

        self.mode = "test"


# data_path="E:/study_series/2020_3/re_write_classify/data/"
# data_path="/mnt/data3/wuchunsheng/code/nlper/NLP_task/text_classification/my_classification_cnews/2020_3_30/text_classify/data/"

config = Config()
train_iter, valid_iter, test_iter, TEXT = generate_data(config)

model = RNNModel(config, TEXT).to(config.device)
model =load_model(config, model)

sen="体育"
res=test_sentence(config, model ,TEXT, sen)
print(res)

