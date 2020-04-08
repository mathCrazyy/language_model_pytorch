import torch.nn as nn
import torch
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import codecs
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self,config):
        super(PositionalEncoding,self).__init__()

        self.dropout=nn.Dropout(p=config.dropout)

        pe=torch.zeros(config.max_len, config.embedding_dim)
        position=torch.arange(0,config.max_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,config.embedding_dim,2).float()*(-math.log(10000.0)/config.embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe=pe.unsqueeze(0).transpose(0,1)## 少了这一条，直接导致forward维度对应不上，一个batch使用了一组position向量。

        self.register_buffer("pe",pe)

    def forward(self,x):
        #x = x.permute(1, 0, 2)
        #print("x_shape: ::  ", x.shape)
        #print("pe shape : ",self.pe[:x.size(0), :].shape)
        x = x + self.pe[:x.size(0), :]
        #x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self,config,TEXT):
        super(TransformerModel,self).__init__()
        self.embedding_dim=config.embedding_dim
        self.mode_type="transformer"
        self.src_mask=None
        self.pos_encoder=PositionalEncoding(config)
        encoder_layers=TransformerEncoderLayer(config.embedding_dim, config.nhead, config.hidden_size, config.dropout)
        self.transformer_encoder=TransformerEncoder(encoder_layers, config.nlayers)

        lines = codecs.open(config.data_ori + config.embedding_path, encoding="utf-8")
        #print("长度:::::::::", len(list(lines)))

        #print("start.........")
        #print(TEXT.vocab.stoi)
        #embedding_vec = [line.replace("\n", "") for line in lines if line.split(" ")[0] in TEXT.vocab.stoi][1:-1]
        #embedding_vec = [line.replace("\n", "").split(" ")[0] for line in lines if line.split(" ")[0] in TEXT.vocab.stoi]
        #embedding_vec = [line.replace("\n", "") for line in lines][1:-1]

        #print("根据词典筛选: ", len(embedding_vec)," 词典的大小: ",len(TEXT.vocab.stoi))
        #for one in TEXT.vocab.stoi:
        #    if one not in embedding_vec:
                #print("not in : ", one,len(one))
        embedding_vec=TEXT.vocab.vectors
        print("embedding_vec_shape: ",len(embedding_vec),"词典传过来的大小: ",len(TEXT.vocab.vectors))

        #print(embedding_vec)
        embeddings = np.random.rand(len(embedding_vec), config.embedding_dim)
        #print(len(TEXT.vocab))
        #print("embedddings: shap e: ",embeddings.shape)



        pretrained_weight = np.array(embeddings)


        self.encoder=nn.Embedding(embeddings.shape[0],config.embedding_dim)
        self.decoder=nn.Linear(config.embedding_dim,embeddings.shape[0])
        if config.mode!="test":
            self.init_weights()
        config.n_tokens=embeddings.shape[0]


        pass
    def _generate_square_subsequent_mask(self,sz):
        mask=(torch.triu(torch.ones(sz,sz))==1).transpose(0,1)
        mask=mask.float().masked_fill(mask==0,float("-inf")).masked_fill(mask==1,float(0.0))
        return mask

    def init_weights(self):
        initrange=0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange,initrange)

        pass
    def forward(self,inputs,has_mask=True):
        if has_mask:
            device=inputs.device
            if self.src_mask is None or self.src_mask.size(0)!=len(inputs):
                mask=self._generate_square_subsequent_mask(len(inputs)).to(device)
                self.src_mask=mask
        else:
            self.src_mask=None
        #print("1: ", inputs)
        inputs=self.encoder(inputs)*math.sqrt(self.embedding_dim)
        inputs=self.pos_encoder(inputs)
        #print("2: ", inputs)
        output=self.transformer_encoder(inputs, self.src_mask)
        #print("3: output: ", output)
        #print("output shape11: ", output.shape)
        output=self.decoder(output)
        #print("4:", output)
        #print("output shape22: ", output.shape)
        #print(F.log_softmax(output, dim=2))
        return F.log_softmax(output, dim=2)

