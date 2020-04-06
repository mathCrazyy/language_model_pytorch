import torch
import torch.nn as nn
from torch.autograd import Variable as V
import codecs
import numpy as np

class RNNModel(nn.Module):
    def __init__(self,config, TEXT, dropout=0.5, tie_weights=True):
        super(RNNModel, self).__init__()

        self.nhid, self.nlayers = config.hidden_size, config.nlayers
        if config.mode=="train" or config.mode=="eval":
            self.bsz= config.batch_size
        else:
            self.bsz=1
        dropout=config.dropout
        self.drop = nn.Dropout(dropout)
        self.device=config.device

        lines=codecs.open(config.data_ori+config.embedding_path,encoding="utf-8")

        embedding_vec = [line.replace("\n", "") for line in lines if line[0] in TEXT.vocab.stoi][1:-1]
        embeddings=np.random.rand(len(embedding_vec),config.embedding_dim)

        for index, line in enumerate(embedding_vec):
            line_seg = line.split(" ")
            try:
                embeddings[index] = [float(one) for one in line_seg[1:]]
            except:
                pass

        pretrained_weight = np.array(embeddings)

        ## 词典大小，单词的维度
        self.encoder = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.encoder.weight.data.copy_(torch.from_numpy(pretrained_weight))

        #self.rnn = nn.LSTM(self.ntokens, self.nhid, self.nlayers, dropout=dropout)
        self.rnn = nn.LSTM(embeddings.shape[1], self.nhid, self.nlayers, dropout=dropout)
        self.decoder = nn.Linear(self.nhid, embeddings.shape[0])
        self.init_weights()
        self.hidden = self.init_hidden(self.bsz)  # the input is a batched consecutive corpus
        # therefore, we retain the hidden state across batches
        config.n_tokens=embeddings.shape[0]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        emb = self.drop(self.encoder(input)).to(self.device)
        #print("inputs shape: ", emb.shape)
        output, self.hidden = self.rnn(emb, self.hidden)
        #print("output_shape: ", output.shape)
        #print("hidden_shape: ", self.hidde02n.shape)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (V(weight.new(self.nlayers, self.bsz, self.nhid).zero_().cuda()),
                V(weight.new(self.nlayers, self.bsz, self.nhid).zero_()).cuda())

    def reset_history(self):
        """Wraps hidden states in new Variables, to detach them from their history."""
        # self.hidden = tuple(V(v.data) for v in self.hidden)
        self.hidden = tuple(V(v.detach()) for v in self.hidden)
