import numpy as np
import torch


USE_GPU = True
BATCH_SIZE = 32
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torchtext
from torchtext import data

import spacy

from spacy.symbols import ORTH
def spacy_tok(x):
    return [tok.lower() for tok in x]

TEXT = data.Field(lower=True, tokenize=spacy_tok)

from torchtext.datasets import WikiText2

train, valid, test = WikiText2.splits(TEXT) # loading custom datasets requires passing in the field, but nothing else.

TEXT.build_vocab(train, vectors="glove.6B.200d")
train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
    (train, valid, test),
    batch_size=BATCH_SIZE,
    bptt_len=30, # this is where we specify the sequence length
    device=(0 if USE_GPU else -1),
    repeat=False)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as V


class RNNModel(nn.Module):
    def __init__(self, ntoken, ninp,
                 nhid, nlayers, bsz,
                 dropout=0.5, tie_weights=True):
        super(RNNModel, self).__init__()
        self.nhid, self.nlayers, self.bsz = nhid, nlayers, bsz
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        #self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.rnn = nn.LSTM(ninp, nhid, nlayers)
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()
        self.hidden = self.init_hidden(bsz) # the input is a batched consecutive corpus
                                            # therefore, we retain the hidden state across batches

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        #emb = self.drop(self.encoder(input)).to(device)
        emb = self.encoder(input).to(device)
        output, self.hidden = self.rnn(emb, self.hidden)
        #output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (V(weight.new(self.nlayers, bsz, self.nhid).zero_().cuda()),
                V(weight.new(self.nlayers, bsz, self.nhid).zero_()).cuda())
        #return (V(weight.new(self.nlayers, bsz, self.nhid).zero_()),
        #        V(weight.new(self.nlayers, bsz, self.nhid).zero_()))
    
    def reset_history(self):
        """Wraps hidden states in new Variables, to detach them from their history."""
        self.hidden = tuple(V(v.data) for v in self.hidden)


weight_matrix = TEXT.vocab.vectors
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = RNNModel(weight_matrix.size(0),
                 weight_matrix.size(1), 200, 1, BATCH_SIZE)

model.encoder.weight.data.copy_(weight_matrix);

if USE_GPU:
    print("--")
    #model.cuda()
    model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.7, 0.99))

n_epochs = 2
n_tokens = weight_matrix.size(0)
from tqdm import tqdm

def train_epoch(epoch):
    """One epoch of a training loop"""
    epoch_loss = 0
    model.train()
    #model.eval()
    for batch in train_iter:
        # reset the hidden state or else the model will try to backpropagate to the
        # beginning of the dataset, requiring lots of time and a lot of memory
        model.reset_history()
        
        optimizer.zero_grad()
        
        text, targets = batch.text.to(device), batch.target.to(device)
        
        prediction = model(text).to(device)
        # pytorch currently only supports cross entropy loss for inputs of 2 or 4 dimensions.
        # we therefore flatten the predictions out across the batch axis so that it becomes
        # shape (batch_size * sequence_length, n_tokens)
        # in accordance to this, we reshape the targets to be
        # shape (batch_size * sequence_length)
        #print("targets--: ", targets.shape)
        #print("prediction-- : ", targets.shape)
        loss = criterion(prediction.view(-1, n_tokens), targets.view(-1)).to(device)
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item() * prediction.size(0) * prediction.size(1)

    epoch_loss /= len(train.examples[0].text)

    # monitor the loss
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in valid_iter:
            model.reset_history()
            text, targets = batch.text.to(device), batch.target.to(device)
            prediction = model(text).to(device)
            loss = criterion(prediction.view(-1, n_tokens), targets.view(-1))
            val_loss += loss.item() * text.size(0)
        val_loss /= len(valid.examples[0].text)
    
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))


for epoch in range(1, n_epochs + 1):
    train_epoch(epoch)


def word_ids_to_sentence(id_tensor, vocab, join=None):
    """Converts a sequence of word ids to a sentence"""
    print(type(id_tensor))
    #print(id_tensor)
    if isinstance(id_tensor, torch.LongTensor):
        ids = id_tensor.transpose(0, 1).contiguous().view(-1)
    elif isinstance(id_tensor, np.ndarray):
        ids = id_tensor.transpose().reshape(-1)
    else:
        ids = id_tensor.transpose(0, 1).contiguous().view(-1)
    #print("ids", ids)

    batch = [vocab.itos[ind] for ind in ids]  # denumericalize
    if join is None:
        return batch
    else:
        return join.join(batch)


b = next(iter(valid_iter))

inputs_word=word_ids_to_sentence(b.text.cuda().data, TEXT.vocab, join=' ')[:210]
print(inputs_word)

arrs = model(b.text.cuda()).cuda().data.cpu().numpy()
preds=word_ids_to_sentence(np.argmax(arrs, axis=2), TEXT.vocab, join=' ')[:210]

print(preds)









