from torchtext.data import Field
from torchtext.data import Iterator,BucketIterator
from torchtext.datasets import LanguageModelingDataset
from torchtext.vocab import Vectors
from torchtext import data
import torch
import numpy as np
from torchtext.data import TabularDataset

def generate_data(config):
    ## 不同字段的操作定义
    path_train = config.data_ori + config.train_path
    path_valid = config.data_ori + config.valid_path
    path_test = config.data_ori + config.test_path

    tokenizer = lambda x: [one for one in x]

    TEXT = Field(batch_first=False, tokenize=tokenizer)
    train = LanguageModelingDataset(path=path_train, text_field=TEXT)
    valid = LanguageModelingDataset(path=path_valid, text_field=TEXT)
    test = LanguageModelingDataset(path=path_test, text_field=TEXT)

    config.train_len=len(train)
    print("example len::  ", len(train.examples))
    print("train_examples[0] len: ", len(train.examples[0].text))
    print("valid_examples[0] len: ", len(valid.examples[0].text))

    train_iter, valid_iter, test_iter=data.BPTTIterator.splits(
        (train,valid, test),
        batch_size=config.batch_size,
        bptt_len=50,
        device=config.device
    )
    #TEXT.build_vocab(train)
    vectors=Vectors(name=config.data_ori+config.embedding_path,cache="./")
    TEXT.build_vocab(train,max_size=config.vocab_maxsize, min_freq=config.vocab_minfreq, vectors=vectors)
    TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)

    config.train_len=len(train.examples[0].text)
    config.test_len=len(test.examples[0].text)
    config.valid_len=len(valid.examples[0].text)

    print("词汇量： ", len(TEXT.vocab))

    return train_iter, valid_iter, test_iter, TEXT

def check_data(iter, TEXT):
    for batch in iter:
        print("batch >>>>> ", type(batch))
        print("text shape: ", batch.text.size())
        print("target shape: ", batch.target.size())
        print(batch.text.size())
        # print(batch.text)
        for example_index in range(batch.text.size()[1]):
            ## 一个batch里面有些什么样本呢
            # for index in batch.text[:,example_index]:

            line = "".join([TEXT.vocab.itos[index] for index in batch.text[:, example_index]])
            print("sentence: ", line)
            # print("target : ", batch.target)
            target = "".join([TEXT.vocab.itos[index] for index in batch.target[:, example_index]])
            print("target: ", target)


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


def word_sentence_to_ids(sentence, vocab):
    """Converts a sequence of word ids to a sentence"""

    #print("ids", ids)


    sentence_di = [vocab.stoi[ele] for ele in sentence]  # denumericalize
    return sentence_di

#if __name__=="__main__":

    #train_iter, valid_iter, test_iter， =generate_data(file_path)

    #a=list(train_iter)
    #print(a[0])
    #print(a[0].context)
