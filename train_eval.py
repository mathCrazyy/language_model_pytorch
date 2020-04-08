
import torch
import torch.nn as nn
import torch.optim as optim
from utils import word_ids_to_sentence,word_sentence_to_ids
import numpy as np

def train(config,model,train_iter, valid_iter,test_iter):

    #optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.7, 0.99))
    optimizer = optim.Adam(model.parameters(), lr=1e-2, betas=(0.7, 0.99))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, config.epoch + 1):
        """One epoch of a training loop"""
        epoch_loss = 0
        model.train()
        for batch in train_iter:
            # reset the hidden state or else the model will try to backpropagate to the
            # beginning of the dataset, requiring lots of time and a lot of memory
            if(model.mode_type!="transformer"):
                model.reset_history()

            optimizer.zero_grad()

            text, targets = batch.text.to(config.device), batch.target.to(config.device)

            prediction = model(text).to(config.device)
            # pytorch currently only supports cross entropy loss for inputs of 2 or 4 dimensions.
            # we therefore flatten the predictions out across the batch axis so that it becomes
            # shape (batch_size * sequence_length, n_tokens)
            # in accordance to this, we reshape the targets to be
            # shape (batch_size * sequence_length)
            loss = criterion(prediction.view(-1, config.n_tokens), targets.view(-1)).to(config.device)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item() * prediction.size(0) * prediction.size(1)  ## 共有batch_size*bptt_len个loss
        epoch_loss /= config.train_len  ## 这些loss对训练文本长度做一个归一化，其他计算方式应该也是可以的

        # monitor the loss
        val_loss = 0
        model.eval()
        # model.train()
        for batch in valid_iter:
            if(model.mode_type!="transformer"):
                model.reset_history()
            text, targets = batch.text.to(config.device), batch.target.to(config.device)
            prediction = model(text).to(config.device)
            loss = criterion(prediction.view(-1, config.n_tokens), targets.view(-1)).to(config.device)
            val_loss += loss.item() * prediction.size(0) * prediction.size(1)
        val_loss /= config.valid_len

        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))
        torch.save(model.state_dict(), config.save_path)

def test(config, model, TEXT, test_iter):
    print("save_path: ", config.save_path)
    model.load_state_dict(torch.load(config.save_path))
    #print(model)
    b = next(iter(test_iter))
    print(b.text.shape)
    #print("输入: ", b.text[0])
    #print("输入的句子: ", word_ids_to_sentence(b.text[0],TEXT.vocab))
    #print("", word_sentence_to_ids(b.text[0],TEXT.vocab))


    print("单条数据: ",b.text[:,1].shape)

    inputs_word = word_ids_to_sentence(b.text.cuda().data, TEXT.vocab)
    #print(inputs_word)
    #print(len(inputs_word))

    arrs = model(b.text.cuda()).cuda().data.cpu().numpy()
    print(arrs.shape)
    preds = word_ids_to_sentence(np.argmax(arrs, axis=2), TEXT.vocab)
    return preds
    #print(preds)

def test_one_sentence(config, model , TEXT,test_iter):
    print("save_path: ", config.save_path)
    model.load_state_dict(torch.load(config.save_path))
    #print(model)
    b = next(iter(test_iter))
    print(b.text.shape)
    print("单条数据: ",b.text[:,1].shape)
    print("单条数据: ",b.text[:,1].view(-1,1).shape)
    inputs_word = word_ids_to_sentence(b.text[:,1].view(-1,1).cuda().data, TEXT.vocab)
    print("inputs_word: ", inputs_word)
    arrs = model(b.text[:,1].view(-1,1).cuda()).cuda().data.cpu().numpy()
    preds = word_ids_to_sentence(np.argmax(arrs, axis=2), TEXT.vocab)
    print("preds----------", preds)



def test_sentence(config, model, TEXT, sentence):
    model.load_state_dict(torch.load(config.save_path))
    #print(model)
    inputs = torch.Tensor([TEXT.vocab.stoi[one] for one in sentence]).long().to(config.device)
    print("inputs: ", inputs)
    inputs = inputs.view(-1, 1)
    #print("inputs: ", inputs)
    print("inputs shape: ", inputs.shape)
    #print(inputs)
    arrs = model(inputs)
    print("arrs shape: ",arrs.shape)
    #print(arrs)
    x=np.sum(np.array(arrs.detach().cpu()),axis=2)
    preds = word_ids_to_sentence(np.argmax(arrs.detach().cpu(), axis=2), TEXT.vocab)
    #print(x)
    print("preds: ",preds)

    return preds

def load_model(config, model):
    model.load_state_dict(torch.load(config.save_path))
    return model

