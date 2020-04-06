from tensorflow.python import pywrap_tensorflow
import numpy as np
ckpt_path="/mnt/data3/wuchunsheng/code/nlper/NLP_task/word2vec/bert/chineseGLUE/baselines/models/roberta/prev_trained_model/bert/publish/bert_model.ckpt"
reader=pywrap_tensorflow.NewCheckpointReader(ckpt_path)
param_dict=reader.get_variable_to_shape_map()
emb=reader.get_tensor("bert/embeddings/word_embeddings")
vocab_file="/mnt/data3/wuchunsheng/code/nlper/NLP_task/word2vec/bert/chineseGLUE/baselines/models/roberta/prev_trained_model/bert/publish/vocab.txt"
vocab=open(vocab_file).read().split("\n")

out=open("bert_embedding","w")
out.write(str(emb.shape[0])+" "+str(emb.shape[1])+"\n")
for index in range(0, emb.shape[0]):
    out.write(vocab[index]+" "+" ".join([str(i) for i in emb[index,:]])+"\n")
out.close()
