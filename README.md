# language_model_pytorch
使用pytorch 的languageModelingDataset生成语言模型训练用到的train, valid, 以及test，然后使用LSTM训练一个基本的语言模型。

## 使用到的数据
包括训练集，验证集，测试集以及抽取的bert向量。
链接：https://pan.baidu.com/s/1iv44fBV5cUbYvahsrie0Kw 
提取码：q3ua

## 加载数据
依然使用cnews的数据，没有对数据集进行详细的琢磨。

## 代码主体
main.py 参考博文的原始代码，以英文为基础的，里面也有挺多问题，没有详细修正。  
main_zh.py 训练主文件  
model.py 模型的具体实现  
test.py 测试模型，输入一个字符串，输出一个等长的字符串。  
train_eval.py 训练、验证主代码  
utils.py 文件加载以及字符转换。  


##参考
http://mlexplained.com/2018/02/15/language-modeling-tutorial-in-torchtext-practical-torchtext-part-2/
