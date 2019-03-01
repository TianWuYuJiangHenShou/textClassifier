# textClassifier
pytorch解决文本多分类问题
项目使用工具：

Python3

分词与词向量：jieba、word2vec、gensim

Pytorch版本：1.0(一开始使用的是pytorch0.4.1,但是在做显存优化的时候使用到了Pytorch1.0的checkpoint特性)

模型：LSTM、GRU、RCNN、bigru_att等

项目采用了2种方法来解决显存溢出的问题，有效减少了50%的显存。

1、尽可能的inplace化，修改网络结构，将BN层、激活层打包成inplace，打包时再重新计算

训练使用main.py ,model = *_ABN.py

2、使用NVIDIA的apex进行float16精度混合运算  

main.py 是加入了apex后进行的修改
