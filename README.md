# textClassifier
pytorch解决文本多分类问题
项目使用工具：

Python3

分词与词向量：jieba、word2vec、gensim

Pytorch版本：1.0(一开始使用的是pytorch0.4.1,但是在做显存优化的时候使用到了Pytorch1.0的checkpoint特性)

模型：LSTM、GRU、RCNN、bigru_att等

项目采用了6种方法来解决显存溢出的问题，有效减少了的显存的使用。

1、尽可能的inplace化，修改网络结构，将BN层、激活层打包成inplace，打包时再重新计算

训练使用main.py ,model = *_ABN.py

2、使用NVIDIA的apex进行float16精度混合运算  

main_float_16.py 是加入了apex后进行的修改

3、使用pytorch1.0的checkpoint特性
checkpoint通过交换计算内存来工作，并不会存储整个计算图的所有中间激活用于后向传播，而是会在反向传播中重新计算他们
例：models.GRU_checkpoint.py

总的效果，inplace_abn、apex可以减少50%的显存，checkpoint可以减少90%的显存！！！

还有一些其他的trick，但是减少显存效果很小。
pytorch减少显存参考：
https://www.zhihu.com/question/274635237/answer/582278076?utm_source=wechat_session&utm_medium=social&utm_oi=42374529024000

pytorch checkpoint源码参考：
https://pytorch.org/docs/master/_modules/torch/utils/checkpoint.html#checkpoint

ps:尝试了checkpoint与inplace_abn的组合使用，前期效果较好，后期模型渐渐饱和后，参数量剧增，很快就out of memory
