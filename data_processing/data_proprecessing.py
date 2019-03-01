#_*_coding:utf-8_*_

import numpy as np
import pandas as pd
from torchtext.vocab import Vectors
from tqdm import tqdm
from torchtext import data
from torch.nn import init
import random
import os

class DataSetLoad(data.Dataset):
    name = 'DataSet Load'
    def sort_key(ex):
        return len(ex.text)

    def shuffle(self,text):
        #np.random.permutation  == shuffle 打乱顺序
        text = np.random.permutation(text.strip().split())
        return ' '.join(text)

    #神经网络中Dropout的思想
    def dropout(self,text,p = 0.5):
        text = text.strip().split()
        len_ = len(text)
        #np.random.choice()   从数组中随机选数据
        indexs = np.random.choice(len_,int(len_ * p))
        for i in indexs:
            text[i] = ''
        return ' '.join(text)
    '''
    构造函数
    *args：（表示的就是将实参中按照位置传值，多出来的值都给args，且以元祖的方式呈现）
    **kwargs：（表示的就是形参中按照关键字传值把多余的传值以字典的方式呈现）
    '''
    def __init__(self,path,text_field,label_field,text_type='word',test=False,aug=False,**kwargs):
        fields = [('text',text_field),('label',label_field)]
        examples = []
        csv_data = pd.read_csv(path)

        if text_type == 'word':
            text_type = 'word_seg'

        if test:
            #测试集，没有label，不加载
            for word in tqdm(csv_data[text_type]):
                #word是一个list
                examples.append(data.Example.fromlist([word,0],fields))
        else:
            #python3 zip(A,B) 返回的是可迭代的元祖对象
            for text,label in tqdm(zip(csv_data[text_type],csv_data['class'])):
                if aug:
                    #数据增强
                    rate = random.random()
                    if rate > 0.5:
                        text = self.dropout(text)
                    else:
                        text = self.shuffle(text)
                examples.append(data.Example.fromlist([text,label - 1],fields))

        super(DataSetLoad,self).__init__(examples,fields,**kwargs)


#加载数据并构建数据集
def load_data(opt):

    TEXT = data.Field(sequential=True,fix_length=opt.max_text_len)
    LABEL = data.Field(sequential=False,use_vocab=False)
    train_path = opt.data_path + opt.text_type +'/'+ 'train_set.csv'
    val_path = opt.data_path + opt.text_type +'/'+ 'val_set.csv'
    test_path = opt.data_path + opt.text_type +'/'+ 'test_set.csv'

    #aug=True -> 做数据增强
    if opt.aug:
        print('make augmentation datasets!')

    train = DataSetLoad(train_path,text_field=TEXT,label_field=LABEL,text_type=opt.text_type,test=False,aug=opt.aug)
    #验证集和测试集不需要做数据增强
    val = DataSetLoad(val_path,text_field=TEXT,label_field=LABEL,text_type=opt.text_type,test=False)
    test = DataSetLoad(test_path,text_field=TEXT,label_field=LABEL,text_type=opt.text_type,test=True)

    '''
    构建词表的数据集
    '''
    cache = '.vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    embedding_path = opt.embedding_path+opt.text_type+'/'+opt.text_type+'_300'+'.txt'
    '''
    加载Word2vec训练的词向量
    '''
    vectors = Vectors(name=embedding_path,cache=cache)
#    vectors.unk_init = init.xavier_uniform_  #不存在的token  pytorch定义网络输入的初始化方式
    #构建词表
    TEXT.build_vocab(train,val,test,min_freq=5,vectors=vectors)

    '''
    训练神经网咯时，是对一个batch的数据进行操作，因此需要对构建的数据集创建迭代器
    '''
    train_iter = data.BucketIterator(dataset=train,batch_size=opt.batch_size,shuffle=True,sort_within_batch=False,repeat=False,device=opt.device)
    val_iter = data.BucketIterator(dataset=val,batch_size=opt.batch_size,shuffle=False,sort=False,repeat=False,device=opt.device)
    # 在 test_iter, shuffle, sort, repeat一定要设置成 False, 要不然会被 torchtext 搞乱样本顺序
    test_iter = data.BucketIterator(dataset=test,batch_size=opt.batch_size,shuffle=False,sort=False,repeat=False,device=opt.device)
    return train_iter,val_iter,test_iter,len(TEXT.vocab),TEXT.vocab.vectors


