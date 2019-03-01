#_*_coding:utf-8_*_

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from models.BasicModule import BasicModule
from util.config import DefaultConfig

#卷积核的尺寸
kernel_sizes = [1,2,3,4,5]


class TextCNN(BasicModule):
    def __init__(self,opt,vectors = None):
        super(TextCNN,self).__init__()

        '''
        Embedding Layer
        '''
        self.embedding = nn.Embedding(opt.vocab_size,opt.embedding_dim)
        if vectors is not None:
            #读取外部值赋值
            self.embedding.weight.data.copy_(vectors)

        cons = [
            nn.Sequential(
                nn.Conv1d(in_channels=opt.embedding_dim,
                          out_channels=opt.kernal_num,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(opt.kernal_num),
                #inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels=opt.kernal_num,
                          out_channels=opt.kernal_num,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(opt.kernal_num),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=(opt.max_text_len - kernel_size * 2 + 2))
            )
            for kernel_size in kernel_sizes]

        self.convs = nn.Sequential(cons)

        self.fc = nn.Sequential(
            nn.Linear(5*opt.kernal_num,opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.label_size)
        )

        def forward(self, inputs):
            embeds = self.embedding(inputs)  #seq * batch *embed
            #nn.Embedding.permute()  将tensor的维度转换   1，2，0是原数据的维度下标
            conv_out = [conv(embeds.permute(1,2,0)) for conv in self.cons]
            conv_out = torch.cat(conv_out,dim=1)
            flatten = conv_out.view(conv_out.size(0),-1)
            logits = self.fc(flatten)
            return logits