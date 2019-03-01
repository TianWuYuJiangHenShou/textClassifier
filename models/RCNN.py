#_*_coding:utf-8_*_

#RCNN  -> RNN、LSTM后接一层CNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BasicModule import BasicModule
from util.config import DefaultConfig

'''
torch.gather(a,dim,index)聚合功能
index的尺寸与a一样，index表示的是索引
'''


def kmax_pooling(x,dim,k):
    index = x.topk(k,dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim,index)

class RCNN(BasicModule):
    def __init__(self,args,vectors=None):
        super(RCNN,self).__init__()
        self.kmax_k = args.kmax_pooling
        self.config = args

        self.embedding = nn.Embedding(args.vocab_size,args.embedding_dim)
        if vectors is not None:
            self.embedding.weight.data.copy_(vectors)

        self.gru = nn.GRU(
            input_size= args.embedding_dim,
            hidden_size = args.hidden_dim,
            num_layers = args.lstm_layers,
            batch_first=False,
            dropout=args.lstm_dropout,
            bidirectional=True
        )
        #加一个卷积层
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=args.hidden_dim * 2 + args.embedding_dim,out_channels=args.rcnn_kernel,kernel_size=3),
            nn.BatchNorm1d(args.rcnn_kernel),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=args.rcnn_kernel,out_channels=args.rcnn_kernel,kernel_size=3),
            nn.BatchNorm1d(args.rcnn_kernel),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(args.kmax_pooling * args.rcnn_kernel,args.linear_hidden_size),
            nn.BatchNorm1d(args.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(args.linear_hidden_size,args.label_size)
        )


    def forward(self, input):
        embed = self.embedding(input)
        out = self.gru(embed)[0].permute(1,2,0)
        out = torch.cat((out,embed.permute(1,2,0)),dim = 1)
        conv_out = kmax_pooling(self.conv(out),2,self.kmax_k)
        flatten = conv_out.view(conv_out.size(0),-1)
        logits = self.fc(flatten)
        return logits