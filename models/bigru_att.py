#_*_coding:utf-8_*_


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BasicModule import BasicModule

class bigru_att(BasicModule):
    def __init__(self,args,vectors=None):
        self.args = args
        super(bigru_att, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.gru_layers = args.lstm_layers

        self.embedding = nn.Embedding(args.vocab_size,args.embedding_dim)

        if vectors is not None:
            self.embedding.weight.data.copy_(vectors)

        self.bigru = nn.GRU(args.embedding_dim,self.hidden_dim // 2,num_layers=self.gru_layers,bidirectional=True)
        self.weight_w = nn.Parameter(torch.Tensor(self.hidden_dim,self.hidden_dim))
        self.weight_proj = nn.Parameter(torch.Tensor(self.hidden_dim,1))
        self.fc = nn.Linear(self.hidden_dim,args.label_size)

        #初始化函数
        nn.init.uniform(self.weight_w,-0.1,0.1)
        nn.init.uniform(self.weight_proj,-0.1,0.1)

    def forward(self, input):
        embeds = self.embedding(input)
        out,hidden = self.bigru(embeds)
        x = out.permute(1,0,2)
        u = torch.tanh(torch.matmul(x,self.weight_w))
        att = torch.matmul(u,self.weight_proj)
        att_score = F.softmax(att,dim=1)
        score = x * att_score  #概率
        feat = torch.sum(score,dim=1)
        y = self.fc(feat)
        return y