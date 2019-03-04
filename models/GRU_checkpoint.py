#_*_coding:utf-8_*_

import torch.nn as nn
import torch.nn.functional as F
from models.BasicModule import BasicModule
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.checkpoint import checkpoint

def kmax_pooling(x,dim,k):
    # torch.Tensor.topk()的输出有两项，前一项为值，后一项为索引
    index = x.topk(k,dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim,index)

class GRU_checkpoint(BasicModule):
    def __init__(self,config,vectors=None):
        super(GRU_checkpoint,self).__init__()
        self.opt = config
        self.kmax_pooling = config.kmax_pooling

        #GRU
        self.embedding = nn.Embedding(config.vocab_size,config.embedding_dim)
        if vectors is not None:
            self.embedding.weight.data.copy_(vectors)
        self.bigru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.lstm_layers,
            batch_first=False,
            dropout=config.lstm_dropout,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(self.kmax_pooling * (config.hidden_dim * 2),config.linear_hidden_size),
            nn.BatchNorm1d(config.linear_hidden_size),
            nn.ReLU(inplace=True),
            # nn.Linear(config.linear_hidden_size,config.label_size)
        )
        self.linear = nn.Linear(config.linear_hidden_size, config.label_size)
    #对LSTM所有隐含层的输出做kmax_pooling
    def forward(self, input):
        embed = self.embedding(input)
        input_var = embed.detach()
        input_var.requires_grad = True
        out = checkpoint(self.bigru,input_var)
        out1 = out[0].permute(1,2,0)
        pooling = kmax_pooling(out1, 2, self.kmax_pooling)  # batch * hidden *kmax
        flatten = pooling.view(pooling.size(0), -1)
        out2 = checkpoint_sequential(self.fc,2,flatten)
        logits = self.linear(out2)
        return logits


