#_*_coding:utf-8_*_

import torch
import time

class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()
        self.modle_name = str(type(self))
#        print(self.modle_name)

    def get_optimizer(self,lr1,lr2 = 0,weight_decay=0):
        #id() 函数用于获取对象的内存地址。
        embed_params = list(map(id,self.embedding.parameters()))
        #filter  过滤函数
        base_params = filter(lambda p:id(p) not in embed_params,self.parameters())
        optimizer = torch.optim.Adam([
            {'params':self.embedding.parameters()},
            {'params':base_params,'lr':lr1,'weight_decay':weight_decay}
        ],lr=lr2)
        return optimizer

