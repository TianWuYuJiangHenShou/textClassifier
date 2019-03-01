#_*_coding:utf-8_*_

import pandas as pd
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('../data/raw_data/train_set.csv',sep=',')
test_data = pd.read_csv('../data/raw_data/test_set.csv',sep=',')
#统计所有的类别数
class_label_num = train_data.groupby('class').count()
'''
train_test_split:   randow_state:随机种子，random_state=0 ，每次split的结果都不一样
'''
train_dataset,val_dataset = train_test_split(train_data.iloc[1:],test_size=0.1,random_state=0)

train_dataset.to_csv('../data/split_data/train_set.csv',index=False)
val_dataset.to_csv('../data/split_data/val_set.csv',index=False)

train_dataset[['article','class']].to_csv('../data/article/train_set.csv',index=False)
val_dataset[['article','class']].to_csv('../data/article/val_set.csv',index=False)
test_data[['article']].to_csv('../data/article/test_set.csv',index=False)

train_dataset[['word_seg','class']].to_csv('../data/word/train_set.csv',index=False)
val_dataset[['word_seg','class']].to_csv('../data/word/val_set.csv',index=False)
test_data[['word_seg']].to_csv('../data/word/test_csv.csv',index=False)
