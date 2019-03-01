#_*_coding:utf-8_*_

import pandas as pd
import codecs

train_data = pd.read_csv('../data/raw_data/train_set.csv')
test_data = pd.read_csv('../data/raw_data/test_set.csv')

word = codecs.open('../data/raw_data/word.csv','a','utf-8')
article = codecs.open('../data/raw_data/article.csv','a','utf-8')

word.writelines([text+'\n' for text in train_data['word_seg']])
word.writelines([text+'\n' for text in test_data['word_seg']])

article.writelines([text+'\n' for text in train_data['article']])
article.writelines([text+'\n' for text in test_data['article']])

