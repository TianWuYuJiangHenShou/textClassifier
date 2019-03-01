#_*_coding:utf-8_*_

from gensim import models
from gensim.models.word2vec import LineSentence
import codecs
import multiprocessing
import logging
import os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        
word = '../data/raw_data/word.csv'
article = '../data/raw_data/article.csv'

file_word= codecs.open(word,'r','utf-8')
model = models.Word2Vec(LineSentence(file_word),sg=0,size=192,window=5,min_count=5,workers=multiprocessing.cpu_count())
#model.save('../data/model/word.bin')
model.wv.save_word2vec_format('../data/model/word/word.txt')

file_article= codecs.open(article,'r','utf-8')
model = models.Word2Vec(LineSentence(file_article),sg=0,size=192,window=5,min_count=5,workers=multiprocessing.cpu_count())
#model.save('../data/model/article.bin')
model.wv.save_word2vec_format('../data/model/article/article.txt')
