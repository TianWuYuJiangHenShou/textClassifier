#_*_coding:utf-8_*_

from gensim import models
from gensim.models.word2vec import LineSentence
import codecs
import multiprocessing
import logging
import os
import word2vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

word = '../data/raw_data/word.csv'
article = '../data/raw_data/article.csv'

#google word2vec
word_model = '../data/model/word/google_word.bin'
word2vec.word2vec(word, word_model, min_count=5, size=300, verbose=True)


article_model = '../data/model/article/google_article.bin'
word2vec.word2vec(article, article_model, min_count=5, size=300, verbose=True)



