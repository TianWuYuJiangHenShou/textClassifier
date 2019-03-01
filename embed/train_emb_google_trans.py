import word2vec
import fire

paths = ['../data/raw_data/word.csv', '../data/raw_data/article.csv']
sizes = [300]


def tran(path):
    model = word2vec.load(path)
    vocab, vectors = model.vocab, model.vectors

    new_path = path.split('.bin')[0] + '.txt'
    f = open(new_path, 'w')
    for word, vector in zip(vocab, vectors):
        f.write(str(word) + ' ' + ' '.join(map(str, vector)) + '\n')


for path in paths:
    for size in sizes:
        root = '../data/model/'
        label = path.split('.csv')[0].split('/')[3]
        emb_path = root + label+'/'+label+'_'+str(size)+'.bin'
        word2vec.word2vec(path, emb_path, min_count=5, size=size, verbose=True)
        tran(emb_path)
