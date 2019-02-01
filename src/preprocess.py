#convert raw corpus, core dict and full dict to feedable form
from utils import *
from collections import Counter
import os

def read_corpus(raw_fname, one_wrd=False):
    """
    :raw_fname: file name of the raw text file, one word per line or otherwise.
    :one_wrd: whether file has one word per line and is separated by \n
    :returns: list of lines of the corpus
    """
    lines = read(raw_fname)
    if not one_wrd: return lines

    joined_corpus = []
    line          = []
    for wrd in lines:
        if len(wrd) != 0: 
            line.append(wrd)
        else:             
            joined_corpus.append(' '.join(line))
            line = []

    return joined_corpus

def create_vocab(corpus, dataset):
    """
    :corpus: list of lines, space tokenized
    :returns: nothing. creates vocab file in data/dataset path
    """

    if os.path.exists('../data/{}/vocab.txt'):
        print("overwriting existing vocab at ../data/{}/vocab.txt".format(dataset))

    vocab = Counter()
    for line in corpus:
        vocab.update(line.split(' '))                           #might need better tokenization here.

    sorted_wrds = [wrd for wrd, freq in vocab.most_common()]
    write(sorted_wrds, '../data/{}/vocab.txt'.format(dataset))
    

if __name__ == "__main__":
    dataset = 'BC5CDR' 
    corpus  = read_corpus('../data/{}/raw_corpus.txt'.format(dataset), one_wrd=True)
    create_vocab(corpus, dataset)
