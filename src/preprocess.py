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

    assert(len(line) == 0)                  #ensure file ends in a blank line
    return joined_corpus

def create_vocab(corpus, dataset):
    """
    :corpus: list of lines, space tokenized
    :returns: nothing. creates vocab file in data/dataset path
    """

    if os.path.exists('../data/{}/vocab.txt'.format(dataset)):
        print("overwriting existing vocab at ../data/{}/vocab.txt".format(dataset))

    vocab = Counter()
    for line in corpus:
        vocab.update(line.split(' '))                           #might need better tokenization here.

    sorted_wrds = [wrd for wrd, freq in vocab.most_common()]
    write(sorted_wrds, '../data/{}/vocab.txt'.format(dataset))

def ck_to_txt(ck_fname, dataset, split='dev'):
    """
    :ck_fname: dev/test file provided by autoNER
    :returns: nothing. Writes dev/test file to ../data/dataset/dev/in.txt, out_tb.txt and out_lbl.txt
    """
    lines = read(ck_fname)

    text, tb, lbl             = [], [], []
    tmp_text, tmp_tb, tmp_lbl = [], [], []

    for line in lines:
        assert(len(line.split()) <= 3)

        if len(line) == 0:
            text.append(' '.join(tmp_text))
            tb.append(' '.join(tmp_tb))
            lbl.append(' '.join(tmp_lbl))
            tmp_text, tmp_tb, tmp_lbl = [], [], []

        else:
            tmp_text.append(line.split(' ')[0])
            tmp_tb.append(line.split(' ')[1])
            tmp_lbl.append(line.split(' ')[2])


    assert(len(tmp_text) == 0)                  #ensure file ends in a blank line
    write(text, '../data/{}/{}/in.txt'.format(dataset, split))
    write(tb, '../data/{}/{}/in_tb.txt'.format(dataset, split))
    write(lbl, '../data/{}/{}/in_lbl.txt'.format(dataset, split))
    
def create_train_data(corpus, core_dict, full_dict, dataset):
    """
    :corpus: list of lines
    :core_dict: each line is label <tab> entity
    :full_dict: each line is entity
    :returns: list of shape of corpus. 1 denotes break b'w w_i and w_{i-1}, 0 denotes tie, -1 denotes unk
    """
    pass
    

if __name__ == "__main__":
    dataset = 'BC5CDR' 
    corpus  = read_corpus('../data/{}/raw_corpus.txt'.format(dataset), one_wrd=True)
    create_vocab(corpus, dataset)
    ck_to_txt('../data/BC5CDR/truth_dev.ck', dataset, split='dev')
    ck_to_txt('../data/BC5CDR/truth_test.ck', dataset, split='test')
