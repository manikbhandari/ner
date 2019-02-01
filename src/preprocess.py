#convert raw corpus, core dict and full dict to feedable form
from utils import *
from collections import Counter
from nltk import ngrams
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

    assert(len(line) == 0)                                                  #ensure file ends in a blank line
    return joined_corpus

def read_core_dict(core_dict_fname):
    """
    :core_dict_fname: name of domain specific dictionary
    :returns: dict of label: entity
    """
    lines = read(core_dict_fname)
    core_dict = {}
    for line in lines:
        core_dict[line.split('\t')[1]] = line.split('\t')[0]

    return core_dict

def read_full_dict(full_dict_fname):
    """
    :full_dict_fname: name of full dictionary (this includes core dictionary provided by AutoNER)
    :returns: dict of entity: 1
    """
    lines = read(full_dict_fname)
    lines = {wrd: 1 for wrd in lines}
    return lines


def create_vocab(corpus, dataset):
    """
    :corpus: list of lines, space tokenized
    :returns: nothing. creates vocab file in data/dataset path
    """

    if os.path.exists('../data/{}/vocab.txt'.format(dataset)):
        print("overwriting existing vocab at ../data/{}/vocab.txt".format(dataset))

    vocab = Counter()
    for line in corpus:
        vocab.update(line.split(' '))                                       #might need better tokenization here.

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


    assert(len(tmp_text) == 0)                                              #ensure file ends in a blank line
    write(text, '../data/{}/{}/in.txt'.format(dataset, split))
    write(tb,   '../data/{}/{}/out_tb.txt'.format(dataset, split))
    write(lbl,  '../data/{}/{}/out_lbl.txt'.format(dataset, split))
    
def create_train_data(corpus, core_dict, full_dict, dataset):
    """
    :corpus: list of lines
    :core_dict: each line is label <tab> entity
    :full_dict: each line is entity
    :returns: list of shape of corpus. 1 denotes break b'w w_i and w_{i-1}, 0 denotes tie, -1 denotes unk
    """
    N = max([len(ent.split()) for ent in core_dict])
    N = max([len(ent.split()) for ent in full_dict] + [N])

    tie_or_break = []
    labels       = []
    for i, line in enumerate(corpus):
        n     = max(N, len(line.split()))
        grams = []
        tb    = ['I' for _ in range(len(line.split()))]
        lbl   = ['None' for _ in range(len(line.split()))]

        while n > 0:
            grams = list(ngrams(line.split(), n))

            for pos, gram in enumerate(grams):                              #pos is the word number in line of the first word of this ngram
                gram = ' '.join(list(gram))

                if gram in core_dict:                                       #case senseitive checking
                    if tb[pos] == 'I' and tb[pos + n - 1] == 'I':           #to avoid consecutive ngrams to be brought together.
                        tb[pos + 1: pos + n] = ['O']*(n-1)

                    lbl[pos: pos + n] = [core_dict[gram]]*n

                elif gram in full_dict:
                    if tb[pos] == 'I' and tb[pos + n - 1] == 'I':
                        tb[pos + 1: pos + n] = ['O']*(n-1)                  #not adding labels here since they are None
            n -= 1

        tie_or_break.append(' '.join(tb))
        labels.append(' '.join(lbl))
        if i % 1000 == 0: print("finished line {}".format(i))

    write(corpus,       '../data/{}/train/in.txt'.format(dataset))
    write(tie_or_break, '../data/{}/train/out_tb.txt'.format(dataset))
    write(labels,       '../data/{}/train/out_lbl.txt'.format(dataset))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NER')
    parser.add_argument('-dataset', dest="dataset", default='BC5CDR', help='')
    args = parser.parse_args()

    corpus     = read_corpus('../data/{}/raw_corpus.txt'.format(dataset), one_wrd=True)
    core_dict  = read_core_dict('../data/{}/dict_core.txt'.format(dataset))
    full_dict  = read_full_dict('../data/{}/dict_full.txt'.format(dataset))

    create_vocab(corpus, dataset)
    ck_to_txt('../data/BC5CDR/truth_dev.ck', dataset, split='dev')
    ck_to_txt('../data/BC5CDR/truth_test.ck', dataset, split='test')
    create_train_data(corpus, core_dict, full_dict, dataset)
