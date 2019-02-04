import argparse
import json
from utils import *

def make_batches():
    """
    :returns: Dumps pickle files in respective dataset/split folder.
    """
    text = read("../data/{}/{}/in.txt".format(args.dataset, args.split))
    tb   = read("../data/{}/{}/out_tb.txt".format(args.dataset, args.split))
    lbl  = read("../data/{}/{}/out_lbl.txt".format(args.dataset, args.split))

    assert(len(text) == len(tb) == len(lbl))

    data = zip(text, tb, lbl)
    data = sorted(data, key=lambda x: len(x[0].split()))
    
    
def dump_vocab():
    """
    :returns: Dumps wvoc2id, id2wvoc, cvoc2id, id2cvoc, lbl2id, id2lbl
              Always: I = 1 = break and O = 0 = tie
    """    
    wvocab = read('../data/{}/vocab.txt'.format(args.dataset))
    wvocab = wvocab + [' ']                                                                   #add space as a word to vocab

    wvoc2id = {i: wrd for i, wrd in enumerate(wvocab)}
    id2wvoc = {wrd: i for i, wrd in enumerate(wvocab)}

    chars = [c for wrd in wvocab for c in wrd]
    chars = list(set(chars))

    cvoc2id = {i: c for i, c in enumerate(chars)}
    id2cvoc = {c: i for i, c in enumerate(chars)}

    labels = read('../data/{}/train/out_lbl.txt'.format(args.dataset))
    labels = [lbl for line in labels for lbl in line.split()]
    labels = list(set(labels))

    lbl2id = {i: lbl for i, lbl in enumerate(labels)}
    id2lbl = {lbl: i for i, lbl in enumerate(labels)}

    json.dump(wvoc2id, open('../data/{}/wvoc2id.json'.format(args.dataset), 'w'))
    json.dump(id2wvoc, open('../data/{}/id2wvoc.json'.format(args.dataset), 'w'))

    json.dump(cvoc2id, open('../data/{}/cvoc2id.json'.format(args.dataset), 'w'))
    json.dump(id2cvoc, open('../data/{}/id2cvoc.json'.format(args.dataset), 'w'))

    json.dump(lbl2id, open('../data/{}/lbl2id.json'.format(args.dataset), 'w'))
    json.dump(id2lbl, open('../data/{}/id2lbl.json'.format(args.dataset), 'w'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NER')

    parser.add_argument('-dataset', dest="dataset", default='BC5CDR',               help='')
    parser.add_argument('-split',   dest="split",   default='train',                help='train/dev/test')
    parser.add_argument('-batch',   dest="batch",   default=64,         type=int,  help='train/dev/test')

    args = parser.parse_args()
    dump_vocab()
    make_batches()
    