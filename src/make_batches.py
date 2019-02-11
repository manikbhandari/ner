import argparse
import json
import random

from utils import *

class Vocab(object):
    def __init__(self, args):
        self.wvoc2id = json.load(open('../data/{}/wvoc2id.json'.format(args.dataset)))
        self.id2wvoc = json.load(open('../data/{}/id2wvoc.json'.format(args.dataset)))
        self.id2wvoc = {int(key): val for key, val in self.id2wvoc.items()}
        
        self.cvoc2id = json.load(open('../data/{}/cvoc2id.json'.format(args.dataset)))
        self.id2cvoc = json.load(open('../data/{}/id2cvoc.json'.format(args.dataset)))
        self.id2cvoc = {int(key): val for key, val in self.id2cvoc.items()}

        self.lbl2id = json.load(open('../data/{}/lbl2id.json'.format(args.dataset)))
        self.id2lbl = json.load(open('../data/{}/id2lbl.json'.format(args.dataset)))
        self.id2lbl = {int(key): val for key, val in self.id2lbl.items()}

        self.tb2id  = {'I': 1, 'O': 0, '<pad>': 0}
        self.id2tb  = {1: 'I',  0: 'O'}

        self.pad    = '<pad>'


def make_batches(args, vocab):
    """
    :returns: Dumps pickle files in respective dataset/split folder.
    """
    text            = read("../data/{}/{}/in.txt".format(args.dataset, args.split))
    tie_or_break    = read("../data/{}/{}/out_tb.txt".format(args.dataset, args.split))
    labels          = read("../data/{}/{}/out_lbl.txt".format(args.dataset, args.split))


    in_wrds, in_chars, out_tb, out_lbl = [], [], [], []

    for line, tb, lbl in zip(text, tie_or_break, labels):
        in_chars.append([char for char in line])

        line_wrds = []

        #initializw with all ties
        line_tb   = ['O' for _ in range(len(line))]

        for i, wrd in enumerate(line.split()):
            for char in wrd:
                line_wrds.append(wrd)

            #don't append ' ' after the last word
            if i != len(line.split()) - 1:
                line_wrds.append(' ')
                line_tb[len(line_wrds) - 1]  = tb.split()[i]

        in_wrds.append(line_wrds)
        out_tb.append(line_tb)
        out_lbl.append(lbl.split())

        assert(len(in_wrds[-1]) == len(in_chars[-1]) == len(out_tb[-1]))
        assert(len(out_lbl[-1]) == len(line.split()))


    assert(len(in_wrds) == len(in_chars) == len(out_tb) == len(out_lbl))

    data = list(zip(in_wrds, in_chars, out_tb, out_lbl))
    random.shuffle(data)
    data = [list(tup) for tup in data]

    st_idx = 0
    while st_idx + args.batch < len(data):
        batch  = data[st_idx : st_idx + args.batch]
        st_idx = st_idx + args.batch

        in_wrds, in_chars, out_tb, out_lbl, lbl_mask = [], [], [], [], []

        #make padded batch
        max_len     = max([len(b[0]) for b in batch])
        max_lbl_len = max([len(b[3]) for b in batch])
        for el in batch:
            in_wrds.append(el[0]         + [vocab.pad] * (max_len - len(el[0])))
            in_chars.append(el[1]        + [vocab.pad] * (max_len - len(el[1])))
            out_tb.append(el[2]          + [vocab.pad] * (max_len - len(el[2])))

            out_lbl.append(el[3][1: -1]             + ['None'] * (max_lbl_len - len(el[3])))
            lbl_mask.append([1] * len(el[3][1: -1]) + [0]      * (max_lbl_len - len(el[3])))

        # in_lbl = []
        # for el in out_tb:
        #     in_lbl.append([y for y, tb in enumerate(el) if tb == 'I'])

        # in_lbl_x = [el[:-1] for el in in_lbl]
        # in_lbl_y = [el[1: ] for el in in_lbl]

        # pdb.set_trace()

        #ignore labels for now. Integrate them later.
        yield (in_wrds, in_chars, out_tb, out_lbl, lbl_mask)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NER')

    parser.add_argument('-dataset', dest="dataset", default='BC5CDR',               help='')
    parser.add_argument('-split',   dest="split",   default='train',                help='train/dev/test')
    parser.add_argument('-batch',   dest="batch",   default=64,         type=int,   help='train/dev/test')

    args    = parser.parse_args()
    vocab   = Vocab(args)
    batches = make_batches(args, vocab)

    pdb.set_trace()
    