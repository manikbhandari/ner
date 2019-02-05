import argparse
import json
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

        self.pad    = '<pad>'


def make_batches(vocab):
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

        #initializw with all breaks
        line_tb   = ['O'    for _ in range(len(line))]

        for i, wrd in enumerate(line.split()):
            for char in wrd:
                line_wrds.append(wrd)

            #don't append ' ' after the last word
            if i != len(line.split()) - 1:
                line_wrds.append(' ')
                line_tb[len(line_wrds)]  = tb.split()[i]

        in_wrds.append(line_wrds)
        out_tb.append(line_tb)
        out_lbl.append(lbl.split())

        assert(len(in_wrds[-1]) == len(in_chars[-1]) == len(out_tb[-1]))
        assert(len(out_lbl[-1]) == len(line.split()))


    assert(len(in_wrds) == len(in_chars) == len(out_tb) == len(out_lbl))

    data = zip(in_wrds, in_chars, out_tb, out_lbl)
    data = sorted(data, key=lambda x: len(x[0]))
    data = [list(tup) for tup in data]

    st_idx = 0
    while st_idx + args.batch < len(data):
        batch  = data[st_idx : st_idx + args.batch]
        st_idx = st_idx + args.batch

        in_wrds, in_chars, out_tb, out_lbl = [], [], [], []

        #make padded batch
        max_len = len(batch[-1][0])
        for el in batch:
            in_wrds.append(el[0] + [vocab.pad] * (max_len - len(el[0])))
            in_chars.append(el[1] + [vocab.pad] * (max_len - len(el[1])))
            out_tb.append(el[2] + [vocab.pad] * (max_len - len(el[2])))

        pdb.set_trace()

        return batch
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NER')

    parser.add_argument('-dataset', dest="dataset", default='BC5CDR',               help='')
    parser.add_argument('-split',   dest="split",   default='train',                help='train/dev/test')
    parser.add_argument('-batch',   dest="batch",   default=64,         type=int,   help='train/dev/test')

    args  = parser.parse_args()
    vocab = Vocab(args)

    batches = make_batches(vocab)
    pdb.set_trace()
    