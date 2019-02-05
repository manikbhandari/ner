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
    
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NER')

    parser.add_argument('-dataset', dest="dataset", default='BC5CDR',               help='')
    parser.add_argument('-split',   dest="split",   default='train',                help='train/dev/test')
    parser.add_argument('-batch',   dest="batch",   default=64,         type=int,  help='train/dev/test')

    args = parser.parse_args()
    dump_vocab()
    make_batches()
    