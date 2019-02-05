import argparse
import json
from utils import *

def make_batches():
    """
    :returns: Dumps pickle files in respective dataset/split folder.
    """
    text            = read("../data/{}/{}/in.txt".format(args.dataset, args.split))
    tie_or_break    = read("../data/{}/{}/out_tb.txt".format(args.dataset, args.split))
    labels          = read("../data/{}/{}/out_lbl.txt".format(args.dataset, args.split))

    assert(len(text) == len(tie_or_break) == len(labels))

    in_chars, in_wrds, out_tb, out_lbl = [], [], [], []

    for line, tb, lbl in zip(text, tie_or_break, labels):
        in_chars.append([char for char in line])

        line_wrds = []
        line_tb   = ['I'    for _ in range(len(line))]                        #initializw with all breaks
        line_lbl  = ['None' for _ in range(len(line))]                        #initialize with all Nones

        for i, wrd in enumerate(line.split()):
            for char in wrd:
                line_wrds.append(wrd)

            if i != len(line.split()) - 1:                            #don't append ' ' after the last word
                line_wrds.append(' ')
                line_tb[len(line_wrds)]  = tb.split()[i]
                line_lbl[len(line_wrds)] = lbl.split()[i]

        in_wrds.append(line_wrds)
        assert(len(in_wrds[-1]) == len(in_chars[-1]))

        pdb.set_trace()

    data = zip(text, tie_or_break, labels)
    data = sorted(data, key=lambda x: len(x[0].split()))
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NER')

    parser.add_argument('-dataset', dest="dataset", default='BC5CDR',               help='')
    parser.add_argument('-split',   dest="split",   default='train',                help='train/dev/test')
    parser.add_argument('-batch',   dest="batch",   default=64,         type=int,   help='train/dev/test')

    args = parser.parse_args()
    make_batches()
    