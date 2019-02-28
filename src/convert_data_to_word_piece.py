import pdb
import argparse

from utils import *
from pytorch_pretrained_bert.tokenization import BertTokenizer


def convert_file(fname, lbl_fname, tb_fname, tokenizer):
    raw_sents = read(fname)
    raw_lbls  = read(lbl_fname)
    raw_tbs   = read(tb_fname)

    sents, labels, lbl_mask, tbs = [], [], [], []
    for i, sent in enumerate(raw_sents):
        new_sent  = sent
        new_label = raw_lbls[i]
        new_tb    = raw_tbs[i]

        if ("<eos>" in sent and "<sos>" in sent) or ("<s>" in sent and "<eof>" in sent):
            new_sent  = ' '.join(sent.split()[1:-1])
            new_label = raw_lbls[i].split()[1:-1]
            new_tb    = raw_tbs[i].split()[1:-1]

        assert len(new_sent.split()) == len(new_label) == len(new_tb)
        sents.append(tokenizer.tokenize(new_sent))

        label, mask, tb = [], [], []
        lbl_ind = 0
        for tok in sents[-1]:
            if tok.startswith('##'):
                mask.append(0)
                label.append('None') # should be masked out
                tb.append('I') # should be masked out
            else:
                mask.append(1)
                label.append(new_label[lbl_ind])
                tb.append(new_tb[lbl_ind])
                lbl_ind += 1

        labels.append(label)
        tbs.append(tb)
        lbl_mask.append(mask)

    assert all([len(sent) == len(lbl) for sent, lbl in zip(sents, labels)])
    assert all([len(mask) == len(lbl) for mask, lbl in zip(lbl_mask, labels)])
    assert all([len(sent) == len(tb) for sent, tb in zip(sents, tbs)])

    sents = [' '.join(sent) for sent in sents]
    labels = [' '.join(lbl) for lbl in labels]
    tbs = [' '.join(tb) for tb in tbs]
    lbl_mask = [' '.join([str(m) for m in mask]) for mask in lbl_mask]


    out_fname      = fname.replace('data', 'word_piece_data')
    out_lbl_fname  = lbl_fname.replace('data', 'word_piece_data')
    out_tb_fname   = tb_fname.replace('data', 'word_piece_data')
    out_mask_fname = lbl_fname.replace('data', 'word_piece_data').replace('out_lbl.txt', 'out_mask.txt')

    open(out_fname, 'w').write('\n'.join(sents))
    open(out_lbl_fname, 'w').write('\n'.join(labels))
    open(out_tb_fname, 'w').write('\n'.join(tbs))
    open(out_mask_fname, 'w').write('\n'.join(lbl_mask))


def main():
    parser = argparse.ArgumentParser(description='converts data to word peice format required by BERT')
    parser.add_argument('-dataset', dest="dataset", default='BC5CDR', help='')
    parser.add_argument('-split', dest="split", default='train', help='')
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)

    convert_file('../data/{}/{}/in.txt'.format(args.dataset, args.split), 
        		'../data/{}/{}/out_lbl.txt'.format(args.dataset, args.split),
        		'../data/{}/{}/out_tb.txt'.format(args.dataset, args.split),
        		tokenizer)
    # input_ids = tokenizer.convert_tokens_to_ids(tokens)


if __name__ == '__main__':
    main()

