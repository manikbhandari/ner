import json
import utils
import pdb
import argparse
from collections import Counter

def main():
   sents = utils.read('../word_piece_data/{}/train/in.txt'.format(args.dataset)) 
   sents += utils.read('../word_piece_data/{}/dev/in.txt'.format(args.dataset)) 
   words = [wrd for sent in sents for wrd in sent.split()] + ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"] 
   vocab = Counter(words)

   print("total words: {} total unique words {}".format(len(words), len(vocab)))
   voc2id = {wrd: idx for idx, (wrd, freq) in enumerate(vocab.most_common())}
   id2voc = {idx: wrd for wrd, idx in voc2id.items()}
   vocab  = [wrd for wrd, _ in vocab.most_common()]

   with open('../word_piece_data/{}/voc2id.json'.format(args.dataset), 'w') as f:
       json.dump(voc2id, f)
   with open('../word_piece_data/{}/id2voc.json'.format(args.dataset), 'w') as f:
       json.dump(id2voc, f)
   with open('../word_piece_data/{}/vocab.txt'.format(args.dataset), 'w') as f:
       f.write('\n'.join(vocab))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='converts data to word peice format required by BERT')
    parser.add_argument('-dataset', dest="dataset", default='BC5CDR', help='')
    args = parser.parse_args()

    # input_ids = tokenizer.convert_tokens_to_ids(tokens)
    main()
