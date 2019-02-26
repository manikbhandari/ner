import pdb
import argparse

from utils import *
from pytorch_pretrained_bert.tokenization import BertTokenizer


def convert_file(fname, lbl_fname, tokenizer):
	raw_sents = read(fname)
	raw_lbls  = read(lbl_fname)

	sents, labels, lbl_mask = [], [], []
	for i, sent in enumerate(raw_sents):
		new_sent  = sent
		new_label = raw_lbls[i]

		if ("<eos>" in sent and "<sos>" in sent) or ("<s>" in sent and "<eof>" in sent):
			new_sent  = ' '.join(sent.split()[1:-1])
			new_label = raw_lbls[i].split()[1:-1]

		assert len(new_sent.split()) == len(new_label)
		sents.append(tokenizer.tokenize(new_sent))

		label, mask = [], []
		lbl_ind = 0
		for tok in sents[-1]:
			if tok.startswith('##'):
				label.append('None')
				mask.append(0)
			else:
				label.append(new_label[lbl_ind])
				mask.append(1)
				lbl_ind += 1

		labels.append(label)
		lbl_mask.append(mask)

	assert all([len(sent) == len(lbl) for sent, lbl in zip(sents, labels)])
	assert all([len(mask) == len(lbl) for mask, lbl in zip(lbl_mask, labels)])

	sents = [' '.join(sent) for sent in sents]
	labels = [' '.join(lbl) for lbl in labels]
	lbl_mask = [' '.join([str(m) for m in mask]) for mask in lbl_mask]


	out_fname      = fname.replace('data', 'word_piece_data')
	out_lbl_fname  = lbl_fname.replace('data', 'word_piece_data')
	out_mask_fname = lbl_fname.replace('data', 'word_piece_data').replace('out_lbl.txt', 'out_mask.txt')

	open(out_fname, 'w').write('\n'.join(sents))
	open(out_lbl_fname, 'w').write('\n'.join(labels))
	open(out_mask_fname, 'w').write('\n'.join(lbl_mask))


def main():
	parser = argparse.ArgumentParser(description='converts data to word peice format required by BERT')
	parser.add_argument('-dataset', dest="dataset", default='BC5CDR', help='')
	parser.add_argument('-split', dest="split", default='train', help='')
	args = parser.parse_args()

	tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)

	convert_file('../data/{}/{}/in.txt'.format(args.dataset, args.split), 
				'../data/{}/{}/out_lbl.txt'.format(args.dataset, args.split),
				tokenizer)
	# input_ids = tokenizer.convert_tokens_to_ids(tokens)


if __name__ == '__main__':
	main()

