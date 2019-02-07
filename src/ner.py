from make_batches import *

def lookup(arr, vocab, tyoe_of_input):
	if tyoe_of_input == 'in_wrds':
		mapping = vocab.wvoc2id

	elif tyoe_of_input == 'in_chars':
		mapping = vocab.cvoc2id

	elif tyoe_of_input == 'out_tb':
		mapping = vocab.tb2id

	elif tyoe_of_input == 'label':
		mapping = vocab.lbl2id

	else:
	 raise NotImplementedError('tyoe_of_input must be in_wrds or in_chars or out_tb or label, \
									but given {}'.format(tyoe_of_input))

	for i, line in enumerate(arr):
		for j, el in enumerate(line):

			#thers is no unk token 
			arr[i][j] = mapping[el]

	return arr

def train(batches, vocab):
	for batch in batches:
		in_wrds, in_chars, out_tb = batch

		in_wrds  = lookup(in_wrds, vocab, 'in_wrds'), 
		in_chars =lookup(in_chars, vocab, 'in_chars'), 
		out_tb   = lookup(out_tb, vocab, 'out_tb')

		#convert to tensor

		#pass through RNN

		pdb.set_trace()


parser = argparse.ArgumentParser(description='NER')

parser.add_argument('-dataset', dest="dataset", default='BC5CDR',               help='')
parser.add_argument('-split',   dest="split",   default='train',                help='train/dev/test')
parser.add_argument('-batch',   dest="batch",   default=64,         type=int,   help='train/dev/test')

args    = parser.parse_args()
vocab   = Vocab(args)
batches = make_batches(args, vocab)

train(batches, vocab)

pdb.set_trace()