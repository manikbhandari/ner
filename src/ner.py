import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

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

	return torch.tensor(arr, dtype=torch.long)

class NER(nn.Module):
	def __init__(self, args, vocab):
		super(NER, self).__init__()
		self.args = args

		if self.args.cell_type == 'lstm':
			cell = nn.LSTM
		elif self.args.cell_type == 'gru':
			cell = nn.GRU

		self.cell = cell(input_size    = self.args.wrd_embed_dim + self.args.char_embed_dim, 
						 hidden_size   = self.args.hidden_size, 
						 num_layers    = self.args.num_layers,
						 batch_first   = True, 		   
						 dropout	   = self.args.drop,  			 
						 bidirectional = self.args.bidir)

		self.hidden2out = nn.Linear(self.args.hidden_size * (2 if self.args.bidir else 1), self.args.num_classes)
		self.hidden 	= self.init_hidden
		self.vocab 		= vocab

		self.wrd_embedding_lookup  = nn.Embedding(len(self.vocab.wvoc2id), self.args.wrd_embed_dim,  padding_idx=self.vocab.wvoc2id[self.vocab.pad])
		self.char_embedding_lookup = nn.Embedding(len(self.vocab.cvoc2id), self.args.char_embed_dim, padding_idx=self.vocab.cvoc2id[self.vocab.pad])

	def init_hidden(self):
		#would random initialization help?
		if self.args.cell_type == 'lstm':
			return (torch.zeros(self.args.num_layers * (2 if self.args.bidir else 1), self.args.batch, self.args.hidden_size).cuda(),
					torch.zeros(self.args.num_layers * (2 if self.args.bidir else 1), self.args.batch, self.args.hidden_size).cuda())
			
		else: return torch.zeros(self.args.num_layers * (2 if self.args.bidir else 1), self.args.batch, self.args.hidden_size).cuda()
					

	def forward(self, in_wrds, in_chars):
		cell_in  			  = torch.cat((self.wrd_embedding_lookup(in_wrds), self.char_embedding_lookup(in_chars)), dim=-1)
		cell_out, self.hidden = self.cell(cell_in, self.hidden)
		scores   			  = F.log_softmax(self.hidden2out(cell_out), dim=-1)

		return scores


def train(model, vocab, args):
	loss_fn   = nn.NLLLoss(reduction='none')
	optimizer = (optim.Adam if args.optim == 'adam' else optim.SGD)
	optimizer = optimizer(model.parameters(), lr=args.lr)

	for epoch in range(args.epochs):
		batches = make_batches(args, vocab)
		epoch_loss = 0
		st_time = time.time()

		for batch_num, batch in enumerate(batches):
			in_wrds, in_chars, out_tb = batch

			in_wrds  = lookup(in_wrds, vocab, 'in_wrds').cuda()
			in_chars = lookup(in_chars, vocab, 'in_chars').cuda()
			out_tb   = lookup(out_tb, vocab, 'out_tb').cuda()

			model.zero_grad()
			model.hidden = model.init_hidden()
			probs 		 = model(in_wrds, in_chars)

			batch_loss = loss_fn(probs.view(args.batch, args.num_classes, -1), out_tb)
			batch_loss = torch.sum(out_tb.float() * batch_loss)
			epoch_loss += batch_loss

			batch_loss.backward()
			optimizer.step()

		time_taken = time.time() - st_time
		print('epoch {} loss {:.4f} time_taken {:.4f}s'.format(epoch, epoch_loss, time_taken))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='NER')

	parser.add_argument('-dataset', 		dest="dataset", 		default='BC5CDR',               help='')
	parser.add_argument('-split',   		dest="split",   		default='train',                help='train/dev/test')
	
	parser.add_argument('-wrd_embed_dim',   dest="wrd_embed_dim",   default=64,         type=int,   help='embedding dimension')
	parser.add_argument('-char_embed_dim',  dest="char_embed_dim",  default=64,         type=int,   help='embedding dimension')
	parser.add_argument('-cell_type',   	dest="cell_type",   	default='lstm',   				help='embedding dimension')
	parser.add_argument('-hidden_size',   	dest="hidden_size", 	default=64,         type=int,   help='hidden size of lstm/gru')
	parser.add_argument('-num_classes',   	dest="num_classes", 	default=2,        	type=int,   help='number of output classes')
	parser.add_argument('-num_layers',   	dest="num_layers",  	default=1,          type=int,   help='number of lstm/gru layers')
	parser.add_argument('-drop',   			dest="drop",   			default=0,          type=float, help='dropout = 1 - keep_prob')
	parser.add_argument('-bidir',   		dest="bidir",   		action='store_true',            help='use bidirectional rnn')

	parser.add_argument('-epochs',   		dest="epochs",   		default=100,        type=int,   help='use bidirectional rnn')
	parser.add_argument('-batch',   		dest="batch",   		default=64,         type=int,   help='batch size')
	parser.add_argument('-gpu',   			dest="gpu",   			default='0',       	 			help='batch size')
	parser.add_argument('-optim',   		dest="optim",   		default='adam',            		help='batch size')
	parser.add_argument('-lr',   			dest="lr",   			default=0.01,       type=float, help='batch size')

	args    = parser.parse_args()
	set_gpu(args.gpu)

	vocab   = Vocab(args)
	model   = NER(args, vocab)

	model = model.cuda()

	train(model, vocab, args)

	print('Model trained successfully')
