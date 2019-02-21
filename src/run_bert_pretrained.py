
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, collections, logging, json, re, pdb
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
logger = logging.getLogger(__name__)


class NERModel(torch.nn.Module):
    def __init__(self, in_dim, dataset):
        super(NERModel, self).__init__()
        self.in_dim  = in_dim
        self.lbl2id  = json.load(open('../data/{}/lbl2id.json'.format(dataset)))
        self.linear  = torch.nn.Linear(in_dim, len(self.lbl2id))

    def forward(self, bert_out):
        out = F.softmax(self.linear(bert_out), dim=-1)
        return out

class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a    = text_a
        self.text_b    = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id      = unique_id
        self.tokens         = tokens
        self.input_ids      = input_ids
        self.input_mask     = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer, tokenize=True):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if not tokenize: tokens_a = example.text_a
        else:            tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            if not tokenize: tokens_b = example.text_b
            else:            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0

        tokens         = []
        input_type_ids = []
        tokens.append("[CLS]")

        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)

        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids)       == seq_length
        assert len(input_mask)      == seq_length
        assert len(input_type_ids)  == seq_length

        if ex_index < 5: # display first 5 examples
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(InputFeatures(unique_id=example.unique_id, tokens=tokens, input_ids=input_ids, 
                                      input_mask=input_mask, input_type_ids=input_type_ids))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break

        if len(tokens_a) > len(tokens_b): tokens_a.pop()
        else:                             tokens_b.pop()


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line   = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples

def read_labels(lbl_file, lbl2id, max_seq_length):
    all_labels = utils.read(lbl_file)
    final_labels = []

    for line in all_labels:
        final_line = []
        line = line.split()
        while(len(line) > max_seq_length): line.pop()
        while(len(line) < max_seq_length): line.append('None')
        for el in line: 
            final_line.append(lbl2id[el])

        final_labels.append(final_line)

    return torch.tensor(final_labels, dtype=torch.long)

def read_mask(mask_file, max_seq_length):
    all_masks = utils.read(mask_file)
    final_masks = []

    for line in all_masks:
        line = line.split()
        line = [int(el) for el in line]
        while(len(line) > max_seq_length): line.pop()
        while(len(line) < max_seq_length): line.append(0)
        final_masks.append(line)

    return torch.tensor(final_masks, dtype=torch.long)

def train_ner(ner_model, model, args, tokenizer):

    examples = read_examples('../word_piece_data/{}/train/in.txt'.format(args.dataset))

    features = convert_examples_to_features(examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer)

    all_input_ids       = torch.tensor([f.input_ids for f in features],  dtype=torch.long)
    all_input_mask      = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index   = torch.arange(all_input_ids.size(0),            dtype=torch.long)

    all_labels   = read_labels('../word_piece_data/{}/train/out_lbl.txt'.format(args.dataset), ner_model.lbl2id, args.max_seq_length)
    all_lbl_mask = read_mask('../word_piece_data/{}/train/out_mask.txt'.format(args.dataset), args.max_seq_length)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    if args.local_rank == -1: eval_sampler = SequentialSampler(eval_data) # samples in order
    else:                     eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    lr      = args.lr
    loss_fn = torch.nn.CrossEntropyLoss()
    optim   = torch.optim.Adam(ner_model.parameters(), lr=lr)


    for input_ids, input_mask, example_indices in eval_dataloader:
        
        input_ids  = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)

        all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)

        bert_out = all_encoder_layers[-1] # bs X seq_len X dim

        y_pred = ner_model(bert_out) # bs X seq_len X len(lbl2id)
        pdb.set_trace()

        loss   = loss_fn()


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    # parser.add_argument("--input_file",  default=None, type=str, required=True)
    # parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--dataset",     default='BC5CDR', type=str)
    parser.add_argument("--bert_model",  default='bert-base-cased', type=str, help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                                                                     "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    ## Other parameters
    parser.add_argument("--do_lower_case",  action='store_true',   help="Set this flag if you are using an uncased model.")
    parser.add_argument("--layers",         default="-1,-2,-3,-4", type=str)
    parser.add_argument("--max_seq_length", default=128,           type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                                                                                    "than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--batch_size", default=32,         type=int, help="Batch size for predictions.")
    parser.add_argument("--local_rank", default=-1,         type=int, help = "local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",    action='store_true',          help="Whether not to use CUDA when available")

    parser.add_argument("--emb_dim",    default=768,        type=int,   help = "embedding dimension of BERT")
    parser.add_argument("--lr",         default=0.001,      type=float, help = "embedding dimension of BERT")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        args.device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {} distributed training: {}".format(args.device, n_gpu, bool(args.local_rank != -1)))

    layer_indexes = [int(x) for x in args.layers.split(",")] # like -1, -2, -3, -4
    tokenizer     = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    model = BertModel.from_pretrained(args.bert_model)
    model.to(args.device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.eval()

    ner_model = NERModel(args.emb_dim, args.dataset)
    ner_model.to(args.device)

    train_ner(ner_model, model, args, tokenizer)



if __name__ == "__main__":
    main()