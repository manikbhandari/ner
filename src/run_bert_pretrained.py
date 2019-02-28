
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, collections, logging, json, re, pdb, numpy as np, time
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt='%m/%d/%Y %H:%M:%S', 
                    level=logging.INFO,
                    filename='logs/bert_pretrained_{}.log'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())), filemode='a')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
console.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(console)

EPS = 1e-9

class NERModel(torch.nn.Module):
    def __init__(self, in_dim, dataset):
        super(NERModel, self).__init__()
        self.in_dim  = in_dim
        self.lbl2id  = json.load(open('../data/{}/lbl2id.json'.format(dataset)))
        self.linear  = torch.nn.Linear(in_dim, len(self.lbl2id))

    def forward(self, bert_out):
        out = self.linear(bert_out)
        return out

class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a    = text_a
        self.text_b    = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, labels, lbl_ids, lbl_mask):
        self.unique_id      = unique_id
        self.tokens         = tokens
        self.input_ids      = input_ids
        self.input_mask     = input_mask
        self.input_type_ids = input_type_ids
        self.labels         = labels
        self.lbl_ids         = lbl_ids
        self.lbl_mask       = lbl_mask


def convert_examples_to_features(examples, seq_length, tokenizer, label_file, label_mask_file, lbl2id, tokenize=True):
    """Loads a data file into a list of `InputBatch`s."""

    labels   = utils.read(label_file)
    lbl_mask = utils.read(label_mask_file)
    features = []
    for (ex_index, example) in enumerate(examples):
        if not tokenize: tokens_a = example.text_a
        else:            tokens_a = tokenizer.tokenize(example.text_a)

        labels_a   = labels[ex_index].split()
        lbl_mask_a = lbl_mask[ex_index].split()
        assert len(tokens_a) == len(labels_a)

        tokens_b = None
        if example.text_b:
            if not tokenize: tokens_b = example.text_b
            else:            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
                labels_a = labels_a[0:(seq_length - 2)]
                lbl_mask_a = lbl_mask_a[0:(seq_length - 2)]

        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0

        tokens          = []
        final_labels    = []
        final_lbl_masks = []
        input_type_ids  = []

        tokens.append("[CLS]")
        final_labels.append('None')
        input_type_ids.append(0)
        final_lbl_masks.append(0)

        for i, token in enumerate(tokens_a):
            tokens.append(token)
            input_type_ids.append(0)
            final_labels.append(labels_a[i])
            final_lbl_masks.append(int(lbl_mask_a[i]))

        tokens.append("[SEP]")
        input_type_ids.append(0)
        final_labels.append('None')
        final_lbl_masks.append(0)

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
            final_labels.append('None')
            final_lbl_masks.append(0)

        assert len(input_ids)       == seq_length
        assert len(input_mask)      == seq_length
        assert len(input_type_ids)  == seq_length
        assert len(final_labels)    == seq_length
        assert len(final_lbl_masks) == seq_length

        if ex_index < 5: # display first 5 examples
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("labels: %s" % " ".join([str(x) for x in final_labels]))
            logger.info("labal masks: %s" % " ".join([str(x) for x in final_lbl_masks]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(InputFeatures(unique_id=example.unique_id, tokens=tokens, input_ids=input_ids, 
                                      input_mask=input_mask, input_type_ids=input_type_ids, labels=final_labels, 
                                      lbl_mask=final_lbl_masks, lbl_ids=[lbl2id[lbl] for lbl in final_labels]))

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

def metrics(y_pred, y_true, masks, lbl2id):
    preds = np.argmax(y_pred.cpu().detach().numpy(), axis=-1).flatten() 
    acts  = y_true.numpy().flatten()
    masks = masks.flatten()

    assert preds.shape == acts.shape

    true_pos  = np.sum([(pred == act) and (pred != lbl2id['None']) and (mask == 1) for pred, act, mask in zip(preds, acts, masks)]) # true and positive
    false_pos = np.sum([(pred != act) and (pred != lbl2id['None']) and (mask == 1) for pred, act, mask in zip(preds, acts, masks)]) # true and negative
    false_neg = np.sum([(pred != act) and (pred == lbl2id['None']) and (mask == 1) for pred, act, mask in zip(preds, acts, masks)])   # false and negative

    prec = (true_pos + EPS) / (true_pos + false_pos + EPS)
    rec  = (true_pos + EPS) / (true_pos + false_neg + EPS)
    f1   = (2 * prec * rec + EPS) / (prec + rec + EPS)

    return prec, rec, f1, true_pos, false_pos, false_neg


def train_ner(ner_model, model, args, tokenizer, epoch):

    examples = read_examples('../word_piece_data/{}/train/in.txt'.format(args.dataset))
    features = convert_examples_to_features(examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer,
                                            label_file='../word_piece_data/{}/train/out_lbl.txt'.format(args.dataset),
                                            label_mask_file='../word_piece_data/{}/train/out_mask.txt'.format(args.dataset),
                                            lbl2id=ner_model.lbl2id)

    all_input_ids       = torch.tensor([f.input_ids for f in features],  dtype=torch.long)
    all_input_mask      = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index   = torch.arange(all_input_ids.size(0),            dtype=torch.long)
    all_labels          = torch.tensor([f.lbl_ids for f in features],     dtype=torch.long)
    all_label_masks     = torch.tensor([f.lbl_mask for f in features],   dtype=torch.float)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index, all_labels, all_label_masks)
    if args.local_rank == -1: eval_sampler = SequentialSampler(eval_data) # samples in order
    else:                     eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    lr      = args.lr
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    optim   = torch.optim.Adam(ner_model.parameters(), lr=lr)

    for batch_num, (input_ids, input_mask, example_indices, labels, lbl_mask) in enumerate(eval_dataloader):
        
        input_ids  = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)

        all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)

        bert_out    = all_encoder_layers[-1] # bs X seq_len X dim
        y_pred      = ner_model(bert_out) # bs X seq_len X len(lbl2id)
        loss        = loss_fn(y_pred.transpose(1, 2), labels.to(args.device))
        masked_loss = torch.sum(loss * lbl_mask.to(args.device))


        prec, rec, f1, tp, fp, fn = metrics(y_pred, labels, lbl_mask, ner_model.lbl2id)
        logger.info("TRAIN: e: {} b: {} NER loss: {:.5f} prec: {:.5f} rec: {:.5f} f1: {:.5f} tp: {} fp: {} fn: {}".format(
                            epoch, batch_num, masked_loss, prec, rec, f1, tp, fp, fn))

        masked_loss.backward()
        optim.step()

def evaluate_ner(ner_model, model, args, tokenizer, epoch, split):
    examples = read_examples('../word_piece_data/{}/{}/in.txt'.format(args.dataset, split))
    features = convert_examples_to_features(examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer,
                                            label_file='../word_piece_data/{}/{}/out_lbl.txt'.format(args.dataset, split),
                                            label_mask_file='../word_piece_data/{}/{}/out_mask.txt'.format(args.dataset, split),
                                            lbl2id=ner_model.lbl2id)

    all_input_ids       = torch.tensor([f.input_ids for f in features],  dtype=torch.long)
    all_input_mask      = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index   = torch.arange(all_input_ids.size(0),            dtype=torch.long)
    all_labels          = torch.tensor([f.lbl_ids for f in features],     dtype=torch.long)
    all_label_masks     = torch.tensor([f.lbl_mask for f in features],   dtype=torch.float)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index, all_labels, all_label_masks)
    if args.local_rank == -1: eval_sampler = SequentialSampler(eval_data) # samples in order
    else:                     eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)


    total_tp, total_fp, total_fn = 0, 0, 0
    for batch_num, (input_ids, input_mask, example_indices, labels, lbl_mask) in enumerate(eval_dataloader):
        
        input_ids  = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)

        all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)

        bert_out    = all_encoder_layers[-1] # bs X seq_len X dim
        y_pred      = ner_model(bert_out) # bs X seq_len X len(lbl2id)

        prec, rec, f1, tp, fp, fn = metrics(y_pred, labels, lbl_mask, ner_model.lbl2id)
        logger.info("{}: e: {} b: {} prec: {:.5f} rec: {:.5f} f1: {:.5f} tp: {} fp: {} fn: {}".format(
                            split, epoch, batch_num, prec, rec, f1, tp, fp, fn))
        total_tp += tp
        total_fp += fp
        total_fn += fn

    prec = (total_tp + EPS) / (total_tp + total_fp + EPS)
    rec  = (total_tp + EPS) / (total_tp + total_fn + EPS)
    f1   = (2 * prec * rec + EPS) / (prec + rec + EPS)
    logger.info("{} results: TP: {} FP: {} FN: {} prec: {} rec: {} f1: {}".format(split, total_tp, total_fp, total_fn, prec, rec, f1))

    return f1



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",     default='BC5CDR', type=str)
    parser.add_argument("--bert_model",  default='bert-base-cased', type=str, help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                                                                     "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--state_dict",  default=None, help="path to state dict of a trained model if present. Load pretrained weights otherwise.")

    parser.add_argument("--do_lower_case",  action='store_true',              help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=128,           type=int,  help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                                                                                    "than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--batch_size", default=4,         type=int,         help="Batch size for predictions.")
    parser.add_argument("--epochs",     default=100,          type=int,         help="number of epochs")
    parser.add_argument("--local_rank", default=-1,         type=int,         help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",    action='store_true',                  help="Whether not to use CUDA when available")

    parser.add_argument("--emb_dim",    default=768,        type=int,         help = "embedding dimension of BERT")
    parser.add_argument("--lr",         default=0.001,      type=float,       help = "learning rate of final layer")

    args = parser.parse_args()

    logger.info(vars(args))

    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        args.device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {} distributed training: {}".format(args.device, n_gpu, bool(args.local_rank != -1)))

    dataset   = args.dataset
    voc_fname = '../word_piece_data/{}/vocab.txt'.format(args.dataset)
    tokenizer = BertTokenizer(voc_fname, do_lower_case=args.do_lower_case)

    model = BertModel.from_pretrained('../trained_model/{}'.format(args.dataset))
    model.to(args.device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.eval()

    ner_model = NERModel(args.emb_dim, args.dataset)
    ner_model.to(args.device)

    best_f1 = 0
    for epoch in range(args.epochs):
        train_ner(ner_model, model, args, tokenizer, epoch)
        f1 = evaluate_ner(ner_model, model, args, tokenizer, epoch, 'dev')
        if f1 > best_f1:
            best_f1 = evaluate_ner(ner_model, model, args, tokenizer, epoch, 'test')

if __name__ == "__main__":
    main()