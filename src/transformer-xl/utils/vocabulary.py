import os, ipdb as pdb
from collections import Counter, OrderedDict

import torch

class Vocab(object):
    def __init__(self, special=[], min_freq=0, max_size=None, lower_case=True, delimiter=None, vocab_file=None):
        #Default arguments are being used in all calls
        self.counter    = Counter()
        self.special    = special           # ['<eos>'] for wt103
        self.min_freq   = min_freq          # 0
        self.max_size   = max_size          # None
        self.lower_case = lower_case        # boolean whether to lower case or not. False for wt103
        self.delimiter  = delimiter         # split line by this delimiter
        self.vocab_file = vocab_file        # None

    def tokenize(self, line, add_eos=False, add_double_eos=False):
        line = line.strip()
        if self.lower_case: line = line.lower()

        # None will make this go to else always
        if self.delimiter == '': symbols = line
        else:                    symbols = line.split(self.delimiter)

        if add_double_eos: return ['<S>'] + symbols + ['<S>']
        elif add_eos:      return symbols + ['<eos>']
        else:              return symbols

    def count_file(self, path, verbose=True, add_eos=False):
        """
        Adds words in path file to the counter of the vocab obhject
        """
        if verbose: print('counting file {} ...'.format(path))      
        assert os.path.exists(path)                                 # paht to train/dev/test file

        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0: print('    line {}'.format(idx))
                symbols = self.tokenize(line, add_eos=add_eos)      # Add eos is always false
                self.counter.update(symbols)                        # Add tokens to vocab
                sents.append(symbols)                               

        return sents                                                # Sents are being returned but not caught anywhere

    def count_sents(self, sents, verbose=False):
        """
            sents : a list of sentences, each a list of tokenized symbols
        """
        if verbose: print('counting {} sents ...'.format(len(sents)))
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            self.counter.update(symbols)

    def _build_from_file(self, vocab_file):
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        self.unk_idx = self.sym2idx['<UNK>']

    def build_vocab(self):
        """
        Creates idx2sym and sym2idx and adds symbols to them.
        """
        if self.vocab_file: #Usually None
            print('building vocab from {}'.format(self.vocab_file))
            self._build_from_file(self.vocab_file)
            print('final vocab size {}'.format(len(self)))

        else:
            print('building vocab with min_freq={}, max_size={}'.format(self.min_freq, self.max_size))
            self.idx2sym = []
            self.sym2idx = OrderedDict()                    # TODO: Why the need for OrderedDict()?

            for sym in self.special: self.add_special(sym)  # adds eos

            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq: break
                self.add_symbol(sym)

            print('final vocab size {} from {} unique tokens'.format(len(self), len(self.counter)))

    def encode_file(self, path, ordered=False, verbose=False, add_eos=True, add_double_eos=False):
        """
        Returns a LongTensor for the full file if ordered and a list of tensors otherwise.
        The LongTensor is a huge tensor containing just words.
        """
        if verbose: print('encoding file {} ...'.format(path))
        assert os.path.exists(path)

        encoded = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0: print('    line {}'.format(idx))
                symbols = self.tokenize(line, add_eos=add_eos, add_double_eos=add_double_eos) # By default add eos
                encoded.append(self.convert_to_tensor(symbols))

        if ordered: encoded = torch.cat(encoded)

        return encoded

    def encode_sents(self, sents, ordered=False, verbose=False):
        if verbose: print('encoding {} sents ...'.format(len(sents)))
        encoded = []
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            encoded.append(self.convert_to_tensor(symbols))

        if ordered: encoded = torch.cat(encoded)

        return encoded

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

    def add_symbol(self, sym):
        """
        Adds sym to self.idx2sym and self.sym2idx
        """
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def get_sym(self, idx):
        assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)
        return self.idx2sym[idx]

    def get_idx(self, sym):
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            # print('encounter unk {}'.format(sym))
            assert '<eos>' not in sym
            assert hasattr(self, 'unk_idx')
            return self.sym2idx.get(sym, self.unk_idx)

    def get_symbols(self, indices):
        return [self.get_sym(idx) for idx in indices]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    def convert_to_tensor(self, symbols):
        return torch.LongTensor(self.get_indices(symbols))

    def convert_to_sent(self, indices, exclude=None):
        if exclude is None:
            return ' '.join([self.get_sym(idx) for idx in indices])
        else:
            return ' '.join([self.get_sym(idx) for idx in indices if idx not in exclude])

    def __len__(self):
        return len(self.idx2sym)
