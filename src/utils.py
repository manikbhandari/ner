import ipdb as pdb
import os

pdb_multi = '!import code; code.interact(local=vars())'

def read(fname):
    lines = []
    with open(fname, 'r', encoding='utf8') as f:
        for line in f:
            lines.append(line.strip())

    return lines

def write(ls, fname):
    with open(fname, 'w', encoding='utf8') as f:
        f.write('\n'.join(ls))

def set_gpu(gpus):
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus
