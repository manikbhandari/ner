import ipdb as pdb

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
