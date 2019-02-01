import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NER')

    parser.add_argument('-dataset', dest="dataset", default='BC5CDR', help='')
    parser.add_argument('-split',   dest="split",   default='train',  help='train/dev/test')

    args = parser.parse_args()
