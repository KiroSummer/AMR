import re
import argparse
from stog.data.dataset_readers.amr_parsing.io import AMRIO


def analyse(amr):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser('input_cleaner.py')
    parser.add_argument('--amr_files', nargs='+', default=[])

    args = parser.parse_args()

    for file_path in args.amr_files:
        with open(file_path + '.input_clean', 'w', encoding='utf-8') as f:
            for amr in AMRIO.read(file_path):
                try:
                    analyse(amr)
                except:
                    continue

