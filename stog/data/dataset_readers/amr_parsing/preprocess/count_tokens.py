# add by kiro @2020.10.16
from supar import Parser
from stog.data.dataset_readers.amr_parsing.io import AMRIO
from stog.utils import logging


logger = logging.init_logger()


class DependencyParser:
    """
    DependencyParser that based on supar, to predict the automatic dependency parsing results for AMR text
    """
    def __init__(self):
        self.parser = Parser.load("biaffine-dep-bert-en")

    def parse(self, file_path):
        total_tokens, total_concepts = 0, 0
        for i, amr in enumerate(AMRIO.read(file_path), 1):
            if i % 1000 == 0:
                logger.info('Processed {} examples.'.format(i))
            # self.parser_sentence(amr)
            total_tokens += len(amr.tokens)
            total_concepts += 0
            yield amr
        print("total tokens", total_tokens)

    def parser_sentence(self, amr):
        tokens = amr.tokens
        result = self.parser.predict([tokens], verbose=False, tree=True, proj=True)
        amr.dependency_edges = result.arcs[0]
        amr.dependency_rels = result.rels[0]


if __name__ == '__main__':
    import argparse

    from stog.data.dataset_readers.amr_parsing.node_utils import NodeUtilities as NU

    parser = argparse.ArgumentParser('filter_valid_amr_graphs.py')
    parser.add_argument('--amr_files', nargs='+', required=True)
    parser.add_argument('--util_dir', default='./temp')

    args = parser.parse_args()

    node_utils = NU.from_json(args.util_dir, 0)

    parser = DependencyParser()

    for file_path in args.amr_files:
        with open(file_path + '.valid', 'w', encoding='utf-8') as f:
            for amr in parser.parse(file_path):
                pass
                # f.write(str(amr) + '\n\n')
