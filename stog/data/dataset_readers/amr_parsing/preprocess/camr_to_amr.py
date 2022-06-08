import re
import argparse
from turtle import pos
from stog.data.dataset_readers.amr_parsing.amr import AMR, AMRGraph

"""
This scirpt aims to
1) convert to CAMR format file to amr format
2) analyse whether only one ":arg1(x18_x20/在之上)"-like relation exists in one sub graph.
"""


class graph:
    def __init__(self, name, father_node=None):
        self.name = name
        self.father_node = father_node
        self.children = []


class NodeIdGenerator:
    def __init__(self):
        self.pre = 'a'
        self.post = 'a'

    def get_node_id(self):
        node_id = self.pre + self.post
        if self.post == 'z':
            self.pre = chr(ord(self.pre) + 1)  # the next char following the a, b, c, d, e, f, g, ... order
            self.post = 'a'
        else:
            self.post = chr(ord(self.post) + 1)
        return node_id



class CAMR:
    def __init__(self, id=None, snt=None, wid=None, graph_lines=None):
        self.id = id
        self.snt = snt
        self.wid = wid
        self.graph_lines=graph_lines
        self.graph_split=' '*6
        self.segmentor = '@@@'
        self.amr_graph_lines = []
 
    @staticmethod
    def extract_relations(line):
        line = line.strip()
        relations, rel = [], []
        start_relation = False
        for char in line:
            if start_relation is True:
                assert char != ':'
                if char != '(' and char != ' ':
                    rel.append(char)
                else:
                    assert len(rel) > 0
                    relations.append(''.join(rel))
                    rel = []
                    start_relation = False
            else:  # False
                if char == ':':
                    start_relation = True
                elif char == ')':
                    start_relation = False
                elif char == ' ':
                    start_relation = False
                else:
                    pass
        return relations


    def analyse(self):
        root = None
        nodes_in_depths = []
        for line in self.graph_lines:
            line = line.rstrip()
            toks = line.split(self.graph_split)
            depth = sum([item == '' for item in toks])
            if depth == 0:
                assert root is None
                root = graph('root')
                nodes_in_depths.append([root])
                continue
            rels = CAMR.extract_relations(line)
            father_node = nodes_in_depths[depth - 1][-1]
            for rel in rels:
                node = graph(rel, father_node)
                # for the instances that arg0-of(x1/的),  arg0-of(x1/的)
                if rel in [child.name for child in father_node.children]:
                    if '(' in rel and ')' in rel and rel.index('(') + 1 < rel.index(')'):
                            print(f"{self.id} rel {rel} is already existing in the same depth, check please")
                else:
                    father_node.children.append(node)
                # for the instances that arg0-of(), arg0-of(x1/的)
                if '(' in rel and ')' in rel and rel.index('(') + 1 < rel.index(')'):
                    if rel[:rel.index('(')] in [child.name[:child.name('(')] for child in father_node.children if '(' in child.name]:
                        print(f"{self.id} rel {rel} is already existing in the same depth, check please")
                if depth >= len(nodes_in_depths):
                    nodes_in_depths.append([])
                nodes_in_depths[depth].append(node)

    def to_amr_with_pseudo_node_id(self):
        ni_g = NodeIdGenerator()

        last_depth, need_removing_right_bracket = 0, 0
        wid_node_set = set()
        bad_num = 0  # for debug
        # print(self.graph_lines)
        for line in self.graph_lines:
            line = line.rstrip()
            toks = line.split(self.graph_split)
            depth = sum([item == '' for item in toks])

            if need_removing_right_bracket > 0 and depth < last_depth:
                # print(f"need to remove ) {need_removing_right_bracket, depth, last_depth, self.amr_graph_lines[-1]}")
                self.amr_graph_lines[-1] = self.amr_graph_lines[-1][:-2] + '\n'  # remove one ')'
                need_removing_right_bracket -= 1
            # remove the '()' in relations
            if '()' in line:  
                line = line.replace('()', '')

            # / 12:00 -> / "12:00"
            pattern = re.compile(r'/ [^\" ]+:[^)\" ]+')
            res = pattern.findall(line)
            for r in res:
                line = line.replace(r, r[:2] + '"' + r[2:] + '"')
            
            # change the relation like arg0-of(x1/的) (xx / yy)
            pattern = re.compile(r'\(\S+\)')
            res = pattern.search(line)
            if res is not None:
                positions = res.span()
                pre_string = line[:positions[0]]
                line1 = pre_string + '-xu' + ' ' + '(' + line[positions[0] + 1: positions[1]] + '\n'
                line2 = pre_string + line[positions[1]:]
                line = line1 + line2
            # xxx/xxx -> xxx / xxx
            pattern = re.compile(r'\S/\S')
            res = pattern.findall(line)
            for r in res:
                line = line.replace(r, r.replace('/', ' / '))
            # change the condition :arg0 (x21
            pattern = re.compile(r':\S+ \([^/)]+$')
            res = pattern.search(line)
            if res is not None:
                assert line.count('(') == 1
                line = line.replace('(', '')
                need_removing_right_bracket += 1
                # print(depth, need_removing_right_bracket, line)
            # :op1 x1 / word -> :op1 x1@@@word
            pattern = re.compile(r':op\d+ [^(]+ / [^)]+')
            res = pattern.findall(line)
            for r in res:
                line = line.replace(r, r.replace(' / ', self.segmentor))
            
            def func(l):
                # print(f"\nfunc begin '{l}'")
                # xxx / www -> xxx@@@www
                pattern = re.compile(r'[^(]+ / [^)]+')
                res = pattern.findall(l)
                for r in res:
                    # print(f"\txxx, {l}")
                    l = l.replace(r, r.replace(' / ', self.segmentor))
                    # print(f"\tyyy, {l}")
                    if r not in wid_node_set:
                        wid_node_set.add(r)
                        # print("adhfliuajlndfads")
                    else:
                        # print("bad_num += 1")
                        # bad_num += 1
                        pass
                #     print(f"\tzzz, {l}")
                # print('0')
                if '/' not in l and self.segmentor not in l and '(' in l and ')' in l:
                    l = l.replace('(', '', 1).replace(')', '', 1)
                # print('1')
                # wid@@@www -> node_id / wid@www
                pattern = re.compile(r'[^(]+@@@[^)]+')
                res = pattern.findall(l)
                for r in res:
                    n_id = ni_g.get_node_id()
                    r_index = l.index(r)
                    l = l[:r_index] + n_id + ' / ' + l[r_index:]
                    # line = line.replace(r, r.replace(' / ', self.segmentor))
                # print('2')
                # (wid@@@www -> (node_id / wid@@@www
                pattern = re.compile(r'\([^/]+@{3}[^)/]+$')
                res = pattern.findall(l)
                for r in res:
                    n_id = ni_g.get_node_id()
                    r_index = l.index(r)
                    l = l[:r_index + 1] + n_id + ' / ' + l[r_index + 1:]
                    # line = line.replace(r, r.replace(' / ', self.segmentor))
                # print(f"func end '{l}'")
                return l
            
            lines = line.split('\n')
            for i, l in enumerate(lines):
                lines[i] = func(l)
            line = '\n'.join(lines)
            # print(line)

            # some annotation errors
            # :location() (x1 / 松原市) in export_amr.1438
            contain_bad = False
            pattern = ':location (x1 / 松原市)'
            if pattern in line:
                contain_bad = True
                line = ''
            
            if not contain_bad:
                self.amr_graph_lines.append(line + '\n')
                last_depth = depth
            # print(f"aaa, {line}")   
        assert need_removing_right_bracket == 0
        
        # graph = AMRGraph.decode(' '.join(self.amr_graph_lines))
        # pemman_graph = str(graph)

        def make_the_same_concept_with_the_same_node_id(lines):
            node_dict = dict()
            results = []
            for line in lines:
                pattern = re.compile(r'[^(]+ / \S+@@@[^)\s]+')
                res = pattern.findall(line)
                for r in res:
                    toks = r.split(' / ')
                    node_id = toks[0]
                    node = toks[1]
                    if 'name' in node:
                        continue
                    if node not in node_dict.keys():
                        node_dict[node] = node_id
                    else:
                        first_node_id = node_dict[node]
                        new_r = first_node_id
                        line = line.replace(r, new_r)
                results.append(line)
            return results

        self.amr_graph_lines = make_the_same_concept_with_the_same_node_id(self.amr_graph_lines)
        
        def func_remove_wid(lines):
            # special cases: loop
            if "export_amr.10778 ::cid export_amr.452" in self.id:
                # print(self.amr_graph_lines)
                # exit()
                line1 = lines.pop(-4)
                lines[-4] = lines[-4][:-4] + lines[-4][-3:]

                toks = line1.split('\n')
                line1 = '\n'.join([l.replace(' ' * 6 * 3, '', 1) for l in toks])
                lines.insert(-10, line1[:-2] + '\n')

            results = []
            last_depth, need_removing_right_bracket = 0, 0
            for i, line in enumerate(lines):
                toks = line.split(self.graph_split)
                depth = sum([item == '' for item in toks])

                pattern = re.compile(r'\S+@@@')
                res = pattern.findall(line)
                for r in res:
                    line = line.replace(r, '')
                
                if need_removing_right_bracket > 0 and depth < last_depth:
                    # print(f"need to remove ) {need_removing_right_bracket, depth, last_depth, self.amr_graph_lines[-1]}")
                    lines[i - 1] = lines[i - 1][:-2] + '\n'  # remove one ')'
                    need_removing_right_bracket -= 1
                
                # change the condition :arg0 (x21
                pattern = re.compile(r':\S+ \([^/\)]+$')
                res = pattern.search(line)
                if res is not None:
                    # if line.count('(') > 1:
                    #     print(line)
                    # assert line.count('(') == 1
                    line = line.replace('(', '')
                    need_removing_right_bracket += 1

                lines[i] = line
            
            # the last one  )))
            if need_removing_right_bracket > 0:
                print(lines[-1])
                lines[-1] = lines[-1][:-2] + '\n'  # remove one ')'
                need_removing_right_bracket -= 1
            assert need_removing_right_bracket == 0

            # (xxx) -> xxx
            for i, line in enumerate(lines):
                pattern = re.compile(r'\(\S+\)')
                res = pattern.findall(line)
                for r in res:
                    line = line.replace(r, r[1:-1])
                lines[i] = line
                results.append(line)
            return results
        
        self.amr_graph_lines = func_remove_wid(self.amr_graph_lines)

        line = ''
        line += self.id
        line += self.snt
        line += self.wid
        line += ''.join(self.amr_graph_lines)
        return line


def read(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"read from the original format cmar file {file_path}")
        id, snt, wid, graph_lines = None, None, None, []
        count = 0
        for line in f.readlines():
            if line.strip() == '':
                assert id is not None and snt is not None and wid is not None and len(graph_lines) > 0
                camr = CAMR(id, snt, wid, graph_lines)
                yield camr
                count += 1
                id, snt, wid, graph_lines = None, None, None, []
                continue
            if line.startswith('# ::id'):
                id = line
            elif line.startswith('# ::snt'):
                snt = line
            elif line.startswith('# ::wid'):
                wid = line
            else:
                graph_lines.append(line)
        print(f"Total {count} sentences in file {file_path}")


def camr_to_amr(amr):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser('input_cleaner.py')
    parser.add_argument('--amr_files', nargs='+', default=[])

    args = parser.parse_args()

    for file_path in args.amr_files:
        with open(file_path + '.pre', 'w', encoding='utf-8') as f:
            for camr in read(file_path):
                # camr.analyse()
                amr = camr.to_amr_with_pseudo_node_id()
                f.write(amr + '\n')

