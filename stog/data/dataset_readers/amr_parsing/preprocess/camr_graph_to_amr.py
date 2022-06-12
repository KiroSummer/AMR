import re
import argparse
from turtle import pos

"""
This scirpt aims to
1) convert to CAMR format file to amr format
2) analyse whether only one ":arg1(x18_x20/在之上)"-like relation exists in one sub graph.
"""


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


NORMAL_NODE = 'Normal'
ATTRIBUTE_NODE = 'Attribute'
ROOT_RELATION = 'ROOT'


class Node:
    def __init__(self, in_relation, wid, word, depth=-1, type=NORMAL_NODE, father_node=None):
        self.in_relations = [in_relation]
        self.wid = wid
        self.word = word
        self.type = type
        self.concept_id = None
        self.depths = [depth]
        self.father_nodes = [father_node]
        self.visited = False
        self.visited_times = -1

        self.children = []
        self.children_relations = []
        self.visited_children = 0

    def __str__(self) -> str:
        info = ''
        info += '(' + self.type + ' node'
        info += ' ' + self.wid + ' <word>' + self.word + '</word> ' + f"self.visited_times {self.visited_times}"
        assert len(self.depths) == len(self.father_nodes) == len(self.in_relations)
        for depth, father, relation in zip(self.depths, self.father_nodes, self.in_relations):
            if father is not None:
                info += ' depth:' + str(depth) + ' father node wid:' + father.wid + ' relation:' + relation + '\t'
        info += '\n'
        for child, relation in zip(self.children, self.children_relations):
            info += ' child wid:' + child.wid + ' relation:' + relation + '\t'
        info += '\n'
        info += ')'
        return info

    @property
    def has_one_father(self):
        return len(self.father_nodes) == 1

    def add_father(self, father, rel, depth):
        self.father_nodes.append(father)
        self.in_relations.append(rel)
        self.depths.append(depth)

    def add_child(self, child, relation):
        self.children.append(child)
        self.children_relations.append(relation)

    def remove_child(self, target_child):
        for i, child in enumerate(self.children):
            if child.wid == target_child.wid and child.word == target_child.word:
                self.children.pop(i)
                self.children_relations.pop(i)
                break

    @property
    def has_no_child(self):
        return len(self.children) == 0

    def set_visited(self, father_node):
        self.visited = True
        self.visited_times += 1
        if father_node is not None:
            father_node.visited_children += 1

    def visited_children_num(self):
        return self.visited_children

    @property
    def has_loop(self):
        self_id = id(self)

        children = self.children
        while len(children) > 0:
            tmp = []
            for n in children:
                n_id = id(n)
                if n_id == self_id:
                    return True
                tmp.extend(n.children)
            children = tmp
        return False
                

head_pattern = re.compile(r'\(\S+ / \S+$')
normal_node_pattern = re.compile(r'[^\(\s]+ / [^\)\s]+')
reference_node_pattern = re.compile(r'x\d+[x_\d]* / x\d+[x_\d]*')
normal_relation_with_xu_pattern = re.compile(r':\S+\(\S+\)\s+')
wid_pattern = re.compile(r'[ \(]x\d+[\d_x]*')
pure_wid_pattern = re.compile(r'x\d+[x_\d]*')
digit_pattern = re.compile(r'[\d.]+')

def extract_normal_node_info(node_string):
    toks = node_string.split(' / ')
    return toks[0], toks[1]


normal_relation_pattern = re.compile(r':\S+\(\) ')
def extract_normal_relation_info(relation:str):
    tok = relation[0:-3]  # :mod()
    return tok

pure_relation_pattern = re.compile(r':\S+ ')
def extract_pure_relation_info(line:str):
    tok = pure_relation_pattern.findall(line)
    assert len(tok) == 1
    return tok[0]


visited_node_pattern = re.compile(r' [^:\(\)\s]+')
def extracted_visited_node_info(line:str):
    tok = visited_node_pattern.findall(line)
    assert len(tok) == 1
    return tok[1:]


def extract_normal_relation_with_xu_info(relation:str):
    # print(f"line {relation}")
    pattern = re.compile(r':\S+[^\(\s]+\/[^\)\s]+')  # :arg1(x14/出) 
    info = pattern.findall(relation)
    # print(f"info {info}")
    assert len(info) == 1
    pattern = re.compile(r'[^\(\s]+\/[^\)\s]+')  # x14/出
    xu = pattern.findall(info[0])
    assert len(xu) == 1
    toks = xu[0].split('/')
    # print(toks)
    pattern = re.compile(r':\S+\(')
    res = pattern.findall(relation)
    assert len(res) == 1
    relation = res[0][0:-1] + '-xu'
    return relation, toks


op_relation_pattern = re.compile(r':op\d+ [x\d_]+\/[^\s\)]+')
quotation_time_pattern = re.compile(r"\"\S+:[\S:]+")
time_pattern = re.compile(r"\S+:[\S:]+")
url_pattern = re.compile(r"www\.\S+")


def extract_op_relation_info(line:str):
    info = op_relation_pattern.findall(line)
    # print(line, info)
    assert len(info) >= 1
    res = []
    for op in info:
        toks = op.split(' ')
        relation = toks[0][:]
        wid, word = toks[1].split('/')
        res.append((relation, wid, word))
    return res


class CAMRGraph:
    def __init__(self, id, snt, wid, graph_lines) -> None:
        self.id = id
        self.snt = snt
        self.wid = wid
        self.graph_lines = graph_lines
        
        self.graph_split=' '*6
        self.segmentor = '@@@'
        self.amr_graph_lines = []

        self.node_in_each_depths = []
        self.wid_node_dict = {}
        self.extra_wid_to_wid = {}  # for special cases like "(x6 / x52" in sample 2031 -> {x6: x52}

    def func_fix_bad_case_2(self, line):
        res = wid_pattern.findall(line)
        nodes = normal_node_pattern.findall(line)
        if len(res) > 0:
            key = res[0].replace('(', '').strip()
            # print(key)
        if (len(nodes) == 0) and (len(res) == 1) and (key not in self.wid_node_dict.keys()) and key not in self.extra_wid_to_wid:  # encounter bad case like x10 in ::id export_amr.10089        
            # print(f"res[0] {res[0]}")
            # # print(f"Debug, {'x274' == res[0].strip()}")
            # print(f"info {res[0].strip()} not in {self.wid_node_dict.keys()}, {res[0].strip() not in self.wid_node_dict}")
            # print(f"info {type(res[0].strip())} not in {[type(k) for k in self.wid_node_dict.keys()]}")  
            pattern = re.compile(r'x\d+\s*')
            res = pattern.findall(line)
            # print(f"line: {line}, nodes: {nodes}, res: '{res[0].strip()}'")
            if len(res) > 0:
                assert len(res) == 1
                matched = res[0]
                wid = matched.strip()[1:]
                words = self.snt[len('# ::snt '):].strip().split(' ')
                print(f"wid {wid}")
                word = words[int(wid) - 1]
                line = line.replace(wid, wid + ' / ' + word)
                print(f"Bad case (2) fixed: {self.id}\nwid: {wid} added word {word}\n After fix: {line}")
        return line

    def get_word_by_position(self, wid):
        if wid.startswith('x'):
            wid = wid[1:]
        words = self.snt[len('# ::snt '):].strip().split(' ')
        print(f"wid {wid}")
        if int(wid) >= len(words):
            print(f"wid:{wid} larger than words number:{len(words)}")
            exit()
        word = words[int(wid) - 1]
        return word

    def process_normal_node(self, line, depth, relation, has_only_wid_then_add_word=False):
        father_node = self.node_in_each_depths[depth - 1][-1]
        # fix bug case 2
        line = self.func_fix_bad_case_2(line)

        def process_normal_node(relation, wid, word, depth, father_node):
            node_type = NORMAL_NODE
            if word == '-':
                node_type = ATTRIBUTE_NODE
            node = Node(relation, wid, word, depth=depth, father_node=father_node, type=node_type)
            # print(f"father node:{father_node.wid} add a child:{node.wid} with the relation:{relation}")
            father_node.add_child(node, relation)

            if len(self.node_in_each_depths) <= depth:
                self.node_in_each_depths.append([node])
            else:
                self.node_in_each_depths[depth].append(node)
            # if wid in self.wid_node_dict:  # change it
            #     existed_n = self.wid_node_dict[wid]
            #     self.wid_node_dict[wid + '/' + existed_n.word] = existed_n
            #     del self.wid_node_dict[wid]
            assert wid not in self.wid_node_dict
            self.wid_node_dict[wid] = node
            return node
        
        if reference_node_pattern.search(line):
            # print(f"reference_node_pattern {line}")
            two_wids = reference_node_pattern.findall(line)
            assert len(two_wids) == 1
            wids = two_wids[0].split(' / ')
            refer_wid, acutal_wid = wids
            # print(f"refer_wid {refer_wid}, acutal_wid {acutal_wid}, relation {relation}")
            if refer_wid == acutal_wid:
                word = self.get_word_by_position(acutal_wid)
                node = process_normal_node(relation, acutal_wid, word, depth, father_node)
                return node
            self.extra_wid_to_wid[refer_wid] = acutal_wid
            
            # print(f"extra_wid_to_wid:{self.extra_wid_to_wid}")
            if acutal_wid not in self.wid_node_dict and acutal_wid not in self.extra_wid_to_wid:
                # print(f"actual wid not in self.wid_node_dict {acutal_wid}, wid_node_dict keys:{self.wid_node_dict.keys()}")
                node = Node(relation, acutal_wid, 'None', depth=depth, father_node=father_node)
            elif acutal_wid not in self.wid_node_dict and acutal_wid in self.extra_wid_to_wid:
                while acutal_wid not in self.wid_node_dict:
                    acutal_wid = self.extra_wid_to_wid[acutal_wid]
                node = self.wid_node_dict[acutal_wid]
                node.add_father(father_node, relation, depth)
            else:
                node = self.wid_node_dict[acutal_wid]
                node.add_father(father_node, relation, depth)
            father_node.add_child(node, relation)
            # print(f"generate a reference node: {node}")

            if len(self.node_in_each_depths) <= depth:
                self.node_in_each_depths.append([node])
            else:
                self.node_in_each_depths[depth].append(node)
            self.wid_node_dict[acutal_wid] = node
            return node
        
        def process_visited_node(line, word=None):
            res = wid_pattern.findall(line)
            # print(f"line {line}, res {res}")
            assert len(res) >= 1
            wid = res[0][1:]  # we only need the first one, if len(res) >= 2, there could be some attribute concepts like :op xxx
            if len(res) >= 2:
                assert len(res) - 1 == len(op_relation_pattern.findall(line))

            # wid = wid if wid in self.wid_node_dict else self.extra_wid_to_wid[wid]
            if wid in self.wid_node_dict:
                pass
            else:
                assert wid in self.extra_wid_to_wid
                wid = self.extra_wid_to_wid[wid]
                while wid not in self.wid_node_dict:
                    wid = self.extra_wid_to_wid[wid]
                # print(f"Info self.extra wid to wid {self.extra_wid_to_wid}")
            # print(f"find the node with wid: '{wid}', full dict: {self.wid_node_dict.keys()}")
            
            node = self.wid_node_dict[wid]
            if word is not None and node.word == 'None':
                node.word = word
            # for cases like export_amr.1560 in test data
            if node.word == 'name' and len(node.children) > 0:
                # assert len(node.father_nodes) == 1
                node = node.father_nodes[-1]  # the nearest father node
            fn = father_node
            in_relation, in_depth = relation, depth
            # print(f"{'1' * 4} relation:'{relation}', in_relations:'{node.in_relations}', node:{node}")
            # if relation[:len(":arg")] == ":arg" and relation in node.in_relations:  # check whether has already existed one relation with the same argX
            #     # print(f"{'2' * 4} relation:'{relation}', in_relations:'{node.in_relations}', node:{node}")
            #     node, fn = fn, node
            #     in_depth = father_node.depths[-1]
            #     in_relation = relation + '-of'
                # return node
            # for key in self.wid_node_dict:
            #     print(f"check self.wid_node_dict {key, self.wid_node_dict[key].wid}")
            if len(self.node_in_each_depths) <= in_depth:
                self.node_in_each_depths.append([node])
            else:
                self.node_in_each_depths[in_depth].append(node)
            
            fn.add_child(node, in_relation)
            node.add_father(fn, in_relation, in_depth)
            # print(f"visited node {node}")
            return node
    
        nodes = normal_node_pattern.findall(line)
        if len(nodes) == 0:  # the node that has been visited before
            node = process_visited_node(line)
            return node
        else: # its a new concept node
            assert len(nodes) >= 1
            wid, word = extract_normal_node_info(nodes[0])
            # if time_pattern.search(word) and not quotation_time_pattern.search(word):
            #     res = time_pattern.findall(word)
            #     assert len(res) == 1
            #     word = word.replace(res[0], '"' + res[0] + '"')
            # if url_pattern.search(word):
            #     res = url_pattern.findall(word)
            #     assert len(res) == 1
            #     word = word.replace(res[0], '"' + res[0] + '"')
            # print(f"Maybe it is a new concept node wid:{wid}, word:{word}")
            
            # maybe it is also a visited node, because in the CAMR data, it is (position / concept) ...
            if wid in self.extra_wid_to_wid or (wid in self.wid_node_dict and word == self.wid_node_dict[wid].word):
                print(f"wid has been visited before line {line}")
                node = process_visited_node(line, word=word)
                return node

            if wid in self.extra_wid_to_wid.values():  # fullfill the ndoe with wid "wid"
                node = self.wid_node_dict[wid]
                # print(f"line {line}, word {word}")
                node.word = word
                if len(self.node_in_each_depths) <= depth:
                    self.node_in_each_depths.append([node])
                else:
                    self.node_in_each_depths[depth].append(node)
                
                father_node.add_child(node, relation)
                node.add_father(father_node, relation, depth)
            else:
                # father node is the last node with the depth-1
                # print(f"depth {depth}, {self.node_in_each_depths}")
                if has_only_wid_then_add_word:
                    # print(f"some concept only has wid, we add a word after it! some info wid={wid}, word={word}")
                    count = 0
                    previous_node = None
                    # print('\n')
                    for key in self.wid_node_dict:
                        value = self.wid_node_dict[key]
                        if value.word == word and value.wid != wid:
                            count += 1
                            previous_node = value
                            # print(wid, word, key, value)
                    # assert count <= 1
                    # print(f"count {count}")
                    if count == 1:
                        node = previous_node
                        old_wid = previous_node.wid
                        previous_node.wid = wid  # update wid
                        self.wid_node_dict[previous_node.wid] = previous_node
                        del self.wid_node_dict[old_wid]
                        node = self.wid_node_dict[wid]
                        node.add_father(father_node, relation, depth)
                        # process wid dict
                        if old_wid not in self.extra_wid_to_wid:
                            # print(f"add a extra wid to wid old_wid: {old_wid}, wid: {wid}")
                            self.extra_wid_to_wid[old_wid] = wid
                    else:
                        node = Node(relation, wid, word, depth=depth, father_node=father_node)
                else:
                    node_type = NORMAL_NODE
                    if word == '-':
                        node_type = ATTRIBUTE_NODE
                    node = Node(relation, wid, word, depth=depth, father_node=father_node, type=node_type)
                # print(f"father node:{father_node.wid} add a child:{node.wid} with the relation:{relation}")
                father_node.add_child(node, relation)

                if len(self.node_in_each_depths) <= depth:
                    self.node_in_each_depths.append([node])
                else:
                    self.node_in_each_depths[depth].append(node)
                if wid in self.wid_node_dict:  # change it
                    existed_n = self.wid_node_dict[wid]
                    self.wid_node_dict[wid + '/' + existed_n.word] = existed_n
                    del self.wid_node_dict[wid]
                    # print(self.wid_node_dict)
                assert wid not in self.wid_node_dict
                self.wid_node_dict[wid] = node
    
                # node.add_father(father_node, relation, depth)
                # print(f"new node {node}, key is: '{wid}'")
            return node

    def generate_english_amr_format_string(self, 
                        remove_xu_rel=False, remove_relation_brackets=False, remove_wid=False):  # depth-first traversal
        special_mode_concept_name = ['expressive', 'interrogative', 'imperative']
        root_node = self.node_in_each_depths[0][0]
        tmp_node_in_each_depths = []  # temporarily store nodes in each depth
        ng = NodeIdGenerator()
        wid_to_random_node_id = dict()

        def is_attribute(concept_name):
            if time_pattern.match(concept_name) or url_pattern.match(concept_name) or digit_pattern.match(concept_name):
                return True
            return False

        def re_format_line(line, r_xu_rel, r_relation_brackets):
            if r_xu_rel and '-xu' in line:
                line = ''
            if r_relation_brackets and '()' in line:
                line = line.replace('()', '')
            # line = line.replace(' / ', '@@@')
            # pattern = re.compile(r'x\d+[x\d_^\s]*')
            # wid = pattern.findall(line)
            # for idx in wid:            
            #     nid = ng.get_node_id()
            #     line = line.replace(idx, nid + ' / ' + idx, 1)
            return line

        def father_node_visitezd_all_children(father_node:Node):
            # print(len(father_node.children), father_node.visited_children_num())
            return len(father_node.children) == father_node.visited_children_num()

        def bracktrace(node: Node, bracket_num=0, depth=0, original_node_id=None):
            # print(f"All nodes in the depth {depth}, number of nodes {len(tmp_node_in_each_depths[depth])}")
            # print(f"Nodes in depth {depth}, {[str(n) for n in tmp_node_in_each_depths[depth]]}")
            # print(f"{node}")
            # print(f"In bracktrace {node}")
            if node is None:
                return bracket_num
            if node.word == '-':
                bracket_num -= 1
            if is_attribute(node.word):
                bracket_num -= 1
            if node.in_relations[node.visited_times] == ':mode' and node.word in special_mode_concept_name:
                bracket_num -= 1
            # print(f"Visited all children {father_node_visitezd_all_children(node)}")
            if not father_node_visitezd_all_children(node):
                return bracket_num
            else:
                depth = depth - 1
                if depth < 0:  # next node is the father node None of the root node
                    return bracket_num + 1
                father_node = tmp_node_in_each_depths[depth][-1]
                return bracktrace(father_node, bracket_num + 1, depth=depth)
        
        def func(father_node: Node, node: Node, node_depth=0):
            # print(f"\nnode {node}")
            line = ''
            rel, wid, word, depths = node.in_relations, node.wid, node.word, node.depths

            # get the node id for the current node
            if wid not in wid_to_random_node_id:  # have not meet it before
                nid = ng.get_node_id()
                wid_to_random_node_id[wid] = nid
            else:
                if not node.visited:
                    nid = ng.get_node_id() if node.concept_id is None else node.concept_id
                else:
                    nid = wid_to_random_node_id[wid]
            if remove_wid:
                word = word  # remove the wid for first code project that adapts to most amr parsers
            else:
                word = wid + '@@' + word
            wid = nid  # set the node id to the new wid
            node.concept_id = wid  # set the node id as the new wid
            # TODO for example (x22 / x6)
            has_reference = False
            if remove_wid:
                wid_pattern = re.compile(r'x\d+[x_\d]*')
                if wid_pattern.search(word):
                    if word not in wid_to_random_node_id:
                        if word not in self.wid_node_dict:
                            # print(f"{word} doesn't exisit in the graph, allocate a new node ID!")
                            nid = ng.get_node_id()
                            word = nid
                            node.concept_id = nid
                        else:
                            assert word in self.wid_node_dict
                            nid = ng.get_node_id() if node.concept_id is None else node.concept_id
                            refered_node = self.wid_node_dict[word]
                            refered_node.concept_id = nid
                            word = refered_node.concept_id
                    else:
                        word = wid_to_random_node_id[word]
                    has_reference = True

            if rel[0] is ROOT_RELATION:  # root node
                if not node.visited:
                    node.set_visited(father_node)
                    line += '(' + wid + ' / ' + word
                    if node.has_no_child:
                        line += ')'
                    tmp_node_in_each_depths.append([node])
                else:
                    node.set_visited(father_node)
                    depth = depths[node.visited_times]
                    depth = depth if depth == node_depth else node_depth
                    # print(f"With depth info: node {node}, visited: {node.visited}, visited_times: {node.visited_times}, depth: {depth}")

                    # depth = depths[node.visited_times]
                    if len(tmp_node_in_each_depths) <= depth:
                        tmp_node_in_each_depths.append([node])
                    else:
                        tmp_node_in_each_depths[depth].append(node)
            
                    line += self.graph_split * depth + rel[node.visited_times] + ' ' + wid
                    x = set()
                    x.add(id(node))
                    # right_b_num = bracktrace(node.father_nodes[node.visited_times], depth=depth - 1, original_node_id=x)
                    right_b_num = bracktrace(father_node, depth=depth - 1, original_node_id=x)
                    # print("right_b_num", right_b_num)
                    line += ')' * right_b_num
                    line += '\n'
                    line = re_format_line(line, remove_xu_rel, remove_relation_brackets)
                    return line
            else:
                if not node.visited:
                    # print(f"node {node}, visited: {node.visited}, visited_times: {node.visited_times}")
                    node.set_visited(father_node)
                    depth = depths[node.visited_times]
                    depth = depth if depth == node_depth else node_depth
                    # print(f"\n{'='*4}Not visited With depth info: node {node}, visited: {node.visited}, visited_times: {node.visited_times}, depth: {depth}")

                    # depth = depths[node.visited_times]
                    if len(tmp_node_in_each_depths) <= depth:
                        tmp_node_in_each_depths.append([node])
                    else:
                        tmp_node_in_each_depths[depth].append(node)          
                    
                    if word == '-':
                        line += self.graph_split * depth + rel[node.visited_times] + ' ' + word
                    elif rel[node.visited_times] == ':mode' and word in special_mode_concept_name:
                        line += self.graph_split * depth + rel[node.visited_times] + ' ' + word
                    elif is_attribute(word):
                        if '"' not in word:
                            line += self.graph_split * depth + rel[node.visited_times] + ' "' + word + '"'
                        else:
                            line += self.graph_split * depth + rel[node.visited_times] + ' ' + word
                    elif (node.has_no_child and ':op' in rel[node.visited_times] and node.type == ATTRIBUTE_NODE) or has_reference:
                        line += self.graph_split * depth + rel[node.visited_times] + ' "' + word + '"'
                    else:
                        if has_reference:
                            line += self.graph_split * depth + rel[node.visited_times] + ' (' + word
                        else:
                            line += self.graph_split * depth + rel[node.visited_times] + ' (' + wid + ' / ' + word
                    # print(f"line {line}")
                    if node.has_no_child:
                        if word == '-':
                            pass
                        elif rel[node.visited_times] == ':mode' and word in special_mode_concept_name:
                            pass
                        elif is_attribute(word):
                            pass
                        elif ':op' in rel[node.visited_times] and node.type == ATTRIBUTE_NODE or has_reference:
                            pass
                        else:
                            line += ')'
                        # print(node, len(node.father_nodes), node.visited_times)
                        x = set()
                        x.add(id(node))
                        # right_b_num = bracktrace(node.father_nodes[node.visited_times], depth=depth - 1, original_node_id=x)
                        right_b_num = bracktrace(father_node, depth=depth - 1, original_node_id=x)
                        # print("right_b_num", right_b_num)
                        line += ')' * right_b_num
                    else:
                        pass
                else:  # for those visited nodes
                    # print(f"node {node}, visited: {node.visited}, visited_times: {node.visited_times}")
                    node.set_visited(father_node)
                    depth = depths[node.visited_times]
                    depth = depth if depth == node_depth else node_depth
                    # print(f"\n{'='*4}Visited, has reference:{has_reference}, With depth info: node {node}, visited: {node.visited}, visited_times: {node.visited_times}, depth: {depth}")

                    # depth = depths[node.visited_times]
                    if len(tmp_node_in_each_depths) <= depth:
                        tmp_node_in_each_depths.append([node])
                    else:
                        tmp_node_in_each_depths[depth].append(node)
            
                    line += self.graph_split * depth + rel[node.visited_times] + ' ' + wid
                    x = set()
                    x.add(id(node))
                    # right_b_num = bracktrace(node.father_nodes[node.visited_times], depth=depth - 1, original_node_id=x)
                    right_b_num = bracktrace(father_node, depth=depth - 1, original_node_id=x)
                    # print("right_b_num", right_b_num)
                    line += ')' * right_b_num
                    line += '\n'
                    line = re_format_line(line, remove_xu_rel, remove_relation_brackets)
                    return line
            
            line += '\n'
            line = re_format_line(line, remove_xu_rel, remove_relation_brackets)
            for n in node.children:
                line += func(node, n, node_depth=node_depth + 1)
            return line
        
        line = func(None, root_node)
        # print(f"amr graph\n {line}")
        return self.id + self.snt + self.wid + line

    def func_fix_bad_case(self, line):
        pattern = re.compile(r'x\d+ \/\s*$')
        res = pattern.findall(line)
        flag = False
        if len(res) > 0:
            assert len(res) == 1
            matched = res[0]
            wid = matched.split('/')[0].strip()[1:]
            words = self.snt[len('# ::snt '):].strip().split(' ')
            word = words[int(wid) - 1]
            print(f"Bad case fixed: {self.id}\nwid: line '{line}', {wid} added word {word}")
            line = line.rstrip() + ' ' + word + '\n'
            flag = True
        pattern = re.compile(r'^x\d+\s+$')
        if pattern.search(line):
            line = ''
        # if '# ::id export_amr.1438 ' == self.id[:len('# ::id export_amr.1438 ')] and line.strip() == ':location() (x1 / 松原市)':
        #     line = ''
        return line, flag
    
    def fix_concept_that_has_its_subgraph_two_positions(self, node:Node):
        depth = node.depths[0] - 1
        return depth         

    def generate_graph_from_lines(self):
        print(f"{self.id}")
        nearest_visited_node_depth = -1
        for i, line in enumerate(self.graph_lines):
            # fix some bad cases in the data, TODO, good data doesn't need it
            # print(f"line before fix: {line}")
            line, flag = self.func_fix_bad_case(line)
            if line == '':
                continue
            # print(f"line after fix: {line}")
            # print(f"\n{i}-th line {line}")
            line = line.rstrip()
            toks = line.split(self.graph_split)
            depth = sum([item == '' for item in toks])
            
            if i == 0:  # root node
                node = normal_node_pattern.findall(line)
                assert len(node) == 1
                wid, word = extract_normal_node_info(node[0])
                root_node = Node(ROOT_RELATION, wid, word, depth=depth)

                self.node_in_each_depths.append([root_node])
                self.wid_node_dict[wid] = root_node
            else:  # other nodes
                # print(f"one line {line}")
                father_node = self.node_in_each_depths[depth - 1][-1]
                res = normal_relation_pattern.findall(line)
                if len(res) > 0:  # normal relation like "            :mod() (x11 / 财务)"
                    assert len(res) == 1
                    relation = extract_normal_relation_info(res[0])
                    # print(f"relation {relation}, depth {depth}, line {line}")
                    node = self.process_normal_node(line, depth, relation, flag)
                else:  # relation that is xu
                    # xu can also be visited!
                    # print(f"xu-relation line {line}")
                    assert normal_relation_with_xu_pattern.search(line) is not None
                    # xu-relation process
                    xu_relation, (xu_wid, xu_word) = extract_normal_relation_with_xu_info(line)
                    xu_wid += '-xu'  # temporary TODO sample 1436
                    if xu_wid in self.wid_node_dict:
                        node = self.wid_node_dict[xu_wid]
                        node.add_father(father_node, xu_relation, depth)
                    else:
                        node = Node(xu_relation, xu_wid, xu_word, depth=depth, father_node=father_node) 
                    # print(f"what relation {relation}, Father node {father_node}\t Node {node}")
                    father_node.add_child(node, xu_relation)
                    if len(self.node_in_each_depths) <= depth:
                        self.node_in_each_depths.append([node])
                    else:
                        self.node_in_each_depths[depth].append(node)
                    self.wid_node_dict[xu_wid] = node
                    # relation after xu-relation process
                    res = normal_relation_with_xu_pattern.findall(line)
                    assert len(res) == 1  # only one xu-relation
                    line = line.replace(res[0], '')
                    # print(f"Node after xu, res[0]: {res[0]}, line: {line}")
                    node = self.process_normal_node(line, depth, xu_relation[:-3], flag)
                # print(f"{i}-th line {line}, generate node {node}")      
                # process for the relations like :op1 xx/ww
                pattern = re.compile(':op\d+ ')
                if pattern.search(line):
                    father_node = self.node_in_each_depths[depth][-1]
                    nodes = extract_op_relation_info(line)
                    for n in nodes:
                        relation, wid, word = n

                        already_has_one_attribute_concept = False
                        for child in father_node.children:
                            # print(wid, child.wid, word, child.word)
                            if wid == child.wid and word == child.word:  # if already exists one attribute concpet that belongs to the same father node
                                already_has_one_attribute_concept = True
                                break
                        if already_has_one_attribute_concept:
                            continue
                        
                        if father_node.word == 'name':
                            node_type = ATTRIBUTE_NODE
                        else:
                            node_type = NORMAL_NODE
                        # if wid in self.wid_node_dict:
                        #     print(f"Info, {wid}, {word}, {self.wid_node_dict[wid]}")
                        composed_key = wid + '/' + word
                        if composed_key in self.wid_node_dict:
                            # print(f"OP node {composed_key} already visited: {node}, remove it!")
                            # node = self.wid_node_dict[wid]
                            # node.add_father(father_node, relation, depth + 1)
                            bug_node = self.wid_node_dict[composed_key]
                            assert len(bug_node.father_nodes) == 1  # currently set, maybe contains more that one father
                            father_n = bug_node.father_nodes[0]
                            # print(f"before father {father_n}")
                            father_n.remove_child(bug_node)
                            del self.wid_node_dict[composed_key]
                            # print(f"after father {father_n}")
                            # save this node
                            # self.wid_node_dict[wid] = node  
                        # else:
                        node = Node(relation, wid, word, depth=depth + 1, type=node_type, father_node=father_node)
                        
                        father_node.add_child(node, relation)
                        if len(self.node_in_each_depths) <= depth + 1:
                            self.node_in_each_depths.append([node])
                        else:
                            self.node_in_each_depths[depth + 1].append(node)
                        if wid not in self.wid_node_dict:
                            self.wid_node_dict[wid] = node  
                        else:
                            # use the composed key to store the OP attribute concept
                            self.wid_node_dict[composed_key] = node
                            # remove the previous node
                            # bug_node = self.wid_node_dict[wid]
                            # assert len(bug_node.father_nodes) == 1  # currently set, maybe contains more that one father
                            # father_n = bug_node.father_nodes[0]
                            # father_n.remove_child(bug_node)
                            # del self.wid_node_dict[wid]
                            # # save this node
                            # self.wid_node_dict[wid] = node  
        #                 print(f"{i}-th line {line}, generate node {node}") 
        # print(f"\nGenerate graph done! wid_node_dict:{self.wid_node_dict.keys()}")     
        # print(f"\nGenerate graph done! extra_wid_to_wid:{self.extra_wid_to_wid.keys()}")  
        # print(f"x28_x29 {self.wid_node_dict['x28_x29']}")


def read(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"read from the original format cmar file {file_path}")
        id, snt, wid, graph_lines = None, None, None, []
        count = 0
        for line in f.readlines():
            if line.strip() == '':
                assert id is not None and snt is not None and wid is not None and len(graph_lines) > 0
                camr = CAMRGraph(id, snt, wid, graph_lines)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('input_cleaner.py')
    parser.add_argument('--amr_files', nargs='+', default=[])

    args = parser.parse_args()

    for file_path in args.amr_files:
        with open(file_path + '.pre', 'w', encoding='utf-8') as f:
            for camr in read(file_path):
                camr.generate_graph_from_lines()
                english_amr_format = camr.generate_english_amr_format_string(remove_xu_rel=True, 
                            remove_relation_brackets=True, remove_wid=True)
                f.write(english_amr_format + '\n')
#     lines = '''# ::id export_amr.3327 ::cid export_amr.3327 ::2017-02-15 11:07:38
# # ::snt 其实 这 也 不 会 成为 俄罗斯 的 战略 企图 ， 未来 的 俄罗斯 完全 可能 在 与 西方 的 地缘 争夺战 中 变得 坚决 而 不 妥协 ， 但 不管 双方 的 关系 多么 冷淡 ， 也 不管 普京 多么 厌恶 西方 ， 俄罗斯 在 主观 上 也 不 可能 愿意 首当其冲 、 独缨其锋 。
# # ::wid x1_其实 x2_这 x3_也 x4_不 x5_会 x6_成为 x7_俄罗斯 x8_的 x9_战略 x10_企图 x11_， x12_未来 x13_的 x14_俄罗斯 x15_完全 x16_可能 x17_在 x18_与 x19_西方 x20_的 x21_地缘 x22_争夺战 x23_中 x24_变得 x25_坚决 x26_而 x27_不 x28_妥协 x29_， x30_但 x31_不管 x32_双方 x33_的 x34_关系 x35_多么 x36_冷淡 x37_， x38_也 x39_不管 x40_普京 x41_多么 x42_厌恶 x43_西方 x44_， x45_俄罗斯 x46_在 x47_主观 x48_上 x49_也 x50_不 x51_可能 x52_愿意 x53_首当其冲 x54_、 x55_独缨其锋 x56_。
# (x58 / contrast
#       :arg1() (x60 / and
#             :op1() (x5 / 会-02
#                   :polarity() (x4 / -)
#                   :arg0() (x6 / 成为-01
#                         :arg1() (x10 / 企图-01
#                               :mod() (x9 / 战略)
#                               :arg0(x8/的) (x66 / country
#                                     :name() (x7 / name :op1 x7/俄罗斯 )))
#                         :arg0() (x2 / 这)
#                         :mod() (x3 / 也))
#                   :mod() (x1 / 其实))
#             :op2() (x16 / 可能-01
#                   :degree() (x15 / 完全)
#                   :arg0() (x24 / 变得-01
#                         :arg1() (x74 / and
#                               :op1() (x25 / 坚决-01
#                                     :arg0() (x14 /
#                                           :time(x13/的) (x12 / 未来)))
#                               :op2() (x28 / 妥协-01
#                                     :polarity() (x27 / -)
#                                     :arg0() x14 ))
#                         :arg0() x14 
#                         :cause(x17_x23/在中) (x22 / 争夺战
#                               :mod() (x21 / 地缘)
#                               :accompanier(x18/与) (x19 / 西方)))))
#       :arg2(x30/但) (x83 / concession
#             :arg1() (x84 / and
#                   :op1(x31/不管) (x36 / 冷淡-01
#                         :arg0() (x34 / 关系
#                               :poss(x33/的) (x32 / 双方
#                                     :arg0-of() (x90 / mean
#                                           :arg1() (x91 / and
#                                                 :op1() x19 
#                                                 :op2() x66 ))))
#                         :degree() (x35 / 多么))
#                   :op2(x39/不管) (x42 / 厌恶-01
#                         :arg0() (x92 / person
#                               :name() (x40 / name :op1 x40/普京 ))
#                         :degree() (x41 / 多么)
#                         :arg1() (x43 / x19)))
#             :arg2(x49/也) (x51 / 可能-01
#                   :polarity() (x50 / -)
#                   :arg0() (x99 / and
#                         :op1() (x53 / 首当其冲-01
#                               :arg0() (x45 / x66)
#                               :mod(x46_x48/在上) (x47 / 主观))
#                         :op2() (x55 / 独缨其锋
#                               :arg0() x66 
#                               :mod() x47 )))))'''
#     # print(lines)
#     lines = lines.split('\n')
#     camr = CAMRGraph(lines[0], lines[1], lines[2], lines[3:])
#     camr.generate_graph_from_lines()
#     res = camr.generate_english_amr_format_string(remove_xu_rel=True, remove_relation_brackets=True, remove_wid=True)
#     print(res)

