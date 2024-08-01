#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
import os
import random
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp


class BColours:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class HGBDataLoaderLP:
    def __init__(self, path, edge_types=[]):
        self.path = path
        self.splited = False
        self.nodes = self.load_nodes()
        self.links = self.load_links('link.dat')
        self.links_test = self.load_links('link.dat.test')
        self.test_types = list(
            self.links_test['data'].keys()) if edge_types == [] else edge_types
        self.types = self.load_types('node.dat')
        self.train_pos, self.valid_pos = self.get_train_valid_pos()
        self.train_neg, self.valid_neg = self.get_train_neg(), self.get_valid_neg()
        self.gen_transpose_links()
        self.nonzero = False

    def get_train_valid_pos(self, train_ratio=0.9):
        if self.splited:
            return self.train_pos, self.valid_pos
        else:
            edge_types = self.links['data'].keys()
            train_pos, valid_pos = dict(), dict()
            for r_id in edge_types:
                train_pos[r_id] = [[], []]
                valid_pos[r_id] = [[], []]
                row, col = self.links['data'][r_id].nonzero()
                last_h_id = -1
                for (h_id, t_id) in zip(row, col):
                    if h_id != last_h_id:
                        train_pos[r_id][0].append(h_id)
                        train_pos[r_id][1].append(t_id)
                        last_h_id = h_id

                    else:
                        if random.random() < train_ratio:
                            train_pos[r_id][0].append(h_id)
                            train_pos[r_id][1].append(t_id)
                        else:
                            valid_pos[r_id][0].append(h_id)
                            valid_pos[r_id][1].append(t_id)
                            self.links['data'][r_id][h_id, t_id] = 0
                            self.links['count'][r_id] -= 1
                            self.links['total'] -= 1
                self.links['data'][r_id].eliminate_zeros()
            self.splited = True
            return train_pos, valid_pos

    def get_sub_graph(self, node_types_tokeep):
        """
        node_types_tokeep is a list or set of node types that you want to keep in the sub-graph
        We only support whole type sub-graph for now.
        This is an in-place update function!
        return: old node type id to new node type id dict, old edge type id to new edge type id dict
        """
        keep = set(node_types_tokeep)
        new_node_type = 0
        new_node_id = 0
        new_nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}}
        new_links = {'total': 0, 'count': Counter(), 'meta': {},
                     'data': defaultdict(list)}
        new_labels_train = {'num_classes': 0, 'total': 0, 'count': Counter(),
                            'data': None, 'mask': None}
        new_labels_test = {'num_classes': 0, 'total': 0, 'count': Counter(),
                           'data': None, 'mask': None}
        old_nt2new_nt = {}
        old_idx = []
        for node_type in self.nodes['count']:
            if node_type in keep:
                nt = node_type
                nnt = new_node_type
                old_nt2new_nt[nt] = nnt
                cnt = self.nodes['count'][nt]
                new_nodes['total'] += cnt
                new_nodes['count'][nnt] = cnt
                new_nodes['attr'][nnt] = self.nodes['attr'][nt]
                new_nodes['shift'][nnt] = new_node_id
                beg = self.nodes['shift'][nt]
                old_idx.extend(range(beg, beg + cnt))

                cnt_label_train = self.labels_train['count'][nt]
                new_labels_train['count'][nnt] = cnt_label_train
                new_labels_train['total'] += cnt_label_train
                cnt_label_test = self.labels_test['count'][nt]
                new_labels_test['count'][nnt] = cnt_label_test
                new_labels_test['total'] += cnt_label_test

                new_node_type += 1
                new_node_id += cnt

        new_labels_train['num_classes'] = self.labels_train['num_classes']
        new_labels_test['num_classes'] = self.labels_test['num_classes']
        for k in ['data', 'mask']:
            new_labels_train[k] = self.labels_train[k][old_idx]
            new_labels_test[k] = self.labels_test[k][old_idx]

        old_et2new_et = {}
        new_edge_type = 0
        for edge_type in self.links['count']:
            h, t = self.links['meta'][edge_type]
            if h in keep and t in keep:
                et = edge_type
                net = new_edge_type
                old_et2new_et[et] = net
                new_links['total'] += self.links['count'][et]
                new_links['count'][net] = self.links['count'][et]
                new_links['meta'][net] = tuple(
                    map(lambda x: old_nt2new_nt[x], self.links['meta'][et]))
                new_links['data'][net] = self.links['data'][et][old_idx][:, old_idx]
                new_edge_type += 1

        self.nodes = new_nodes
        self.links = new_links
        self.labels_train = new_labels_train
        self.labels_test = new_labels_test
        return old_nt2new_nt, old_et2new_et

    def get_meta_path(self, meta=[]):
        """
        Get meta path matrix
            meta is a list of edge types (also can be denoted by a pair of node types)
            return a sparse matrix with shape [node_num, node_num]
        """
        ini = sp.eye(self.nodes['total'])
        meta = [self.get_edge_type(x) for x in meta]
        for x in meta:
            ini = ini.dot(self.links['data'][x]) if x >= 0 else ini.dot(
                self.links['data_trans'][-x - 1])
        return ini

    def get_nonzero(self):
        self.nonzero = True
        self.re_cache = defaultdict(dict)
        for k in self.links['data']:
            th_mat = self.links['data'][k]
            for i in range(th_mat.shape[0]):
                th = th_mat[i].nonzero()[1]
                self.re_cache[k][i] = th
        for k in self.links['data_trans']:
            th_mat = self.links['data_trans'][k]
            for i in range(th_mat.shape[0]):
                th = th_mat[i].nonzero()[1]
                self.re_cache[-k - 1][i] = th

    def dfs(self, now, meta, meta_dict):
        if len(meta) == 0:
            meta_dict[now[0]].append(now)
            return
        # th_mat = self.links['data'][meta[0]] if meta[0] >= 0 else self.links['data_trans'][-meta[0] - 1]
        th_node = now[-1]
        for col in self.re_cache[meta[0]][th_node]:  # th_mat[th_node].nonzero()[1]:
            self.dfs(now + [col], meta[1:], meta_dict)

    def get_full_meta_path(self, meta=[], symmetric=False):
        """
        Get full meta path for each node
            meta is a list of edge types (also can be denoted by a pair of node types)
            return a dict of list[list] (key is node_id)
        """
        if not self.nonzero:
            self.get_nonzero()
        meta = [self.get_edge_type(x) for x in meta]
        if len(meta) == 1:
            meta_dict = {}
            start_node_type = self.links['meta'][meta[0]][0] if meta[0] >= 0 else \
            self.links['meta'][-meta[0] - 1][1]
            trav = range(self.nodes['shift'][start_node_type],
                         self.nodes['shift'][start_node_type] + self.nodes['count'][
                             start_node_type])
            for i in trav:
                meta_dict[i] = []
                self.dfs([i], meta, meta_dict)
        else:
            meta_dict1 = {}
            meta_dict2 = {}
            mid = len(meta) // 2
            meta1 = meta[:mid]
            meta2 = meta[mid:]
            start_node_type = self.links['meta'][meta1[0]][0] if meta1[0] >= 0 else \
            self.links['meta'][-meta1[0] - 1][1]
            trav = range(self.nodes['shift'][start_node_type],
                         self.nodes['shift'][start_node_type] + self.nodes['count'][
                             start_node_type])
            for i in trav:
                meta_dict1[i] = []
                self.dfs([i], meta1, meta_dict1)
            start_node_type = self.links['meta'][meta2[0]][0] if meta2[0] >= 0 else \
            self.links['meta'][-meta2[0] - 1][1]
            trav = range(self.nodes['shift'][start_node_type],
                         self.nodes['shift'][start_node_type] + self.nodes['count'][
                             start_node_type])
            for i in trav:
                meta_dict2[i] = []
            if symmetric:
                for k in meta_dict1:
                    paths = meta_dict1[k]
                    for x in paths:
                        meta_dict2[x[-1]].append(list(reversed(x)))
            else:
                for i in trav:
                    self.dfs([i], meta2, meta_dict2)
            meta_dict = {}
            start_node_type = self.links['meta'][meta1[0]][0] if meta1[0] >= 0 else \
            self.links['meta'][-meta1[0] - 1][1]
            for i in range(self.nodes['shift'][start_node_type],
                           self.nodes['shift'][start_node_type] + self.nodes['count'][
                               start_node_type]):
                meta_dict[i] = []
                for beg in meta_dict1[i]:
                    for end in meta_dict2[beg[-1]]:
                        meta_dict[i].append(beg + end[1:])
        return meta_dict

    def gen_file_for_evaluate(self, edge_list, confidence, edge_type, file_path, flag):
        """
        :param edge_list: shape(2, edge_num)
        :param confidence: shape(edge_num,)
        :param edge_type: shape(1)
        :param file_path: string
        """
        op = "w" if flag else "a"
        with open(file_path, op) as f:
            for l, r, c in zip(edge_list[0], edge_list[1], confidence):
                f.write(f"{l}\t{r}\t{edge_type}\t{c}\n")

    def get_node_type(self, node_id):
        for i in range(len(self.nodes['shift'])):
            if node_id < self.nodes['shift'][i] + self.nodes['count'][i]:
                return i

    def get_edge_type(self, info):
        if type(info) is int or len(info) == 1:
            return info
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return i
        info = (info[1], info[0])
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return -i - 1
        raise Exception('No available edge type')

    def get_edge_info(self, edge_id):
        return self.links['meta'][edge_id]

    def list_to_sp_mat(self, li):
        data = [x[2] for x in li]
        i = [x[0] for x in li]
        j = [x[1] for x in li]
        return sp.coo_matrix((data, (i, j)),
                             shape=(self.nodes['total'], self.nodes['total'])).tocsr()

    def load_types(self, name):
        """
        return types dict
            types: list of types
            total: total number of nodes
            data: a dictionary of type of all nodes)
        """
        types = {'types': list(), 'total': 0, 'data': dict()}
        with open(os.path.join(self.path, name), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.strip().split('\t')
                node_id, node_name, node_type = int(th[0]), th[1], int(th[2])
                types['data'][node_id] = node_type
                types['types'].append(node_type)
                types['total'] += 1
        types['types'] = list(set(types['types']))
        return types

    def get_train_neg(self, edge_types=[]):
        edge_types = self.test_types if edge_types == [] else edge_types
        train_neg = dict()
        for r_id in edge_types:
            h_type, t_type = self.links['meta'][r_id]
            t_range = (self.nodes['shift'][t_type],
                       self.nodes['shift'][t_type] + self.nodes['count'][t_type])
            '''get neg_neigh'''
            train_neg[r_id] = [[], []]
            for h_id in self.train_pos[r_id][0]:
                train_neg[r_id][0].append(h_id)
                neg_t = random.randrange(t_range[0], t_range[1])
                train_neg[r_id][1].append(neg_t)
        return train_neg

    def get_valid_neg(self, edge_types=[]):
        edge_types = self.test_types if edge_types == [] else edge_types
        valid_neg = dict()
        for r_id in edge_types:
            h_type, t_type = self.links['meta'][r_id]
            t_range = (self.nodes['shift'][t_type],
                       self.nodes['shift'][t_type] + self.nodes['count'][t_type])
            '''get neg_neigh'''
            valid_neg[r_id] = [[], []]
            for h_id in self.valid_pos[r_id][0]:
                valid_neg[r_id][0].append(h_id)
                neg_t = random.randrange(t_range[0], t_range[1])
                valid_neg[r_id][1].append(neg_t)
        return valid_neg

    def get_test_neigh_2hop(self):
        return self.get_test_neigh()

    def get_test_neigh(self):
        random.seed(1)
        neg_neigh, pos_neigh, test_neigh, test_label = dict(), dict(), dict(), dict()
        edge_types = self.test_types
        '''get sec_neigh'''
        pos_links = 0
        for r_id in self.links['data'].keys():
            pos_links += self.links['data'][r_id] + self.links['data'][r_id].T
        for r_id in self.links_test['data'].keys():
            pos_links += self.links_test['data'][r_id] + self.links_test['data'][r_id].T
        for r_id in self.valid_pos.keys():
            values = [1] * len(self.valid_pos[r_id][0])
            valid_of_rel = sp.coo_matrix((values, self.valid_pos[r_id]),
                                         shape=pos_links.shape)
            pos_links += valid_of_rel

        r_double_neighs = np.dot(pos_links, pos_links)
        data = r_double_neighs.data
        data[:] = 1
        r_double_neighs = \
            sp.coo_matrix((data, r_double_neighs.nonzero()), shape=np.shape(pos_links),
                          dtype=int) \
            - sp.coo_matrix(pos_links, dtype=int) \
            - sp.lil_matrix(np.eye(np.shape(pos_links)[0], dtype=int))
        data = r_double_neighs.data
        pos_count_index = np.where(data > 0)
        row, col = r_double_neighs.nonzero()
        r_double_neighs = sp.coo_matrix(
            (data[pos_count_index], (row[pos_count_index], col[pos_count_index])),
            shape=np.shape(pos_links))

        row, col = r_double_neighs.nonzero()
        data = r_double_neighs.data
        sec_index = np.where(data > 0)
        row, col = row[sec_index], col[sec_index]

        relation_range = [self.nodes['shift'][k] for k in
                          range(len(self.nodes['shift']))] + [self.nodes['total']]
        for r_id in self.links_test['data'].keys():
            neg_neigh[r_id] = defaultdict(list)
            h_type, t_type = self.links_test['meta'][r_id]
            r_id_index = np.where(
                (row >= relation_range[h_type]) & (row < relation_range[h_type + 1])
                & (col >= relation_range[t_type]) & (col < relation_range[t_type + 1]))[
                0]
            # r_num = np.zeros((3, 3))
            # for h_id, t_id in zip(row, col):
            #     r_num[self.get_node_type(h_id)][self.get_node_type(t_id)] += 1
            r_row, r_col = row[r_id_index], col[r_id_index]
            for h_id, t_id in zip(r_row, r_col):
                neg_neigh[r_id][h_id].append(t_id)

        for r_id in edge_types:
            '''get pos_neigh'''
            pos_neigh[r_id] = defaultdict(list)
            (row, col), data = self.links_test['data'][r_id].nonzero(), \
            self.links_test['data'][r_id].data
            for h_id, t_id in zip(row, col):
                pos_neigh[r_id][h_id].append(t_id)

            '''sample neg as same number as pos for each head node'''
            test_neigh[r_id] = [[], []]
            pos_list = [[], []]
            test_label[r_id] = []
            for h_id in sorted(list(pos_neigh[r_id].keys())):
                pos_list[0] = [h_id] * len(pos_neigh[r_id][h_id])
                pos_list[1] = pos_neigh[r_id][h_id]
                test_neigh[r_id][0].extend(pos_list[0])
                test_neigh[r_id][1].extend(pos_list[1])
                test_label[r_id].extend([1] * len(pos_list[0]))

                neg_list = random.choices(neg_neigh[r_id][h_id],
                                          k=len(pos_list[0])) if len(
                    neg_neigh[r_id][h_id]) != 0 else []
                test_neigh[r_id][0].extend([h_id] * len(neg_list))
                test_neigh[r_id][1].extend(neg_list)
                test_label[r_id].extend([0] * len(neg_list))
        return test_neigh, test_label

    def get_test_neigh_w_random(self):
        random.seed(1)
        all_had_neigh = defaultdict(list)
        neg_neigh, pos_neigh, neigh, label = dict(), dict(), dict(), dict()
        edge_types = self.test_types
        '''get pos_links of train and test data'''
        pos_links = 0
        for r_id in self.links['data'].keys():
            pos_links += self.links['data'][r_id] + self.links['data'][r_id].T
        for r_id in self.links_test['data'].keys():
            pos_links += self.links_test['data'][r_id] + self.links_test['data'][r_id].T
        for r_id in self.valid_pos.keys():
            values = [1] * len(self.valid_pos[r_id][0])
            valid_of_rel = sp.coo_matrix((values, self.valid_pos[r_id]),
                                         shape=pos_links.shape)
            pos_links += valid_of_rel

        row, col = pos_links.nonzero()
        for h_id, t_id in zip(row, col):
            all_had_neigh[h_id].append(t_id)
        for h_id in all_had_neigh.keys():
            all_had_neigh[h_id] = set(all_had_neigh[h_id])
        for r_id in edge_types:
            h_type, t_type = self.links_test['meta'][r_id]
            t_range = (self.nodes['shift'][t_type],
                       self.nodes['shift'][t_type] + self.nodes['count'][t_type])
            '''get pos_neigh and neg_neigh'''
            pos_neigh[r_id], neg_neigh[r_id] = defaultdict(list), defaultdict(list)
            (row, col), data = self.links_test['data'][r_id].nonzero(), \
            self.links_test['data'][r_id].data
            for h_id, t_id in zip(row, col):
                pos_neigh[r_id][h_id].append(t_id)
                neg_t = random.randrange(t_range[0], t_range[1])
                while neg_t in all_had_neigh[h_id]:
                    neg_t = random.randrange(t_range[0], t_range[1])
                neg_neigh[r_id][h_id].append(neg_t)
            '''get the neigh'''
            neigh[r_id] = [[], []]
            pos_list = [[], []]
            neg_list = [[], []]
            label[r_id] = []
            for h_id in sorted(list(pos_neigh[r_id].keys())):
                pos_list[0] = [h_id] * len(pos_neigh[r_id][h_id])
                pos_list[1] = pos_neigh[r_id][h_id]
                neigh[r_id][0].extend(pos_list[0])
                neigh[r_id][1].extend(pos_list[1])
                label[r_id].extend([1] * len(pos_neigh[r_id][h_id]))
                neg_list[0] = [h_id] * len(neg_neigh[r_id][h_id])
                neg_list[1] = neg_neigh[r_id][h_id]
                neigh[r_id][0].extend(neg_list[0])
                neigh[r_id][1].extend(neg_list[1])
                label[r_id].extend([0] * len(neg_neigh[r_id][h_id]))
        return neigh, label

    def get_test_neigh_full_random(self):
        edge_types = self.test_types
        random.seed(1)
        '''get pos_links of train and test data'''
        all_had_neigh = defaultdict(list)
        pos_links = 0
        for r_id in self.links['data'].keys():
            pos_links += self.links['data'][r_id] + self.links['data'][r_id].T
        for r_id in self.links_test['data'].keys():
            pos_links += self.links_test['data'][r_id] + self.links_test['data'][r_id].T
        for r_id in self.valid_pos.keys():
            values = [1] * len(self.valid_pos[r_id][0])
            valid_of_rel = sp.coo_matrix((values, self.valid_pos[r_id]),
                                         shape=pos_links.shape)
            pos_links += valid_of_rel

        row, col = pos_links.nonzero()
        for h_id, t_id in zip(row, col):
            all_had_neigh[h_id].append(t_id)
        for h_id in all_had_neigh.keys():
            all_had_neigh[h_id] = set(all_had_neigh[h_id])
        test_neigh, test_label = dict(), dict()
        for r_id in edge_types:
            test_neigh[r_id] = [[], []]
            test_label[r_id] = []
            h_type, t_type = self.links_test['meta'][r_id]
            h_range = (self.nodes['shift'][h_type],
                       self.nodes['shift'][h_type] + self.nodes['count'][h_type])
            t_range = (self.nodes['shift'][t_type],
                       self.nodes['shift'][t_type] + self.nodes['count'][t_type])
            (row, col), data = self.links_test['data'][r_id].nonzero(), \
            self.links_test['data'][r_id].data
            for h_id, t_id in zip(row, col):
                test_neigh[r_id][0].append(h_id)
                test_neigh[r_id][1].append(t_id)
                test_label[r_id].append(1)
                neg_h = random.randrange(h_range[0], h_range[1])
                neg_t = random.randrange(t_range[0], t_range[1])
                while neg_t in all_had_neigh[neg_h]:
                    neg_h = random.randrange(h_range[0], h_range[1])
                    neg_t = random.randrange(t_range[0], t_range[1])
                test_neigh[r_id][0].append(neg_h)
                test_neigh[r_id][1].append(neg_t)
                test_label[r_id].append(0)

        return test_neigh, test_label

    def gen_transpose_links(self):
        self.links['data_trans'] = defaultdict()
        for r_id in self.links['data'].keys():
            self.links['data_trans'][r_id] = self.links['data'][r_id].T

    def load_links(self, name):
        """
        return links dict
            total: total number of links
            count: a dict of int, number of links for each type
            meta: a dict of tuple, explaining the link type is from what type of node to what type of node
            data: a dict of sparse matrices, each link type with one matrix. Shapes are all (nodes['total', nodes['total'])
        """
        links = {'total': 0, 'count': Counter(), 'meta': {}, 'data': defaultdict(list)}
        with open(os.path.join(self.path, name), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id, r_id, link_weight = int(th[0]), int(th[1]), int(
                    th[2]), float(th[3])
                if r_id not in links['meta']:
                    h_type = self.get_node_type(h_id)
                    t_type = self.get_node_type(t_id)
                    links['meta'][r_id] = (h_type, t_type)
                links['data'][r_id].append((h_id, t_id, link_weight))
                links['count'][r_id] += 1
                links['total'] += 1
        new_data = {}
        for r_id in links['data']:
            new_data[r_id] = self.list_to_sp_mat(links['data'][r_id])
        links['data'] = new_data
        return links

    def load_nodes(self):
        """
        return nodes dict
        total: total number of nodes
        count: a dict of int, number of nodes for each type
        attr: a dict of np.array (or None), attribute matrices for each type of nodes
        shift: node_id shift for each type. You can get the id range of a type by
                    [ shift[node_type], shift[node_type]+count[node_type] )
        """
        nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}}
        with open(os.path.join(self.path, 'node.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                if len(th) == 4:
                    # Then this line of node has attribute
                    node_id, node_name, node_type, node_attr = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    node_attr = list(map(float, node_attr.split(',')))
                    nodes['count'][node_type] += 1
                    nodes['attr'][node_id] = node_attr
                    nodes['total'] += 1
                elif len(th) == 3:
                    # Then this line of node doesn't have attribute
                    node_id, node_name, node_type = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    nodes['count'][node_type] += 1
                    nodes['total'] += 1
                else:
                    raise Exception("Too few information to parse!")
        shift = 0
        attr = {}
        for i in range(len(nodes['count'])):
            nodes['shift'][i] = shift
            if shift in nodes['attr']:
                mat = []
                for j in range(shift, shift + nodes['count'][i]):
                    mat.append(nodes['attr'][j])
                attr[i] = np.array(mat)
            else:
                attr[i] = None
            shift += nodes['count'][i]
        nodes['attr'] = attr
        return nodes


class HGBDataLoaderNC:
    def __init__(self, path):
        self.path = path
        self.nodes = self.load_nodes()
        self.links = self.load_links()
        self.labels_train = self.load_labels('label.dat')
        self.labels_test = self.load_labels('label.dat.test')

    def get_sub_graph(self, node_types_tokeep):
        """
        node_types_tokeep is a list or set of node types that you want to keep in the sub-graph
        We only support whole type sub-graph for now.
        This is an in-place update function!
        return: old node type id to new node type id dict, old edge type id to new edge type id dict
        """
        keep = set(node_types_tokeep)
        new_node_type = 0
        new_node_id = 0
        new_nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}}
        new_links = {'total': 0, 'count': Counter(), 'meta': {},
                     'data': defaultdict(list)}
        new_labels_train = {'num_classes': 0, 'total': 0, 'count': Counter(),
                            'data': None, 'mask': None}
        new_labels_test = {'num_classes': 0, 'total': 0, 'count': Counter(),
                           'data': None, 'mask': None}
        old_nt2new_nt = {}
        old_idx = []
        for node_type in self.nodes['count']:
            if node_type in keep:
                nt = node_type
                nnt = new_node_type
                old_nt2new_nt[nt] = nnt
                cnt = self.nodes['count'][nt]
                new_nodes['total'] += cnt
                new_nodes['count'][nnt] = cnt
                new_nodes['attr'][nnt] = self.nodes['attr'][nt]
                new_nodes['shift'][nnt] = new_node_id
                beg = self.nodes['shift'][nt]
                old_idx.extend(range(beg, beg + cnt))

                cnt_label_train = self.labels_train['count'][nt]
                new_labels_train['count'][nnt] = cnt_label_train
                new_labels_train['total'] += cnt_label_train
                cnt_label_test = self.labels_test['count'][nt]
                new_labels_test['count'][nnt] = cnt_label_test
                new_labels_test['total'] += cnt_label_test

                new_node_type += 1
                new_node_id += cnt

        new_labels_train['num_classes'] = self.labels_train['num_classes']
        new_labels_test['num_classes'] = self.labels_test['num_classes']
        for k in ['data', 'mask']:
            new_labels_train[k] = self.labels_train[k][old_idx]
            new_labels_test[k] = self.labels_test[k][old_idx]

        old_et2new_et = {}
        new_edge_type = 0
        for edge_type in self.links['count']:
            h, t = self.links['meta'][edge_type]
            if h in keep and t in keep:
                et = edge_type
                net = new_edge_type
                old_et2new_et[et] = net
                new_links['total'] += self.links['count'][et]
                new_links['count'][net] = self.links['count'][et]
                new_links['meta'][net] = tuple(
                    map(lambda x: old_nt2new_nt[x], self.links['meta'][et]))
                new_links['data'][net] = self.links['data'][et][old_idx][:, old_idx]
                new_edge_type += 1

        self.nodes = new_nodes
        self.links = new_links
        self.labels_train = new_labels_train
        self.labels_test = new_labels_test
        return old_nt2new_nt, old_et2new_et

    def get_meta_path(self, meta=[]):
        """
        Get meta path matrix
            meta is a list of edge types (also can be denoted by a pair of node types)
            return a sparse matrix with shape [node_num, node_num]
        """
        ini = sp.eye(self.nodes['total'])
        meta = [self.get_edge_type(x) for x in meta]
        for x in meta:
            ini = ini.dot(self.links['data'][x]) if x >= 0 else ini.dot(
                self.links['data'][-x - 1].T)
        return ini

    def dfs(self, now, meta, meta_dict):
        if len(meta) == 0:
            meta_dict[now[0]].append(now)
            return
        th_mat = self.links['data'][meta[0]] if meta[0] >= 0 else self.links['data'][
            -meta[0] - 1].T
        th_node = now[-1]
        for col in th_mat[th_node].nonzero()[1]:
            self.dfs(now + [col], meta[1:], meta_dict)

    def get_full_meta_path(self, meta=[], symmetric=False):
        """
        Get full meta path for each node
            meta is a list of edge types (also can be denoted by a pair of node types)
            return a dict of list[list] (key is node_id)
        """
        meta = [self.get_edge_type(x) for x in meta]
        if len(meta) == 1:
            meta_dict = {}
            start_node_type = self.links['meta'][meta[0]][0] if meta[0] >= 0 else \
                self.links['meta'][-meta[0] - 1][1]
            for i in range(self.nodes['shift'][start_node_type],
                           self.nodes['shift'][start_node_type] + self.nodes['count'][
                               start_node_type]):
                meta_dict[i] = []
                self.dfs([i], meta, meta_dict)
        else:
            meta_dict1 = {}
            meta_dict2 = {}
            mid = len(meta) // 2
            meta1 = meta[:mid]
            meta2 = meta[mid:]
            start_node_type = self.links['meta'][meta1[0]][0] if meta1[0] >= 0 else \
                self.links['meta'][-meta1[0] - 1][1]
            for i in range(self.nodes['shift'][start_node_type],
                           self.nodes['shift'][start_node_type] + self.nodes['count'][
                               start_node_type]):
                meta_dict1[i] = []
                self.dfs([i], meta1, meta_dict1)
            start_node_type = self.links['meta'][meta2[0]][0] if meta2[0] >= 0 else \
                self.links['meta'][-meta2[0] - 1][1]
            for i in range(self.nodes['shift'][start_node_type],
                           self.nodes['shift'][start_node_type] + self.nodes['count'][
                               start_node_type]):
                meta_dict2[i] = []
            if symmetric:
                for k in meta_dict1:
                    paths = meta_dict1[k]
                    for x in paths:
                        meta_dict2[x[-1]].append(list(reversed(x)))
            else:
                for i in range(self.nodes['shift'][start_node_type],
                               self.nodes['shift'][start_node_type] +
                               self.nodes['count'][start_node_type]):
                    self.dfs([i], meta2, meta_dict2)
            meta_dict = {}
            start_node_type = self.links['meta'][meta1[0]][0] if meta1[0] >= 0 else \
                self.links['meta'][-meta1[0] - 1][1]
            for i in range(self.nodes['shift'][start_node_type],
                           self.nodes['shift'][start_node_type] + self.nodes['count'][
                               start_node_type]):
                meta_dict[i] = []
                for beg in meta_dict1[i]:
                    for end in meta_dict2[beg[-1]]:
                        meta_dict[i].append(beg + end[1:])
        return meta_dict

    def gen_file_for_evaluate(self, test_idx, label, file_path, mode='bi'):
        if test_idx.shape[0] != label.shape[0]:
            return
        if mode == 'multi':
            multi_label = []
            for i in range(label.shape[0]):
                label_list = [str(j) for j in range(label[i].shape[0]) if
                              label[i][j] == 1]
                multi_label.append(','.join(label_list))
            label = multi_label
        elif mode == 'bi':
            label = np.array(label)
        else:
            return
        with open(file_path, "w") as f:
            for nid, l in zip(test_idx, label):
                f.write(f"{nid}\t\t{self.get_node_type(nid)}\t{l}\n")

    def load_labels(self, name):
        """
        return labels dict
            num_classes: total number of labels
            total: total number of labeled data
            count: number of labeled data for each node type
            data: a numpy matrix with shape (self.nodes['total'], self.labels['num_classes'])
            mask: to indicate if that node is labeled, if False, that line of data is masked
        """
        labels = {'num_classes': 0, 'total': 0, 'count': Counter(), 'data': None,
                  'mask': None}
        nc = 0
        mask = np.zeros(self.nodes['total'], dtype=bool)
        data = [None for i in range(self.nodes['total'])]
        with open(os.path.join(self.path, name), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                node_id, node_name, node_type, node_label = int(th[0]), th[1], int(
                    th[2]), list(map(int, th[3].split(',')))
                for label in node_label:
                    nc = max(nc, label + 1)
                mask[node_id] = True
                data[node_id] = node_label
                labels['count'][node_type] += 1
                labels['total'] += 1
        labels['num_classes'] = nc
        new_data = np.zeros((self.nodes['total'], labels['num_classes']), dtype=int)
        for i, x in enumerate(data):
            if x is not None:
                for j in x:
                    new_data[i, j] = 1
        labels['data'] = new_data
        labels['mask'] = mask
        return labels

    def get_node_type(self, node_id):
        for i in range(len(self.nodes['shift'])):
            if node_id < self.nodes['shift'][i] + self.nodes['count'][i]:
                return i

    def get_edge_type(self, info):
        if type(info) is int or len(info) == 1:
            return info
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return i
        info = (info[1], info[0])
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return -i - 1
        raise Exception('No available edge type')

    def get_edge_info(self, edge_id):
        return self.links['meta'][edge_id]

    def list_to_sp_mat(self, li):
        data = [x[2] for x in li]
        i = [x[0] for x in li]
        j = [x[1] for x in li]
        return sp.coo_matrix((data, (i, j)),
                             shape=(self.nodes['total'], self.nodes['total'])).tocsr()

    def load_links(self):
        """
        return links dict
            total: total number of links
            count: a dict of int, number of links for each type
            meta: a dict of tuple, explaining the link type is from what type of node to what type of node
            data: a dict of sparse matrices, each link type with one matrix. Shapes are all (nodes['total'], nodes['total'])
        """
        links = {'total': 0, 'count': Counter(), 'meta': {}, 'data': defaultdict(list)}
        r_ids = []
        with open(os.path.join(self.path, 'link.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id, r_id, link_weight = int(th[0]), int(th[1]), int(
                    th[2]), float(th[3])

                if h_id in self.old_to_new_id_mapping.keys() and t_id in self.old_to_new_id_mapping.keys():
                    h_id = self.old_to_new_id_mapping[h_id]
                    t_id = self.old_to_new_id_mapping[t_id]
                    if r_id not in links['meta']:
                        h_type = self.get_node_type(h_id)
                        t_type = self.get_node_type(t_id)
                        links['meta'][r_id] = (h_type, t_type)
                    links['data'][r_id].append((h_id, t_id, link_weight))
                    links['count'][r_id] += 1
                    links['total'] += 1
                    if r_id not in r_ids:
                        r_ids.append(r_id)
        r_ids = sorted(r_ids)

        temp_meta = {}
        for i in range(len(links['meta'].keys())):
            temp_meta[i] = links['meta'][r_ids[i]]
        links['meta'] = temp_meta

        temp_count = {}
        for i in range(len(links['count'].keys())):
            temp_count[i] = links['count'][r_ids[i]]
        links['count'] = temp_count

        temp_data = {}
        for i in range(len(links['data'].keys())):
            temp_data[i] = links['data'][r_ids[i]]
        links['data'] = temp_data

        new_data = {}
        for r_id in links['data']:
            new_data[r_id] = self.list_to_sp_mat(links['data'][r_id])
        links['data'] = new_data
        return links

    def load_nodes(self):
        """
        return nodes dict
            total: total number of nodes
            count: a dict of int, number of nodes for each type
            attr: a dict of np.array (or None), attribute matrices for each type of nodes
            shift: node_id shift for each type. You can get the id range of a type by
                        [ shift[node_type], shift[node_type]+count[node_type] )
        """
        nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}}
        node_ids = []
        print(self.path)
        # with open(os.path.join(self.path, 'new.txt'), 'w') as f:
        # f.write("1")

        with open(os.path.join(self.path, 'node.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                if len(th) == 4:
                    # Then this line of node has attribute
                    node_id, node_name, node_type, node_attr = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    node_attr = list(map(float, node_attr.split(',')))
                    node_ids.append(node_id)
                    nodes['count'][node_type] += 1
                    nodes['attr'][node_id] = node_attr
                    nodes['total'] += 1
                elif len(th) == 3:
                    # Then this line of node doesn't have attribute
                    node_id, node_name, node_type = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    node_ids.append(node_id)
                    nodes['count'][node_type] += 1
                    nodes['total'] += 1
                else:
                    raise Exception("Too few information to parse!")

        # type_id mapping
        temp_count = Counter()
        for k in nodes['count']:
            temp_count[k] = nodes['count'][k]
        nodes['count'] = temp_count

        # node_id mapping
        self.old_to_new_id_mapping = {}
        self.new_to_old_id_mapping = {}
        node_ids = sorted(node_ids)
        for new_id in range(len(node_ids)):
            self.old_to_new_id_mapping[node_ids[new_id]] = new_id
            self.new_to_old_id_mapping[new_id] = node_ids[new_id]
        temp_attr = {}
        for old_id in node_ids:
            if old_id in nodes['attr'].keys():
                temp_attr[self.old_to_new_id_mapping[old_id]] = nodes['attr'][old_id]
        nodes['attr'] = temp_attr

        shift = 0
        attr = {}
        for i in range(len(nodes['count'])):
            nodes['shift'][i] = shift
            if shift in nodes['attr']:
                mat = []
                for j in range(shift, shift + nodes['count'][i]):
                    mat.append(nodes['attr'][j])
                attr[i] = np.array(mat)
            else:
                attr[i] = None
            shift += nodes['count'][i]
        nodes['attr'] = attr
        return nodes


def load_data_nc(data_loader: HGBDataLoaderNC, multi_label: bool = False):
    features = []
    for i in range(len(data_loader.nodes['count'])):
        th = data_loader.nodes['attr'][i]
        print(th)
        if th is None:
            features.append(sp.eye(data_loader.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(data_loader.links['data'].values())
    labels = np.zeros((data_loader.nodes['count'][0], data_loader.labels_train['num_classes']), dtype=int)
    val_ratio = 0.2
    train_idx = np.nonzero(data_loader.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0] * val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(data_loader.labels_test['mask'])[0]
    labels[train_idx] = data_loader.labels_train['data'][train_idx]
    labels[val_idx] = data_loader.labels_train['data'][val_idx]
    if multi_label:
        labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    return features, \
        adjM, \
        labels, \
        train_val_test_idx,
