from collections import Counter

import dgl
import numpy as np
import torch as th


def label_last(g, last_nid):
    for i in range(len(last_nid)):
        is_last = th.zeros(g.num_nodes('s' + str(i + 1)), dtype=th.int32)
        is_last[last_nid[i]] = 1
        g.nodes['s' + str(i + 1)].data['last'] = is_last

    return g


def seq_to_graph(seq, order=1):
    train_order = order
    order = min(len(seq), train_order)
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    seq_nid = [iid2nid[iid] for iid in seq]
    last_item = [iid2nid[seq[-1]]]
    combine_seqs = []
    item_dicts = [iid2nid]

    def com(i, order):
        item = str(seq[i:i + order])
        return item

    class combine:
        def __init__(self):
            self.dict = {}

        def __call__(self, *input):
            return self.forward(*input)

        def forward(self, i, order):
            if str(i) not in self.dict:
                self.dict[str(i)] = {}
            if order not in self.dict[str(i)]:
                self.dict[str(i)][order] = com(i, order)
            return self.dict[str(i)][order]

    combine = combine()

    for i in range(1, train_order):
        combine_seq = []
        item_dict = {}
        cnt = 0
        for j in range(len(seq_nid) - i):
            item = combine(j, i + 1)
            if item not in item_dict:
                item_dict[item] = cnt
                cnt += 1
                combine_seq.append([seq[idx] for idx in range(j, j + i + 1)])

        if len(item_dict) > 0:
            last_item.append(item_dict[item])
        else:
            last_item.append(0)
        combine_seqs.append(combine_seq)

        item_dicts.append(item_dict)

    graph_data = {}

    for k in range(order):
        if k == 0:
            counter = Counter([(seq_nid[i], seq_nid[i + 1]) for i in range(len(seq) - 1)])
        else:
            counter = Counter([(item_dicts[k][combine(i, k + 1)], item_dicts[k][combine(i + 1, k + 1)]) for i in
                               range(len(seq) - k - 1)])

        edges = counter.keys()

        if len(edges) > 1000:
            src, dst = zip(*edges)
        else:
            src, dst = [], []

        graph_data[('s' + str(k + 1), 'intra' + str(k + 1), 's' + str(k + 1))] = (
            th.tensor(src).long(), th.tensor(dst).long())

    for k in range(1, order):

        counter = Counter([(seq_nid[i], item_dicts[k][combine(i + 1, k + 1)]) for i in range(len(seq) - k - 1)])
        # print(counter)

        edges = counter.keys()

        if len(edges) > 0:
            src, dst = zip(*edges)
        else:
            src, dst = th.LongTensor([]), th.LongTensor([])

        graph_data[('s1', 'inter', 's' + str(k + 1))] = (src, dst)

        counter = Counter([(item_dicts[k][combine(i, k + 1)], seq_nid[i + k + 1]) for i in range(len(seq) - k - 1)])

        edges = counter.keys()

        if len(edges) > 0:
            src, dst = zip(*edges)
        else:
            src, dst = th.LongTensor([]), th.LongTensor([])

        graph_data[('s' + str(k + 1), 'inter', 's1')] = (src, dst)

    if order < train_order:
        for i in range(order, train_order):
            graph_data[('s' + str(i + 1), 'intra' + str(i + 1), 's' + str(i + 1))] = (
                th.LongTensor([]), th.LongTensor([]))
            graph_data[('s' + str(i + 1), 'inter', 's1')] = (th.LongTensor([]), th.LongTensor([]))
            graph_data[('s1', 'inter', 's' + str(i + 1))] = (th.LongTensor([]), th.LongTensor([]))

    g = dgl.heterograph(graph_data)
    if g.num_nodes('s1') < len(items):
        g.add_nodes(len(items) - g.num_nodes('s1'), ntype='s1')
    g.nodes['s1'].data['iid'] = th.from_numpy(items)

    if order < train_order:
        for i in range(order, train_order):
            if 's' + str(i + 1) not in g.ntypes or g.num_nodes('s' + str(i + 1)) == 0:
                g.add_nodes(1, ntype='s' + str(i + 1))
                g.nodes['s' + str(i + 1)].data['iid'] = th.ones(1, i + 1).long() * g.nodes['s1'].data['iid'][0]
    for i in range(1, order):
        if g.num_nodes('s' + str(i + 1)) == 0:
            g.add_nodes(1, ntype='s' + str(i + 1))

        g.nodes['s' + str(i + 1)].data['iid'] = th.from_numpy(np.array(combine_seqs[i - 1])).type(th.int64)

    label_last(g, last_item)

    return g


def collate_fn(seq_to_graph, order):
    def collate_fn(samples):
        seqs_train, labels = zip(*samples)
        inputs_train = []


        batch_train = list(map(seq_to_graph, seqs_train, [order for _ in range(len(seqs_train))]))
        bg_train = dgl.batch(batch_train)
        inputs_train.append(bg_train)
        labels = th.LongTensor(labels)

        return inputs_train, labels

    return collate_fn
