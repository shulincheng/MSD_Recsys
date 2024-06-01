import itertools
import pandas as pd
import numpy as np


def read_sessions(filepath):
    sessions = pd.read_csv(filepath, sep='\t', header=None, squeeze=True)
    sessions = sessions.apply(lambda x: list(map(int, x.split(',')))).values
    return sessions


def read_dataset(dataset_dir):
    train_sessions = read_sessions(dataset_dir / 'train.txt')
    test_sessions = read_sessions(dataset_dir / 'test.txt')
    with open(dataset_dir / 'num_items.txt', 'r') as f:
        num_items = int(f.readline())
    return train_sessions, test_sessions, num_items


def creat_index(sessions):
    session_len = np.fromiter(map(len, sessions), dtype=np.int)
    session_idx = np.repeat(np.arange(len(sessions)), session_len-1)
    idx = map(lambda x: range(1, x), session_len)
    idx = itertools.chain.from_iterable(idx)
    idx = np.fromiter(idx, dtype=np.int)
    index = np.column_stack((session_idx, idx))
    return index


class AugmentedDataset:
    def __init__(self, sessions):
        self.sessions = sessions
        index = creat_index(self.sessions)
        self.index = index

    def __getitem__(self, idx):
        session_id, idx = self.index[idx]
        seq_train = self.sessions[session_id][:idx]
        label = self.sessions[session_id][idx]
        return  seq_train, label

    def __len__(self):
        return len(self.index)
