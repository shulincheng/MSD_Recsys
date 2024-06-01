import argparse
import os
import random
import numpy as np
from pathlib import Path
import torch as th
from torch.utils.data import DataLoader, SequentialSampler

from model import MyModel
from utils.train import TrainRunner
from utils.dataset import read_dataset, AugmentedDataset
from collate import collate_fn, seq_to_graph

def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.enabled = True
    
seed_torch(2022)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='Diginetica', help='the dataset directory')
parser.add_argument('--embedding_dim', type=int, default=256, help='the embedding size')
parser.add_argument('--num_layers', type=int, default=1, help='the number of layers')
parser.add_argument('--feat_drop', type=float, default=0.1, help='the dropout ratio for features')
parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
parser.add_argument('--batch_size', type=int, default=512, help='the batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='the number of training epochs')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='the parameter for L2 regularization')
parser.add_argument('--patience', type=int, default=3, help='the training stops when the performance does not improve')
parser.add_argument('--log_interval', type=int, default=10, help='print the loss after this number of iterations')
parser.add_argument('--order', type=int, default=5, help='order of msg')
parser.add_argument('--reducer', type=str, default='mean', help='method for reducer')
parser.add_argument('--norm', type=bool, default=True, help='whether use l2 norm')
parser.add_argument('--fusion', default=True, action='store_true', help='whether use IFR')
parser.add_argument('--extra', type=bool, default=True, help='whether use l2 norm')

args = parser.parse_args()
print(args)

device = th.device('cuda:2' if th.cuda.is_available() else'cpu')
dataset = args.dataset
if dataset == 'Diginetica':
    dataset_dir = Path('../data/Diginetica')
elif dataset == 'Yoochoose1_4':
    dataset_dir = Path('../data/Yoochoose1_4')
elif dataset == 'Yoochoose1_64':
    dataset_dir = Path('../data/Yoochoose1_64')
elif dataset == 'Gowalla':
    dataset_dir = Path('../data/Gowalla')
elif dataset == 'Last.FM':
    dataset_dir = Path('../data/Lastfm')
print('Reading dataset!')
train_sessions, test_sessions, num_items = read_dataset(dataset_dir)


train_set = AugmentedDataset(train_sessions)
test_set = AugmentedDataset(test_sessions)

collate_fn = collate_fn(seq_to_graph, order=args.order)

train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    collate_fn=collate_fn,
    pin_memory=True,
    sampler=SequentialSampler(train_set)
)

test_loader = DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True
)

model = MyModel(num_items, args.embedding_dim, args.num_layers, dropout=args.feat_drop, reducer=args.reducer,
                order=args.order, norm=args.norm, fusion=args.fusion, extra=args.extra, device=device)
model = model.to(device)

runner = TrainRunner(
    model,
    train_loader,
    test_loader,
    device=device,
    lr=args.lr,
    weight_decay=args.weight_decay,
    patience=args.patience
)
print('Start Training!')
mrr, hit = runner.train(args.epochs, args.log_interval)
print('MRR@20\tHR@20')
print(f'{mrr * 100:.3f}%\t{hit * 100:.3f}%')











