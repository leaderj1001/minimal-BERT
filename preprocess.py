from torch.utils.data import Dataset, DataLoader
import torch

from config import load_args

import pandas as pd
import pickle
import os
import numpy as np
import csv
from collections import Counter, defaultdict
import random
import glob


class Vocab(object):
    def __init__(self, args, corpus_filename=None):
        self.out_filename = '{}/corpus.pkl'.format(args.base_dir)

        if not os.path.isfile(self.out_filename):
            self.corpus = Counter()
            self.itos, self.stoi = defaultdict(int), defaultdict(str)
            self.itos[0], self.itos[1], self.itos[2], self.itos[3], self.itos[4] = '<PAD>', '<CLS>', '<SEP>', '<END>', '<MASK>'
            self.stoi['<PAD>'], self.stoi['<CLS>'], self.stoi['<SEP>'], self.stoi['<END>'], self.stoi['<MASK>'] = 0, 1, 2, 3, 4
            self.cnt = 5

            for filename in corpus_filename:
                _data = pd.read_csv('{}/{}'.format(args.base_dir, filename), delimiter='\t', header=0)

                for idx, s1, s2, label in zip(_data['index'], _data['sentence1'], _data['sentence2'], _data['label']):
                    self._build_vocab(s1)
                    self._build_vocab(s2)

            data = {
                'corpus': self.corpus,
                'itos': self.itos,
                'stoi': self.stoi
            }

            with open(self.out_filename, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def _get_data(self):
        with open(self.out_filename, 'rb') as f:
            return pickle.load(f)

    def _build_vocab(self, sentence):
        words = sentence.split()
        for word in words:
            self.corpus[word] += 1

            if word not in self.stoi.keys():
                self.stoi[word] = self.cnt
                self.itos[self.cnt] = word
                self.cnt += 1


def handle_pickle(args, data=None, mode='train'):
    if data is not None:
        with open('{}/{}.pkl'.format(args.base_dir, mode), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        return data
    else:
        with open('{}/{}.pkl'.format(args.base_dir, mode), 'rb') as f:
            return pickle.load(f)


def prepare_dataset(args, mode='train'):
    filename = '{}/{}.pkl'.format(args.base_dir, mode)

    if not os.path.isfile(filename):
        print('Make Data pkl file: {}'.format(mode))
        csv_data = pd.read_csv('{}/{}.tsv'.format(args.base_dir, mode), delimiter='\t', header=0)

        pkl_data = {
            'data': [],
            'label': [],
            'len': 0
        }

        labels = {
            'not_entailment': 0,
            'entailment': 1,
        }

        for idx, s1, s2, label in zip(csv_data['index'], csv_data['sentence1'], csv_data['sentence2'], csv_data['label']):
            pkl_data['data'].append([s1, s2])
            pkl_data['label'].append([labels[label]])
            pkl_data['len'] += 1

        return handle_pickle(args, pkl_data, mode=mode)
    else:
        print('Ready to {}'.format(mode))
        return handle_pickle(args, mode=mode)


class BERTDataset(Dataset):
    def __init__(self, args, mode='train', remove_pkl=False):
        if remove_pkl:
            os.remove('{}/{}.pkl'.format(args.base_dir, mode))
        self.data = prepare_dataset(args, mode=mode)
        self.mode = mode
        self.args = args

        self.data['convert_data'] = []

        for idx in range(self.data['len']):
            s1, s2, task_label = self.data['data'][idx][0], self.data['data'][idx][1], self.data['label'][idx]
            self.data['convert_data'].append([self._convert_sentence(s1), self._convert_sentence(s2), task_label])

    def __getitem__(self, idx):
        s1, s2, task_label = self.data['convert_data'][idx]

        if self.args.task:
            rm_s1, rm_s2 = s1, s2
            rm_s1_label, rm_s2_label = s1, s2
            rs_label = None
        else:
            rs_s1, rs_s2, rs_label = self._random_sentence(s1, s2)
            rm_s1, rm_s1_label = self._random_masking(rs_s1)
            rm_s2, rm_s2_label = self._random_masking(rs_s2)

        segment = [1 for _ in range(len(rm_s1))] + [2 for _ in range(len(rm_s2))]
        data = [self.args.vocab['stoi']['<CLS>']] + rm_s1 + [self.args.vocab['stoi']['<SEP>']] + rm_s2 + [self.args.vocab['stoi']['<SEP>']]
        label = [self.args.vocab['stoi']['<CLS>']] + rm_s1_label + [self.args.vocab['stoi']['<SEP>']] + rm_s2_label + [self.args.vocab['stoi']['<SEP>']]

        segment = segment[:self.args.max_len]
        data = data[:self.args.max_len]
        label = label[:self.args.max_len]

        padding = [self.args.vocab['stoi']['<PAD>'] for _ in range(self.args.max_len - len(data))]
        segment += [self.args.vocab['stoi']['<PAD>'] for _ in range(self.args.max_len - len(segment))]
        data += padding
        label += padding

        data = torch.tensor(data)
        segment = torch.tensor(segment)
        if self.mode == 'test':
            return {
                'data': data,
                'segment': segment
            }

        label = torch.tensor(label)
        return {
            'data': data,
            'segment': segment,
            'label': label,
            'rs_label': rs_label,
            'task_label': task_label
        }

    def __len__(self):
        return self.data['len']

    def _random_sentence(self, _s1, _s2):
        if random.random() < self.args.nsp_ratio:
            return _s1, _s2, 1
        random_idx = random.randrange(self.data['len'])
        return _s1, self.data['convert_data'][random_idx][1], 0

    def _convert_sentence(self, sentence):
        words, data = sentence.split(), []
        for word in words:
            data.append(self.args.vocab['stoi'][word])

        return data

    def _random_masking(self, words):
        data, label = [], []
        for word in words:
            if random.random() < self.args.mlm_ratio:
                rand = random.random()
                if rand < 0.8:
                    data_token = self.args.vocab['stoi']['<MASK>']
                elif rand < 0.9:
                    data_token = random.randrange(len(self.args.vocab['stoi'].values()))
                else:
                    data_token = word
                label_token = word
            else:
                data_token, label_token = word, word

            data.append(data_token)
            label.append(label_token)

        return data, label


def load_data(args):
    vocab = Vocab(args, corpus_filename=['train.tsv', 'dev.tsv'])
    args.vocab = vocab._get_data()

    train_data = BERTDataset(args, mode='train')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_data = BERTDataset(args, mode='dev')
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, test_loader


from config import load_args
train_loader, test_loader = load_data(load_args())

for _ in test_loader:
    print(_)
    break