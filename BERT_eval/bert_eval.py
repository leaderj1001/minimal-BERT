import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import Dataset, DataLoader

import argparse
import os
import pandas as pd
import pickle


def load_args():
    parser = argparse.ArgumentParser('BERT')

    parser.add_argument('--max_len', type=int, default=512)

    # data loader
    parser.add_argument('--base_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)

    # training
    parser.add_argument('--lr', type=float, default=3e-5, help="5e-5 (2), 3e-5 (3), 2e-5 (4)")
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=3, help="5e-5 (2), 3e-5 (3), 2e-5 (4)")
    parser.add_argument('--gradient_clip', type=float, default=1.)

    args = parser.parse_args()

    return args


def save_checkpoint(model, optimizer, args, epoch):
    print('Model Saving...')
    if args.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join('checkpoints', 'checkpoint_model_best.pth'))


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

        for idx, s1, s2, label in zip(csv_data['index'], csv_data['sentence1'], csv_data['sentence2'],
                                      csv_data['label']):
            pkl_data['data'].append([s1, s2])
            pkl_data['label'].append([labels[label]])
            pkl_data['len'] += 1

        return handle_pickle(args, pkl_data, mode=mode)
    else:
        print('Ready to {}'.format(mode))
        return handle_pickle(args, mode=mode)


class BERTDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.data = prepare_dataset(args, mode=mode)
        self.mode = mode
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __getitem__(self, idx):
        s1, s2, task_label = self.data['data'][idx][0], self.data['data'][idx][1], self.data['label'][idx]
        data = s1 + s2

        encoded_data = self.tokenizer.encode(data, add_special_tokens=True)
        inputs = encoded_data + [0] * (self.args.max_len - len(encoded_data))
        token_type = [0] * len(inputs)
        attention_mask = [1] * len(encoded_data) + [0] * (self.args.max_len - len(encoded_data))

        return {
            'data': torch.tensor(inputs),
            'attention_mask': torch.tensor(attention_mask),
            'token_type_ids': torch.tensor(token_type),
            'label': task_label[0],
        }

    def __len__(self):
        return self.data['len']


class BERTClassifier(nn.Module):
    def __init__(self):
        super(BERTClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 2)
        )

    def forward(self, x, attention_mask, token_type_ids):
        out, _ = self.bert(x, attention_mask=attention_mask, token_type_ids=token_type_ids)

        out = self.fc(out[:, 0])

        return out


def _train(epoch, train_loader, model, optimizer, criterion, args):
    model.train()

    losses, step, acc, total = 0., 0., 0., 0.
    for _ in train_loader:
        data, attention_mask, token_type_ids, label = _['data'], _['attention_mask'], _['token_type_ids'], _['label']

        if args.cuda:
            data, attention_mask, token_type_ids, label = data.cuda(), attention_mask.cuda(), token_type_ids.cuda(), label.cuda()

        outputs = model(data, attention_mask=attention_mask, token_type_ids=token_type_ids)

        optimizer.zero_grad()
        loss = criterion(outputs, label)
        losses += loss.item()
        loss.backward()
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()

        pred = F.softmax(outputs, dim=-1).max(-1)[1]
        acc += pred.eq(label).sum().item()

        step += 1
        total += label.size(0)

    print('Train Epoch: {0:4d}, losses: {1:.3f}, acc: {2:.3f}'.format(epoch, losses / step, acc / total))


def _eval(epoch, test_loader, model, criterion, args):
    model.eval()

    with torch.no_grad():
        losses, step, acc, total = 0., 0., 0., 0.
        for _ in test_loader:
            data, attention_mask, token_type_ids, label = _['data'], _['attention_mask'], _['token_type_ids'], _['label']

            if args.cuda:
                data, attention_mask, token_type_ids, label = data.cuda(), attention_mask.cuda(), token_type_ids.cuda(), label.cuda()

            outputs = model(data, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(outputs, label)
            losses += loss.item()

            pred = F.softmax(outputs, dim=-1).max(-1)[1]
            acc += pred.eq(label).sum().item()

            step += 1
            total += label.size(0)

        print('Test Epoch: {0:4d}, losses: {1:.3f}, acc: {2:.3f}'.format(epoch, losses / step, acc / total))


def _main(args):
    model = BERTClassifier()

    if args.cuda:
        model = model.cuda()

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train_dataset = BERTDataset(args, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_dataset = BERTDataset(args, mode='dev')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        _train(epoch, train_loader, model, optimizer, criterion, args)
        _eval(epoch, test_loader, model, criterion, args)
        save_checkpoint(model, optimizer, args, epoch)


if __name__ == '__main__':
    args = load_args()
    _main(args)