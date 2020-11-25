import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import load_args
from preprocess import load_data
from model import BERT

import os
from sklearn.metrics import f1_score, precision_score, recall_score


def _metric(y_true, y_pred, average='weighted'):
    f1 = f1_score(y_true, y_pred, average=average)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)

    return precision, recall, f1


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


def _eval(epoch, test_loader, model, criterion, args):
    model.eval()

    mlm_acc, nsp_acc, mlm_losses, nsp_losses, losses, step = 0., 0., 0., 0., 0., 0.
    total = 0.
    with torch.no_grad():
        for data in test_loader:
            inputs, segment, mlm_label, nsp_label = data['data'], data['segment'], data['label'], data['rs_label']
            if args.task:
                inputs, segment, mlm_label, nsp_label = data['data'], data['segment'], data['label'], data['task_label']

            if args.cuda:
                inputs, segment, mlm_label, nsp_label = inputs.cuda(), segment.cuda(), mlm_label.cuda(), nsp_label.cuda()

            mlm_logits, nsp_logits = model(inputs, segment)
            mlm_logits = mlm_logits.view(-1, args.vocab_len)
            mlm_label = mlm_label.view(-1)

            nsp_loss = criterion['nsp'](nsp_logits, nsp_label)
            nsp_losses += nsp_loss.item()

            if not args.task:
                mlm_loss = criterion['mlm'](mlm_logits, mlm_label)
                mlm_losses += mlm_loss.item()
                loss = mlm_loss + nsp_loss
            else:
                loss = nsp_loss
            losses += loss.item()

            mlm_pred = F.softmax(mlm_logits, dim=-1).max(-1)[1]
            nsp_pred = F.softmax(nsp_logits, dim=-1).max(-1)[1]

            inds = (mlm_label != 0).view(-1)
            mlm_acc += mlm_pred[inds].eq(mlm_label[inds]).sum().item()
            nsp_acc += nsp_pred.eq(nsp_label).sum().item()

            step += 1
            total += inds.size(0)
        mlm_losses /= step
        nsp_losses /= step
        losses /= step
        mlm_acc = mlm_acc / total * 100.
        nsp_acc = nsp_acc / len(test_loader.dataset) * 100.
        print('[Test Epoch: {0:4d}] mlm loss: {1:.3f}, nsp loss: {2:.3f}, loss: {3:.3f}, mlm acc: {4:.4f}, nsp acc: {5:.4f}'.format(epoch, mlm_losses,
                                                                                                  nsp_losses, losses,
                                                                                                  mlm_acc, nsp_acc))
        return mlm_losses, nsp_losses, losses, mlm_acc, nsp_acc


def _train(epoch, train_loader, model, optimizer, criterion, args):
    model.train()

    mlm_acc, nsp_acc, mlm_losses, nsp_losses, losses, step = 0., 0., 0., 0., 0., 1
    total = 0.
    for idx, data in enumerate(train_loader):
        inputs, segment, mlm_label, nsp_label = data['data'], data['segment'], data['label'], data['rs_label']
        if args.task:
            inputs, segment, mlm_label, nsp_label = data['data'], data['segment'], data['label'], data['task_label']

        if args.cuda:
            inputs, segment, mlm_label, nsp_label = inputs.cuda(), segment.cuda(), mlm_label.cuda(), nsp_label.cuda()

        mlm_logits, nsp_logits = model(inputs, segment)
        mlm_logits = mlm_logits.view(-1, args.vocab_len)
        mlm_label = mlm_label.view(-1)

        optimizer.zero_grad()
        nsp_loss = criterion['nsp'](nsp_logits, nsp_label)
        nsp_losses += nsp_loss.item()

        if not args.task:
            mlm_loss = criterion['mlm'](mlm_logits, mlm_label)
            mlm_losses += mlm_loss.item()
            loss = mlm_loss + nsp_loss
        else:
            loss = nsp_loss
        losses += loss.item()
        loss.backward()
        optimizer.step()

        mlm_pred = F.softmax(mlm_logits, dim=-1).max(-1)[1]
        nsp_pred = F.softmax(nsp_logits, dim=-1).max(-1)[1]

        inds = (mlm_label != 0).view(-1)
        mlm_acc += mlm_pred[inds].eq(mlm_label[inds]).sum().item()
        nsp_acc += nsp_pred.eq(nsp_label).sum().item()

        step += 1
        total += inds.size(0)

    mlm_losses /= step
    nsp_losses /= step
    losses /= step
    mlm_acc = mlm_acc / total * 100.
    nsp_acc = nsp_acc / len(train_loader.dataset) * 100.
    print('[Train Epoch: {0:4d}] mlm loss: {1:.3f}, nsp loss: {2:.3f}, loss: {3:.3f}, mlm acc: {4:.4f}, nsp acc: {5:.4f}'.format(epoch, mlm_losses, nsp_losses, losses, mlm_acc, nsp_acc))

    return mlm_losses, nsp_losses, losses, mlm_acc, nsp_acc


def main(args):
    train_loader, test_loader = load_data(args)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    args.vocab_len = len(args.vocab['stoi'].keys())

    model = BERT(args.vocab_len, args.max_len, args.heads, args.embedding_dim, args.N)
    if args.cuda:
        model = model.cuda()

    if args.task:
        print('Start Down Stream Task')
        args.epochs = 3
        args.lr = 3e-5

        state_dict = torch.load(args.checkpoints)
        model.load_state_dict(state_dict['model_state_dict'])

        criterion = {
            'mlm': None,
            'nsp': nn.CrossEntropyLoss()
        }

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(1, args.epochs + 1):
            train_mlm_loss, train_nsp_loss, train_loss, train_mlm_acc, train_nsp_acc = _train(epoch, train_loader, model, optimizer, criterion, args)
            test_mlm_loss, test_nsp_loss, test_loss, test_mlm_acc, test_nsp_acc = _eval(epoch, test_loader, model, criterion, args)
            save_checkpoint(model, optimizer, args, epoch)
    else:
        print('Start Pre-training')
        criterion = {
            'mlm': nn.CrossEntropyLoss(ignore_index=0),
            'nsp': nn.CrossEntropyLoss()
        }
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(1, args.epochs):
            train_mlm_loss, train_nsp_loss, train_loss, train_mlm_acc, train_nsp_acc = _train(epoch, train_loader, model, optimizer, criterion, args)
            test_mlm_loss, test_nsp_loss, test_loss, test_mlm_acc, test_nsp_acc = _eval(epoch, test_loader, model, criterion, args)
            save_checkpoint(model, optimizer, args, epoch)


if __name__ == '__main__':
    args = load_args()
    main(args)
