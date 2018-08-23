#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#@Filename : utils
#@Date : 2018-07-17-21-39
#@Poject: ArabicCQA
#@AUTHOR : Yassine EL ADLOUNI

import os
import argparse
from itertools import groupby

import torch

def save_checkpoint(state, filename):
    torch.save(state, filename)

def get_args():

    parser = argparse.ArgumentParser(description='PyTorch/Semeval CQA parameters')

    parser.add_argument('--model', type=str, default='BiMPM', choices=['mpcnn', 'bimpm'], help='Model to use')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='Momentum')
    parser.add_argument('--regularization', type=float, default=3e-4, help='Regularization')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--n_perspectives', type=int, default=20, help='Perspectives')
    parser.add_argument('--n_hidden', type=int, default=20, help='Hidden')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--device', type=int, default=-1, help='device to use, default cpu')
    parser.add_argument('--cache', type=str, default=os.path.join(os.getcwd(), '.cache'))
    parser.add_argument('--vectors', type=str, default='/home/usuaris/yassine/SemevalSLS/.cache/fasttext.webteb.100d.vec')
    parser.add_argument('--datasets-dir', type=str, default='/home/usuaris/yassine/SemevalSLS/.data')
    parser.add_argument('--train_file', type=str, default='terminology-train.txt')
    parser.add_argument('--dev-file', type=str, default='terminology-dev.txt')
    parser.add_argument('--test-file', type=str, default='terminology-test.txt')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--logs-dir', type=str, default=os.path.join(os.getcwd(), 'Logs directory'))
    parser.add_argument('--log-interval', type=int, default=50)
    args = parser.parse_args()

    return args

def MAP(qids, relevances, confidences):
    precisions = []
    records = 0

    for key, group in groupby(sorted(zip(qids, relevances, confidences)), lambda x: x[0]):
        data = list(group)

        ytrue = [item[1] for item in data]
        ypred = [item[2] for item in data]


        precisions.append(AP(ytrue, ypred))

        records += len(ytrue)

    mAP = sum(precisions) / len(precisions)

    return mAP


def AP(y_true, y_pred, k=10):
    c = zip(y_true, y_pred)
    c = sorted(c, key=lambda x: x[1], reverse=True)

    ipos = 0.
    s = 0.
    for i, (g, p) in enumerate(c):
        if g > 0.:
            ipos += 1.
            s += ipos / (1. + i)
        if i >= k:
            break
    if ipos == 0:
        return 0.
    else:
        return s / ipos
