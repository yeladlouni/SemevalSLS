#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#@Filename : main
#@Date : 2018-08-17-15-48
#@Poject: SemevalSLS
#@AUTHOR : Yassine EL ADLOUNI

import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from models.bimpm import BiMPM
from semeval import Semeval
from runner import Runner
from metrics.retrieval_metrics import MAP, MRR
from utils import get_args


def resolved_pred_to_score(y, batch):
    num_classes = batch.dataset.num_classes
    predict_classes = torch.arange(0, num_classes, dtype=torch.float).expand(len(batch.qid), num_classes)

    if y.is_cuda:
        with torch.cuda.device(y.get_device()):
            predict_classes = predict_classes.cuda()

    return (predict_classes * y.exp()).sum(dim=1)


def y_to_score(y, batch):
    return y[:, 1]


args = get_args()

train_loader, dev_loader, test_loader = Semeval.iters(args)

model = eval(args.model)(args)

if args.device != -1:
    with torch.cuda.device(args.device):
        model = model.cuda()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

total_params = 0
for param in model.parameters():
    if param.requires_grad:
        size = [s for s in param.size()]
        total_params += np.prod(size)
logger.info('Total number of parameters: %s', total_params)

criterion = nn.KLDivLoss()
metrics = {
    'map': MAP(),
    'mrr': MRR()
}

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                       weight_decay=args.regularization)
runner = Runner(model, criterion, metrics, optimizer, y_to_score, resolved_pred_to_score, args.device, None)
runner.run(args.epochs, train_loader, dev_loader, test_loader, args.log_interval)