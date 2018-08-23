#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#@Filename : train
#@Date : 22/08/18
#@Poject: SemevalSLS
#@AUTHOR : Yassine EL ADLOUNI


from ignite.engine.engine import Engine
import torch


def _prepare_batch(batch):
    x, y = batch, batch.qarel
    return x, y


def create_supervised_trainer(model, optimizer, loss_fn, cuda=False):
    """
    Factory function for creating a trainer for supervised models
    Args:
        model (torch.nn.Module): the model to train
        optimizer (torch.optim.Optimizer): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        cuda (bool, optional): whether or not to transfer batch to GPU (default: False)
    Returns:
        Engine: a trainer engine with supervised update function
    """
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.data.cpu().item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics=None, y_to_score=None, pred_to_score=None, cuda=False):
    """
    Factory function for creating an evaluator for supervised models
    Args:
        model (torch.nn.Module): the model to train
        metrics (dict of str: Metric): a map of metric names to Metrics
        cuda (bool, optional): whether or not to transfer batch to GPU (default: False)
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    def _inference(engine, batch):
        model.eval()
        x, y = _prepare_batch(batch)
        y_pred = model(x)
        if y_to_score is not None:
            y = y_to_score(y, batch)

        if pred_to_score is not None:
            y_pred = pred_to_score(y_pred, batch)

        return batch.qid, y_pred, y

    engine = Engine(_inference)

    if metrics is None:
        metrics = {}

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
