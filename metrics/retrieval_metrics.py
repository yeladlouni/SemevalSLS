#!/usr/lename : retrieval_metrics
#@Date : 22/08/18
#@Poject: SemevalSLS
#@AUTHOR : Yassine EL ADLOUNI

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
import torch

from utils import MAP as map_metric

class RetrievalMetrics(Metric):
    """
    Calculates retrieval metrics using trec_eval
    `update` must receive output of the form (ids, y_pred, y).
    """
    def reset(self):
        self._ids = []
        self._predictions = []
        self._gold = []

    def update(self, output):
        ids, y_pred, y = output
        self._ids.extend(ids)
        self._predictions.append(y_pred)
        self._gold.append(y)

    def compute(self):
        if len(self._predictions) == 0:
            raise NotComputableError('MAP/MRR must have at least one example before it can be computed')

        predicted_scores = torch.cat(self._predictions).data.cpu().numpy()
        gold_scores = torch.cat(self._gold).data.cpu().numpy()

        mAP = map_metric(self._ids, gold_scores, predicted_scores)

        print(mAP)

        return {'map': mAP, 'mrr': 0.0}


class MAP(RetrievalMetrics):
    """
    Calculates the MAP.
    `update` must receive output of the form (ids, y_pred, y).
    """
    def compute(self):
        retrieval_metrics = super(MAP, self).compute()
        return retrieval_metrics['map']


class MRR(RetrievalMetrics):
    """
    Calculates the MRR.
    `update` must receive output of the form (ids, y_pred, y).
    """
    def compute(self):
        retrieval_metrics = super(MRR, self).compute()
        return retrieval_metrics['mrr']

