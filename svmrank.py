"""
Implementation of pairwise ranking using scikit-learn LinearSVC

Reference: "Large Margin Rank Boundaries for Ordinal Regression", R. Herbrich,
    T. Graepel, K. Obermayer.

Authors: Fabian Pedregosa <fabian@fseoane.net>
         Alexandre Gramfort <alexandre.gramfort@inria.fr>
"""
import os
import itertools
import json

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score

from similarity.metric_lcs import MetricLCS
from similarity.jarowinkler import JaroWinkler
from similarity.levenshtein import Levenshtein
from similarity.damerau import Damerau
from similarity.qgram import QGram
from similarity.ngram import NGram
from similarity.cosine import Cosine
from similarity.sorensen_dice import SorensenDice
from similarity.jaccard import Jaccard

from py_stringmatching.similarity_measure import hybrid_similarity_measure

from utils import get_args


def extract_features(record):
    sidf = hybrid_similarity_measure.HybridSimilarityMeasure()
    question = record['question']
    qaquestion = record['qaquestion']

    return sidf.(question, qaquestion)

def load_dataset(path):

    with open(path, 'r') as f:
        for line in f:
            record = json.loads(line)
            yield {
                'qid': record['qid'],
                'qaid': record['qaid'],
                'question': record['question'],
                'qaquestion': record['qaquestion'],
                'qaanswer': record['qaanswer'],
                'qpos': record['qpos'],
                'qaqpos': record['qaqpos'],
                'qaapos': record['qaapos'],
                'qarel': record['qarel']
            }

def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking

    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.

    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.

    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    return np.asarray(X_new), np.asarray(y_new).ravel()

class RankSVM(svm.LinearSVC):
    """Performs pairwise ranking with an underlying LinearSVC model

    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.

    See object :ref:`svm.LinearSVC` for a full description of parameters.
    """

    def fit(self, X, y):
        """
        Fit a pairwise ranking model.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)

        Returns
        -------
        self
        """
        X_trans, y_trans = transform_pairwise(X, y)

        super(RankSVM, self).fit(X_trans, y_trans)
        return self

    def predict(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        Returns
        -------
        ord : array, shape (n_samples,)
            Returns a list of integers representing the relative order of
            the rows in X.
        """
        if hasattr(self, 'coef_'):
            return np.dot(X, self.coef_.ravel())
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X, y):
        """
        Because we transformed into a pairwise problem, chance level is at 0.5
        """
        X_trans, y_trans = transform_pairwise(X, y)
        return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)



if __name__ == '__main__':
    args = get_args()

    train_df = pd.read_json('.data/train_2016.json',
                            lines=True)

    train_df = train_df.assign(cosine=train_df.apply(extract_features, axis=1))


    print(train_df.shape)

    # train_df[['qlen', 'qaqlen', 'qaalen', 'qchars', 'qaqchars', 'qaachars',
    #           'levenshtein', 'metric_lcs', 'jarowinkler', 'damerau'
    #     , 'sorensen_dice', 'jarowinkler2', 'jaccard', 'ngram', 'qgram', 'cosine']] = \
    # pd.DataFrame(train_df.features.values.tolist())

    #print('Training dataset loaded, {} samples.'.format(len(train_df)))

    test_df = pd.read_json('.data/test_2017.json', lines=True)

    test_df = test_df.assign(cosine=test_df.apply(extract_features, axis=1))

    # test_df[['qlen', 'qaqlen', 'qaalen', 'qchars', 'qaqchars', 'qaachars',
    #           'levenshtein', 'metric_lcs', 'jarowinkler', 'damerau'
    #     , 'sorensen_dice', 'jarowinkler2', 'jaccard', 'ngram', 'qgram', 'cosine']] = \
    #     pd.DataFrame(test_df.features.values.tolist())

    # print('Test dataset loaded, {} samples.'.format(len(test_df)))
    #
    X_train, X_dev, y_train, y_dev = train_test_split(train_df[['cosine']], train_df[['qarel', 'qid']], stratify=train_df[['qarel']], test_size=0.25)
    print(X_train.shape, y_train.shape)
    print(X_dev.shape, y_dev.shape)

    rank_svm = RankSVM().fit(X_train.values, y_train.values)

    aps = []
    with open('./output/lightgbm.semeval', 'w') as f:
        with open('./output/map.txt', 'w') as g:
            for key, group in test_df.groupby('qid'):
                y_test = group['qarel'].tolist()
                qids = group['qid'].tolist()
                qaids = group['qaid'].tolist()
                y_pred = rank_svm.predict(group[['cosine']].values)

                ap = average_precision_score(y_test, y_pred)
                for qid, qaid, score in zip(qids, qaids, y_pred):
                    f.write('{}\t{}\t{}\t{}\t{}\n'.format(qid, qaid, 0, score, 'true' if score > 0.5 else 'false'))
                g.write('{}\t{}\n'.format(key, ap))
                aps.append(ap)

    map = np.nanmean(np.nan_to_num(aps))
    print(map)