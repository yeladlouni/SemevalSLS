#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : data
# @Date : 2018-08-17-13-34
# @Poject: SemevalSLS
# @AUTHOR : Yassine EL ADLOUNI

import logging
import json

import numpy as np

import torch

from torchtext import data
from torchtext import vocab


def get_class_probs(sim, *args):
    """
    Convert a single label into class probabilities.
    """
    class_probs = np.zeros(2)
    class_probs[0], class_probs[1] = 1 - sim, sim

    return class_probs

class Semeval(data.Dataset):
    """

    """
    dirname =''
    name = 'semeval'
    num_classes = 2

    @staticmethod
    def sort_key(ex):
        return len(ex.question)

    def __init__(self, path, id_field, text_field, rel_field, conf_field, raw_field, tag_field, **kwargs):
        """

        :param path:
        :param id_field:
        :param text_field:
        :param rel_field:
        :param conf_field:
        :param raw_field:
        :param kwargs:
        """

        logger = logging.getLogger('SemEval')

        fields = [
            ('qid', id_field),
            ('qaid', id_field),
            ('qarel', rel_field),
            ('qaconf', conf_field),
            ('question', text_field),
            ('qaquestion', text_field),
            ('qaanswer', text_field),
        ]

        examples = []

        with open(path) as f:
            for line in f:
                content = json.loads(line)
                values = [content['qid'],
                          content['qaid'],
                          content['qarel'],
                          content['qaconf'],
                          content['question'],
                          content['qaquestion'],
                          content['qaanswer']]

                examples.append(data.Example.fromlist(values, fields))

                #logger.warning('Processing Example qid: {} qaid: {}'.format(
                #    content['qid'],
                #    content['qaid']
                #))

        super(Semeval, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, id_field, text_field, rel_field, conf_field, raw_field,
               tag_field, root='.data', train='train_2016.json',
               validation='dev_2016.json', test='test_2017.json', **kwargs):
        """
        Create dataset objects for splits of the SemEval dataset.
        :param path: The directory containing the datasets.
        :param id_field: The field that will be used for id data.
        :param text_field: The field that will be used for text data.
        :param label_field: The field that will be used for label data.
        :param root: The root directory containing the datasets files.
        :param train: The filename of the train data.
        :param validation: The filename of the validation data.
        :param test: The filename of the test data.
        :param kwargs:
        :return:
        """
        path = root

        #text_field.postprocessing = data.Pipeline(shrink_chunk)

        return super(Semeval, cls).splits(
            path=path, root=root, id_field=id_field, text_field=text_field,
            rel_field=rel_field, conf_field=conf_field, raw_field=raw_field,
            tag_field=tag_field, train=train, validation=validation, test=test
        )

    @classmethod
    def iters(cls, config, **kwargs):
        """
        Create the iterator objects for splits of the SemEval dataset.
        :param batch_size: Batch_size
        :param device: Device to create batches, -1 for CPU and None for GPU.
        :param root: The root directory containing datasets files.
        :param vectors: Load pretrained vectors
        :param kwargs:
        :return:
        """

        vectors = vocab.Vectors(name=config.vectors, cache=config.cache)

        ID = data.RawField()
        TEXT = data.Field(batch_first=True, tokenize=lambda x:x, fix_length=20)
        TAG = data.Field(batch_first=True, tokenize=lambda x:x, fix_length=20)
        RAW = data.RawField()
        REL = data.Field(sequential=False, use_vocab=False, batch_first=True, tensor_type=torch.FloatTensor, postprocessing=data.Pipeline(get_class_probs))
        CONF = data.RawField()

        #TAG.preprocessing = shrink_chunk

        train, val, test = cls.splits(ID, TEXT, REL, CONF, RAW, TAG, root=config.datasets_dir, **kwargs)

        TEXT.build_vocab(train)
        config.n_embed = len(TEXT.vocab)
        config.d_embed = vectors.dim
        TEXT.vocab.load_vectors(vectors)

        config.weights = TEXT.vocab.vectors

        config.n_classes = 2

        return data.BucketIterator.splits(
            (train, val, test), batch_size=config.batch_size, shuffle=config.shuffle,
            device=config.device, repeat=False
        )
