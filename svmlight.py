#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : svmlight
# @Date : 2018-08-17-13-34
# @Poject: SemevalSLS
# @AUTHOR : Yassine EL ADLOUNI


from utils import save_svmlight

# save_svmlight('./features/train.lex')
# save_svmlight('./features/val.lex')
# save_svmlight('./features/test.lex')
#
# save_svmlight('./features/train.cbow')
# save_svmlight('./features/val.cbow')
# save_svmlight('./features/test.cbow')

save_svmlight('./features/train.cos')
save_svmlight('./features/val.cos')
save_svmlight('./features/test.cos')
#
# qids = []
# qaids = []
# with open('./features/test.cbow.svmlight') as f:
#     for line in f:
#         fields = line.split()
#         qids.append(fields[1][4:])
#         qaids.append(fields[-1][1:])
#
# confs = []
# with open('/home/yassine/quickrank/scores.txt') as f:
#     for line in f:
#         confs.append(float(line))
#
# with open('./features/test.cbow.semeval', 'w') as f:
#     for qid, qaid, conf in zip(qids, qaids, confs):
#         f.write('{}\t{}\t0\t{}\t{}\n'.format(qid, qaid,conf, 'true' if conf > 0.5 else 'false'))