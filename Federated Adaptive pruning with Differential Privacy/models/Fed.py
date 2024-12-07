#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import random

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FedProx(w, mu):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] + mu * (w_avg[k] - w[0][k])
    return w_avg


def FedMA(w_locals):
    # 初始化全局模型的权重为零向量
    w_glob = {k: torch.zeros_like(v) for k, v in w_locals[0].items()}

    # 对每个本地模型的权重进行累加
    for w in w_locals:
        for k, v in w.items():
            w_glob[k] += v

    # 对累加结果求平均
    for k in w_glob.keys():
        w_glob[k] /= len(w_locals)

    return w_glob
