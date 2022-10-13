"""Borrowed from https://github.com/singhgautam/slate/blob/master/utils.py"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_one_hot(logits, dim=-1):
    index = logits.argmax(dim, keepdim=True)
    one_hot_label = torch.zeros_like(logits).scatter_(dim, index, 1.)
    return one_hot_label


def gumbel_max(logits, dim=-1):

    eps = torch.finfo(logits.dtype).tiny

    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = logits + gumbels

    return gumbels.argmax(dim)


def gumbel_softmax(logits, tau=1., hard=False, dim=-1):

    eps = torch.finfo(logits.dtype).tiny

    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = (logits + gumbels) / tau

    y_soft = F.softmax(gumbels, dim)

    if hard:
        index = y_soft.argmax(dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


def log_prob_gaussian(value, mean, std):

    var = std**2
    if isinstance(var, float):
        return -0.5 * ((
            (value - mean)**2) / var + math.log(var) + math.log(2 * math.pi))
    else:
        return -0.5 * ((
            (value - mean)**2) / var + var.log() + math.log(2 * math.pi))


def cosine_anneal(step, start_value, final_value, start_step, final_step):

    assert start_value >= final_value
    assert start_step <= final_step

    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = 0.5 * (start_value - final_value)
        b = 0.5 * (start_value + final_value)
        progress = (step - start_step) / (final_step - start_step)
        value = a * math.cos(math.pi * progress) + b

    return value


def conv2d(in_channels,
           out_channels,
           kernel_size,
           stride=1,
           padding=0,
           dilation=1,
           groups=1,
           bias=True,
           padding_mode='zeros',
           weight_init='xavier'):

    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  dilation, groups, bias, padding_mode)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight)

    if bias:
        nn.init.zeros_(m.bias)

    return m


class Conv2dBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super().__init__()

        self.m = conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            weight_init='kaiming')
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):

        x = self.m(x)
        return F.relu(F.group_norm(x, 1, self.weight, self.bias))


class Conv2dBlockBN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super().__init__()

        self.m = conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            weight_init='kaiming')
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.m(x)
        return F.relu(self.bn(x))


def linear(in_features,
           out_features,
           bias=True,
           weight_init='xavier',
           gain=1.):

    m = nn.Linear(in_features, out_features, bias)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)

    if bias:
        nn.init.zeros_(m.bias)

    return m


def gru_cell(input_size, hidden_size, bias=True):

    m = nn.GRUCell(input_size, hidden_size, bias)

    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)

    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)

    return m
