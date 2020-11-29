# -*- coding: utf-8 -*-

from flair.parser.modules.dropout import SharedDropout

import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, n_in, n_hidden, dropout=0, identity = False):
        super(MLP, self).__init__()

        self.linear = nn.Linear(n_in, n_hidden)
        self.identity = identity
        if not self.identity:
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = SharedDropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        if not self.identity:
            x = self.activation(x)
        x = self.dropout(x)

        return x
