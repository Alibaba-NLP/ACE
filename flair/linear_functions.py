"""
Quadrilinear and Hexalinear function for sequence labeling
Author: Xinyu Wang
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.sparse as sparse
import math
import flair.nn
import pdb

class QuadriLinearScore(nn.Module):
    """
    Outer product version of quadrilinear function for sequence labeling.
    """

    def __init__(self, wemb_size, tagset_size, temb_size=20, rank=396, std=0.1545, window_size=1, normalization=True, **kwargs):
        """
        Args:
            wemb_size: word embedding hidden size
            tagset_size: tag set size
            temb_size: tag embedding size
            rank: rank of the weight tensor
            std: standard deviation of the tensor
        """
        super(QuadriLinearScore, self).__init__()
        self.wemb_size = wemb_size
        self.tagset_size = tagset_size
        self.temb_size = temb_size
        self.rank = rank
        self.std = std
        self.window_size = window_size
        self.normalization = normalization

        self.tag_emd = nn.Parameter(torch.Tensor(self.tagset_size, self.temb_size))
        self.T = nn.Parameter(torch.Tensor(self.wemb_size, self.rank))
        self.U = nn.Parameter(torch.Tensor(self.wemb_size, self.rank))
        self.V = nn.Parameter(torch.Tensor(self.temb_size, self.rank))
        self.W = nn.Parameter(torch.Tensor(self.temb_size, self.rank))
        self.rand_init()
        self.to(flair.device)

    def rand_init(self):
        '''random initialization
        '''

        # utils.init_trans(self.tag_emd)
        # utils.init_tensor(self.T, self.std)
        # utils.init_tensor(self.U, self.std)
        # utils.init_tensor(self.V, self.std)
        # utils.init_tensor(self.W, self.std)
        # std = 1.0
        nn.init.uniform_(self.tag_emd, a=math.sqrt(6/self.temb_size),b=math.sqrt(6/self.temb_size))
        nn.init.normal_(self.T, std=self.std)
        nn.init.normal_(self.U, std=self.std)
        nn.init.normal_(self.V, std=self.std)
        nn.init.normal_(self.W, std=self.std)

    def forward(self, word_emb):
        """
        Args:
            word_emb: [batch, sent_length, wemb_size]
        Returns: Tensor
            [batch, sent_length-window_size, tagset_size, tagset_size]
        """
        assert word_emb.size(2) == self.wemb_size, 'batch sizes of encoder and decoder are requires to be equal.'
        
        # print(f'{word_emb1.size()}, {last_word.size()}, {word_emb.size()}')
        # (n x m - w x d) * (d x k) -> (n x m - w x k)
        g0 = torch.matmul(word_emb[:,:-self.window_size], self.U)
        # (n x m - w x d) * (d x k) -> (n x m - w x k)
        g1 = torch.matmul(word_emb[:,self.window_size:], self.T)
        # (l x d) * (d x k) -> (l x k)
        g2 = torch.matmul(self.tag_emd, self.V)
        # (l' x d) * (d x k) -> (l x k)
        g3 = torch.matmul(self.tag_emd, self.W)
        # (n x m - w x k) -> (n x m - w x k)
        temp01 = g0 * g1 #torch.einsum('nak, nak->nak', [g1, g0])
        # (n x m - w x k) * (l x k) -> (n x m - w x l x k)
        temp012 = torch.einsum('nak,bk->nabk', [temp01, g2])
        # (n x m - w x l x k) * (l' x k) -> (n x m - w x l x l')
        score = torch.einsum('nabk,ck->nabc', [temp012, g3])
        if self.normalization:
            score = score/math.sqrt(self.rank)
        return score

class HexaLinearScore(nn.Module):
    """
    Outer product version of hexalinear function for sequence labeling.
    """

    def __init__(self, wemb_size, tagset_size, temb_size=20, rank=396, std=0.1545, normalization=True, **kwargs):
        """
        Args:
            wemb_size: word embedding hidden size
            tagset_size: tag set size
            temb_size: tag embedding size
            rank: rank of the weight tensor
            std: standard deviation of the tensor
        """
        super(HexaLinearScore, self).__init__()
        self.wemb_size = wemb_size
        self.tagset_size = tagset_size
        self.temb_size = temb_size
        self.rank = rank
        self.std = std
        self.normalization = normalization

        self.tag_emd = nn.Parameter(torch.Tensor(self.tagset_size, self.temb_size))
        self.W1 = nn.Parameter(torch.Tensor(self.wemb_size, self.rank))
        self.W2 = nn.Parameter(torch.Tensor(self.wemb_size, self.rank))
        self.W3 = nn.Parameter(torch.Tensor(self.wemb_size, self.rank))
        self.T1 = nn.Parameter(torch.Tensor(self.temb_size, self.rank))
        self.T2 = nn.Parameter(torch.Tensor(self.temb_size, self.rank))
        self.T3 = nn.Parameter(torch.Tensor(self.temb_size, self.rank))
        self.rand_init()
        self.to(flair.device)

    
    def rand_init(self):
        '''random initialization
        '''

        # utils.init_trans(self.tag_emd)
        # utils.init_tensor(self.T, self.std)
        # utils.init_tensor(self.U, self.std)
        # utils.init_tensor(self.V, self.std)
        # utils.init_tensor(self.W, self.std)
        # std = 1.0
        nn.init.uniform_(self.tag_emd, a=math.sqrt(6/self.temb_size),b=math.sqrt(6/self.temb_size))
        nn.init.normal_(self.T1, std=self.std)
        nn.init.normal_(self.T2, std=self.std)
        nn.init.normal_(self.T3, std=self.std)
        nn.init.normal_(self.W1, std=self.std)
        nn.init.normal_(self.W2, std=self.std)
        nn.init.normal_(self.W3, std=self.std)

    def forward(self, word_emb):
        """
        Args:
            word_emb: [batch, sent_length, wemb_size]
        Returns: Tensor
            [batch, sent_length-window_size, tagset_size, tagset_size]
        """
        assert word_emb.size(2) == self.wemb_size, 'batch sizes of encoder and decoder are requires to be equal.'
        
        # print(f'{word_emb1.size()}, {last_word.size()}, {word_emb.size()}')
        # (n x m-2 x d) * (d x k) -> (n x m-2 x k)
        # score (1 * 2 * 3)
        # 1: [0:-2]
        g1 = torch.matmul(word_emb[:,:-2], self.W1)
        # 2: [1:-1]
        g2 = torch.matmul(word_emb[:,1:-1], self.W2)
        # 3: [2:]
        g3 = torch.matmul(word_emb[:,2:], self.W3)


        # (l x d) * (d x k) -> (l x k)
        g4 = torch.matmul(self.tag_emd, self.T1)
        # (l' x d) * (d x k) -> (l' x k)
        g5 = torch.matmul(self.tag_emd, self.T2)
        # (l' x d) * (d x k) -> (l'' x k)
        g6 = torch.matmul(self.tag_emd, self.T3)
        # (n x m-2 x k) -> (n x m-2 x k)
        temp01 = g1 * g2 * g3
        # (l x k) * (l' x k) * (l'' x k) -> (l x l' x l'' x k)
        temp02 = torch.einsum('ak,bk,ck->abck', [g4, g5, g6])
        # (n x m-2 x k) * (l x l' x l'' x k) -> (n x m-2 x l x l' x l'')
        score = torch.einsum('nmk,abck->nmabc', [temp01,temp02])

        if self.normalization:
            score = score/math.sqrt(self.rank)
        return score