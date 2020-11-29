# pytorch verison of mean field variational inference for sequence labeling
# Author: Xinyu Wang
# Email: wangxy1@shanghaitech.edu.cn

import copy
import numpy as np
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from .linear_functions import *
import flair.nn

class MFVI(nn.Module):
	def __init__(self,
		hidden_dim: int,
		tagset_size: int,
		iterations: int = 3,
		use_second_order: bool = True,
		use_third_order: bool = False,
		use_quadrilinear: bool = False,
		use_hexalinear: bool = False,
		window_size: int = 1,
		quad_rank: int = 150,
		quad_std: float = 0.25,
		hexa_rank: int = 150,
		tag_dim: int = 150,
		hexa_std: float = 0.25,
		normalize_weight: bool = True,
		add_start_end: bool = False,
		):
		super(MFVI, self).__init__()
		self.hidden_dim: int = hidden_dim
		self.tagset_size: int = tagset_size
		self.iterations: int = iterations
		self.use_second_order: bool = use_second_order
		self.use_third_order: bool = use_third_order
		self.use_quadrilinear: bool = use_quadrilinear
		self.use_hexalinear: bool = use_hexalinear
		self.window_size: int = window_size
		self.quad_rank: int = quad_rank
		self.quad_std: float = quad_std
		self.hexa_rank: int = hexa_rank
		self.hexa_std: float = hexa_std
		self.normalize_weight: bool = normalize_weight
		self.tag_dim: bool = tag_dim
		self.add_start_end: bool = add_start_end
		if self.use_second_order:
			if self.use_quadrilinear:
				for i in range(self.window_size):
					setattr(self,'quadrilinear'+str(i),QuadriLinearScore(self.hidden_dim,
												self.tagset_size,
												self.tag_dim,
												self.quad_rank,
												self.quad_std,
												window_size=i+1,
												normalization=self.normalize_weight,
												))
				# self.quadrilinears = [QuadriLinearScore(self.hidden_dim,
				# 								self.tagset_size,
				# 								self.tag_dim,
				# 								self.quad_rank,
				# 								self.quad_std,
				# 								window_size=i+1,
				# 								normalization=self.normalize_weight,
				# 								) for i in range(self.window_size)]
			else:
				self.transitions = nn.Parameter(torch.randn(self.window_size,self.tagset_size, self.tagset_size))
				if self.add_start_end:
					#\psi(start,y_{1}), \psi(start,y_{2})
					self.start_transitions = nn.Parameter(torch.randn(self.window_size,self.tagset_size))
					#\psi(y_{n-1},end), \psi(y_{n},end)
					self.end_transitions = nn.Parameter(torch.randn(self.window_size,self.tagset_size))
		if self.use_third_order:
			if self.use_hexalinear:
				self.hexalinear = HexaLinearScore(self.hidden_dim,
												self.tagset_size,
												self.tag_dim,
												self.hexa_rank,
												self.hexa_std,
												normalization=self.normalize_weight
												)
			else:
				self.tri_transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size, self.tagset_size))
				if self.add_start_end:
					self.start_transitions = nn.Parameter(torch.randn(self.tagset_size,self.tagset_size))
					self.end_transitions = nn.Parameter(torch.randn(self.tagset_size,self.tagset_size))
		self.to(flair.device)
	def forward(self,token_feats,unary_score,mask, lengths):
		# pdb.set_trace()
		sent_len=token_feats.shape[1]
		unary_score = unary_score * mask.unsqueeze(-1)
		token_feats=token_feats*mask.unsqueeze(-1)
		if self.use_second_order:
			if sent_len<=1:
				# pdb.set_trace()
				return unary_score
			if self.use_quadrilinear:
				# [batch_size, tokens, window_size, tagset_size, tagset_size]
				# Representing the i-th token and i+1-th/... token score.
				# The last part is padded with 0.
				binary_score=[]
				for i in range(min(self.window_size,sent_len-1)):
					linear_func=getattr(self,'quadrilinear'+str(i))
					binary_score.append(linear_func(token_feats))
			else:
				binary_score=[]
				for i in range(min(self.window_size,sent_len-1)):
					binary_score.append(self.transitions[i])
				# binary_score=self.transitions
		else:
			binary_score=None
		# if sent_len==2:
		# 	pdb.set_trace()
		if self.use_third_order:
			if sent_len<=2 and not self.use_second_order:
				return unary_score
			elif sent_len > 2:
				if self.use_hexalinear:
					# [batch_size, tokens, tagset_size, tagset_size, tagset_size]
					# Representing the i-th token and i+1-th/... token score.
					ternary_score=self.hexalinear(token_feats)
				else:
					ternary_score=self.tri_transitions
			else:
				ternary_score=None
		else:
			ternary_score=None
		scores=self._mean_field_variational_infernece(unary_score,binary_score,ternary_score,mask, lengths=lengths)
		return scores



	def _mean_field_variational_infernece(self, unary: torch.Tensor, binary: torch.Tensor = None, ternary: torch.Tensor = None, mask: torch.Tensor = None, lengths = None):
		# unary: [batch, sent_length, labels]
		# binary: (list)[batch, sent_length-window_size, labels, labels']/[labels, labels']. i-th place encodes the score between (i,i+1)
		# ternary: [batch, sent_length-2, labels, labels', labels'']/[labels, labels', labels'']. i-th place encodes the score between (i,i+1,i+2)
		# pdb.set_trace()
		# unary = unary*mask.unsqueeze(-1)
		unary_potential = unary.clone()
		sent_len=unary_potential.shape[1]
		# [batch, sent_length, labels]
		q_value = unary_potential.clone()

		# if binary is not None and self.use_quadrilinear:
		# 	for i in range(len(binary)):
		# 		# pdb.set_trace()
		# 		binary[i] = binary[i] * mask[:,:-i-1].unsqueeze(-1).unsqueeze(-1)
		# if ternary is not None and self.use_hexalinear:
		# 	ternary = ternary * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		batch_range=torch.arange(q_value.shape[0])
		for iteration in range(self.iterations):
			q_value=F.softmax(q_value,-1)
			if self.use_second_order:
				# [batch, sent_length, labels'] * [batch, sent_length, window_size, labels, labels'] -> [batch, sent_length, labels]
				# [batch, sent_length, labels]
				left_sum = torch.zeros_like(q_value)
				right_sum = torch.zeros_like(q_value)

				'''
				debug:
				vleft_val = torch.zeros_like(q_value)
				vright_val = torch.zeros_like(q_value)
				vj=2
				vn=0
				va=15
				temp1=q_value[vn,va+vj]
				temp2=binary[vj-1][vn,va]
				vright=(temp1.unsqueeze(-2)*temp2).sum(-1)
				vright_val[:,:-j]=torch.einsum('nsb,nsab->nsa',[q_value[:,j:],binary[j-1]])
				(vright-vright_val[vn,va]).sum()
				temp1=q_value[vn,va-vj]
				temp2=binary[vj-1][vn,va-vj]
				vleft=(temp1.unsqueeze(-1)*temp2).sum(-2)
				vleft_val[:,j:]=torch.einsum('nsa,nsab->nsb',[q_value[:,:-j],binary[j-1]])
				(vleft-vleft_val[vn,va]).sum()
				'''
				# each bianry[i] encodes the score of (X_i,X_{i+window_size})
				for j in range(1,len(binary)+1):
					if self.use_quadrilinear:						
						# [batch, sent_length-j, labels'] * [batch, sent_length-j, labels, labels'] -> [batch, sent_length-j, labels]
						# M[i] = \sum score_{i-j,i}Q_{i-j}(X_{i-j})
						# [j:] <- [0:n-j]
						left_sum[:,j:]+=torch.einsum('nsa,nsab->nsb',[q_value[:,:-j],binary[j-1]])
						# M[i] = \sum score_{i,i+j}Q_{i+j}(X_i)
						# [0:n-j] <- [j:]
						right_sum[:,:-j]+=torch.einsum('nsb,nsab->nsa',[q_value[:,j:],binary[j-1]])
						# if j==2:
						# [batch, sent_length, labels] * [labels, labels'] -> [batch, sent_length, labels]
					else:
						# [batch, sent_length-i, labels'] * [labels, labels'] -> [batch, sent_length-i, labels]
						# M[i] = \sum score_{i-j,i}Q_{i-j}(X_{i-j})
						# [j:] <- [0:n-j]
						left_sum[:,j:]+=torch.einsum('nsa,ab->nsb',[q_value[:,:-j],binary[j-1]])
						# M[i] = \sum score_{i,i+j}Q_{i+j}(X_{i+j})
						# [0:n-j] <- [j:]
						right_sum[:,:-j]+=torch.einsum('nsb,ab->nsa',[q_value[:,j:],binary[j-1]])
						if self.add_start_end:
							right_sum[batch_range,lengths-j]+=self.end_transitions[j-1]
				# pdb.set_trace()
				second_order_msg=left_sum+right_sum
				if self.add_start_end:
					second_order_msg[:,0:self.window_size]+=self.start_transitions
				
			else:
				second_order_msg=0

			if self.use_third_order and ternary is not None:
				left_sum = torch.zeros_like(q_value)
				middle_sum = torch.zeros_like(q_value)
				right_sum = torch.zeros_like(q_value)
				# [batch, sent_length-2, labels'] * [batch, sent_length-2, labels''] * [batch, sent_length-2, labels, labels',labels''] -> [batch, sent_length, labels]
				# pdb.set_trace()
				# pdb.set_trace()
				if self.use_hexalinear:
					'''
					debug:
					vn=0
					va=15
					temp1=q_value[vn,va-2].unsqueeze(-1).unsqueeze(-1)
					temp2=q_value[vn,va-1].unsqueeze(-1).unsqueeze(0)
					temp3=q_value[vn,va+1].unsqueeze(-1).unsqueeze(0)
					temp4=q_value[vn,va+2].unsqueeze(0).unsqueeze(0)
					vleft=(temp1*temp2*ternary[vn,va-2]).sum(0).sum(0)
					vmiddle=(temp3*temp2*ternary[vn,va-1]).sum(0).sum(-1)
					vright=(temp3*temp4*ternary[vn,va]).sum(-1).sum(-1)
					(vleft-left_sum[vn,va]).sum()
					(vmiddle-middle_sum[vn,va]).sum()
					(vright-right_sum[vn,va]).sum()
					'''
					# for the score of i-th token, sum over the i-2, i-1-th place.
					# M[i] = \sum score_{i-2,i-1,i}Q_{i-2}(X_{i-2})Q_{i-1}(X_{i-1})
					left_sum[:,2:]+=torch.einsum('nsa,nsb,nsabc->nsc',[q_value[:,:-2],q_value[:,1:-1],ternary])
					# for the score of i-th token, sum over the i-1, i+1-th place.
					middle_sum[:,1:-1]+=torch.einsum('nsa,nsc,nsabc->nsb',[q_value[:,:-2],q_value[:,2:],ternary])
					# for the score of i-th token, sum over the i+1, i+2-th place.
					right_sum[:,:-2]+=torch.einsum('nsb,nsc,nsabc->nsa',[q_value[:,1:-1],q_value[:,2:],ternary])
				# [batch, sent_length, labels'] * [batch, sent_length, labels', labels''] * [labels, labels',labels''] -> [batch, sent_length, labels]
				else:
					# for the score of i-th token, sum over the i-2, i-1-th place.
					left_sum[:,2:]+=torch.einsum('nsa,nsb,abc->nsc',[q_value[:,:-2],q_value[:,1:-1],ternary])
					# for the score of i-th token, sum over the i-1, i+1-th place.
					middle_sum[:,1:-1]+=torch.einsum('nsa,nsc,abc->nsb',[q_value[:,:-2],q_value[:,2:],ternary])
					# for the score of i-th token, sum over the i+1, i+2-th place.
					right_sum[:,:-2]+=torch.einsum('nsb,nsc,abc->nsa',[q_value[:,1:-1],q_value[:,2:],ternary])

				if self.add_start_end:
					left_sum[:,1]+=torch.einsum('na,ab->nb',[q_value[:,0],self.start_transitions])
					middle_sum[:,0]+=torch.einsum('nb,ab->na',[q_value[:,1],self.start_transitions])

					middle_sum[batch_range,lengths-1]+=torch.einsum('na,ab->nb',[q_value[batch_range,lengths-2],self.end_transitions])
					right_sum[batch_range,lengths-2]+=torch.einsum('nb,ab->na',[q_value[batch_range,lengths-1],self.end_transitions])
				# pdb.set_trace()
				third_order_msg=left_sum+middle_sum+right_sum
			else:
				third_order_msg=0

			q_value = unary_potential + second_order_msg + third_order_msg
			q_value = q_value*mask.unsqueeze(-1)
			
		return q_value