import torch.nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.autograd as autograd
import flair.nn
import torch

import numpy as np
import pdb
import copy
import time

import sys



class EmbedController(flair.nn.Model):
	def __init__(
		self,
		num_actions,
		model_structure = None,
		state_size = 20,
		hidden_size = 100,
	):
		"""
		Initializes a SequenceTagger
		:param hidden_size: number of hidden states in RNN
		:param embeddings: word embeddings used in tagger
		:param tag_dictionary: dictionary of tags you want to predict
		:param tag_type: string identifier for tag type
		:param use_crf: if True use CRF decoder, else project directly to tag space
		:param use_rnn: if True use RNN layer, otherwise use word embeddings directly
		:param rnn_layers: number of RNN layers
		:param dropout: dropout probability
		:param word_dropout: word dropout probability
		:param locked_dropout: locked dropout probability
		:param distill_crf: CRF information distillation
		:param crf_attention: use CRF distillation weights
		:param biaf_attention: use bilinear attention for word-KD distillation
		"""
		super(EmbedController, self).__init__()
		self.previous_selection = None
		self.best_action = None
		self.num_actions = num_actions
		self.model_structure = model_structure
		self.state_size = state_size
		if self.model_structure is None:
			self.selector = Parameter(
						torch.zeros(num_actions),
						requires_grad=True,
					)
		else:
			# self.rnn = getattr(torch.nn, 'LSTM')(
			# 		state_size,
			# 		hidden_size,
			# 		num_layers=1,
			# 		dropout=0.0,
			# 		bidirectional=True,
			# 	)
			self.selector = torch.nn.Linear(state_size, num_actions)
			torch.nn.init.zeros_(self.selector.weight)
			torch.nn.init.zeros_(self.selector.bias)
		self.to(flair.device)
		
		

	def _init_model_with_state_dict(state):
		

		model = EmbedController(
			num_actions = state['num_actions'],
			model_structure = state['model_structure'],
			state_size = state['state_size'],
		)
		model.load_state_dict(state["state_dict"])
		return model
	def _get_state_dict(self):
		model_state = {
			"state_dict": self.state_dict(),
			"num_actions": self.num_actions,
			"model_structure": self.model_structure,
			"state_size": self.state_size,
		}
		return model_state
	def sample(self, states=None, mask = None):
		value = self.get_value(states,mask)
		one_prob = torch.sigmoid(value)
		m = torch.distributions.Bernoulli(one_prob)
		selection = m.sample()
		#avoid all values are 0, or avoid the selection is the same as previous iteration in training
		if self.model_structure is None:
			while selection.sum()==0 or (self.previous_selection is not None and (self.previous_selection == selection).all()):
				selection = m.sample()
		else:
			for idx in range(len(selection)):
				while selection[idx].sum() == 0:
					m_temp = torch.distributions.Bernoulli(one_prob[idx])
					selection[idx] = m_temp.sample()

		log_prob = m.log_prob(selection)
		self.previous_selection = selection.clone()
		return selection, log_prob
	def forward(self, states=None, mask = None):
		value = self.get_value(states,mask)
		
		return torch.sigmoid(value)

	def get_value(self, states=None, mask = None):
		if self.model_structure is None:
			value=self.selector
		else:
			states = (states*mask.unsqueeze(-1)).sum(-2)/mask.sum(-1,keepdim=True)
			value=self.selector(states)
		return value