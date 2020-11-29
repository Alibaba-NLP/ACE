import warnings
import logging
from pathlib import Path

import torch.nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.autograd as autograd
import flair.nn
import torch

from flair.data import Dictionary, Sentence, Token, Label
from flair.datasets import DataLoader
from flair.embeddings import TokenEmbeddings
from flair.file_utils import cached_path

from typing import List, Tuple, Union

from flair.training_utils import Result, store_embeddings
from .biaffine_attention import BiaffineAttention

from tqdm import tqdm
from tabulate import tabulate
import numpy as np
import pdb
import copy
import time

import sys
# sys.path.insert(0,'/home/wangxy/workspace/flair/parser')
# sys.path.append('./flair/parser/modules')

from flair.parser.modules import CHAR_LSTM, MLP, BertEmbedding, Biaffine, BiLSTM, TrilinearScorer
from flair.parser.modules.dropout import IndependentDropout, SharedDropout
from flair.parser.utils.alg import eisner, crf
from flair.parser.utils.metric import Metric
from flair.parser.utils.fn import ispunct, istree, numericalize_arcs
# from flair.parser.utils.fn import ispunct
import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
								pad_sequence)

from .mst_decoder import MST_inference
def process_potential(log_potential):
	# (batch, sent_len+1, sent_len+1) or (batch, sent_len+1, sent_len+1, labels)
	
	# (batch, sent_len)
	root_score = log_potential[:,1:,0]
	# convert (dependency, head) to (head, dependency)
	# (batch, sent_len, sent_len)
	log_potential = log_potential.transpose(1,2)[:,1:,1:]
	batch, sent_len = log_potential.shape[:2]
	# Remove the <ROOT> and put the root probability in the diagonal part 
	log_potential[:,torch.arange(sent_len),torch.arange(sent_len)] = root_score
	return log_potential


def get_struct_predictions(dist):
	# (batch, sent_len, sent_len) | head, dep
	argmax_val = dist.argmax
	batch, sent_len, _ = argmax_val.shape
	res_val = torch.zeros([batch,sent_len+1,sent_len+1]).type_as(argmax_val)
	res_val[:,1:,1:] = argmax_val
	res_val = res_val.transpose(1,2)
	# set diagonal part to heads
	res_val[:,:,0] = res_val[:,torch.arange(sent_len+1),torch.arange(sent_len+1)]
	res_val[:,torch.arange(sent_len+1),torch.arange(sent_len+1)] = 0
	
	return res_val.argmax(-1)

def convert_score_back(marginals):
	# (batch, sent_len, sent_len) | head, dep
	batch = marginals.shape[0]
	sent_len = marginals.shape[1]
	res_val = torch.zeros([batch,sent_len+1,sent_len+1]+list(marginals.shape[3:])).type_as(marginals)
	res_val[:,1:,1:] = marginals
	res_val = res_val.transpose(1,2)
	# set diagonal part to heads
	res_val[:,:,0] = res_val[:,torch.arange(sent_len+1),torch.arange(sent_len+1)]
	res_val[:,torch.arange(sent_len+1),torch.arange(sent_len+1)] = 0
	
	return res_val




def is_punctuation(word, pos, punct_set=None):
	if punct_set is None:
		return is_uni_punctuation(word)
	else:
		return pos in punct_set

import uuid
uid = uuid.uuid4().hex[:6]
  


log = logging.getLogger("flair")

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"


def to_scalar(var):
	return var.view(-1).detach().tolist()[0]


def argmax(vec):
	_, idx = torch.max(vec, 1)
	return to_scalar(idx)


def log_sum_exp(vec):
	max_score = vec[0, argmax(vec)]
	max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
	return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax_batch(vecs):
	_, idx = torch.max(vecs, 1)
	return idx


def log_sum_exp_batch(vecs):
	maxi = torch.max(vecs, 1)[0]
	maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
	recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
	return maxi + recti_

def log_sum_exp_vb(vec, m_size):
	"""
	calculate log of exp sum

	args:
		vec (batch_size, vanishing_dim, hidden_dim) : input tensor
		m_size : hidden_dim
	return:
		batch_size, hidden_dim
	"""
	_, idx = torch.max(vec, 1)  # B * 1 * M
	max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M

	return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1,
																												m_size)  # B * M

def pad_tensors(tensor_list):
	ml = max([x.shape[0] for x in tensor_list])
	shape = [len(tensor_list), ml] + list(tensor_list[0].shape[1:])
	template = torch.zeros(*shape, dtype=torch.long, device=flair.device)
	lens_ = [x.shape[0] for x in tensor_list]
	for i, tensor in enumerate(tensor_list):
		template[i, : lens_[i]] = tensor

	return template, lens_


# Part of Codes are from https://github.com/yzhangcs/biaffine-parser
class SemanticDependencyParser(flair.nn.Model):
	def __init__(
		self,
		hidden_size: int,
		embeddings: TokenEmbeddings,
		tag_dictionary: Dictionary,
		tag_type: str,
		use_crf: bool = False,
		use_rnn: bool = False,
		train_initial_hidden_state: bool = False,
		punct: bool = False, # ignore all punct in default
		tree: bool = False, # keep the dpendency with tree structure
		n_mlp_arc = 500,
		n_mlp_rel = 100,
		mlp_dropout = .33,
		use_second_order = False,
		token_loss = False,
		n_mlp_sec = 150,
		init_std = 0.25,
		factorize = True,
		use_sib = True,
		use_gp = True,
		use_cop = False,
		iterations = 3,
		binary = True,
		is_mst = False,
		rnn_layers: int = 3,
		lstm_dropout: float = 0.33,
		dropout: float = 0.0,
		word_dropout: float = 0.33,
		locked_dropout: float = 0.5,
		pickle_module: str = "pickle",
		interpolation: float = 0.5,
		factorize_interpolation: float = 0.025,
		config = None,
		use_decoder_timer = True,
		debug = False,
		target_languages = 1,
		word_map = None,
		char_map = None,
		relearn_embeddings = False,
		distill_arc: bool = False,
		distill_rel: bool = False,
		distill_crf: bool = False,
		distill_posterior: bool = False,
		distill_prob: bool = False,
		distill_factorize: bool = False,
		crf_attention: bool = False,
		temperature: float = 1,
		diagonal: bool = False,
		is_srl: bool = False,
		embedding_selector = False,
		use_rl: bool = False,
		use_gumbel: bool = False,
		identity: bool = False,
		embedding_attention: bool = False,
		testing: bool = False,
		is_sdp: bool = False,
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

		super(SemanticDependencyParser, self).__init__()
		self.debug = False
		self.biaf_attention = False
		self.token_level_attention = False
		self.use_language_attention = False
		self.use_language_vector = False
		self.use_crf = use_crf
		self.use_decoder_timer = False
		self.sentence_level_loss = False
		self.train_initial_hidden_state = train_initial_hidden_state
		#add interpolation for target loss and distillation loss
		self.token_loss = token_loss

		self.interpolation = interpolation
		self.debug = debug
		self.use_rnn = use_rnn
		self.hidden_size = hidden_size

		self.rnn_layers: int = rnn_layers
		self.embeddings = embeddings
		self.config = config
		self.punct = punct 
		self.punct_list = ['``', "''", ':', ',', '.', 'PU', 'PUNCT']
		self.tree = tree
		self.is_mst = is_mst
		self.is_srl = is_srl
		self.use_rl = use_rl
		self.use_gumbel = use_gumbel
		self.embedding_attention = embedding_attention
		# set the dictionaries
		self.tag_dictionary: Dictionary = tag_dictionary
		self.tag_type: str = tag_type
		self.tagset_size: int = len(tag_dictionary)

		self.word_map = word_map
		self.char_map = char_map
		self.is_sdp = is_sdp
		# distillation part
		self.distill_arc = distill_arc
		self.distill_rel = distill_rel
		self.distill_crf = distill_crf
		self.distill_posterior = distill_posterior
		self.distill_prob = distill_prob
		self.distill_factorize = distill_factorize
		self.factorize_interpolation = factorize_interpolation
		self.temperature = temperature
		self.crf_attention = crf_attention
		self.diagonal = diagonal
		self.embedding_selector = embedding_selector

		# initialize the network architecture
		self.nlayers: int = rnn_layers
		self.hidden_word = None
		self.identity = identity
		# dropouts
		self.use_dropout: float = dropout
		self.use_word_dropout: float = word_dropout
		self.use_locked_dropout: float = locked_dropout

		self.pickle_module = pickle_module

		if dropout > 0.0:
			self.dropout = torch.nn.Dropout(dropout)

		if word_dropout > 0.0:
			self.word_dropout = flair.nn.WordDropout(word_dropout)

		if locked_dropout > 0.0:
			self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

		rnn_input_dim: int = self.embeddings.embedding_length

		self.relearn_embeddings: bool = relearn_embeddings

		if (self.embedding_selector and not self.use_rl) or self.embedding_attention:
			if use_gumbel:
				self.selector = Parameter(
						torch.zeros(len(self.embeddings.embeddings),2),
						requires_grad=True,
					)
			else:
				self.selector = Parameter(
						torch.zeros(len(self.embeddings.embeddings)),
						requires_grad=True,
					)
		if self.relearn_embeddings:
			self.embedding2nn = torch.nn.Linear(rnn_input_dim, rnn_input_dim)

		self.bidirectional = True
		self.rnn_type = "LSTM"
		if not self.use_rnn:
			self.bidirectional = False
		# bidirectional LSTM on top of embedding layer
		num_directions = 1

		# hiddens
		self.n_mlp_arc = n_mlp_arc
		self.n_mlp_rel = n_mlp_rel
		self.mlp_dropout = mlp_dropout
		self.n_mlp_sec = n_mlp_sec
		self.init_std = init_std
		self.lstm_dropout = lstm_dropout
		self.factorize = factorize
		# Initialization of Biaffine Parser
		self.embed_dropout = IndependentDropout(p=word_dropout)
		if self.use_rnn:
			self.rnn = BiLSTM(input_size=rnn_input_dim,
							   hidden_size=hidden_size,
							   num_layers=self.nlayers,
							   dropout=self.lstm_dropout)
			self.lstm_dropout_func = SharedDropout(p=self.lstm_dropout)
			# num_directions = 2 if self.bidirectional else 1

			# if self.rnn_type in ["LSTM", "GRU"]:

			#   self.rnn = getattr(torch.nn, self.rnn_type)(
			#       rnn_input_dim,
			#       hidden_size,
			#       num_layers=self.nlayers,
			#       dropout=0.0 if self.nlayers == 1 else 0.5,
			#       bidirectional=True,
			#   )
			#   # Create initial hidden state and initialize it
			#   if self.train_initial_hidden_state:
			#       self.hs_initializer = torch.nn.init.xavier_normal_

			#       self.lstm_init_h = Parameter(
			#           torch.randn(self.nlayers * num_directions, self.hidden_size),
			#           requires_grad=True,
			#       )

			#       self.lstm_init_c = Parameter(
			#           torch.randn(self.nlayers * num_directions, self.hidden_size),
			#           requires_grad=True,
			#       )

			#       # TODO: Decide how to initialize the hidden state variables
			#       # self.hs_initializer(self.lstm_init_h)
			#       # self.hs_initializer(self.lstm_init_c)

			# final linear map to tag space
			mlp_input_hidden = hidden_size * 2
		else:
			mlp_input_hidden = rnn_input_dim

		# the MLP layers
		self.mlp_arc_h = MLP(n_in=mlp_input_hidden,
							 n_hidden=n_mlp_arc,
							 dropout=mlp_dropout,
							 identity=self.identity)
		self.mlp_arc_d = MLP(n_in=mlp_input_hidden,
							 n_hidden=n_mlp_arc,
							 dropout=mlp_dropout,
							 identity=self.identity)
		self.mlp_rel_h = MLP(n_in=mlp_input_hidden,
							 n_hidden=n_mlp_rel,
							 dropout=mlp_dropout,
							 identity=self.identity)
		self.mlp_rel_d = MLP(n_in=mlp_input_hidden,
							 n_hidden=n_mlp_rel,
							 dropout=mlp_dropout,
							 identity=self.identity)
		# the Biaffine layers
		self.arc_attn = Biaffine(n_in=n_mlp_arc,
								 bias_x=True,
								 bias_y=False)
		self.rel_attn = Biaffine(n_in=n_mlp_rel,
								 n_out=self.tagset_size,
								 bias_x=True,
								 bias_y=True,
								 diagonal=self.diagonal,)
		self.binary = binary
		# the Second Order Parts
		self.use_second_order=use_second_order
		self.iterations=iterations
		self.use_sib = use_sib
		self.use_cop = use_cop
		self.use_gp = use_gp
		if self.use_second_order:
			if use_sib:
				self.mlp_sib_h = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout,
							 	 identity=self.identity)
				self.mlp_sib_d = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout,
							  	 identity=self.identity)
				self.trilinear_sib = TrilinearScorer(n_mlp_sec,n_mlp_sec,n_mlp_sec,init_std=init_std, rank = n_mlp_sec, factorize = factorize)
			if use_cop:
				self.mlp_cop_h = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout,
							 	 identity=self.identity)
				self.mlp_cop_d = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout,
							 	 identity=self.identity)
				self.trilinear_cop = TrilinearScorer(n_mlp_sec,n_mlp_sec,n_mlp_sec,init_std=init_std, rank = n_mlp_sec, factorize = factorize)
			if use_gp:
				self.mlp_gp_h = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout,
							 	 identity=self.identity)
				self.mlp_gp_d = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout,
							 	 identity=self.identity)
				self.mlp_gp_hd = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout,
							 	 identity=self.identity)
				self.trilinear_gp = TrilinearScorer(n_mlp_sec,n_mlp_sec,n_mlp_sec,init_std=init_std, rank = n_mlp_sec, factorize = factorize)
				

		# self.pad_index = pad_index
		# self.unk_index = unk_index
		self.rel_criterion = nn.CrossEntropyLoss()
		self.arc_criterion = nn.CrossEntropyLoss()

		if self.binary:
			self.rel_criterion = nn.CrossEntropyLoss(reduction='none')
			self.arc_criterion = nn.BCEWithLogitsLoss(reduction='none')
		if self.crf_attention:
			self.distill_criterion = nn.CrossEntropyLoss(reduction='none')
			self.distill_rel_criterion = nn.CrossEntropyLoss(reduction='none')
		if not testing:
			self.to(flair.device)


	def _init_model_with_state_dict(state, testing = False):
		use_dropout = 0.0 if not "use_dropout" in state.keys() else state["use_dropout"]
		use_word_dropout = (
			0.0 if not "use_word_dropout" in state.keys() else state["use_word_dropout"]
		)
		use_locked_dropout = (
			0.0
			if not "use_locked_dropout" in state.keys()
			else state["use_locked_dropout"]
		)
		if 'biaf_attention' in state:
			biaf_attention = state['biaf_attention']
		else:
			biaf_attention = False
		if 'token_level_attention' in state:
			token_level_attention = state['token_level_attention']
		else:
			token_level_attention = False
		if 'teacher_hidden' in state:
			teacher_hidden = state['teacher_hidden']
		else:
			teacher_hidden = 256
		use_cnn=state["use_cnn"] if 'use_cnn' in state else False

		model = SemanticDependencyParser(
			hidden_size=state["hidden_size"],
			embeddings=state["embeddings"],
			tag_dictionary=state["tag_dictionary"],
			tag_type=state["tag_type"],
			use_crf=state["use_crf"],
			use_rnn=state["use_rnn"],
			tree=state["tree"],
			punct=state["punct"],
			train_initial_hidden_state=state["train_initial_hidden_state"],
			n_mlp_arc = state["n_mlp_arc"],
			n_mlp_rel = state["n_mlp_rel"],
			mlp_dropout = state["mlp_dropout"],
			token_loss = False if 'token_loss' not in state else state["token_loss"],
			use_second_order = state["use_second_order"],
			n_mlp_sec = state["n_mlp_sec"],
			init_std = state["init_std"],
			factorize = state["factorize"],
			use_sib = state["use_sib"],
			use_gp = state["use_gp"],
			use_cop = state["use_cop"],
			iterations = state["iterations"],
			is_mst = False if "is_mst" not in state else state["is_mst"],
			binary = state["binary"],
			rnn_layers=state["rnn_layers"],
			dropout=use_dropout,
			word_dropout=use_word_dropout,
			locked_dropout=use_locked_dropout,
			config=state['config'] if "config" in state else None,
			word_map=None if 'word_map' not in state else state['word_map'],
			char_map=None if 'char_map' not in state else state['char_map'],
			relearn_embeddings = True if 'relearn_embeddings' not in state else state['relearn_embeddings'],
			distill_arc = False if 'distill_arc' not in state else state['distill_arc'],
			distill_rel = False if 'distill_rel' not in state else state['distill_rel'],
			distill_crf = False if 'distill_crf' not in state else state['distill_crf'],
			distill_posterior = False if 'distill_posterior' not in state else state['distill_posterior'],
			distill_prob = False if 'distill_prob' not in state else state['distill_prob'],
			distill_factorize = False if 'distill_factorize' not in state else state['distill_factorize'],
			factorize_interpolation = False if 'factorize_interpolation' not in state else state['factorize_interpolation'],
			diagonal = False if 'diagonal' not in state else state['diagonal'],
			embedding_selector = False if "embedding_selector" not in state else state["embedding_selector"],
			use_rl = False if "use_rl" not in state else state["use_rl"],
			use_gumbel = False if "use_gumbel" not in state else state["use_gumbel"],
			identity = False if "identity" not in state else state["identity"],
			embedding_attention = False if "embedding_attention" not in state else state["embedding_attention"],
			testing = testing,
			is_sdp = False if "is_sdp" not in state else state["is_sdp"],
		)
		model.load_state_dict(state["state_dict"])
		return model
	def _get_state_dict(self):
		model_state = {
			"state_dict": self.state_dict(),
			"embeddings": self.embeddings,
			"hidden_size": self.hidden_size,
			"tag_dictionary":self.tag_dictionary,
			"tag_type":self.tag_type,
			"tree":self.tree,
			"punct":self.punct,
			"use_crf": self.use_crf,
			"use_rnn":self.use_rnn,
			"train_initial_hidden_state": self.train_initial_hidden_state,
			"n_mlp_arc": self.n_mlp_arc,
			"n_mlp_rel": self.n_mlp_rel,
			"mlp_dropout": self.mlp_dropout,
			"token_loss": self.token_loss,
			"use_second_order": self.use_second_order,
			"n_mlp_sec": self.n_mlp_sec,
			"init_std": self.init_std,
			"factorize": self.factorize,
			"use_sib": self.use_sib,
			"use_gp": self.use_gp,
			"use_cop": self.use_cop,
			"iterations": self.iterations,
			"is_mst": self.is_mst,
			"binary": self.binary,
			"rnn_layers": self.rnn_layers,
			"dropout": self.use_dropout,
			"word_dropout": self.use_word_dropout,
			"locked_dropout": self.use_locked_dropout,
			"config": self.config,
			"word_map": self.word_map,
			"char_map": self.char_map,
			"relearn_embeddings": self.relearn_embeddings,
			"distill_arc": self.distill_arc,
			"distill_rel": self.distill_rel,
			"distill_crf": self.distill_crf,
			"distill_posterior": self.distill_posterior,
			"distill_prob": self.distill_prob,
			"distill_factorize": self.distill_factorize,
			"factorize_interpolation": self.factorize_interpolation,
			"diagonal": self.diagonal,
			"embedding_selector": self.embedding_selector,
			"use_rl": self.use_rl,
			"use_gumbel": self.use_gumbel,
			"embedding_attention": self.embedding_attention,
			"identity": self.identity,
			"is_sdp": self.is_sdp,
		}
		return model_state
	def forward(self, sentences: List[Sentence], prediction_mode = False):
		# self.zero_grad()

		lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

		longest_token_sequence_in_batch: int = max(lengths)

		if prediction_mode and self.embedding_selector:
			self.embeddings.embed(sentences,embedding_mask=self.selection)
		else:
			self.embeddings.embed(sentences)
		if self.embedding_selector:
			if self.use_rl:
				if self.embedding_attention:
					embatt=torch.sigmoid(self.selector)
					sentence_tensor = torch.cat([sentences.features[x].to(flair.device) * self.selection[idx] * embatt[idx] for idx, x in enumerate(sorted(sentences.features.keys()))],-1)
				else:
					sentence_tensor = torch.cat([sentences.features[x].to(flair.device) * self.selection[idx] for idx, x in enumerate(sorted(sentences.features.keys()))],-1)
					
					# sentence_tensor = torch.cat([sentences.features[x].to(flair.device) * self.selection[idx] for idx, x in enumerate(sentences.features.keys())],-1)
			else:
				# if self.training:
				# 	selection=torch.nn.functional.gumbel_softmax(self.selector,hard=True)
				# 	sentence_tensor = torch.cat([sentences.features[x].to(flair.device) * selection[idx][1] for idx, x in enumerate(sorted(sentences.features.keys()))],-1)
				# else:
				# selection=torch.sigmoid(self.selector)
				# sentence_tensor = torch.cat([sentences.features[x].to(flair.device) * selection[idx] for idx, x in enumerate(sorted(sentences.features.keys()))],-1)
				if self.use_gumbel:
					if self.training:
						selection=torch.nn.functional.gumbel_softmax(self.selector,hard=True)
						sentence_tensor = torch.cat([sentences.features[x].to(flair.device) * selection[idx][1] for idx, x in enumerate(sorted(sentences.features.keys()))],-1)
					else:
						selection=torch.argmax(self.selector,-1)
						sentence_tensor = torch.cat([sentences.features[x].to(flair.device) * selection[idx] for idx, x in enumerate(sorted(sentences.features.keys()))],-1)
				else:
					selection=torch.sigmoid(self.selector)
					sentence_tensor = torch.cat([sentences.features[x].to(flair.device) * selection[idx] for idx, x in enumerate(sorted(sentences.features.keys()))],-1)
		else:
			# sentence_tensor = torch.cat([sentences.features[x].to(flair.device) for x in sentences.features],-1)
			sentence_tensor = torch.cat([sentences.features[x].to(flair.device) for x in sorted(sentences.features.keys())],-1)
		# print('===================')
		# for x in sentences.features: print(x)
		# print('===================')
		# pdb.set_trace()
		if hasattr(self,'keep_embedding'):	
			sentence_tensor = [sentences.features[x].to(flair.device) for x in sorted(sentences.features.keys())]
			embedding_name = sorted(sentences.features.keys())[self.keep_embedding]
			if 'forward' in embedding_name or 'backward' in embedding_name:
				# sentence_tensor = torch.cat([sentences.features[x].to(flair.device) for x in sorted(sentences.features.keys()) if 'forward' in x or 'backward' in x],-1)
				for idx, x in enumerate(sorted(sentences.features.keys())):
					if 'forward' not in x and 'backward' not in x:
						sentence_tensor[idx].fill_(0)
			else:
				for idx, x in enumerate(sorted(sentences.features.keys())):
					if x != embedding_name:
						sentence_tensor[idx].fill_(0)
			sentence_tensor = torch.cat(sentence_tensor,-1)
		sentence_tensor = self.embed_dropout(sentence_tensor)[0]


		if self.relearn_embeddings:
			sentence_tensor = self.embedding2nn(sentence_tensor)
			# sentence_tensor = self.embedding2nn(sentence_tensor)

		if self.use_rnn:
			x = pack_padded_sequence(sentence_tensor, lengths, True, False)
			x, _ = self.rnn(x)
			sentence_tensor, _ = pad_packed_sequence(x, True, total_length=sentence_tensor.shape[1])
			sentence_tensor = self.lstm_dropout_func(sentence_tensor)
	
		mask=self.sequence_mask(torch.tensor(lengths),longest_token_sequence_in_batch).cuda().type_as(sentence_tensor)
		self.mask=mask
		# mask = words.ne(self.pad_index)
		# lens = mask.sum(dim=1)

		# get outputs from embedding layers
		x = sentence_tensor

		# apply MLPs to the BiLSTM output states
		arc_h = self.mlp_arc_h(x)
		arc_d = self.mlp_arc_d(x)
		rel_h = self.mlp_rel_h(x)
		rel_d = self.mlp_rel_d(x)

		# get arc and rel scores from the bilinear attention
		# [batch_size, seq_len, seq_len]
		s_arc = self.arc_attn(arc_d, arc_h)
		# [batch_size, seq_len, seq_len, n_rels]
		s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

		# add second order using mean field variational inference
		if self.use_second_order:
			mask_unary, mask_sib, mask_cop, mask_gp = self.from_mask_to_3d_mask(mask)
			unary = mask_unary*s_arc
			arc_sib, arc_cop, arc_gp = self.encode_second_order(x)
			layer_sib, layer_cop, layer_gp = self.get_edge_second_order_node_scores(arc_sib, arc_cop, arc_gp, mask_sib, mask_cop, mask_gp)
			s_arc = self.mean_field_variational_infernece(unary, layer_sib, layer_cop, layer_gp) 
		# set the scores that exceed the length of each sentence to -inf
		if not self.binary:
			s_arc.masked_fill_(~mask.unsqueeze(1).bool(), float(-1e9))
		return s_arc, s_rel

	def mean_field_variational_infernece(self, unary, layer_sib=None, layer_cop=None, layer_gp=None):
		layer_gp2 = layer_gp.permute(0,2,3,1)
		# modify from (dep, head) to (head, dep), in order to fit my code
		unary = unary.transpose(1,2)
		unary_potential = unary.clone()
		q_value = unary_potential.clone()
		for i in range(self.iterations):
			if self.binary:
				q_value=torch.sigmoid(q_value)
			else:
				q_value=F.softmax(q_value,1)
			if self.use_sib:
				second_temp_sib = torch.einsum('nac,nabc->nab', (q_value, layer_sib))
				#(n x ma x mb) -> (n x ma) -> (n x ma x 1) | (n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
				#Q(a,a)*p(a,b,a) 
				diag_sib1 = torch.diagonal(q_value,dim1=1,dim2=2).unsqueeze(-1) * torch.diagonal(layer_sib.transpose(1,2),dim1=-2,dim2=-1).transpose(1,2)
				# (n x ma x mb x mc) -> (n x ma x mb)
				#Q(a,b)*p(a,b,b)
				diag_sib2 = q_value * torch.diagonal(layer_sib,dim1=-2,dim2=-1)
				#(n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
				second_temp_sib = second_temp_sib - diag_sib1 - diag_sib2
			else:
				second_temp_sib=0

			if self.use_gp:
				second_temp_gp = torch.einsum('nbc,nabc->nab', (q_value, layer_gp))
				second_temp_gp2 = torch.einsum('nca,nabc->nab', (q_value, layer_gp2))
				#Q(b,a)*p(a,b,a)
				diag_gp1 = q_value.transpose(1,2) * torch.diagonal(layer_gp.transpose(1,2),dim1=-2,dim2=-1).transpose(1,2)
				#(n x ma x mb) -> (n x mb) -> (n x 1 x mb) | (n x ma x mb x mc) -> (n x ma x mb)
				#Q(b,b)*p(a,b,b)
				diag_gp2 = torch.diagonal(q_value,dim1=-2,dim2=-1).unsqueeze(1) * torch.diagonal(layer_gp,dim1=-2,dim2=-1)
				#(n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
				#Q(a,a)*p(a,b,a)
				diag_gp21 = torch.diagonal(q_value,dim1=-2,dim2=-1).unsqueeze(-1) * torch.diagonal(layer_gp2.transpose(1,2),dim1=-2,dim2=-1).transpose(1,2)
				#(n x ma x mb) -> (n x mb) -> (n x 1 x mb) | (n x ma x mb x mc) -> (n x ma x mb)
				#Q(b,a)*p(a,b,b)
				diag_gp22 = q_value.transpose(1,2) * torch.diagonal(layer_gp2,dim1=-2,dim2=-1)

				second_temp_gp = second_temp_gp - diag_gp1 - diag_gp2
				#c->a->b
				second_temp_gp2 = second_temp_gp2 - diag_gp21 - diag_gp22
			else:
				second_temp_gp=second_temp_gp2=0

			if self.use_cop:
				second_temp_cop = torch.einsum('ncb,nabc->nab', (q_value, layer_cop))
				#(n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
				#Q(a,b)*p(a,b,a)
				diag_cop1 = q_value * torch.diagonal(layer_cop.transpose(1,2),dim1=-2,dim2=-1).transpose(1,2)
				# diag_cop1 = q_value * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_cop,perm=[0,2,1,3])),perm=[0,2,1])
				#(n x ma x mb) -> (n x mb) -> (n x 1 x mb) | (n x ma x mb x mc) -> (n x ma x mb)
				#Q(b,b)*p(a,b,b)
				diag_cop2 = torch.diagonal(q_value,dim1=-2,dim2=-1).unsqueeze(1) * torch.diagonal(layer_cop,dim1=-2,dim2=-1)
				# diag_cop2 = tf.expand_dims(tf.linalg.diag_part(q_value),1) * tf.linalg.diag_part(layer_cop)
				second_temp_cop = second_temp_cop - diag_cop1 - diag_cop2
			else:
				second_temp_cop=0

			second_temp = second_temp_sib + second_temp_gp + second_temp_gp2 + second_temp_cop
			q_value = unary_potential + second_temp
		# transpose from (head, dep) to (dep, head)
		return q_value.transpose(1,2)

	def encode_second_order(self, memory_bank):

		if self.use_sib:
			edge_node_sib_h = self.mlp_sib_h(memory_bank)
			edge_node_sib_m = self.mlp_sib_d(memory_bank)
			arc_sib=(edge_node_sib_h, edge_node_sib_m)
		else:
			arc_sib=None

		if self.use_cop:
			edge_node_cop_h = self.mlp_cop_h(memory_bank)
			edge_node_cop_m = self.mlp_cop_d(memory_bank)
			arc_cop=(edge_node_cop_h, edge_node_cop_m)
		else:
			arc_cop=None

		if self.use_gp:
			edge_node_gp_h = self.mlp_gp_h(memory_bank)
			edge_node_gp_m = self.mlp_gp_d(memory_bank)
			edge_node_gp_hm = self.mlp_gp_hd(memory_bank)
			arc_gp=(edge_node_gp_h, edge_node_gp_hm, edge_node_gp_m)
		else:
			arc_gp=None

		return arc_sib, arc_cop, arc_gp

	def get_edge_second_order_node_scores(self, arc_sib, arc_cop, arc_gp, mask_sib, mask_cop, mask_gp):

		if self.use_sib:
			edge_node_sib_h, edge_node_sib_m = arc_sib
			layer_sib = self.trilinear_sib(edge_node_sib_h, edge_node_sib_m, edge_node_sib_m) * mask_sib
			# keep (ma x mb x mc) -> (ma x mb x mb)
			#layer_sib = 0.5 * (layer_sib + layer_sib.transpose(3,2))
			one_mask=torch.ones(layer_sib.shape[-2:]).cuda()
			tril_mask=torch.tril(one_mask,-1)
			triu_mask=torch.triu(one_mask,1)
			layer_sib = layer_sib-layer_sib*tril_mask.unsqueeze(0).unsqueeze(0) + (layer_sib*triu_mask.unsqueeze(0).unsqueeze(0)).permute([0,1,3,2])
			
		else:
			layer_sib = None
		if self.use_cop:
			edge_node_cop_h, edge_node_cop_m = arc_cop
			layer_cop = self.trilinear_cop(edge_node_cop_h, edge_node_cop_m, edge_node_cop_h) * mask_cop
			# keep (ma x mb x mc) -> (ma x mb x ma)
			one_mask=torch.ones(layer_cop.shape[-2:]).cuda()
			tril_mask=torch.tril(one_mask,-1)
			triu_mask=torch.triu(one_mask,1)
			layer_cop=layer_cop.transpose(1,2)
			layer_cop = layer_cop-layer_cop*tril_mask.unsqueeze(0).unsqueeze(0) + (layer_cop*triu_mask.unsqueeze(0).unsqueeze(0)).permute([0,1,3,2])
			layer_cop=layer_cop.transpose(1,2)
		else:
			layer_cop = None

		if self.use_gp:
			edge_node_gp_h, edge_node_gp_hm, edge_node_gp_m = arc_gp
			layer_gp = self.trilinear_gp(edge_node_gp_h, edge_node_gp_hm, edge_node_gp_m) * mask_gp
		else:
			layer_gp = None
		
		return layer_sib,layer_cop,layer_gp

	def from_mask_to_3d_mask(self,token_weights):
		root_weights = token_weights.clone()
		root_weights[:,0] = 0
		token_weights3D = token_weights.unsqueeze(-1) * root_weights.unsqueeze(-2)
		token_weights2D = root_weights.unsqueeze(-1) * root_weights.unsqueeze(-2)
		# abc -> ab,ac
		#token_weights_sib = tf.cast(tf.expand_dims(root_, axis=-3) * tf.expand_dims(tf.expand_dims(root_weights, axis=-1),axis=-1),dtype=tf.float32)
		#abc -> ab,cb
		if self.use_cop:
			token_weights_cop = token_weights.unsqueeze(-1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(-1) * token_weights.unsqueeze(1).unsqueeze(1)
			token_weights_cop[:,0,:,0] = 0
		else:
			token_weights_cop=None
		#data=np.stack((devprint['printdata']['layer_cop'][0][0]*devprint['token_weights3D'][0].T)[None,:],devprint['printdata']['layer_cop'][0][1:])
		#abc -> ab, bc
		if self.use_gp:
			token_weights_gp = token_weights.unsqueeze(-1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(1)
		else:
			token_weights_gp = None

		if self.use_sib:
			#abc -> ca, ab
			if self.use_gp:
				token_weights_sib = token_weights_gp.clone()
			else:
				token_weights.unsqueeze(-1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(1)
		else:
			token_weights_sib = None
		return token_weights3D, token_weights_sib, token_weights_cop, token_weights_gp



	def forward_loss(
		self, data_points: Union[List[Sentence], Sentence], sort=True
	) -> torch.tensor:
		s_arc, s_rel = self.forward(data_points)
		# lengths = [len(sentence.tokens) for sentence in data_points]
		# longest_token_sequence_in_batch: int = max(lengths)

		# max_len = features.shape[1]
		# mask=self.sequence_mask(torch.tensor(lengths), max_len).cuda().type_as(features)
		loss = self._calculate_loss(s_arc, s_rel, data_points, self.mask)
		return loss

	def simple_forward_distillation_loss(
		self, data_points: Union[List[Sentence], Sentence], teacher_data_points: Union[List[Sentence], Sentence]=None, teacher=None, sort=True,
		interpolation=0.5, train_with_professor=False, professor_interpolation=0.5, language_attention_warmup = False, calc_teachers_target_loss = False,
		language_weight = None, biaffine = None, language_vector = None,
	) -> torch.tensor:
		arc_scores, rel_scores = self.forward(data_points)
		lengths = [len(sentence.tokens) for sentence in data_points]
		max_len = arc_scores.shape[1]
		mask=self.mask.clone()
		posterior_loss = 0
		if self.distill_posterior:
			# mask[:,0] = 0
			if hasattr(data_points,'teacher_features') and 'posteriors' in data_points.teacher_features:
				teacher_scores = data_points.teacher_features['posteriors'].to(flair.device)
			else:
				teacher_scores = torch.stack([sentence.get_teacher_posteriors() for sentence in data_points],0)
			if self.distill_arc:
				root_mask = mask.clone()
				root_mask[:,0] = 0
				binary_mask = root_mask.unsqueeze(-1) * mask.unsqueeze(-2)
				arc_scores.masked_fill_(~binary_mask.bool(), float(-1e9))
				for i in range(teacher_scores.shape[-2]):
					if self.distill_rel:
						assert 0
						marginals = convert_score_back(teacher_scores[:,:,:,i])
						arc_probs = arc_scores.softmax(-1)
						rel_probs = rel_scores.softmax(-1)
						student_probs = arc_probs.unsqueeze(-1) * rel_probs
						student_scores = (student_probs+1e-12).log()
						student_scores = student_scores.view(list(student_scores.shape[0:2])+[-1])
						marginals = marginals.reshape(list(marginals.shape[0:2])+[-1])
						# create the mask
						binary_mask = binary_mask.unsqueeze(-1).expand(list(binary_mask.shape)+[rel_probs.shape[-1]]).reshape(list(binary_mask.shape[0:2])+[-1])
					else:
						marginals = convert_score_back(teacher_scores[:,:,i])
					posterior_loss += self._calculate_distillation_loss(student_scores, marginals, root_mask, binary_mask, T=self.temperature, teacher_is_score = False)
			else:
				root_mask = mask.clone()
				root_mask[:,0] = 0
				binary_mask = root_mask.unsqueeze(-1) * mask.unsqueeze(-2)
				inside_outside_prob = crf(arc_scores, root_mask.bool(),marginal_gradient=True)
				inside_outside_score = (inside_outside_prob + 1e-12).log()
				for i in range(teacher_scores.shape[-2]):
					posterior_loss += self._calculate_distillation_loss(inside_outside_score, teacher_scores[:,:,i], root_mask, binary_mask, T=self.temperature, teacher_is_score = False)
				# temp_mask = mask[:,1:]
				# dist=generate_tree(arc_scores,temp_mask.squeeze(-1).long(),is_mst=self.is_mst)
				# forward_backward_score = dist.marginals
				# # change back to relation of (dependency, head)
				# input_forward_score = (forward_backward_score.transpose(-1,-2)+1e-12).log()
				# binary_mask = temp_mask.unsqueeze(-1) * temp_mask.unsqueeze(-2)
				# input_forward_score.masked_fill_(~binary_mask.bool(), float(-1e9))
				# for i in range(teacher_scores.shape[-2]):
				# 	posterior_loss += self._calculate_distillation_loss(input_forward_score, teacher_scores[:,:,i].transpose(-1,-2), temp_mask, binary_mask, T=self.temperature, teacher_is_score = False)
			posterior_loss/=teacher_scores.shape[-2]
		
		distillation_loss = 0
		if self.distill_crf:
			# [batch, length, kbest]
			mask[:,0] = 0
			if hasattr(data_points,'teacher_features') and 'topk' in data_points.teacher_features:
				teacher_tags = data_points.teacher_features['topk'].to(flair.device)
				teacher_weights = data_points.teacher_features['weights'].to(flair.device)
				if self.distill_rel:
					teacher_rel_tags = data_points.teacher_features['topk_rels'].to(flair.device)
			else:
				teacher_tags = torch.stack([sentence.get_teacher_target() for sentence in data_points],0)
				teacher_weights = torch.stack([sentence.get_teacher_weights() for sentence in data_points],0)
				if self.distill_rel:
					teacher_rel_tags = torch.stack([sentence.get_teacher_rel_target() for sentence in data_points],0)
			# proprocess, convert k best to batch wise
			teacher_mask = (mask.unsqueeze(-1) * (teacher_weights.unsqueeze(1)>0).type_as(mask)).bool()
			
			student_arc_scores = arc_scores.unsqueeze(-2).expand(list(arc_scores.shape[:2])+[teacher_mask.shape[-1],arc_scores.shape[-1]])[teacher_mask]
			teacher_topk_arcs = teacher_tags[teacher_mask]
			if self.distill_rel:
				# gold_arcs = arcs[mask]
				# rel_scores, rels = rel_scores[mask], rels[mask]
				# rel_scores = rel_scores[torch.arange(len(gold_arcs)), gold_arcs]

				student_rel_scores = rel_scores.unsqueeze(-3).expand(list(rel_scores.shape[:2])+[teacher_mask.shape[-1]]+list(rel_scores.shape[-2:]))[teacher_mask]
				teacher_topk_rels = teacher_rel_tags[teacher_mask]
				student_rel_scores = student_rel_scores[torch.arange(len(teacher_topk_arcs)),teacher_topk_arcs]
			if self.crf_attention:
				weights = teacher_weights.unsqueeze(1).expand([teacher_weights.shape[0],arc_scores.shape[1],teacher_weights.shape[1]])[teacher_mask]
				distillation_loss = self.distill_criterion(student_arc_scores, teacher_topk_arcs)
				# the loss calculates only one times because the sum of weight is 1
				distillation_loss = (distillation_loss * weights).sum() / mask.sum()
				if self.distill_rel:
					rel_distillation_loss = self.distill_rel_criterion(student_rel_scores, teacher_topk_rels)
					rel_distillation_loss = (rel_distillation_loss * weights).sum() / mask.sum()
			else:
				# the loss calculates for k times
				distillation_loss = self.arc_criterion(student_arc_scores, teacher_topk_arcs)
				if self.distill_rel:
					rel_distillation_loss = self.rel_criterion(student_rel_scores, teacher_topk_rels)

		arc_loss,rel_loss = self._calculate_loss(arc_scores, rel_scores, data_points, self.mask.clone(), return_arc_rel=True)
		if (self.distill_arc or self.distill_rel) and not self.distill_posterior and not self.distill_crf:
			root_mask = mask.clone()
			root_mask[:,0] = 0
			binary_mask = root_mask.unsqueeze(-1) * mask.unsqueeze(-2)

			if hasattr(data_points,'teacher_features') and 'distributions' in data_points.teacher_features:
				teacher_features = data_points.teacher_features['distributions'].to(flair.device)
			else:
				teacher_features = torch.stack([sentence.get_teacher_prediction() for sentence in data_points],0)

			if self.distill_arc:
				features = arc_scores
			if self.distill_rel:
				# features = arc_scores.unsqueeze(-1) * rel_scores
				if self.distill_factorize:
					rel_binary_mask = binary_mask.unsqueeze(-1).expand(list(binary_mask.shape)+[rel_scores.shape[-1]]).reshape(list(binary_mask.shape[0:2])+[-1])
					if hasattr(data_points,'teacher_features') and 'rel_distributions' in data_points.teacher_features:
						teacher_rel_features = data_points.teacher_features['rel_distributions'].to(flair.device)
					else:
						teacher_rel_features = torch.stack([sentence.get_teacher_rel_prediction() for sentence in data_points],0)
					rel_probs = rel_scores.softmax(-1)
					
					rel_probs = rel_probs.view(list(rel_probs.shape[0:2])+[-1])
					rel_scores = (rel_probs+1e-12).log()

					teacher_rel_features = teacher_rel_features.view(list(teacher_rel_features.shape[0:2])+[-1])

					rel_distillation_loss = self._calculate_distillation_loss(rel_scores, teacher_rel_features, root_mask, rel_binary_mask, T=self.temperature, teacher_is_score=(not self.distill_prob) and (not self.distill_rel))
					features = arc_scores
				else:
					arc_probs = arc_scores.softmax(-1)
					rel_probs = rel_scores.softmax(-1)
					features = arc_probs.unsqueeze(-1) * rel_probs
					features = features.view(list(features.shape[0:2])+[-1])
					features = (features+1e-12).log()
					teacher_features = teacher_features.view(list(teacher_features.shape[0:2])+[-1])
					# create the mask
					binary_mask = binary_mask.unsqueeze(-1).expand(list(binary_mask.shape)+[rel_probs.shape[-1]]).reshape(list(binary_mask.shape[0:2])+[-1])

			else:
				teacher_features.masked_fill_(~self.mask.unsqueeze(1).bool(), float(-1e9))

			distillation_loss = self._calculate_distillation_loss(features, teacher_features, root_mask, binary_mask, T=self.temperature, teacher_is_score=(not self.distill_prob) and (not self.distill_rel))
		# target_loss2 = super()._calculate_loss(features,data_points)
		# distillation_loss2 = super()._calculate_distillation_loss(features, teacher_features,torch.tensor(lengths))
		# (interpolation * (posterior_loss + distillation_loss) + (1-interpolation) * target_loss).backward()
		if self.distill_rel:
			# if distilling both arc and rel distribution, just use the same interpolation
			target_loss = 2 * ((1-self.interpolation) * arc_loss + self.interpolation * rel_loss)
			if self.distill_factorize:
				# balance the relation distillation loss and arc distillation loss through a new interpolation
				distillation_loss = 2 * ((1-self.factorize_interpolation) * distillation_loss + self.factorize_interpolation * rel_distillation_loss)
			if self.distill_crf:
				distillation_loss = 2 * ((1-self.interpolation) * distillation_loss + self.interpolation * rel_distillation_loss)
			return interpolation * (posterior_loss + distillation_loss) + (1-interpolation) * target_loss
		else:
			# otherwise, balance between the (arc distillation loss + arc loss) and (rel loss)
			return 2*((1-self.interpolation) * (interpolation * (posterior_loss + distillation_loss) + (1-interpolation) * arc_loss) + self.interpolation * rel_loss)

	def sequence_mask(self, lengths, max_len=None):
		"""
		Creates a boolean mask from sequence lengths.
		"""
		batch_size = lengths.numel()
		max_len = max_len or lengths.max()
		return (torch.arange(0, max_len)
				.type_as(lengths)
				.repeat(batch_size, 1)
				.lt(lengths.unsqueeze(1)))
	def _calculate_distillation_loss(self, features, teacher_features, mask, binary_mask, T = 1, teacher_is_score=True, student_is_score = True):
		# TODO: time with mask, and whether this should do softmax
		# pdb.set_trace()
		if teacher_is_score:
			teacher_prob=F.softmax(teacher_features/T, dim=-1)
		else:
			if T>1:
				teacher_scores = (teacher_features+1e-12).log()
				teacher_prob=F.softmax(teacher_scores/T, dim=-1)
			else:
				teacher_prob=teacher_features
		KD_loss = torch.nn.functional.kl_div(F.log_softmax(features/T, dim=-1), teacher_prob,reduction='none') * binary_mask * T * T

		# KD_loss = KD_loss.sum()/mask.sum()
		
		if self.sentence_level_loss:
			KD_loss = KD_loss.sum()/KD_loss.shape[0]
		else:
			KD_loss = KD_loss.sum()/mask.sum()
		return KD_loss
		# return torch.nn.functional.MSELoss(features, teacher_features, reduction='mean')
	def _calculate_loss(
		self, arc_scores: torch.tensor, rel_scores: torch.tensor, sentences: List[Sentence], mask: torch.tensor, return_arc_rel = False,
	) -> float:
		if self.binary:
			root_mask = mask.clone()
			root_mask[:,0] = 0
			binary_mask = root_mask.unsqueeze(-1) * mask.unsqueeze(-2)
			# arc_mat=
			if hasattr(sentences,self.tag_type+'_arc_tags'):
				arc_mat=getattr(sentences,self.tag_type+'_arc_tags').to(flair.device).float()
			else:
				arc_mat=torch.stack([getattr(sentence,self.tag_type+'_arc_tags').to(flair.device) for sentence in sentences],0).float()
			if hasattr(sentences,self.tag_type+'_rel_tags'):
				rel_mat=getattr(sentences,self.tag_type+'_rel_tags').to(flair.device).long()
			else:
				rel_mat=torch.stack([getattr(sentence,self.tag_type+'_rel_tags').to(flair.device) for sentence in sentences],0).long()
			
			arc_loss = self.arc_criterion(arc_scores, arc_mat)
			rel_loss = self.rel_criterion(rel_scores.reshape(-1,self.tagset_size), rel_mat.reshape(-1))
			arc_loss = (arc_loss*binary_mask).sum()/binary_mask.sum()

			rel_mask = (rel_mat>0)*binary_mask
			num_rels=rel_mask.sum()
			if num_rels>0:
				rel_loss = (rel_loss*rel_mask.view(-1)).sum()/num_rels
			else:
				rel_loss = 0
			# rel_loss = (rel_loss*rel_mat.view(-1)).sum()/rel_mat.sum()
		else:
			if hasattr(sentences,self.tag_type+'_arc_tags'):
				arcs=getattr(sentences,self.tag_type+'_arc_tags').to(flair.device).long()
			else:
				arcs=torch.stack([getattr(sentence,self.tag_type+'_arc_tags').to(flair.device) for sentence in sentences],0).long()
			if hasattr(sentences,self.tag_type+'_rel_tags'):
				rels=getattr(sentences,self.tag_type+'_rel_tags').to(flair.device).long()
			else:
				rels=torch.stack([getattr(sentence,self.tag_type+'_rel_tags').to(flair.device) for sentence in sentences],0).long()
			self.arcs=arcs
			self.rels=rels
			mask[:,0] = 0
			mask = mask.bool()
			gold_arcs = arcs[mask]
			rel_scores, rels = rel_scores[mask], rels[mask]
			rel_scores = rel_scores[torch.arange(len(gold_arcs)), gold_arcs]
			if self.use_crf:
				arc_loss, arc_probs = crf(arc_scores, mask, arcs)
				arc_loss = arc_loss/mask.sum()
				rel_loss = self.rel_criterion(rel_scores, rels)

				#=============================================================================================
				# dist=generate_tree(arc_scores,mask,is_mst=self.is_mst)
				# labels = dist.struct.to_parts(arcs[:,1:], lengths=mask.sum(-1)).type_as(arc_scores)
				# log_prob = dist.log_prob(labels)
				# if (log_prob>0).any():
					
				#   log_prob[torch.where(log_prob>0)]=0
				#   print("failed to get correct loss!")
				# if self.token_loss:
				#   arc_loss = - log_prob.sum()/mask.sum()
				# else:
				#   arc_loss = - log_prob.mean()
				
				# self.dist=dist
				
				# rel_loss = self.rel_criterion(rel_scores, rels)
				# if self.token_loss:
				#   rel_loss = rel_loss.mean()
				# else:
				#   rel_loss = rel_loss.sum()/len(sentences)

				# if self.debug:
				#   if rel_loss<0 or arc_loss<0:
				#       pdb.set_trace()
				#=============================================================================================
			else:
				arc_scores, arcs = arc_scores[mask], arcs[mask]
				arc_loss = self.arc_criterion(arc_scores, arcs)
			
				# rel_scores, rels = rel_scores[mask], rels[mask]
				# rel_scores = rel_scores[torch.arange(len(arcs)), arcs]
				
				rel_loss = self.rel_criterion(rel_scores, rels)
		if return_arc_rel:
			return (arc_loss,rel_loss)
		loss = 2 * ((1-self.interpolation) * arc_loss + self.interpolation * rel_loss)


		# score = torch.nn.functional.cross_entropy(features.view(-1,features.shape[-1]), tag_list.view(-1,), reduction='none') * mask.view(-1,)



		# if self.sentence_level_loss or self.use_crf:
		#   score = score.sum()/features.shape[0]
		# else:
		#   score = score.sum()/mask.sum()
			
		#   score = (1-self.posterior_interpolation) * score + self.posterior_interpolation * posterior_score
		return loss

	def evaluate(
		self,
		data_loader: DataLoader,
		out_path: Path = None,
		embeddings_storage_mode: str = "cpu",
		prediction_mode: bool = False,
	) -> (Result, float):
		data_loader.assign_embeddings()
		with torch.no_grad():
			if self.binary:
				eval_loss = 0

				batch_no: int = 0

				# metric = Metric("Evaluation")
				# sentence_writer = open('temps/'+str(uid)+'_eval'+'.conllu','w')
				lines: List[str] = []
				utp = 0
				ufp = 0
				ufn = 0
				ltp = 0
				lfp = 0
				lfn = 0
				if out_path is not None:
					outfile = open(out_path, "w", encoding="utf-8")
				for batch in data_loader:
					batch_no += 1
					arc_scores, rel_scores = self.forward(batch, prediction_mode=prediction_mode)
					mask=self.mask
					root_mask = mask.clone()
					root_mask[:,0] = 0
					binary_mask = root_mask.unsqueeze(-1) * mask.unsqueeze(-2)
					
					arc_predictions = (arc_scores.sigmoid() > 0.5) * binary_mask
					rel_predictions = (rel_scores.softmax(-1)*binary_mask.unsqueeze(-1)).argmax(-1)
					# if not prediction_mode:
					arc_mat=torch.stack([getattr(sentence,self.tag_type+'_arc_tags').to(flair.device) for sentence in batch],0).float()
					rel_mat=torch.stack([getattr(sentence,self.tag_type+'_rel_tags').to(flair.device) for sentence in batch],0).long()
					loss = self._calculate_loss(arc_scores, rel_scores, batch, mask)
					if self.is_srl:
						# let the head selection fixed to the gold predicate only
						binary_mask[:,:,0] = arc_mat[:,:,0]
						arc_predictions = (arc_scores.sigmoid() > 0.5) * binary_mask
						

					# UF1
					true_positives = arc_predictions * arc_mat
					# (n x m x m) -> ()
					n_predictions = arc_predictions.sum()
					n_unlabeled_predictions = n_predictions
					n_targets = arc_mat.sum()
					n_unlabeled_targets = n_targets
					n_true_positives = true_positives.sum()
					# () - () -> ()
					n_false_positives = n_predictions - n_true_positives
					n_false_negatives = n_targets - n_true_positives
					# (n x m x m) -> (n)
					n_targets_per_sequence = arc_mat.sum([1,2])
					n_true_positives_per_sequence = true_positives.sum([1,2])
					# (n) x 2 -> ()
					n_correct_sequences = (n_true_positives_per_sequence==n_targets_per_sequence).sum()
					utp += n_true_positives
					ufp += n_false_positives
					ufn += n_false_negatives

					# LF1
					# (n x m x m) (*) (n x m x m) -> (n x m x m)
					true_positives = (rel_predictions == rel_mat) * arc_predictions
					correct_label_tokens = (rel_predictions == rel_mat) * arc_mat
					# (n x m x m) -> ()
					# n_unlabeled_predictions = tf.reduce_sum(unlabeled_predictions)
					# n_unlabeled_targets = tf.reduce_sum(unlabeled_targets)
					n_true_positives = true_positives.sum()
					n_correct_label_tokens = correct_label_tokens.sum()
					# () - () -> ()
					n_false_positives = n_unlabeled_predictions - n_true_positives
					n_false_negatives = n_unlabeled_targets - n_true_positives
					# (n x m x m) -> (n)
					n_targets_per_sequence = arc_mat.sum([1,2])
					n_true_positives_per_sequence = true_positives.sum([1,2])
					n_correct_label_tokens_per_sequence = correct_label_tokens.sum([1,2])
					# (n) x 2 -> ()
					n_correct_sequences = (n_true_positives_per_sequence == n_targets_per_sequence).sum()
					n_correct_label_sequences = ((n_correct_label_tokens_per_sequence == n_targets_per_sequence)).sum()
					ltp += n_true_positives
					lfp += n_false_positives
					lfn += n_false_negatives

					eval_loss += loss

					if out_path is not None:
						masked_arc_scores = arc_scores.masked_fill(~binary_mask.bool(), float(-1e9))
						# if self.target
						# lengths = [len(sentence.tokens) for sentence in batch]
						
						# temp_preds = eisner(arc_scores, mask)
						if not self.is_mst and self.tree:
							temp_preds = eisner(arc_scores, root_mask.bool())
						for (sent_idx, sentence) in enumerate(batch):
							if self.is_mst:
								preds=MST_inference(torch.softmax(masked_arc_scores[sent_idx],-1).cpu().numpy(), len(sentence), binary_mask[sent_idx].cpu().numpy())
							elif self.tree:
								preds=temp_preds[sent_idx]
							else:
								preds = []
							sent_arc_preds=torch.where(arc_predictions[sent_idx]>0)
							
							if len(sent_arc_preds[0])==0:
								graph_score = 0
							else:
								
								sent_arc_scores = arc_scores[sent_idx, sent_arc_preds[0], sent_arc_preds[1]]
							
								sent_rel_scores = rel_scores[sent_idx, sent_arc_preds[0], sent_arc_preds[1]].max(-1)[0]
								
								final_score = sent_arc_scores*sent_rel_scores
								graph_score = final_score.sum().cpu().item()
							if out_path is not None:
								outfile.write(f'# Tree score: {graph_score}\n')
							for token_idx, token in enumerate(sentence):
								if token_idx == 0:
									continue

								# append both to file for evaluation
								arc_heads = torch.where(arc_predictions[sent_idx,token_idx]>0)[0]
								if len(preds)>0 and preds[token_idx] not in arc_heads:
									val=torch.zeros(1).type_as(arc_heads)
									val[0]=preds[token_idx].item()
									arc_heads=torch.cat([arc_heads,val],0)
								# this part should be removed for SDP
								# if len(arc_heads) == 0:
								# 	arc_heads = masked_arc_scores[sent_idx,token_idx].argmax().unsqueeze(0)
								if len(arc_heads) != 0:
									rel_index = rel_predictions[sent_idx,token_idx,arc_heads]
									rel_labels = [self.tag_dictionary.get_item_for_index(x) for x in rel_index]
									arc_list=[]
									token_arc_scores = arc_scores[sent_idx,token_idx, arc_heads]
									token_rel_scores = rel_scores[sent_idx,token_idx, arc_heads].max(-1)[0]
									token_score = (token_arc_scores*token_rel_scores).sum().cpu().item()
									for i, label in enumerate(rel_labels):
										if '+' in label:
											labels = label.split('+')
											for temp_label in labels:
												arc_list.append(str(arc_heads[i].item())+':'+temp_label)
										else:
											arc_list.append(str(arc_heads[i].item())+':'+label)
								else:
									arc_list = ['_']
									token_score = 0
								eval_line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
									token_idx,
									token.text,
									'X',
									'X',
									'X',
									token.get_tag(self.tag_type).value,
									str(token_idx-1),
									'root' if token_idx-1==0 else 'det',
									'|'.join(arc_list),
									f'{token_score}',
								)
								# lines.append(eval_line)
								if out_path is not None:
									outfile.write(eval_line)
							# lines.append("\n")
							if out_path is not None:
								outfile.write('\n')


				eval_loss /= batch_no
				UF1=self.compute_F1(utp,ufp,ufn).cpu().numpy()
				LF1=self.compute_F1(ltp,lfp,lfn).cpu().numpy()

				if out_path is not None:
					outfile.close()
				# 	with open(out_path, "w", encoding="utf-8") as outfile:
				# 		outfile.write("".join(lines))
				# if prediction_mode:
				# 	return None, None

				result = Result(
					main_score=LF1,
					log_line=f"\nUF1: {UF1} - LF1 {LF1}",
					log_header="PRECISION\tRECALL\tF1",
					detailed_results=f"\nUF1: {UF1} - LF1 {LF1}",
				)
			else:
				# if prediction_mode:
				# 	eval_loss, metric=self.dependency_evaluate(data_loader,out_path=out_path,prediction_mode=prediction_mode)
				# 	return eval_loss, metric
				# else:   
				eval_loss, metric=self.dependency_evaluate(data_loader,out_path=out_path)
				
				UAS=metric.uas
				LAS=metric.las
				result = Result(main_score=LAS,log_line=f"\nUAS: {UAS} - LAS {LAS}",log_header="PRECISION\tRECALL\tF1",detailed_results=f"\nUAS: {UAS} - LAS {LAS}",)
			return result, eval_loss
	def compute_F1(self, tp, fp, fn):
		precision = tp/(tp+fp + 1e-12)
		recall = tp/(tp+fn + 1e-12)
		return 2 * (precision * recall) / (precision + recall+ 1e-12)


	@torch.no_grad()
	def dependency_evaluate(self, loader, out_path=None, prediction_mode=False):
		# self.model.eval()

		loss, metric = 0, Metric()
		# total_start_time=time.time()
		# forward_time=0
		# loss_time=0
		# decode_time=0
		# punct_time=0
		lines=[]
		for batch in loader:
			forward_start=time.time()
			arc_scores, rel_scores = self.forward(batch)
			# forward_end=time.time()
			mask = self.mask
			if not prediction_mode:
				loss += self._calculate_loss(arc_scores, rel_scores, batch, mask)
			# loss_end=time.time()
			# forward_time+=forward_end-forward_start
			# loss_time+=loss_end-forward_end
			mask=mask.bool()
			# decode_start=time.time()
			arc_preds, rel_preds, pred_arc_scores, pred_rel_scores = self.decode(arc_scores, rel_scores, mask)
			# decode_end=time.time()
			# decode_time+=decode_end-decode_start
			# ignore all punctuation if not specified
			# if out_path is not None:
			# 	pdb.set_trace()
			if not self.punct:
				for sent_id,sentence in enumerate(batch):
					for token_id, token in enumerate(sentence):
						upos=token.get_tag('upos').value
						xpos=token.get_tag('pos').value
						word=token.text
						if is_punctuation(word,upos,self.punct_list) or is_punctuation(word,upos,self.punct_list):
							mask[sent_id][token_id]=0
				# mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
			
			final_score = pred_arc_scores*pred_rel_scores
			tree_score = final_score.sum(-1)
			if out_path is not None:
				for (sent_idx, sentence) in enumerate(batch):

					lines.append(f'# Tree score: {tree_score[sent_idx].cpu().item()}\n')
					for token_idx, token in enumerate(sentence):
						if token_idx == 0:
							continue

						# append both to file for evaluation
						eval_line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
							token_idx,
							token.text,
							'X',
							'X',
							'X',
							token.get_tag(self.tag_type).value,
							arc_preds[sent_idx,token_idx],
							self.tag_dictionary.get_item_for_index(rel_preds[sent_idx,token_idx]),
							'X',
							final_score[sent_idx,token_idx].cpu().item(),
						)
						lines.append(eval_line)
					lines.append("\n")
				
			
			if not prediction_mode:
				# punct_end=time.time()
				# punct_time+=punct_end-decode_end
				metric(arc_preds, rel_preds, self.arcs, self.rels, mask)
		if out_path is not None:
			with open(out_path, "w", encoding="utf-8") as outfile:
				outfile.write("".join(lines))
		if prediction_mode:
			return None, None
		# total_end_time=time.time()
		# print(total_start_time-total_end_time)
		# print(forward_time)
		# print(punct_time)
		# print(decode_time)
		
		loss /= len(loader)

		return loss, metric

	def decode(self, arc_scores, rel_scores, mask):
		arc_preds = arc_scores.argmax(-1)
		bad = [not istree(sequence, not self.is_mst)
			   for sequence in arc_preds.tolist()]
		if self.tree and any(bad):

			arc_preds[bad] = eisner(arc_scores[bad], mask[bad])
			# if not hasattr(self,'dist') or self.is_mst:
			#   dist = generate_tree(arc_scores,mask,is_mst=False)
			# else:
			#   dist = self.dist
			# arc_preds=get_struct_predictions(dist)
			

			# deal with masking
			# if not (arc_preds*mask == result*mask).all():
			#   pdb.set_trace()

		rel_preds = rel_scores.argmax(-1)
		rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
		# pdb.set_trace()
		return arc_preds, rel_preds, arc_scores.max(-1)[0] * mask, rel_scores.max(-1)[0].gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1) * mask
	def get_state(self,):
		return None