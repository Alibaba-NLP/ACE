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

from flair.training_utils import Metric, Result, store_embeddings
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
import pdb
import copy
import time

import sys
# sys.path.insert(0,'/home/wangxy/workspace/flair/parser')
# sys.path.append('./flair/parser/modules')

# from flair.parser.utils.fn import ispunct
import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
								pad_sequence)


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


class EnsembleModel(flair.nn.Model):
	def __init__(
		self,
		embeddings: TokenEmbeddings,
		tag_dictionary: Dictionary,
		tag_type: str,
		hidden_size: int = 256,
		use_crf: bool = False,
		use_rnn: bool = False,
		train_initial_hidden_state: bool = False,
		rnn_layers: int = 3,
		lstm_dropout: float = 0.0,
		dropout: float = 0.0,
		word_dropout: float = 0.0,
		locked_dropout: float = 0.0,
		pickle_module: str = "pickle",
		config = None,
		use_decoder_timer = True,
		debug = False,
		word_map = None,
		char_map = None,
		relearn_embeddings = False,
		testing = False,
		candidates = -1,
		target_languages = -1,
		binary = False,
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

		super(EnsembleModel, self).__init__()
		self.debug = False
		self.use_crf = use_crf
		self.use_rnn = use_rnn
		self.hidden_size = hidden_size
		self.embeddings = embeddings
		self.config = config
		self.binary = binary

		self.rnn_layers: int = rnn_layers
		# set the dictionaries
		self.tag_dictionary: Dictionary = tag_dictionary
		self.tag_type: str = tag_type
		self.tagset_size: int = len(tag_dictionary)

		self.word_map = word_map
		self.char_map = char_map

		# initialize the network architecture
		self.nlayers: int = rnn_layers
		self.hidden_word = None
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

		if self.relearn_embeddings:
			self.embedding2nn = torch.nn.Linear(rnn_input_dim + candidates, rnn_input_dim + candidates)
		if candidates == -1:
			pdb.set_trace()
		self.candidates = candidates
		self.hidden2score = torch.nn.Linear(rnn_input_dim + candidates, candidates)

		self.bidirectional = True
		self.rnn_type = "LSTM"
		if not self.use_rnn:
			self.bidirectional = False
		# bidirectional LSTM on top of embedding layer
		num_directions = 1

		if self.use_rnn:
			self.rnn = BiLSTM(input_size=rnn_input_dim,
							   hidden_size=hidden_size,
							   num_layers=self.nlayers,
							   dropout=self.lstm_dropout)
			self.lstm_dropout_func = SharedDropout(p=self.lstm_dropout)

			mlp_input_hidden = hidden_size * 2
		else:
			mlp_input_hidden = rnn_input_dim

		
		# self.criterion = nn.CrossEntropyLoss()
		self.criterion = nn.BCEWithLogitsLoss(reduction='none')
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
		use_cnn=state["use_cnn"] if 'use_cnn' in state else False
		model = EnsembleModel(
			hidden_size=state["hidden_size"],
			embeddings=state["embeddings"],
			tag_dictionary=state["tag_dictionary"],
			tag_type=state["tag_type"],
			use_crf=state["use_crf"],
			use_rnn=state["use_rnn"],
			rnn_layers=state["rnn_layers"],
			dropout=use_dropout,
			word_dropout=use_word_dropout,
			locked_dropout=use_locked_dropout,
			config=state['config'] if "config" in state else None,
			word_map=None if 'word_map' not in state else state['word_map'],
			char_map=None if 'char_map' not in state else state['char_map'],
			relearn_embeddings = True if 'relearn_embeddings' not in state else state['relearn_embeddings'],
			testing = testing,
			candidates = state['candidates']
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
			"use_crf": self.use_crf,
			"use_rnn":self.use_rnn,
			"rnn_layers": self.rnn_layers,
			"dropout": self.use_dropout,
			"word_dropout": self.use_word_dropout,
			"locked_dropout": self.use_locked_dropout,
			"config": self.config,
			"word_map": self.word_map,
			"char_map": self.char_map,
			"relearn_embeddings": self.relearn_embeddings,
			"candidates": self.candidates,
		}
		return model_state
	def forward(self, sentences: List[Sentence], prediction_mode = False):
		# self.zero_grad()
		# pdb.set_trace()
		lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

		longest_token_sequence_in_batch: int = max(lengths)

		self.embeddings.embed(sentences)
		sentence_tensor = torch.cat([sentences.features[x].to(flair.device) for x in sorted(sentences.features.keys())],-1)
		sentence_tensor = torch.zeros_like(sentence_tensor)
		system_preds=torch.stack([getattr(sentence,self.tag_type+'_system_scores').to(flair.device) for sentence in sentences],0).float()
		sentence_tensor = torch.cat([sentence_tensor,system_preds],-1)

		# sentence_tensor=torch.stack([getattr(sentence,self.tag_type+'_system_scores').to(flair.device) for sentence in sentences],0).float()
		# if self.use_dropout > 0.0:
		#   sentence_tensor = self.dropout(sentence_tensor)
		if self.use_word_dropout > 0.0:
		  sentence_tensor = self.word_dropout(sentence_tensor)
		# if self.use_locked_dropout > 0.0:
		#   sentence_tensor = self.locked_dropout(sentence_tensor)

		if self.relearn_embeddings:
			sentence_tensor = self.embedding2nn(sentence_tensor)
			# sentence_tensor = self.embedding2nn(sentence_tensor)

		# get the mask and lengths of given batch
		mask=self.sequence_mask(torch.tensor(lengths),longest_token_sequence_in_batch).cuda().type_as(sentence_tensor)
		self.mask=mask
		# convert hidden state to score of each embedding candidates
		scores = self.hidden2score(sentence_tensor)
		return scores

	def forward_loss(
		self, data_points: Union[List[Sentence], Sentence], sort=True
	) -> torch.tensor:
		scores = self.forward(data_points)
		# lengths = [len(sentence.tokens) for sentence in data_points]
		# longest_token_sequence_in_batch: int = max(lengths)

		# max_len = features.shape[1]
		# mask=self.sequence_mask(torch.tensor(lengths), max_len).cuda().type_as(features)
		loss = self._calculate_loss(scores, data_points, self.mask)
		return loss


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

	def _calculate_loss(
		self, scores: torch.tensor, sentences: List[Sentence], mask: torch.tensor, return_arc_rel = False,
	) -> float:

		if self.binary:
			pass
		else:
			# the system preds represents whether the tag is correct
			if hasattr(sentences,self.tag_type+'_system_preds'):
				system_preds=getattr(sentences,self.tag_type+'_system_preds').to(flair.device).long()
			else:
				system_preds=torch.stack([getattr(sentence,self.tag_type+'_system_preds').to(flair.device) for sentence in sentences],0).long()
			
			mask = mask.bool()
			
		loss = self.criterion(scores, system_preds.float()) * mask.unsqueeze(-1)
		loss = loss.sum()/mask.sum()
		# bce_loss = -(torch.log(torch.sigmoid(scores)) * system_preds + torch.log(1-torch.sigmoid(scores)) * (1-system_preds))
		# loss = 2 * ((1-self.interpolation) * arc_loss + self.interpolation * rel_loss)


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
		eval_loss = 0
		batch_no = 0
		data_loader.assign_embeddings()
		if out_path is not None:
			outfile = open(out_path, "w", encoding="utf-8")
		if not self.binary:
			metric = Metric("Evaluation")
		with torch.no_grad():
			for batch in data_loader:
				batch_no+=1
				scores = self.forward(batch, prediction_mode=prediction_mode)
				loss = self._calculate_loss(scores, batch, self.mask)
				eval_loss += loss
				if self.binary:
					pdb.set_trace()
					result = Result(
						main_score=LF1,
						log_line=f"\nUF1: {UF1} - LF1 {LF1}",
						log_header="PRECISION\tRECALL\tF1",
						detailed_results=f"\nUF1: {UF1} - LF1 {LF1}",
					)
				else:
					# if prediction_mode:
					#   eval_loss, metric=self.dependency_evaluate(data_loader,out_path=out_path,prediction_mode=prediction_mode)
					#   return eval_loss, metric
					# else:   

					tags, _ = self._obtain_labels(scores, batch)
					for (sentence, sent_tags) in zip(batch, tags):
						for (token, tag) in zip(sentence.tokens, sent_tags):
							token: Token = token
							token.add_tag_label("predicted", tag)

							# append both to file for evaluation
							eval_line = "{} {} {} {}\n".format(
								token.text,
								token.get_tag(self.tag_type).value,
								tag.value,
								tag.score,
							)
							# lines.append(eval_line)
							if out_path is not None:
								outfile.write(eval_line)
						# lines.append("\n")
						if out_path is not None:
							outfile.write("\n")
					for sentence in batch:
						# make list of gold tags
						gold_tags = [
							(tag.tag, str(tag)) for tag in sentence.get_spans(self.tag_type)
						]
						# make list of predicted tags
						predicted_tags = [
							(tag.tag, str(tag)) for tag in sentence.get_spans("predicted")
						]

						# check for true positives, false positives and false negatives
						for tag, prediction in predicted_tags:
							if (tag, prediction) in gold_tags:
								metric.add_tp(tag)
							else:
								metric.add_fp(tag)

						for tag, gold in gold_tags:
							if (tag, gold) not in predicted_tags:
								metric.add_fn(tag)
							else:
								metric.add_tn(tag)
		eval_loss /= batch_no
		if out_path is not None:
			outfile.close()
		detailed_result = (
			f"\nMICRO_AVG: acc {metric.micro_avg_accuracy()} - f1-score {metric.micro_avg_f_score()}"
			f"\nMACRO_AVG: acc {metric.macro_avg_accuracy()} - f1-score {metric.macro_avg_f_score()}"
		)
		for class_name in metric.get_classes():
			detailed_result += (
				f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
				f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "
				f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
				f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
				f"{metric.f_score(class_name):.4f}"
			)

		result = Result(
			main_score=metric.micro_avg_f_score(),
			log_line=f"{metric.precision()}\t{metric.recall()}\t{metric.micro_avg_f_score()}",
			log_header="PRECISION\tRECALL\tF1",
			detailed_results=detailed_result,
		)
		return result, eval_loss

	def _obtain_labels(
		self, system_preds, sentences, get_all_tags: bool = False
	) -> (List[List[Label]], List[List[List[Label]]]):
		"""
		Returns a tuple of two lists:
		 - The first list corresponds to the most likely `Label` per token in each sentence.
		 - The second list contains a probability distribution over all `Labels` for each token
		   in a sentence for all sentences.
		"""
		lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
		tags = []
		all_tags = []
		feature = system_preds.argmax(-1)
		confidences = system_preds.softmax(-1)
		for i, vals in enumerate(zip(feature, lengths)):
			feats, length=vals
			
	
			tag_list = [Label(token.system_preds[feats[token_id]], confidences[i][token_id][feats[token_id]]) for token_id, token in enumerate(sentences[i])]
			tags.append(tag_list.copy())

			if get_all_tags:
				pdb.set_trace()
				all_tags.append(
					[
						[
							Label(
								self.tag_dictionary.get_item_for_index(score_id), score
							)
							for score_id, score in enumerate(score_dist)
						]
						for score_dist in scores
					]
				)

		return tags, all_tags


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
			arc_preds, rel_preds = self.decode(arc_scores, rel_scores, mask)
			# decode_end=time.time()
			# decode_time+=decode_end-decode_start
			# ignore all punctuation if not specified
			# if out_path is not None:
			#   pdb.set_trace()
			if not self.punct:
				for sent_id,sentence in enumerate(batch):
					for token_id, token in enumerate(sentence):
						upos=token.get_tag('upos').value
						xpos=token.get_tag('pos').value
						word=token.text
						if is_punctuation(word,upos,self.punct_list) or is_punctuation(word,upos,self.punct_list):
							mask[sent_id][token_id]=0
				# mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
			if out_path is not None:
				for (sent_idx, sentence) in enumerate(batch):
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
							'X',
							arc_preds[sent_idx,token_idx],
							self.tag_dictionary.get_item_for_index(rel_preds[sent_idx,token_idx]),
							'X',
							'X',
						)
						lines.append(eval_line)
					lines.append("\n")
				
			
			if not prediction_mode:
				# punct_end=time.time()
				# punct_time+=punct_end-decode_end
				metric(arc_preds, rel_preds, self.arcs, self.rels, mask)
		# if out_path is not None:
		#   with open(out_path, "w", encoding="utf-8") as outfile:
		#       outfile.write("".join(lines))
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

		return arc_preds, rel_preds
	def get_state(self,):
		return None