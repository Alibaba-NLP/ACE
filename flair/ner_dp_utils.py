from __future__ import absolute_import
from __future__ import division

import os, io
import errno
import codecs
import collections
import shutil
import sys
import json

import math
import numpy as np 
import torch
import torch.nn as nn
import pyhocon
# import tensorflow.compat.v1 as tf



import pdb

def initialize_from_env():
  name = sys.argv[1]
  print("Running experiment: {}".format(name))

  config = pyhocon.ConfigFactory.parse_file("experiments.conf")[name]
  config["log_dir"] = mkdirs(os.path.join(config["log_root"], name))

  print(pyhocon.HOCONConverter.convert(config, "hocon"))
  return config

def flatten(l):
  return [item for sublist in l for item in sublist]

def get_shape(t):
	return list(t.shape)

def mkdirs(path):
  try:
	  os.makedirs(path)
  except OSError as exception:
	  if exception.errno != errno.EEXIST:
		  raise
  return path

# def make_summary(value_dict):
#   return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k,v in value_dict.items()])

def copy_checkpoint(source, target):
  shutil.copyfile(source, target)

def save_model(step, model, optimizer, loss, config, save_path):
	state_dict = {
				'step':step,
				'model':model.state_dict(),
				# 'optimizer': optimizer.state_dict(),
				'loss': loss,
				'config': config,
				}
	torch.save(state_dict, save_path)
	print('model saved in {}'.format(save_path))

def load_model(self, save_path):
	checkpoint = torch.load(save_path)
	return checkpoint

def load_char_dict(char_vocab_path):
  vocab = [u"<unk>"]
  with codecs.open(char_vocab_path, encoding="utf-8") as f:
    vocab.extend(l.strip() for l in f.readlines())
  char_dict = collections.defaultdict(int)
  char_dict.update({c:i for i, c in enumerate(vocab)})
  return char_dict

class ScheduledOptim():
	'''A simple wrapper class for learning rate scheduling'''

	def __init__(self, config, optimizer):
		self._optimizer = optimizer
		# self.n_warmup_steps = n_warmup_steps
		# self.n_current_steps = 0
		self.init_lr = config['learning_rate']
		self.lr = self.init_lr
		self.decay_rate = config['decay_rate']
		self.decay_frequency = config['decay_frequency']

	def step_and_update_lr(self, global_step):
		"Step with the inner optimizer"
		self._update_learning_rate(global_step)
		self._optimizer.step()

	def zero_grad(self):
		"Zero out the gradients by the inner optimizer"
		self._optimizer.zero_grad()

	def _get_lr_scale(self, global_step, stairstep=True):
		# return np.min([
		# 	np.power(self.n_current_steps, -0.5),
		# 	np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])
		if stairstep:
			power = global_step//self.decay_frequency
		else:
			power = global_step/self.decay_frequency
		return math.pow(self.decay_rate, power)

	def _update_learning_rate(self, global_step):
		''' Learning rate scheduling per step '''

		self.lr = self.init_lr * self._get_lr_scale(global_step)

		for param_group in self._optimizer.param_groups:
			param_group['lr'] = self.lr
		






class EmbeddingDictionary(object):
	def __init__(self, info, normalize=True, maybe_cache=None):
		self._size = info["size"]
		self._normalize = normalize
		self._path = info["path"]
		if maybe_cache is not None and maybe_cache._path == self._path:
			assert self._size == maybe_cache._size
			self._embeddings = maybe_cache._embeddings
		else:
			self._embeddings = self.load_embedding_dict(self._path)

	@property
	def size(self):
		return self._size

	def load_embedding_dict(self, path):
		print("Loading word embeddings from {}...".format(path))
		default_embedding = np.zeros(self.size)
		embedding_dict = collections.defaultdict(lambda:default_embedding)
		if len(path) > 0:
			vocab_size = None
			with io.open(path,encoding="utf8") as f:
				for i, line in enumerate(f.readlines()):
					if i == 0 and line.count(" ") == 1:  # header row
						continue
					word_end = line.find(" ")
					word = line[:word_end]
					embedding = np.fromstring(line[word_end + 1:], np.float32, sep=" ")
					assert len(embedding) == self.size
					embedding_dict[word] = embedding
			if vocab_size is not None:
				assert vocab_size == len(embedding_dict)
			print("Done loading word embeddings.")
		return embedding_dict

	def is_in_embeddings(self, key):
		return key in self._embeddings
		#return self._embeddings.has_key(key)

	def __getitem__(self, key):
		embedding = self._embeddings[key]
		if self._normalize:
			embedding = self.normalize(embedding)
		return embedding

	def normalize(self, v):
		norm = np.linalg.norm(v)
		if norm > 0:
			return v / norm
		else:
			return v

class prepared_dataloader():
	def __init__(self, config, datatype='train'):
		self.config = config

		self.device = torch.device('cuda' if config['device'] else 'cpu')
		pathname = datatype + '_path'
		self.data_path = config[pathname]
		self.context_embeddings = EmbeddingDictionary(config["context_embeddings"])
		self.context_embeddings_size = self.context_embeddings.size
		if datatype=='train':
			self.is_training = True
		else:
			self.is_training = False
		if self.data_path:
			with open(self.config[pathname]) as f:
				self.examples = [json.loads(jsonline) for jsonline in f.readlines()]
		else:
			pdb.set_trace()

		self.ner_types = self.config['ner_types']
		self.ner_maps = {ner: (i + 1) for i, ner in enumerate(self.ner_types)}
		self.char_dict = load_char_dict(config["char_vocab_path"])
	@property
	def batches(self):
		if self.examples:
			batches = []
			for example in self.examples:
				batch = (self.tensorize_example(example), example)
				batches.append(batch)
		else:
			pdb.set_trace()
		return batches
	@property
	def model_sizes(self):
		return [self.context_embeddings_size]


	def tensorize_example(self, example):
		ners = example["ners"]
		sentences = example["sentences"]

		max_sentence_length = max(len(s) for s in sentences)
		max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.config["filter_widths"]))
		text_len = torch.tensor([len(s) for s in sentences], device=self.device, dtype=torch.float32)
		tokens = [[""] * max_sentence_length for _ in sentences]
		char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
		context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings_size])
		lemmas = []
		if "lemmas" in example:
			lemmas = example["lemmas"]
		for i, sentence in enumerate(sentences):
			for j, word in enumerate(sentence):
				# pdb.set_trace()
				tokens[i][j] = word
				if self.context_embeddings.is_in_embeddings(word):
					context_word_emb[i, j] = self.context_embeddings[word]
				elif lemmas and self.context_embeddings.is_in_embeddings(lemmas[i][j]):
					context_word_emb[i,j] = self.context_embeddings[lemmas[i][j]]
				char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]
		context_word_emb = torch.tensor(context_word_emb, device=self.device, dtype=torch.float32)
		char_index = torch.tensor(char_index, device=self.device, dtype=torch.int64)
		tokens = np.array(tokens)

		# doc_key = example["doc_key"]

		# lm_emb = self.load_lm_embeddings(doc_key)

		gold_labels = []
		if self.is_training:
			for sid, sent in enumerate(sentences):
				ner = {(s,e):self.ner_maps[t] for s,e,t in ners[sid]}
				for s in range(len(sent)):
					for e in range(s,len(sent)):
						gold_labels.append(ner.get((s,e),0))    # 0 for 'None'
		gold_labels = torch.tensor(gold_labels, device=self.device, dtype=torch.int64)
	

		
		example_tensors = [tokens, context_word_emb, char_index, text_len, gold_labels]
		# for idx, array in enumerate(example_tensors):
		# 	try:
		# 		example_tensors[idx] = torch.tensor(array, device=self.device)
		# 	except:
		# 		example_tensors[idx] = array

		return example_tensors

class train_dataloader(prepared_dataloader):
	def __init__(self, config):
		super(train_dataloader, self).__init__(config=config, datatype='train')

class eval_dataloader(prepared_dataloader):
	def __init__(self, config):
		super(eval_dataloader, self).__init__(config=config, datatype='eval')
class test_dataloader(prepared_dataloader):
	def __init__(self, config):
		super(test_dataloader, self).__init__(config=config, datatype='test')

