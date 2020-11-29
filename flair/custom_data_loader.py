import random
import torch

import pdb
from pytorch_transformers import (
	BertTokenizer,)
import re


def convert_dictionary_to_dict(char_map):
	return char_map.item2idx

class BatchedData(list):
	def __init__(self,input):
		super().__init__(input)
		self.features = {}
		self.teacher_features = {}
	pass
	# features = None
	# tags = None
	# assigned = False
	
class ColumnDataLoader:
	## adopt from stanfordnlp, modfied by Xinyu Wang for flair's ColumnDataset
	## link: https://github.com/stanfordnlp/stanfordnlp/tree/d8061501ff14c73734e834a08fa33c58c4a6d917
	def __init__(self, data, batch_size, shuffle=False, args=None,grouped_data=False,use_bert=False, tokenizer=None, sort_data = True, sentence_level_batch = False, model = None):
		self.batch_size = batch_size
		self.args = args
		self.shuffled=shuffle
		data=list(data)
		# self.word_map = word_map
		# self.char_map = char_map
		self.model = model
		# shuffle for training
		# if self.shuffled:
		#     random.shuffle(data)
		self.num_examples = len(data)
		self.grouped_data = grouped_data
		self.sentence_level_batch = sentence_level_batch
		if self.sentence_level_batch:
			if batch_size>500:
				assert 0, 'warning, batch size too large, maybe you are setting wrong batch mode'
		# chunk into batches

		self.use_bert=use_bert
		if self.use_bert:
			if tokenizer is None:
				self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
			else:
				self.tokenizer = tokenizer

		self.data = self.chunk_batches(data,sort_data=sort_data)

		# pdb.set_trace()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, key):
		""" Get a batch with index. """
		if not isinstance(key, int):
			raise TypeError
		if key < 0 or key >= len(self.data):
			raise IndexError
		batch = self.data[key]
		return batch
		
	def __iter__(self):
		for i in range(self.__len__()):
			yield self.__getitem__(i)

	def reshuffle(self):
		# data = [y for x in self.data for y in x]
		# self.data = self.chunk_batches(data)
		random.shuffle(self.data)
	def true_reshuffle(self):
		data = [y for x in self.data for y in x]
		self.data = self.chunk_batches(data)
		random.shuffle(self.data)
	def get_subtoken_length(self,sentence):
		return len(self.tokenizer.tokenize(sentence.to_tokenized_string()))
	def chunk_batches(self, data, sort_data = True):
		res = []
		# sort sentences (roughly) by length for better memory utilization
		if sort_data:
			if self.use_bert:
				# pdb.set_trace()
				if self.grouped_data:
					# pdb.set_trace()
					data = sorted(data, key = lambda x: self.get_subtoken_length(x[0]))
				else:
					data = sorted(data, key = lambda x: self.get_subtoken_length(x))
				
			else:
				if self.grouped_data:
					data = sorted(data, key = lambda x: len(x[0]))
				else:
					data = sorted(data, key = lambda x: len(x))
		# lengths = [len(x) for x in data]
		current = []
		currentlen = 0
		for x in data:
			# avoid too many sentences makes OOM

			if self.grouped_data:
				if self.use_bert:
					len_val=self.get_subtoken_length(x[0])
				else:
					len_val=len(x[0])
				# pdb.set_trace()
				# if (len(x[0]) + currentlen > self.batch_size) or len(current) > self.batch_size/10.0:
				if (len_val + currentlen > self.batch_size):
					res.append(current)
					current = []
					currentlen = 0
			elif self.sentence_level_batch:
				if len(current) >= self.batch_size:
					res.append(current)
					current = []
					currentlen = 0
			else:
				if self.use_bert:
					len_val=self.get_subtoken_length(x)
				else:
					len_val=len(x)
				# if (len(x) + currentlen > self.batch_size) or len(current) > self.batch_size/10.0:
				if (len_val + currentlen > self.batch_size):
					res.append(current)
					current = []
					currentlen = 0
			current.append(x)
			if self.grouped_data:
				if self.use_bert:
					len_val=self.get_subtoken_length(x[0])
				else:
					len_val=len(x[0])
				currentlen += len_val
			else:
				if self.use_bert:
					len_val=self.get_subtoken_length(x)
				else:
					len_val=len(x)
				currentlen += len_val

		if currentlen > 0:
			res.append(current)
		return res
	def assign_embeddings(self):
		input_data=self.data
		for batch_no, batch in enumerate(input_data):
			max_len=-1
			max_char_len = []
			for sentence in batch:
				if len(sentence)>max_len:
					max_len=len(sentence)
				for embedding in self.model.embeddings.embeddings:
					if 'Char' in embedding.name:
						max_char_len.append(max([len(w.text) for w in sentence]))
			batch = BatchedData(batch)
			for embedding in self.model.embeddings.embeddings:
				if 'Word:' in embedding.name:
					word_tensor = torch.zeros([len(batch),max_len],device='cpu').long()	
				if 'lemma' in embedding.name:
					lemma_tensor = torch.zeros([len(batch),max_len],device='cpu').long()	
				if 'pos' == embedding.name:
					pos_tensor = torch.zeros([len(batch),max_len],device='cpu').long()	
				if 'Char' in embedding.name:
					char_tensor = torch.zeros([len(batch),max_len,max(max_char_len)],device='cpu').long()
					char_length_tensor = torch.ones([len(batch),max_len],device='cpu').long()
				for s_id, sentence in enumerate(batch):
					if 'Word:' in embedding.name:
						words=self._get_word_id(embedding.vocab, sentence)
						word_tensor[s_id][:len(sentence)]=words
					if 'Char' in embedding.name:
						chars, char_lens=self._get_char_idx(embedding.char_dictionary, sentence)
						char_tensor[s_id][:len(sentence),:chars.shape[0]] = chars.transpose(0,1)
						char_length_tensor[s_id][:len(sentence)]=char_lens
					if 'lemma' in embedding.name:
						lemmas=self._get_word_id(embedding.lemma_dictionary, sentence, attr = 'lemma')
						lemma_tensor[s_id][:len(sentence)]=lemmas
					if 'pos' == embedding.name:
						poses=self._get_word_id(embedding.pos_dictionary, sentence, attr = 'pos')
						pos_tensor[s_id][:len(sentence)]=poses
				if 'Word:' in embedding.name:
					setattr(batch,embedding.name+'words',word_tensor)
				if 'lemma' in embedding.name:
					setattr(batch,embedding.name,lemma_tensor)
				if 'pos' == embedding.name:
					setattr(batch,embedding.name,pos_tensor)
				if 'Char' in embedding.name:
					# (char_size, batch*word)
					setattr(batch,'char_seqs',char_tensor.reshape(-1,char_tensor.shape[-1]).transpose(1,0))
					# (batch*word)
					setattr(batch,'char_lengths',char_length_tensor.reshape(-1))
					setattr(batch,'max_sent_len',max_len)
			input_data[batch_no]=batch
	def assign_tags(self,tag_type,tag_dictionary,teacher_input=None,grouped_data=False):
		if teacher_input is not None:
			input_data=[teacher_input]
		else:
			input_data=self.data
		for batch_no, batch in enumerate(input_data):
			tag_list: List = []
			max_len=-1
			max_char_len = []
			for sentence in batch:
				if grouped_data:
					sentence = sentence[1]
				if len(sentence)>max_len:
					max_len=len(sentence)
				for embedding in self.model.embeddings.embeddings:
					if 'Char' in embedding.name:
						max_char_len.append(max([len(w.text) for w in sentence]))

			batch = BatchedData(batch)
			for embedding in self.model.embeddings.embeddings:
				if 'Word:' in embedding.name:
					word_tensor = torch.zeros([len(batch),max_len],device='cpu').long()	
				if 'lemma' in embedding.name:
					lemma_tensor = torch.zeros([len(batch),max_len],device='cpu').long()	
				if 'pos' == embedding.name:
					pos_tensor = torch.zeros([len(batch),max_len],device='cpu').long()	
				if 'Char' in embedding.name:
					char_tensor = torch.zeros([len(batch),max_len,max(max_char_len)],device='cpu').long()
					char_length_tensor = torch.ones([len(batch),max_len],device='cpu').long()
				for s_id, sentence in enumerate(batch):
					if 'Word:' in embedding.name:
						words=self._get_word_id(embedding.vocab, sentence)
						word_tensor[s_id][:len(sentence)]=words
					if 'lemma' in embedding.name:
						lemmas=self._get_word_id(embedding.lemma_dictionary, sentence, attr = 'lemma')
						lemma_tensor[s_id][:len(sentence)]=lemmas
					if 'pos' == embedding.name:
						poses=self._get_word_id(embedding.pos_dictionary, sentence, attr = 'pos')
						pos_tensor[s_id][:len(sentence)]=poses
					if 'Char' in embedding.name:
						chars, char_lens=self._get_char_idx(embedding.char_dictionary, sentence)
						char_tensor[s_id][:len(sentence),:chars.shape[0]] = chars.transpose(0,1)
						char_length_tensor[s_id][:len(sentence)]=char_lens
				if 'Word:' in embedding.name:
					setattr(batch,embedding.name+'words',word_tensor)
				if 'lemma' in embedding.name:
					setattr(batch,embedding.name,lemma_tensor)
				if 'pos' == embedding.name:
					setattr(batch,embedding.name,pos_tensor)
				if 'Char' in embedding.name:
					# (char_size, batch*word)
					setattr(batch,'char_seqs',char_tensor.reshape(-1,char_tensor.shape[-1]).transpose(1,0))
					# (batch*word)
					setattr(batch,'char_lengths',char_length_tensor.reshape(-1))
					setattr(batch,'max_sent_len',max_len)
			for s_id, sentence in enumerate(batch):
				# get the tags in this sentence
				# pdb.set_trace()
				if hasattr(sentence[0],'system_preds'):
					system_preds=[x.system_preds for x in sentence]
					system_scores=[x.system_scores for x in sentence]
					num_candiates = len(system_preds[0])
					for val in system_preds:
						assert num_candiates == len(val)
					if tag_type == 'enhancedud' or tag_type == 'srl':
						pdb.set_trace()
					elif tag_type == 'dependency':
						pdb.set_trace()
					else:
						# pdb.set_trace()
						gold_preds = [token.get_tag(tag_type).value for token in sentence]
						tag_template = torch.zeros(max_len, num_candiates,device='cpu')
						score_template = torch.zeros(max_len, num_candiates,device='cpu')
						for token_id, system_pred in enumerate(system_preds):
							score_template[token_id] = torch.Tensor(system_scores[token_id]).type_as(score_template)
							for sys_id, pred in enumerate(system_pred):
								if pred == gold_preds[token_id]:
									tag_template[token_id][sys_id] = 1
								else:
									tag_template[token_id][sys_id] = 0
						
						setattr(sentence,tag_type+'_system_preds',tag_template)
						setattr(sentence,tag_type+'_system_scores',score_template)
					# pdb.set_trace()


				if tag_type=='enhancedud' or tag_type=='srl':
					relations=[token.get_tag(tag_type).value.split('|') for token in sentence]

					arc_template = torch.zeros([max_len,max_len],device='cpu',dtype=torch.int32)
					rel_template = torch.zeros([max_len,max_len],device='cpu',dtype=torch.int32)
					for index, relation_group in enumerate(relations):
						if index==0:
							continue
						for head_rel in relation_group:
							if head_rel == '_':
								continue
							headid = int(head_rel.split(':')[0])
							relid = tag_dictionary.get_idx_for_item(':'.join(head_rel.split(':')[1:]))
							arc_template[index,headid] = 1
							rel_template[index,headid] = relid
					# for rel in relations:
					#     if len(rel)>1:
					#         pdb.set_trace()
					setattr(sentence,tag_type+'_arc_tags',arc_template)
					setattr(sentence,tag_type+'_rel_tags',rel_template)
				elif tag_type=='dependency':
					arcs: List[int] = [token.head_id for token in sentence]
					rels: List[int] = [tag_dictionary.get_idx_for_item(token.get_tag(tag_type).value) for token in sentence]
					# add tags as tensor
					arc_template = torch.zeros(max_len,device='cpu')
					arcs = torch.tensor(arcs, device='cpu')
					arc_template[:len(sentence)]=arcs
					rel_template = torch.zeros(max_len,device='cpu')
					rels = torch.tensor(rels, device='cpu')
					rel_template[:len(sentence)]=rels

					setattr(sentence,tag_type+'_arc_tags',arc_template)
					setattr(sentence,tag_type+'_rel_tags',rel_template)
				elif tag_type == 'ner_dp':
					# if len(sentence)==2:
					# pdb.set_trace()
		
					arc_template = torch.zeros([max_len,max_len],device='cpu')
					rel_template = torch.zeros([max_len,max_len],device='cpu')*tag_dictionary.item2idx[str('None').encode('utf-8')]
					
					sent_length = len(sentence.tokens)

					#set 'None' label target.
					rel_template[:sent_length, :sent_length] = tag_dictionary.item2idx[str('None').encode('utf-8')]

					arc_template = torch.tril(arc_template, diagonal=0)
					rel_template = torch.tril(rel_template, diagonal=0)

					# pdb.set_trace()
					#from sentence span info get gold ner matrix representation. template entry is ner label index.
					spanlist = sentence.get_spans(tag_type='ner_dp')

					for span in spanlist:
						try:
							tag_idx = tag_dictionary.item2idx[str(span.tag).encode('utf-8')]
						except:
							pdb.set_trace()
						n_tokens = len(span.tokens)
						if n_tokens > 1:
							start_idx = span.tokens[0].idx
							end_idx = span.tokens[-1].idx
						else:
							start_idx = end_idx = span.tokens[0].idx
						try:
							arc_template[end_idx-1, start_idx-1] = 1
							rel_template[end_idx-1, start_idx-1] = tag_idx
						except:
							pdb.set_trace()
					# pdb.set_trace()
					setattr(sentence,tag_type+'_arc_tags',arc_template)
					setattr(sentence,tag_type+'_rel_tags',rel_template)
				else:
					tag_idx: List[int] = [
						tag_dictionary.get_idx_for_item(token.get_tag(tag_type).value)
						for token in sentence
					]
					# add tags as tensor
					tag_template = torch.zeros(max_len,device='cpu')
					tag = torch.tensor(tag_idx, device='cpu')
					tag_template[:len(sentence)]=tag
					setattr(sentence,tag_type+'_tags',tag_template)
			# pdb.set_trace()
			if tag_type=='enhancedud' or tag_type=='dependency' or tag_type=='srl' or tag_type=='ner_dp':
				arc_tags=torch.stack([getattr(sentence,tag_type+'_arc_tags') for sentence in batch],0)
				rel_tags=torch.stack([getattr(sentence,tag_type+'_rel_tags') for sentence in batch],0)
				setattr(batch,tag_type+'_arc_tags',arc_tags)
				setattr(batch,tag_type+'_rel_tags',rel_tags)
			else:
				tag_list=torch.stack([getattr(sentence,tag_type+'_tags') for sentence in batch],0).long()
				setattr(batch,tag_type+'_tags',tag_list)
			if teacher_input is None:
				self.data[batch_no]=batch
			else:
				input_data[batch_no]=batch
		if teacher_input is not None:
			return input_data
		else:
			return


	def expand_teacher_predictions(self):
		'''
		expand teacher prection to batch size
		'''
		for batch in self.data:
			tag_list: List = []
			# pdb.set_trace()
			max_len=-1
			for sentence in batch:
				if len(sentence)>max_len:
					max_len=len(sentence)

	def _get_word_id(self, word_map, sent, attr = 'text'):
		word_idx = []
		keys = word_map.keys()
		for word in sent:
			word = getattr(word,attr)
			if word in keys:
				word_idx.append(word_map[word])
			elif word.lower() in keys:
				word_idx.append(word_map[word.lower()])
			elif re.sub(r"\d", "#", word.lower()) in keys:
				word_idx.append(
					word_map[re.sub(r"\d", "#", word.lower())]
				)
			elif re.sub(r"\d", "0", word.lower()) in keys:
				word_idx.append(
					word_map[re.sub(r"\d", "0", word.lower())]
				)
			else: word_idx.append(word_map['unk'])
			# word_idx.append(word_id)
		return torch.LongTensor(word_idx)

	def _get_char_idx(self, char_map, sent):
		if type(char_map)!=type({}):
			char_map=convert_dictionary_to_dict(char_map)
			unk=b'<unk>'
		else:
			unk='<u>'
		max_length = max([len(w.text) for w in sent])
		char_lens = []
		char_idxs = []
		for word in sent:
			# pdb.set_trace()
			c_id = [char_map.get(char, char_map[unk]) for char in word.text]
			char_lens.append(len(c_id))
			c_id += [char_map[unk]] * (max_length - len(c_id))
			char_idxs.append(c_id)
		return torch.LongTensor(char_idxs).transpose(0, 1), torch.LongTensor(char_lens)
