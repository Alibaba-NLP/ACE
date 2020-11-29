# -*- coding: utf-8 -*-

from flair.parser.modules import CHAR_LSTM, MLP, BertEmbedding, Biaffine, BiLSTM, TrilinearScorer
from flair.parser.modules.dropout import IndependentDropout, SharedDropout

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
								pad_sequence)

import pdb
import torch.nn.functional as F


class Model(nn.Module):

	def __init__(self, args):
		super(Model, self).__init__()

		self.args = args
		# the embedding layer
		self.word_embed = nn.Embedding(num_embeddings=args.n_words,
									   embedding_dim=args.n_embed)
		if args.use_char:
			self.char_embed = CHAR_LSTM(n_chars=args.n_char_feats,
										n_embed=args.n_char_embed,
										n_out=args.n_embed)
		if args.use_bert:
			self.bert_embed = BertEmbedding(model=args.bert_model,
											n_layers=args.n_bert_layers,
											n_out=args.n_embed)
		if args.use_pos:
			self.pos_embed = nn.Embedding(num_embeddings=args.n_pos_feats,
										   embedding_dim=args.n_embed)
		self.embed_dropout = IndependentDropout(p=args.embed_dropout)

		# the word-lstm layer
		self.lstm = BiLSTM(input_size=args.n_embed*(args.use_char+args.use_bert+args.use_pos+1),
						   hidden_size=args.n_lstm_hidden,
						   num_layers=args.n_lstm_layers,
						   dropout=args.lstm_dropout)
		self.lstm_dropout = SharedDropout(p=args.lstm_dropout)

		# the MLP layers
		self.mlp_arc_h = MLP(n_in=args.n_lstm_hidden*2,
							 n_hidden=args.n_mlp_arc,
							 dropout=args.mlp_dropout)
		self.mlp_arc_d = MLP(n_in=args.n_lstm_hidden*2,
							 n_hidden=args.n_mlp_arc,
							 dropout=args.mlp_dropout)
		self.mlp_rel_h = MLP(n_in=args.n_lstm_hidden*2,
							 n_hidden=args.n_mlp_rel,
							 dropout=args.mlp_dropout)
		self.mlp_rel_d = MLP(n_in=args.n_lstm_hidden*2,
							 n_hidden=args.n_mlp_rel,
							 dropout=args.mlp_dropout)

		# the Biaffine layers
		self.arc_attn = Biaffine(n_in=args.n_mlp_arc,
								 bias_x=True,
								 bias_y=False)
		self.rel_attn = Biaffine(n_in=args.n_mlp_rel,
								 n_out=args.n_rels,
								 bias_x=True,
								 bias_y=True)
		self.binary = args.binary
		# the Second Order Parts
		if self.args.use_second_order:
			self.use_sib = args.use_sib
			self.use_cop = args.use_cop
			self.use_gp = args.use_gp
			if args.use_sib:
				self.mlp_sib_h = MLP(n_in=args.n_lstm_hidden*2,
								 n_hidden=args.n_mlp_sec,
								 dropout=args.mlp_dropout)
				self.mlp_sib_d = MLP(n_in=args.n_lstm_hidden*2,
								 n_hidden=args.n_mlp_sec,
								 dropout=args.mlp_dropout)
				self.trilinear_sib = TrilinearScorer(args.n_mlp_sec,args.n_mlp_sec,args.n_mlp_sec,init_std=args.init_std, rank = args.n_mlp_sec, factorize = args.factorize)
			if args.use_cop:
				self.mlp_cop_h = MLP(n_in=args.n_lstm_hidden*2,
								 n_hidden=args.n_mlp_sec,
								 dropout=args.mlp_dropout)
				self.mlp_cop_d = MLP(n_in=args.n_lstm_hidden*2,
								 n_hidden=args.n_mlp_sec,
								 dropout=args.mlp_dropout)
				self.trilinear_cop = TrilinearScorer(args.n_mlp_sec,args.n_mlp_sec,args.n_mlp_sec,init_std=args.init_std, rank = args.n_mlp_sec, factorize = args.factorize)
			if args.use_gp:
				self.mlp_gp_h = MLP(n_in=args.n_lstm_hidden*2,
								 n_hidden=args.n_mlp_sec,
								 dropout=args.mlp_dropout)
				self.mlp_gp_d = MLP(n_in=args.n_lstm_hidden*2,
								 n_hidden=args.n_mlp_sec,
								 dropout=args.mlp_dropout)
				self.mlp_gp_hd = MLP(n_in=args.n_lstm_hidden*2,
								 n_hidden=args.n_mlp_sec,
								 dropout=args.mlp_dropout)
				self.trilinear_gp = TrilinearScorer(args.n_mlp_sec,args.n_mlp_sec,args.n_mlp_sec,init_std=args.init_std, rank = args.n_mlp_sec, factorize = args.factorize)
				
		self.pad_index = args.pad_index
		self.unk_index = args.unk_index

	def load_pretrained(self, embed=None):
		if embed is not None:
			self.pretrained = nn.Embedding.from_pretrained(embed)
			nn.init.zeros_(self.word_embed.weight)

		return self

	def forward(self, words, feats):
		batch_size, seq_len = words.shape
		# get the mask and lengths of given batch
		mask = words.ne(self.pad_index)
		lens = mask.sum(dim=1)
		# set the indices larger than num_embeddings to unk_index
		ext_mask = words.ge(self.word_embed.num_embeddings)
		ext_words = words.masked_fill(ext_mask, self.unk_index)

		# get outputs from embedding layers
		word_embed = self.word_embed(ext_words)
		if hasattr(self, 'pretrained'):
			word_embed += self.pretrained(words)
		feat_embeds=[word_embed]
		feats_index=0
		# pdb.set_trace()
		if self.args.use_char:
			input_feats=feats[feats_index]
			feats_index+=1
			char_embed = self.char_embed(input_feats[mask])
			char_embed = pad_sequence(char_embed.split(lens.tolist()), True)
			# char_embed = self.embed_dropout(char_embed)
			feat_embeds.append(char_embed)
		if self.args.use_bert:
			input_feats=feats[feats_index]
			feats_index+=1
			bert_embed = self.bert_embed(*input_feats)
			# bert_embed = self.embed_dropout(bert_embed)
			feat_embeds.append(bert_embed)
		if self.args.use_pos:
			input_feats=feats[feats_index]
			feats_index+=1
			pos_embed = self.pos_embed(input_feats)
			# pos_embed = self.embed_dropout(pos_embed)
			feat_embeds.append(pos_embed)
		feat_embeds=self.embed_dropout(*feat_embeds)
		# for i in range(len(feat_embeds)):
		# 	feat_embeds[i]=self.embed_dropout(feat_embeds[i])

		# word_embed = self.embed_dropout(word_embed)
		# concatenate the word and feat representations
		embed = torch.cat(feat_embeds, dim=-1)

		x = pack_padded_sequence(embed, lens, True, False)
		x, _ = self.lstm(x)
		x, _ = pad_packed_sequence(x, True, total_length=seq_len)
		x = self.lstm_dropout(x)

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
		if self.args.use_second_order:
			mask_unary, mask_sib, mask_cop, mask_gp = self.from_mask_to_3d_mask(mask)
			unary = mask_unary*s_arc
			arc_sib, arc_cop, arc_gp = self.encode_second_order(x)
			layer_sib, layer_cop, layer_gp = self.get_edge_second_order_node_scores(arc_sib, arc_cop, arc_gp, mask_sib, mask_cop, mask_gp)
			s_arc = self.mean_field_variational_infernece(unary, layer_sib, layer_cop, layer_gp) 
		# set the scores that exceed the length of each sentence to -inf
		s_arc.masked_fill_(~mask.unsqueeze(1), float(-1e9))

		return s_arc, s_rel

	@classmethod
	def load(cls, path):
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		state = torch.load(path, map_location=device)
		model = cls(state['args'])
		model.load_pretrained(state['pretrained'])
		model.load_state_dict(state['state_dict'], False)
		model.to(device)

		return model

	def save(self, path):
		state_dict, pretrained = self.state_dict(), None
		if hasattr(self, 'pretrained'):
			pretrained = state_dict.pop('pretrained.weight')
		state = {
			'args': self.args,
			'state_dict': state_dict,
			'pretrained': pretrained
		}
		torch.save(state, path)

	def mean_field_variational_infernece(self, unary, layer_sib=None, layer_cop=None, layer_gp=None):
		layer_gp2 = layer_gp.permute(0,2,3,1)
		# modify from (dep, head) to (head, dep), in order to fit my code
		unary = unary.transpose(1,2)
		unary_potential = unary.clone()
		q_value = unary_potential.clone()
		for i in range(self.args.iterations):
			# pdb.set_trace()
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