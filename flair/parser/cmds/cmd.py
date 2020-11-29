# -*- coding: utf-8 -*-

import os
from parser.utils import Embedding
from parser.utils.alg import eisner
from parser.utils.common import bos, pad, unk
from parser.utils.corpus import CoNLL, Corpus
from parser.utils.field import BertField, CharField, Field
from parser.utils.fn import ispunct
from parser.utils.metric import Metric

import torch
import torch.nn as nn
from transformers import BertTokenizer

import pdb
class CMD(object):

    def __call__(self, args):
        self.args = args
        if not hasattr(self.args, 'interpolation'):
            self.args.interpolation = 0.5
        if not os.path.exists(args.file):
            os.mkdir(args.file)
        if not os.path.exists(args.fields) or args.preprocess:
            print("Preprocess the data")
            self.WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
            # if args.feat == 'char':
            #     self.FEAT = CharField('chars', pad=pad, unk=unk, bos=bos,
            #                           fix_len=args.fix_len, tokenize=list)
            # elif args.feat == 'bert':
            #     tokenizer = BertTokenizer.from_pretrained(args.bert_model)
            #     self.FEAT = BertField('bert', pad='[PAD]', bos='[CLS]',
            #                           tokenize=tokenizer.encode)
            # else:
            #     self.FEAT = Field('tags', bos=bos)

            

            self.CHAR_FEAT=None
            self.POS_FEAT=None
            self.BERT_FEAT=None
            self.FEAT=[self.WORD]
            if args.use_char:
                self.CHAR_FEAT = CharField('chars', pad=pad, unk=unk, bos=bos,
                                      fix_len=args.fix_len, tokenize=list)
                self.FEAT.append(self.CHAR_FEAT)
            if args.use_pos:
                self.POS_FEAT = Field('tags', bos=bos)
            if args.use_bert:
                tokenizer = BertTokenizer.from_pretrained(args.bert_model)
                self.BERT_FEAT = BertField('bert', pad='[PAD]', bos='[CLS]',
                                      tokenize=tokenizer.encode)
                self.FEAT.append(self.BERT_FEAT)

            self.HEAD = Field('heads', bos=bos, use_vocab=False, fn=int)
            self.REL = Field('rels', bos=bos)

            self.fields = CoNLL(FORM=self.FEAT, CPOS=self.POS_FEAT, HEAD=self.HEAD, DEPREL=self.REL)
            # if args.feat in ('char', 'bert'):
            #     self.fields = CoNLL(FORM=(self.WORD, self.FEAT),
            #                         HEAD=self.HEAD, DEPREL=self.REL)
            # else:
            #     self.fields = CoNLL(FORM=self.WORD, CPOS=self.FEAT,
            #                         HEAD=self.HEAD, DEPREL=self.REL)

            train = Corpus.load(args.ftrain, self.fields)
            if args.fembed:
                embed = Embedding.load(args.fembed, args.unk)
            else:
                embed = None
            self.WORD.build(train, args.min_freq, embed)
            if args.use_char:
                self.CHAR_FEAT.build(train)
            if args.use_pos:
                self.POS_FEAT.build(train)
            if args.use_bert:
                self.BERT_FEAT.build(train)
            # self.FEAT.build(train)
            self.REL.build(train)
            torch.save(self.fields, args.fields)
        else:
            self.fields = torch.load(args.fields)
            if args.feat in ('char', 'bert'):
                self.WORD, self.FEAT = self.fields.FORM
            else:
                self.WORD, self.FEAT = self.fields.FORM, self.fields.CPOS
            self.HEAD, self.REL = self.fields.HEAD, self.fields.DEPREL
        self.puncts = torch.tensor([i for s, i in self.WORD.vocab.stoi.items()
                                    if ispunct(s)]).to(args.device)
        self.rel_criterion = nn.CrossEntropyLoss()
        self.arc_criterion = nn.CrossEntropyLoss()
        if args.binary:
            self.arc_criterion = nn.BCEWithLogitsLoss(reduction='none')

        # print(f"{self.WORD}\n{self.FEAT}\n{self.HEAD}\n{self.REL}")
        print(f"{self.WORD}\n{self.HEAD}\n{self.REL}")
        update_info={}
        # pdb.set_trace()
        if args.use_char:
            update_info['n_char_feats']=len(self.CHAR_FEAT.vocab)
        if args.use_pos:
            update_info['n_pos_feats']=len(self.POS_FEAT.vocab)
        args.update({
                'n_words': self.WORD.vocab.n_init,
                # 'n_feats': len(self.FEAT.vocab),
                'n_rels': len(self.REL.vocab),
                'pad_index': self.WORD.pad_index,
                'unk_index': self.WORD.unk_index,
                'bos_index': self.WORD.bos_index
            })
        args.update(update_info)

    def train(self, loader):
        self.model.train()
        for vals in loader:
            words = vals[0]
            feats = vals[1:-2]
            arcs, rels = vals[-2:]
            self.optimizer.zero_grad()

            mask = words.ne(self.args.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            arc_scores, rel_scores = self.model(words, feats)
            loss = self.get_loss(arc_scores, rel_scores, arcs, rels, mask, words=words)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        loss, metric = 0, Metric()

        for vals in loader:
            words = vals[0]
            feats = vals[1:-2]
            arcs, rels = vals[-2:]
            mask = words.ne(self.args.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            arc_scores, rel_scores = self.model(words, feats)
            loss += self.get_loss(arc_scores, rel_scores, arcs, rels, mask, words=words)
            arc_preds, rel_preds = self.decode(arc_scores, rel_scores, mask)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            metric(arc_preds, rel_preds, arcs, rels, mask)
        loss /= len(loader)

        return loss, metric

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()

        all_arcs, all_rels = [], []
        for vals in loader:
            words = vals[0]
            feats = vals[2:]

            mask = words.ne(self.args.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            arc_scores, rel_scores = self.model(words, feats)
            arc_preds, rel_preds = self.decode(arc_scores, rel_scores, mask)
            all_arcs.extend(arc_preds[mask].split(lens))
            all_rels.extend(rel_preds[mask].split(lens))
        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.REL.vocab.id2token(seq.tolist()) for seq in all_rels]

        return all_arcs, all_rels

    def get_loss(self, arc_scores, rel_scores, arcs, rels, mask, words = None):
        if self.args.binary:
            full_mask = mask.clone()
            full_mask[:,0] = 1
            binary_mask = mask.unsqueeze(-1) * full_mask.unsqueeze(-2)
            
            arc_target = torch.zeros_like(arc_scores)
            res = arc_target.scatter(-1,arcs.unsqueeze(-1),1)
            arc_scores=arc_scores*binary_mask
            arc_loss = self.arc_criterion(arc_scores, res)
            '''
            # sampling the zero part
            zero_mask=1-res
            keep_prob=2*res.shape[1]/(res.shape[1]*res.shape[2])
            sample_val=zero_mask.new_empty(zero_mask.shape).bernoulli_(keep_prob)
            binary_mask=sample_val*zero_mask*binary_mask+res
            '''
            arc_loss = (arc_loss*binary_mask).sum()/binary_mask.sum()
            if torch.isnan(arc_loss).any():
                pdb.set_trace()
            arc_scores, arcs = arc_scores[mask], arcs[mask]
        else:
            arc_scores, arcs = arc_scores[mask], arcs[mask]
            arc_loss = self.arc_criterion(arc_scores, arcs)
        rel_scores, rels = rel_scores[mask], rels[mask]
        rel_scores = rel_scores[torch.arange(len(arcs)), arcs]
        
        rel_loss = self.rel_criterion(rel_scores, rels)
        # if self.args.binary:
        loss = 2 * ((1-self.args.interpolation) * arc_loss + self.args.interpolation * rel_loss)
        # else:
        #     loss = arc_loss + rel_loss

        return loss

    def decode(self, arc_scores, rel_scores, mask):
        if self.args.tree:
            arc_preds = eisner(arc_scores, mask)
        else:
            arc_preds = arc_scores.argmax(-1)
        rel_preds = rel_scores.argmax(-1)
        rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds
