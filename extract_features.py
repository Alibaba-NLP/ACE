# This code extract document level features of transformer models

import os

import h5py
import pdb
from typing import List
import flair
from flair.data import Dictionary, Sentence, Token, Label
#from flair.datasets import CONLL_03, CONLL_03_DUTCH, CONLL_03_SPANISH, CONLL_03_GERMAN
import flair.datasets as datasets
from flair.data import MultiCorpus, Corpus
from flair.list_data import ListCorpus
import flair.embeddings as Embeddings
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter
# initialize sequence tagger
from flair.models import SequenceTagger
from pathlib import Path
import argparse
import yaml
from flair.utils.from_params import Params
# from flair.trainers import ModelTrainer
# from flair.trainers import ModelDistiller
# from flair.trainers import ModelFinetuner
from flair.config_parser import ConfigParser
import numpy as np
import argparse

from flair.training_utils import (
    init_output_file,
    WeightExtractor,
    log_line,
    add_file_handler,
    Result,
    store_embeddings,
)

def predict_embeddings(self,doc_dict,embedding,file_name):
    # torch.cuda.empty_cache()
    lm_file = h5py.File(file_name, "r")
    for key in doc_dict:
        if key == 'start':
            for i, sentence in enumerate(doc_dict[key]):
                for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                    word_embedding = torch.zeros(embedding.embedding_length).float()
                    word_embedding = torch.FloatTensor(word_embedding)

                    token.set_embedding(embedding.name, word_embedding)
            continue
        group = lm_file[key]
        num_sentences = len(list(group.keys()))
        sentences_emb = [group[str(i)][...] for i in range(num_sentences)]
        try: 
            assert len(doc_dict[key])==len(sentences_emb)
        except:
            pdb.set_trace()
        for i, sentence in enumerate(doc_dict[key]):
            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                word_embedding = sentences_emb[i][token_idx]
                word_embedding = torch.from_numpy(word_embedding).view(-1)

                token.set_embedding(embedding.name, word_embedding)
            store_embeddings([sentence], 'cpu')


parser = argparse.ArgumentParser('extract_features.py')
parser.add_argument('--config', help='configuration YAML file.')
parser.add_argument('--test', action='store_true', help='Whether testing the pretrained model.')
parser.add_argument('--zeroshot', action='store_true', help='testing with zeroshot corpus.')
parser.add_argument('--all', action='store_true', help='training/testing with all corpus.')
parser.add_argument('--other', action='store_true', help='training/testing with other corpus.')
parser.add_argument('--quiet', action='store_true', help='print results only')
parser.add_argument('--nocrf', action='store_true', help='without CRF')
parser.add_argument('--parse', action='store_true', help='parse files')
parser.add_argument('--parse_train_and_dev', action='store_true', help='chech the performance on the training and development sets')
parser.add_argument('--keep_order', action='store_true', help='keep the parse order for the prediction')
parser.add_argument('--predict', action='store_true', help='predict files')
parser.add_argument('--debug', action='store_true', help='debugging')
parser.add_argument('--target_dir', default='', help='file dir to parse')
parser.add_argument('--spliter', default='\t', help='file dir to parse')
parser.add_argument('--recur_parse', action='store_true', help='recursively parse the file dirs in target_dir')
parser.add_argument('--parse_test', action='store_true', help='parse the test set')
parser.add_argument('--save_embedding', action='store_true', help='save the pretrained embeddings')
parser.add_argument('--mst', action='store_true', help='use mst to parse the result')
parser.add_argument('--test_speed', action='store_true', help='test the running speed')
parser.add_argument('--predict_posterior', action='store_true', help='test the running speed')
parser.add_argument('--batch_size', default=32, type=int, help='set the mini batch size for extraction')
parser.add_argument('--window_size', default=511, type=int, help='transformer window_size')
parser.add_argument('--stride', default=1, type=int, help='transformer stride')
parser.add_argument('--keep_embedding', default=-1, help='mask out all embeddings except the index, for analysis')

args = parser.parse_args()

config = Params.from_file(args.config)
configparser = ConfigParser(config,all=args.all,zero_shot=args.zeroshot,other_shot=args.other,predict=args.predict)
corpus = configparser.corpus
config = configparser.config

trainer = config['trainer']
embeddings, word_map, char_map, lemma_map, postag_map=configparser.create_embeddings(config['embeddings'])
corpus2id = {x:i for i,x in enumerate(corpus.targets)}
doc_sentence_dict = {}
for corpus_id in range(len(corpus2id)):
    corpus_name = corpus.targets[corpus_id].lower()+'_'
    doc_name = 'train_'
    doc_idx = -1
    for sentence in corpus.train_list[corpus_id]:
        if '-DOCSTART-' in sentence[0].text:
            doc_idx+=1
            doc_key='start'
        else:
            doc_key=corpus_name+doc_name+str(doc_idx)
        if doc_key not in doc_sentence_dict:
            doc_sentence_dict[doc_key]=[]
        doc_sentence_dict[doc_key].append(sentence)
    doc_name = 'dev_'
    doc_idx = -1
    for sentence in corpus.dev_list[corpus_id]:
        if '-DOCSTART-' in sentence[0].text:
            doc_idx+=1
            doc_key='start'
        else:
            doc_key=corpus_name+doc_name+str(doc_idx)
        if doc_key not in doc_sentence_dict:
            doc_sentence_dict[doc_key]=[]
        doc_sentence_dict[doc_key].append(sentence)
    doc_name = 'test_'
    doc_idx = -1
    for sentence in corpus.test_list[corpus_id]:
        if '-DOCSTART-' in sentence[0].text:
            doc_idx+=1
            doc_key='start'
        else:
            doc_key=corpus_name+doc_name+str(doc_idx)
        if doc_key not in doc_sentence_dict:
            doc_sentence_dict[doc_key]=[]
        doc_sentence_dict[doc_key].append(sentence)

for idx, embedding in enumerate(embeddings.embeddings):
    if embedding.name not in config[trainer]['pretrained_file_dict']:
        continue
    output_file = config[trainer]['pretrained_file_dict'][embedding.name]
    writer = h5py.File(output_file, 'a')
    for doc_id, doc_key in enumerate(doc_sentence_dict):
        if doc_key!='start':
            # pdb.set_trace()
            sentences=embedding.add_document_embeddings(doc_sentence_dict[doc_key], window_size=args.window_size, stride=args.stride, batch_size = args.batch_size)
            # pdb.set_trace()
            # ====================================== debug =========================================
            # lm_file = h5py.File('../temp/biaffine-ner/bert_features.hdf5', "r")
            # group = lm_file['train_0']
            # num_sentences = len(list(group.keys()))
            # sentences_emb = [group[str(i)][...] for i in range(num_sentences)]
            # idx=-1
            # sentfeat=np.concatenate([sentences_emb[idx][:,:,i] for i in range(sentences_emb[idx].shape[-1])],-1)
            # for i in range(len(sentences_emb[idx])): np.absolute(sentences[idx][i].embedding.cpu().numpy()-sentfeat[i]).max()
            # pdb.set_trace()
            # ====================================== debug =========================================
            file_key = doc_key.replace('/', ':')
            for sentence_index, sentence in enumerate(sentences):
                dataset_key ="{}/{}".format(file_key, sentence_index)
                if dataset_key not in writer:
                    writer.create_dataset(dataset_key,
                                          (len(sentence), embedding.embedding_length),
                                          dtype=np.float32)
                dset = writer[dataset_key]
                for token_id, token in enumerate(sentence):
                    dset[token_id, :] = token.embedding.cpu().numpy()
            store_embeddings(sentences,'none')
        if (doc_id+1) % (len(doc_sentence_dict)//10) == 0:
            print(f'Processed {doc_id+1}/{(len(doc_sentence_dict))} documents')

        # writer = h5py.File(FLAGS.output_file, 'w')
        # with tqdm(total=sum(len(e.tokens) for e in orig_examples)) as t:
        #     for result in estimator.predict(input_fn, yield_single_examples=True):
        #         document_index = int(result["unique_ids"])
        #         bert_example = bert_examples[document_index]
        #         orig_example = orig_examples[document_index]
        #         file_key = bert_example.doc_key.replace('/', ':')

        #         t.update(n=(result['extract_indices'] >= 0).sum())

        #         for output_index, bert_token_index in enumerate(result['extract_indices']):
        #             if bert_token_index < 0:
        #                 continue

        #             token_index = bert_example.bert_to_orig_map[bert_token_index]
        #             sentence_index, token_index = orig_example.unravel_token_index(token_index)

        #             dataset_key ="{}/{}".format(file_key, sentence_index)
        #             if dataset_key not in writer:
        #                 writer.create_dataset(dataset_key,
        #                                       (len(orig_example.sentence_tokens[sentence_index]), bert_config.hidden_size, len(layer_indexes)),
        #                                       dtype=np.float32)

        #             dset = writer[dataset_key]
        #             for j, layer_index in enumerate(layer_indexes):
        #                 layer_output = result["layer_output_%d" % j]
        #                 dset[token_index, :, j] = layer_output[output_index]
        # writer.close()



