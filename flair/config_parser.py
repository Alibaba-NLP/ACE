from typing import List

#from flair.datasets import CONLL_03, CONLL_03_DUTCH, CONLL_03_SPANISH, CONLL_03_GERMAN
from . import datasets as datasets
from .data import MultiCorpus, Corpus, Dictionary
from .list_data import ListCorpus
from . import embeddings as Embeddings
from .training_utils import EvaluationMetric
from .visual.training_curves import Plotter
import torch
from torch.utils.data.dataset import ConcatDataset
from flair.datasets import CoupleDataset
from .custom_data_loader import ColumnDataLoader
from flair.training_utils import store_embeddings
# initialize sequence tagger
from . import models as models
from pathlib import Path
import argparse
import yaml
from .utils.from_params import Params
from . import logging
import pdb
import copy
log = logging.getLogger("flair")

from flair.corpus_mapping import corpus_map,reverse_corpus_map
dependency_tasks={'enhancedud', 'dependency', 'srl', 'ner_dp'}
class ConfigParser:
	def __init__(self, config, all=False, zero_shot=False, other_shot=False, predict=False):
		self.full_corpus={'ner':'CONLL_03_GERMAN:CONLL_03:CONLL_03_DUTCH:CONLL_03_SPANISH', 'upos':'UD_GERMAN:UD_ENGLISH:UD_FRENCH:UD_ITALIAN:UD_DUTCH:UD_SPANISH:UD_PORTUGUESE:UD_CHINESE'}
		# self.zeroshot_corpus={'ner':'PANX-TA:PANX-SL:PANX-PT:PANX-ID:PANX-HE:PANX-FR:PANX-FA:PANX-EU', 'upos':'UD_BASQUE:UD_DUTCH:UD_ARABIC:UD_RUSSIAN:UD_KOREAN:UD_CHINESE:UD_HINDI:UD_FINNISH'}
		self.zeroshot_corpus={}
		for key in corpus_map:
			self.zeroshot_corpus[key]=':'.join(corpus_map[key].values())
		self.zeroshot_corpus['ner']='PANX-SV:PANX-FR:PANX-RU:PANX-PL:PANX-VI:PANX-JA:PANX-ZH:PANX-AR:PANX-PT:PANX-UK:PANX-FA:PANX-CA:PANX-SR:PANX-NO:PANX-ID:PANX-KO:PANX-FI:PANX-HU:PANX-SH:PANX-CS:PANX-RO:PANX-EU:PANX-TR:PANX-MS:PANX-EO:PANX-HY:PANX-DA:PANX-CE:PANX-HE:PANX-SK:PANX-KK:PANX-HR:PANX-ET:PANX-LT:PANX-BE:PANX-EL:PANX-SL:PANX-GL'
		# pdb.set_trace()
		self.zeroshot_corpus={
		'ner':'PANX-TA:PANX-EU:PANX-HE:PANX-FA', 
		# 'ner':'CONLL_03_DUTCH:CONLL_03_SPANISH:CONLL_03:CONLL_03_GERMAN:MIXED_NER-EU:MIXED_NER-FA:MIXED_NER-FI:MIXED_NER-FR:MIXED_NER-HE:MIXED_NER-HI:MIXED_NER-HR:MIXED_NER-ID:MIXED_NER-NO:MIXED_NER-PL:MIXED_NER-PT:MIXED_NER-SL:MIXED_NER-SV:MIXED_NER-TA',
		# 'ner':'MIXED_NER-EN:MIXED_NER-NL:MIXED_NER-ES:MIXED_NER-DE:MIXED_NER-EU:MIXED_NER-FA:MIXED_NER-FI:MIXED_NER-FR:MIXED_NER-HE:MIXED_NER-HI:MIXED_NER-HR:MIXED_NER-ID:MIXED_NER-JA:MIXED_NER-NO:MIXED_NER-PL:MIXED_NER-PT:MIXED_NER-SL:MIXED_NER-SV:MIXED_NER-TA',
		'upos':'UD_TURKISH:UD_SWEDISH:UD_SPANISH:UD_SLOVAK:UD_SERBIAN:UD_RUSSIAN:UD_ROMANIAN:UD_PORTUGUESE:UD_POLISH:UD_NORWEGIAN:UD_KOREAN:UD_ITALIAN:UD_HINDI:UD_GERMAN:UD_FINNISH:UD_DUTCH:UD_DANISH:UD_CZECH:UD_CROATIAN:UD_CHINESE:UD_CATALAN:UD_BULGARIAN:UD_BASQUE:UD_ARABIC:UD_HEBREW:UD_JAPANESE:UD_INDONESIAN:UD_PERSIAN:UD_TAMIL',
		'mixedner':'CONLL_03_DUTCH:CONLL_03_SPANISH:CONLL_03:CONLL_03_GERMAN:MIXED_NER-EU:MIXED_NER-FA:MIXED_NER-FI:MIXED_NER-FR:MIXED_NER-HE:MIXED_NER-HI:MIXED_NER-HR:MIXED_NER-ID:MIXED_NER-NO:MIXED_NER-PL:MIXED_NER-PT:MIXED_NER-SL:MIXED_NER-SV:MIXED_NER-TA',
		'low10ner':'CONLL_03_DUTCH:CONLL_03_SPANISH:CONLL_03:CONLL_03_GERMAN:LOW10_NER-EU:LOW10_NER-FA:LOW10_NER-FI:LOW10_NER-FR:LOW10_NER-HE:LOW10_NER-HI:LOW10_NER-HR:LOW10_NER-ID:LOW10_NER-NO:LOW10_NER-PL:LOW10_NER-PT:LOW10_NER-SL:LOW10_NER-SV:LOW10_NER-TA',
		}
							# 'upos':'UD_TURKISH:UD_SWEDISH:UD_SPANISH:UD_SLOVAK:UD_SERBIAN:UD_RUSSIAN:UD_ROMANIAN:UD_PORTUGUESE:UD_POLISH:UD_NORWEGIAN:UD_KOREAN:UD_ITALIAN:UD_HINDI:UD_GERMAN:UD_FINNISH:UD_DUTCH:UD_DANISH:UD_CZECH:UD_CROATIAN:UD_CHINESE:UD_CATALAN:UD_BULGARIAN:UD_BASQUE:UD_ARABIC'}

		self.othershot_corpus={'ner':'CONLL_03_DUTCH:CONLL_03_SPANISH:CONLL_03:CONLL_03_GERMAN'}
		self.predict_corpus={'ner':{'en':'PANXPRED-EN','ta':'PANXPRED-TA','fi':'PANXPRED-FI','eu':'PANXPRED-EU','he':'PANXPRED-HE','ar':'PANXPRED-AR','id':'PANXPRED-ID','cs':'PANXPRED-CS','it':'PANXPRED-IT','fa':'PANXPRED-FA','ja':'PANXPRED-JA','sl':'PANXPRED-SL','fr':'`PRED-FR'}}
		self.config = config
		self.mini_batch_size=self.config['train']['mini_batch_size']

		self.target: str=self.get_target
		self.tag_type = self.target
		if all:
			self.corpus: ListCorpus=self.get_full_corpus
		elif zero_shot:
			self.corpus: ListCorpus=self.get_zeroshot_corpus
		elif other_shot:
			self.corpus: ListCorpus=self.get_othershot_corpus
		elif predict:
			self.corpus: ListCorpus=self.get_predict_corpus
		else:
			self.corpus: ListCorpus=self.get_corpus
		if 'trainer' in self.config and self.config['trainer'] == 'SWAFTrainer':
			self.assign_system_prediction
		self.tokens = self.corpus.get_train_full_tokenset(-1, min_freq =-1 if 'min_freq' not in self.config['train'] else self.config['train']['min_freq'])
		for embedding in config['embeddings'].keys():
			if 'LemmaEmbeddings' in embedding:
				self.lemmas = self.corpus.get_train_full_tokenset(-1, min_freq =-1 if 'min_lemma_freq' not in self.config['train'] else self.config['train']['min_lemma_freq'], attr='lemma')[0]
			if 'POSEmbeddings' in embedding:
				self.postags = self.corpus.get_train_full_tokenset(-1, min_freq =-1 if 'min_pos_freq' not in self.config['train'] else self.config['train']['min_pos_freq'], attr='pos')[0]
		use_unlabeled_data = False if 'use_unlabeled_data' not in self.config['train'] else self.config['train']['use_unlabeled_data']
		if use_unlabeled_data:

			self.unlabeled_corpus: ListCorpus=self.get_unlabeled_corpus

			unlabeled_data_for_zeroshot = False if 'unlabeled_data_for_zeroshot' not in self.config['train'] else self.config['train']['unlabeled_data_for_zeroshot']
			self.assign_unlabel_tag(self.corpus,1)
			self.assign_unlabel_tag(self.unlabeled_corpus,-1)
			if unlabeled_data_for_zeroshot:
				new_train_set = []
				new_dev_set = []
				zs_corpus = self.config[self.target]['zeroshot_corpus'].split(':')
				for i, val in enumerate(self.corpus.train_list):
					corpus_name = self.corpus.targets[i]
					if corpus_name not in zs_corpus:
						new_train_set.append(val)
						new_dev_set.append(self.corpus.dev_list[i])
				new_train_set+=self.unlabeled_corpus.train_list
				self.corpus._train = ConcatDataset(new_train_set)
				self.corpus.train_list = new_train_set

				new_dev_set+=self.unlabeled_corpus.dev_list
				self.corpus._dev = ConcatDataset(new_dev_set)
				self.corpus.dev_list = new_dev_set
				
			else:
				self.corpus._train = ConcatDataset(self.corpus.train_list+self.unlabeled_corpus.train_list)
				self.corpus.train_list+=self.unlabeled_corpus.train_list
				self.corpus.targets+=self.unlabeled_corpus.targets
				# pdb.set_trace()
		self.corpus_list: list[str] = self.config[self.target]['Corpus'].split(':')
		# keep the consistency of tag dictionary
		if 'tag_dictionary' in self.config[self.target] and Path(self.config[self.target]['tag_dictionary']).exists():
			self.tag_dictionary=Dictionary.load_from_file(self.config[self.target]['tag_dictionary'])
			# pdb.set_trace()
		else:
			self.tag_dictionary = self.corpus.make_tag_dictionary(tag_type=self.target)
			if 'tag_dictionary' in self.config[self.target]:
				self.tag_dictionary.save(self.config[self.target]['tag_dictionary'])

		# self.check_failed_count(1)
		# self.check_failed_count(2)
		log.info(self.tag_dictionary.item2idx)
		self.num_corpus = len(self.corpus.targets)
		log.info(self.corpus)
	def assign_unlabel_tag(self,corpus, is_unlabel):
		for datidx, dataset in enumerate(corpus.train_list):
			for sentidx, sentence in enumerate(dataset):
				sentence.is_unlabel=is_unlabel
			for sentidx, sentence in enumerate(corpus.dev_list[datidx]):
				sentence.is_unlabel=is_unlabel
			for sentidx, sentence in enumerate(corpus.test_list[datidx]):
				sentence.is_unlabel=is_unlabel
	def check_failed_count(self,mincount=1):
		res=self.corpus._obtain_statistics_for(self.corpus.train,"TRAIN",'dependency')
		test=res['number_of_tokens_per_tag']

		res2=self.corpus._obtain_statistics_for(self.corpus.test,"TEST",'dependency')
		test2=res2['number_of_tokens_per_tag']

		results=torch.where(torch.Tensor(list(test.values()))<=mincount)[0]
		failed_count=0
		for index in results:
			key=list(test.keys())[index]
			if key in test2:
				failed_count+=1
		print(failed_count)
		print(len(results))
	def create_embeddings(self, embeddings: dict):
		embedding_list: List[TokenEmbeddings]=[]
		# how to solve the problem of loading model?
		word_map = None
		char_map = None
		lemma_map = None
		postag_map = None
		for embedding in embeddings:
			# pdb.set_trace()
			if isinstance(embeddings[embedding],dict):
				if 'FastWordEmbeddings' in embedding:
					embedding_list.append(getattr(Embeddings,embedding.split('-')[0])(**embeddings[embedding],all_tokens=self.tokens))
					word_map = embedding_list[-1].vocab
				elif 'LemmaEmbeddings' in embedding:
					embedding_list.append(getattr(Embeddings,embedding.split('-')[0])(**embeddings[embedding],vocab=self.lemmas))
					lemma_map = embedding_list[-1].lemma_dictionary
				elif 'POSEmbeddings' in embedding:
					embedding_list.append(getattr(Embeddings,embedding.split('-')[0])(**embeddings[embedding],vocab=self.postags))
					postag_map = embedding_list[-1].pos_dictionary
				elif 'FastCharacterEmbeddings' in embedding:
					embedding_list.append(getattr(Embeddings,embedding.split('-')[0])(vocab=self.tokens[1],**embeddings[embedding]))
					char_map = embedding_list[-1].char_dictionary
				else:
					embedding_list.append(getattr(Embeddings,embedding.split('-')[0])(**embeddings[embedding]))
				# embedding_list.append(getattr(Embeddings,embedding.split('-')[0])(**embeddings[embedding]))
			else:
				if 'FastCharacterEmbeddings' in embedding:
					embedding_list.append(getattr(Embeddings,embedding.split('-')[0])(vocab=self.tokens[1]))
					char_map = embedding_list[-1].char_dictionary
				elif 'LemmaEmbeddings' in embedding:
					embedding_list.append(getattr(Embeddings,embedding.split('-')[0])(vocab=self.lemmas))
					lemma_map = embedding_list[-1].lemma_dictionary
				elif 'POSEmbeddings' in embedding:
					embedding_list.append(getattr(Embeddings,embedding.split('-')[0])(vocab=self.postags))
					postag_map = embedding_list[-1].pos_dictionary
				else:
					embedding_list.append(getattr(Embeddings,embedding.split('-')[0])())

		embeddings: Embeddings.StackedEmbeddings = Embeddings.StackedEmbeddings(embeddings=embedding_list)
		return embeddings, word_map, char_map, lemma_map, postag_map
	def create_model(self, config: dict = None, pretrained=False, is_student=False, crf=True):
		if config is None:
			config=self.config
		# Toy example debugging
		if 'is_toy' in self.config:
			if self.config['is_toy']==True:
				pretrained=False
				pass
		embeddings, word_map, char_map, lemma_map, postag_map=self.create_embeddings(config['embeddings'])
		kwargs=copy.deepcopy(config['model'])
		classname=list(kwargs.keys())[0]
		kwargs=copy.deepcopy(config['model'][classname])
		if classname == 'EnsembleModel':
			kwargs['candidates'] = len(config[self.target]['systems'])
		if crf==False:
			kwargs['use_crf']=crf
		# pdb.set_trace()
		kwargs['embeddings']=embeddings
		kwargs['tag_type']=self.target
		kwargs['tag_dictionary']=self.tag_dictionary
		if not pretrained:
			kwargs['target_languages']=self.num_corpus
		tagger = getattr(models,classname)(**kwargs, config=config)
		tagger.word_map = word_map
		tagger.char_map = char_map
		tagger.lemma_map = lemma_map
		tagger.postag_map = postag_map

		#
		if pretrained:
			if is_student and 'pretrained_model' in config:
				base_path=Path(config['target_dir'])/config['pretrained_model']
			base_path=Path(config['target_dir'])/config['model_name']

			if (base_path / "best-model.pt").exists():
				log.info('Loading pretraining best model')
				tagger = tagger.load(base_path / "best-model.pt")
			elif (base_path / "final-model.pt").exists():
				log.info('Loading pretraining final model')
				tagger = tagger.load(base_path / "final-model.pt")
			else:
				assert 0, str(base_path)+ ' not exist!'
		tagger.use_bert=False
		for embedding in config['embeddings']:
			if 'bert' in embedding.lower():
				tagger.use_bert=True
				# break
		if crf==False:
			tagger.use_crf=crf
		return tagger

	def create_student(self,nocrf=False):
		if nocrf:
			return self.create_model(self.config,pretrained=self.load_pretrained(self.config), is_student=True,crf=False)
		else:
			return self.create_model(self.config,pretrained=self.load_pretrained(self.config), is_student=True)

	def create_teachers(self,is_professor=False):
		teacher_list=[]
		for corpus in self.corpus_list:
			if is_professor:
				config=Params.from_file(self.config[self.target][corpus]['professor_config'])
			else:
				config=Params.from_file(self.config[self.target][corpus]['train_config'])
			teacher_model=self.create_model(config, pretrained=True)
			teacher_model.targets=set([corpus])

			teacher_list.append(teacher_model)
		return teacher_list

	def create_teachers_list(self,is_professor=False):
		# pdb.set_trace()
		teacher_list=[]
		if is_professor:
			configs=self.config[self.target]['professors']
		else:
			configs=self.config[self.target]['teachers']

		for filename in configs:
			corpus_target=set(configs[filename].split(':'))
			if len(set(self.corpus.targets)&corpus_target)==0:
				continue
			config=Params.from_file(filename)

			teacher_model=self.create_model(config, pretrained=True)
			teacher_model.to("cpu")
			teacher_model.targets=corpus_target
			teacher_list.append(teacher_model)
		# pdb.set_trace()
		return teacher_list

	def distill_teachers_prediction(self):
		if self.config['train']['train_with_dev']:
			train_data = [ConcatDataset([train, self.corpus.dev_list[index]]) for index, train in enumerate(self.corpus.train_list)]
		train_data_teacher = copy.deepcopy(train_data)
		coupled_train_data = [CoupleDataset(data,train_data_teacher[index]) for index, data in enumerate(train_data)]
		train_data=self.assign_pretrained_teacher_predictions(coupled_train_data)
		del train_data_teacher
		del coupled_train_data
		# pdb.set_trace()
		return train_data
	def load_pretrained(self,config):
		try:
			return self.config['load_pretrained']
		except:
			return False
	@property
	def get_target(self):
		targets = self.config.get('targets').split(':')
		if len(targets)>1:
			log.info('Warning! Not support multitask now!')
		return targets[0]
	@property
	def get_corpus(self):
		corpus_list={'train':[],'dev':[],'test':[]}
		# pdb.set_trace()
		for corpus in self.config[self.target]['Corpus'].split(':'):
			#corpus_list.append(getattr(datasets,corpus)())
			if 'UD' in corpus and '-' not in corpus and 'enhancedud' != self.target:
				current_dataset=getattr(datasets,corpus)()
			elif 'UD' in corpus:
				kwargs=self.config['model']
				classname=list(kwargs.keys())[0]
				if self.target=='enhancedud':
					if 'UNREL' in corpus:
						current_dataset=getattr(datasets,'UNREL_ENHANCEDUD')(corpus.split('-')[0])
					else:
						current_dataset=getattr(datasets,'ENHANCEDUD')(corpus)
				elif self.target=='dependency' and 'use_crf' in kwargs[classname] and kwargs[classname]['use_crf']:
					current_dataset=getattr(datasets,'UD_PROJ')(corpus,add_root = self.target=='dependency')
				else:
					current_dataset=getattr(datasets,'UD')(corpus,add_root = self.target=='dependency')
			elif 'DM' in corpus or 'PSD' in corpus or 'PAS' in corpus:
				current_dataset=getattr(datasets,'ENHANCEDUD')(corpus)
			elif 'SRL' in corpus:
				corpus,lc=corpus.split('-')
				current_dataset=getattr(datasets,corpus)(lang=lc.lower())
			elif 'PANX' in corpus or 'SEMEVAL16' in corpus or 'CALCS' in corpus or 'MIXED_NER' in corpus or 'LOW10_NER' in corpus or 'COMMNER' in corpus or 'ATIS' in corpus:
				corpus,lc=corpus.split('-')
				current_dataset=getattr(datasets,corpus)(tag_to_bioes=self.target, lang=lc.lower())
			elif 'TWEEBANK' in corpus:
				current_dataset=getattr(datasets,corpus)()
			elif 'ColumnCorpus' in corpus or 'UniversalDependenciesCorpus' in corpus:
				if '-' in corpus:
					corpus_name,idx=corpus.split('-')
				else:
					corpus_name = corpus
				current_dataset=getattr(datasets,corpus_name)(**self.config[self.target][corpus])
			else:
				current_dataset=getattr(datasets,corpus)(tag_to_bioes=self.target)
			corpus_list['train'].append(current_dataset.train)
			corpus_list['dev'].append(current_dataset.dev)
			corpus_list['test'].append(current_dataset.test)
		corpus_list['targets'] = self.config[self.target]['Corpus'].split(':')
		corpus: ListCorpus = ListCorpus(**corpus_list)
		return corpus
	@property
	def get_full_corpus(self):
		corpus_list={'train':[],'dev':[],'test':[]}
		for corpus in self.full_corpus[self.target].split(':'):
			#corpus_list.append(getattr(datasets,corpus)())
			if 'UD' in corpus and '-' not in corpus and 'enhancedud' != self.target:
				current_dataset=getattr(datasets,corpus)()
			elif 'UD' in corpus:
				if self.target=='enhancedud':
					if 'UNREL' in corpus:
						current_dataset=getattr(datasets,'UNREL_ENHANCEDUD')(corpus.split('-')[0])
					else:
						current_dataset=getattr(datasets,'ENHANCEDUD')(corpus)
				if self.target=='dependency':
					current_dataset=getattr(datasets,'UD_PROJ')(corpus,add_root = self.target=='dependency')
				else:
					current_dataset=getattr(datasets,'UD')(corpus)
			elif 'PANX' in corpus or 'SEMEVAL16' in corpus or 'CALCS' in corpus or 'MIXED_NER' in corpus or 'LOW10_NER' in corpus or 'COMMNER' in corpus or 'ATIS' in corpus:
				corpus,lc=corpus.split('-')
				current_dataset=getattr(datasets,corpus)(tag_to_bioes=self.target, lang=lc.lower())
			
			else:
				current_dataset=getattr(datasets,corpus)(tag_to_bioes=self.target)
			corpus_list['train'].append(current_dataset.train)
			corpus_list['dev'].append(current_dataset.dev)
			corpus_list['test'].append(current_dataset.test)
		corpus_list['targets'] = self.full_corpus[self.target].split(':')
		corpus: ListCorpus = ListCorpus(**corpus_list)
		return corpus
	@property
	def get_zeroshot_corpus(self):
		corpus_list={'train':[],'dev':[],'test':[]}
		for corpus in self.zeroshot_corpus[self.target].split(':'):
			#corpus_list.append(getattr(datasets,corpus)())
			if 'UD' in corpus and '-' not in corpus and 'enhancedud' != self.target:
				current_dataset=getattr(datasets,corpus)()
			elif 'UD' in corpus:
				if self.target=='enhancedud':
					if 'UNREL' in corpus:
						current_dataset=getattr(datasets,'UNREL_ENHANCEDUD')(corpus.split('-')[0])
					else:
						current_dataset=getattr(datasets,'ENHANCEDUD')(corpus)
				if self.target=='dependency':
					current_dataset=getattr(datasets,'UD_PROJ')(corpus,add_root = self.target=='dependency')
				else:
					current_dataset=getattr(datasets,'UD')(corpus)
			elif 'PANX' in corpus or 'SEMEVAL16' in corpus or 'CALCS' in corpus or 'MIXED_NER' in corpus or 'LOW10_NER' in corpus or 'COMMNER' in corpus or 'ATIS' in corpus:
				corpus,lc=corpus.split('-')
				current_dataset=getattr(datasets,corpus)(tag_to_bioes=self.target, lang=lc.lower())
			else:
				current_dataset=getattr(datasets,corpus)(tag_to_bioes=self.target)
			corpus_list['train'].append(current_dataset.train)
			corpus_list['dev'].append(current_dataset.dev)
			corpus_list['test'].append(current_dataset.test)
		corpus_list['targets'] = self.zeroshot_corpus[self.target].split(':')
		corpus: ListCorpus = ListCorpus(**corpus_list)
		return corpus
	@property
	def get_othershot_corpus(self):
		corpus_list={'train':[],'dev':[],'test':[]}
		for corpus in self.othershot_corpus[self.target].split(':'):
			#corpus_list.append(getattr(datasets,corpus)())
			if 'UD' in corpus and '-' not in corpus and 'enhancedud' != self.target:
				current_dataset=getattr(datasets,corpus)()
			elif 'UD' in corpus:
				if self.target=='enhancedud':
					if 'UNREL' in corpus:
						current_dataset=getattr(datasets,'UNREL_ENHANCEDUD')(corpus.split('-')[0])
					else:
						current_dataset=getattr(datasets,'ENHANCEDUD')(corpus)
				if self.target=='dependency':
					current_dataset=getattr(datasets,'UD_PROJ')(corpus,add_root = self.target=='dependency')
				else:
					current_dataset=getattr(datasets,'UD')(corpus)
			elif 'PANX' in corpus or 'SEMEVAL16' in corpus or 'CALCS' in corpus or 'MIXED_NER' in corpus or 'LOW10_NER' in corpus or 'COMMNER' in corpus or 'ATIS' in corpus:
				corpus,lc=corpus.split('-')
				current_dataset=getattr(datasets,corpus)(tag_to_bioes=self.target, lang=lc.lower())
			else:
				current_dataset=getattr(datasets,corpus)(tag_to_bioes=self.target)
			corpus_list['train'].append(current_dataset.train)
			corpus_list['dev'].append(current_dataset.dev)
			corpus_list['test'].append(current_dataset.test)
		corpus_list['targets'] = self.othershot_corpus[self.target].split(':')
		corpus: ListCorpus = ListCorpus(**corpus_list)
		return corpus
	@property
	def get_predict_corpus(self):
		corpus_list={'train':[],'dev':[],'test':[]}
		corpus = self.config.get(self.target)['Corpus'].split(':')[0]
		lang=corpus.split('-')[1]
		#corpus_list.append(getattr(datasets,corpus)())
		if 'UD' in corpus and '-' not in corpus and 'enhancedud' != self.target:
			current_dataset=getattr(datasets,corpus)()
		elif 'UD' in corpus:
			if self.target=='enhancedud':
				if 'UNREL' in corpus:
					current_dataset=getattr(datasets,'UNREL_ENHANCEDUD')(corpus.split('-')[0])
				else:
					current_dataset=getattr(datasets,'ENHANCEDUD')(corpus)
			if self.target=='dependency':
				current_dataset=getattr(datasets,'UD_PROJ')(corpus,add_root = self.target=='dependency')
			else:
				current_dataset=getattr(datasets,'UD')(corpus)
		elif 'PANX' in corpus or 'SEMEVAL16' in corpus or 'CALCS' in corpus or 'MIXED_NER' in corpus or 'LOW10_NER' in corpus or 'COMMNER' in corpus or 'ATIS' in corpus:
			corpus,lc=corpus.split('-')
			corpus=corpus+'PRED'
			current_dataset=getattr(datasets,corpus)(tag_to_bioes=self.target, lang=lc.lower())
		else:
			current_dataset=getattr(datasets,corpus)(tag_to_bioes=self.target)
		corpus_list['train'].append(current_dataset.train)
		corpus_list['dev'].append(current_dataset.dev)
		corpus_list['test'].append(current_dataset.test)
		corpus_list['targets'] = [corpus+'-'+lang]
		corpus: ListCorpus = ListCorpus(**corpus_list)
		return corpus
	@property
	def get_unlabeled_corpus(self):
		corpus_list={'train':[],'dev':[],'test':[],'targets':[]}
		corpus = self.config.get(self.target)['Corpus'].split(':')[0]
		teacher_list=[]
		configs=self.config[self.target]['teachers']
		for filename in configs:
			corpus_target=set(configs[filename].split(':'))
			if len(set(self.corpus.targets)&corpus_target)==0:
				continue
			config=Params.from_file(filename)
			for target in corpus_target:
				task_target=self.config['targets']
				if 'PANX' in self.config.get(self.target)['Corpus']:
					task_target = 'panx'
				lang=reverse_corpus_map[task_target][target]
				extra = None if 'extra_name' not in self.config else self.config['extra_name']
				if self.config['targets']=='dependency':
					if lang=='ptb':
						lang='en'
					if lang=='ctb':
						lang='zh'
					current_dataset=getattr(datasets,'UNLABEL_DEPENDENCY')(modelname=config['model_name'], lang=lang, extra=extra)
				else:
					current_dataset=getattr(datasets,'UNLABEL')(tag_to_bioes=self.target, modelname=config['model_name'], lang=lang, extra=extra)
				corpus_list['train'].append(current_dataset.train)
				corpus_list['dev'].append(current_dataset.dev)
				corpus_list['test'].append(current_dataset.test)
				corpus_list['targets'].append('unlabeled'+'-'+config['model_name']+'-'+lang)
				self.config[self.target]['teachers'][filename]+=':'+'unlabeled'+'-'+config['model_name']+'-'+lang
				# pdb.set_trace()
		corpus: ListCorpus = ListCorpus(**corpus_list)
		return corpus
	@property
	def assign_system_prediction(self):
		# pdb.set_trace()
		teacher_list=[]
		configs=self.config[self.target]['systems']
		log.info(f'System Candidates: {sorted(configs.keys())}')
		for filename in sorted(configs.keys()):
			dev_file = Path('system_pred/dev.'+filename+'.conllu')
			test_file = Path('system_pred/test.'+filename+'.conllu')
			if self.target in dependency_tasks:
				dev = datasets.UniversalDependenciesDataset(dev_file, in_memory=True, add_root=True, spliter='\t')
				test = datasets.UniversalDependenciesDataset(test_file, in_memory=True, add_root=True, spliter='\t')
			else:
				dev = datasets.ColumnDataset(
					dev_file,
					column_name_map = {0: "text", 1: "gold_label", 2:"pred_label", 3:"score"},
					tag_to_bioes=None,
					comment_symbol=None,
					in_memory=True,
				)
				test = datasets.ColumnDataset(
					test_file,
					column_name_map = {0: "text", 1: "gold_label", 2:"pred_label", 3:"score"},
					tag_to_bioes=None,
					comment_symbol=None,
					in_memory=True,
				)
			# pdb.set_trace()
			for idx, corpus_name in enumerate(self.corpus.targets):
				if self.config[self.target]['systems'][filename] != corpus_name:
					continue
				if len(self.corpus.dev_list[idx]) != len(dev):
					pdb.set_trace()
				for sentid, sentence in enumerate(self.corpus.dev_list[idx]):

					for tokenid, token in enumerate(sentence):
						if len(sentence) != len(dev[sentid]):
							pdb.set_trace()
						if not hasattr(token, 'system_preds'):
							token.system_preds=[]
							token.system_scores=[]
						token.system_preds.append(dev[sentid][tokenid].tags['pred_label']._value)
						token.system_scores.append(float(dev[sentid][tokenid].tags['score']._value))
						
				if len(self.corpus.test_list[idx]) != len(test):
					pdb.set_trace()
				for sentid, sentence in enumerate(self.corpus.test_list[idx]):

					for tokenid, token in enumerate(sentence):
						if len(sentence) != len(test[sentid]):
							pdb.set_trace()
						if not hasattr(token, 'system_preds'):
							token.system_preds=[]
							token.system_scores=[]
						token.system_preds.append(test[sentid][tokenid].tags['pred_label']._value)
						token.system_scores.append(float(test[sentid][tokenid].tags['score']._value))
		# pdb.set_trace()
	@property
	def check_model_corpus_group(self):
		cfg=self.config['model_name']
		targets=[]
		if 'ner' in cfg:
			targets.append('ner')
		if '_upos' in cfg:
			targets.append('upos')
		if '_ast' in cfg:
			targets.append('ast')
		if 'np' in cfg:
			targets.append('np')
		if '_cs' in cfg and '_cs_' not in cfg:
			targets.append('cs')
		if '_dep' in cfg and '_depscore' not in cfg:
			targets.append('dep')
		targets=':'.join(targets)
		if 'panx' in cfg:
			targets='panx'
		elif 'semeval' in cfg:
			targets='semeval'
		elif 'smallud' in cfg:
			targets='smallud'
		elif 'mixedner' in cfg:
			targets='mixedner'
		elif 'lowner' in cfg:
			targets='lowner'
		elif 'low10ner' in cfg:
			targets='low10ner'
		return targets
	@property
	def get_tag_dictionary(self):
		return self.tag_dictionary
	@property
	def get_target_path(self):
		return Path(self.config['target_dir'])/self.config['model_name']


