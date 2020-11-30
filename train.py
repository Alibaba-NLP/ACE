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
import pdb
import sys
import os
import logging
from flair.custom_data_loader import ColumnDataLoader
from flair.datasets import DataLoader
# Disable
def blockPrint():
		sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
		sys.stdout = sys.__stdout__

parser = argparse.ArgumentParser('train.py')
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
parser.add_argument('--predict_posterior', action='store_true', help='predict the posterior distribution of CRF model')
parser.add_argument('--batch_size', default=-1, help='manually setting the mini batch size for testing')
parser.add_argument('--keep_embedding', default=-1, help='mask out all embeddings except the index, for analysis')

def count_parameters(model):
	import numpy as np
	total_param = 0
	for name,param in model.named_parameters():
		num_param = np.prod(param.size())
		# print(name,num_param)
		total_param+=num_param
	return total_param


log = logging.getLogger("flair")
args = parser.parse_args()
if args.quiet:
	blockPrint()
	log.disabled=True
config = Params.from_file(args.config)
if args.test and args.zeroshot:
	temperory_reject_list=['ast','enhancedud','dependency','atis','chunk']
	if config['targets'] in temperory_reject_list:
		enablePrint()
		print()
		exit()

# pdb.set_trace()
config = ConfigParser(config,all=args.all,zero_shot=args.zeroshot,other_shot=args.other,predict=args.predict)
# pdb.set_trace()


student=config.create_student(nocrf=args.nocrf)
log.info(f"Model Size: {count_parameters(student)}")
corpus=config.corpus


teacher_func=config.create_teachers
if 'is_teacher_list' in config.config:
	if config.config['is_teacher_list']:
		teacher_func=config.create_teachers_list

# pdb.set_trace()
if 'trainer' in config.config:
	trainer_name=config.config['trainer']
else:
	if 'ModelDistiller' in config.config:
		trainer_name='ModelDistiller'
	elif 'ModelFinetuner' in config.config:
		trainer_name='ModelFinetuner'
	elif 'ReinforcementTrainer' in config.config:
		trainer_name='ReinforcementTrainer'
	else:
		trainer_name='ModelDistiller'

trainer_func=getattr(flair.trainers,trainer_name)


if 'distill_mode' not in config.config[trainer_name]:
	config.config[trainer_name]['distill_mode']=False
if not args.test and config.config[trainer_name]['distill_mode']:
	teachers=teacher_func()
	professors=[]
	# corpus=config.distill_teachers_prediction()
	trainer: trainer_func = trainer_func(student, teachers, corpus, config=config.config, professors=professors,**config.config[trainer_name])
elif not args.parse:
	trainer: trainer_func = trainer_func(student, None, corpus, config=config.config, **config.config[trainer_name], is_test=args.test)
else:
	trainer: trainer_func = trainer_func(student, None, corpus, config=config.config, **config.config[trainer_name], is_test=args.test)

# pdb.set_trace()

train_config=config.config['train']
train_config['base_path']=config.get_target_path

# train_config['shuffle']=False
eval_mini_batch_size = int(config.config['train']['mini_batch_size'])
# if args.parse or args.test:
#   if 'sentence_level_batch' in config.config[trainer_name] and config.config[trainer_name]['sentence_level_batch']:
#       eval_mini_batch_size = 2000
# pdb.set_trace()
if int(args.batch_size)>0:
	eval_mini_batch_size = int(args.batch_size)

if args.test_speed:
	student.eval()
	# pdb.set_trace()
	print(count_parameters(student))
	# for embedding in student.embeddings.embeddings:
	# 	embedding.training = False
	test_loader=ColumnDataLoader(list(trainer.corpus.test),32,use_bert=trainer.use_bert,tokenizer=trainer.bert_tokenizer, sort_data=False, model = student, sentence_level_batch = True)
	test_loader.assign_tags(student.tag_type,student.tag_dictionary)
	train_eval_result, train_loss = student.evaluate(test_loader,embeddings_storage_mode='none',speed_test=True)
	# print('Current accuracy: ' + str(train_eval_result.main_score*100))
	# print(train_eval_result.detailed_results)
	

elif args.test:
	student.eval()
	trainer.embeddings_storage_mode = 'cpu'
	trainer.final_test(
		config.get_target_path,
		eval_mini_batch_size=eval_mini_batch_size,
		overall_test=True if int(args.keep_embedding)<0 else False,
		quiet_mode=args.quiet,
		nocrf=args.nocrf,
		# debug=args.debug,
		# keep_embedding = int(args.keep_embedding),
		predict_posterior=args.predict_posterior,
		# sort_data = not args.keep_order,
	)
elif args.parse or args.save_embedding:
	print('Batch Size:',eval_mini_batch_size)
	base_path=Path(config.config['target_dir'])/config.config['model_name']
	if (base_path / "best-model.pt").exists():
		print('Loading pretraining best model')
		if trainer_name == 'ReinforcementTrainer':
			student = student.load(base_path / "best-model.pt", device='cpu')
			for name, module in student.named_modules():
				if 'embeddings' in name or name == '':
					continue
				else:
					module.to(flair.device)
			for name, module in student.named_parameters():
				module.to(flair.device)
		else:
			student = student.load(base_path / "best-model.pt")
		
	elif (base_path / "final-model.pt").exists():
		print('Loading pretraining final model')
		student = student.load(base_path / "final-model.pt")
	else:
		assert 0, str(base_path)+ ' not exist!'
	if trainer_name == 'ReinforcementTrainer':
		import torch
		training_state = torch.load(base_path/'training_state.pt')
		start_episode = training_state['episode']
		student.selection = training_state['best_action']
		name_list=sorted([x.name for x in student.embeddings.embeddings])
		print(name_list)
		print(f"Setting embedding mask to the best action: {student.selection}")
		embedlist = sorted([(embedding.name, embedding) for embedding in student.embeddings.embeddings], key = lambda x: x[0])
		for idx, embedding_tuple in enumerate(embedlist):
			embedding = embedding_tuple[1]
			if student.selection[idx] == 1:
				embedding.to(flair.device)
				if 'elmo' in embedding.name:
					# embedding.reset_elmo()
					# continue
					# pdb.set_trace()
					embedding.ee.elmo_bilm.cuda(device=embedding.ee.cuda_device)
					states=[x.to(flair.device) for x in embedding.ee.elmo_bilm._elmo_lstm._states]
					embedding.ee.elmo_bilm._elmo_lstm._states = states
					for idx in range(len(embedding.ee.elmo_bilm._elmo_lstm._states)):
						embedding.ee.elmo_bilm._elmo_lstm._states[idx]=embedding.ee.elmo_bilm._elmo_lstm._states[idx].to(flair.device)
			else:
				embedding.to('cpu')
				
		for name, module in student.named_modules():
			if 'embeddings' in name or name == '':
				continue
			else:
				module.to(flair.device)
		parameters = [x for x in student.named_parameters()]
		for parameter in parameters:
			name = parameter[0]
			module = parameter[1]
			module.data.to(flair.device)
			if '.' not in name:
				if type(getattr(student, name))==torch.nn.parameter.Parameter:
					setattr(student, name, torch.nn.parameter.Parameter(getattr(student,name).to(flair.device)))
		# pdb.set_trace()
		
	if args.save_embedding:
		for embedding in student.embeddings.embeddings:
			if hasattr(embedding,'fine_tune') and embedding.fine_tune: 
				if not os.path.exists(base_path/embedding.name.split('/')[-1]):
					os.mkdir(base_path/embedding.name.split('/')[-1])
				embedding.tokenizer.save_pretrained(base_path/embedding.name.split('/')[-1])
				embedding.model.save_pretrained(base_path/embedding.name.split('/')[-1])
		exit()
	if not hasattr(student,'use_bert'):
		student.use_bert=False
	if hasattr(student,'word_map'):
		word_map = student.word_map
	else:
		word_map = None
	if hasattr(student,'char_map'):
		char_map = student.char_map
	else:
		char_map = None
	if args.mst:
		student.is_mst=True
	if args.parse_train_and_dev:

		print('Current Model: ', config.config['model_name'])
		print('Current Set: ', 'dev')
		if not os.path.exists('system_pred'):
			os.mkdir('system_pred')
		for index, subcorpus in enumerate(corpus.dev_list):
			# log_line(log)
			# log.info('current corpus: '+self.corpus.targets[index])
			if len(subcorpus)==0:
				continue
			print('Current Lang: ', corpus.targets[index])
			loader=ColumnDataLoader(list(subcorpus),eval_mini_batch_size,use_bert=student.use_bert, model = student, sort_data = not args.keep_order, sentence_level_batch = config.config[trainer_name]['sentence_level_batch'] if 'sentence_level_batch' in config.config[trainer_name] else True)
			loader.assign_tags(student.tag_type,student.tag_dictionary)
			train_eval_result, train_loss = student.evaluate(loader,embeddings_storage_mode='none',
				out_path=Path('system_pred/dev.'+config.config['model_name']+'.conllu'),)
			print('Current accuracy: ' + str(train_eval_result.main_score*100))
			print(train_eval_result.detailed_results)
		print('Current Set: ', 'train')
		for index, subcorpus in enumerate(corpus.train_list):
			# log_line(log)
			# log.info('current corpus: '+self.corpus.targets[index])
			if len(subcorpus)==0:
				continue
			print('Current Lang: ', corpus.targets[index])
			loader=ColumnDataLoader(list(subcorpus),eval_mini_batch_size,use_bert=student.use_bert, model = student, sort_data = not args.keep_order, sentence_level_batch = config.config[trainer_name]['sentence_level_batch'] if 'sentence_level_batch' in config.config[trainer_name] else True)
			loader.assign_tags(student.tag_type,student.tag_dictionary)
			train_eval_result, train_loss = student.evaluate(
				loader,
				embeddings_storage_mode='none',
				out_path=Path('system_pred/train.'+config.config['model_name']+'.conllu'),
			)
			print('Current accuracy: ' + str(train_eval_result.main_score*100))
			print(train_eval_result.detailed_results)
		# print('Current Set: ', 'train+dev')
		# for index, subcorpus in enumerate(corpus.train_list):
		# 	# log_line(log)
		# 	# log.info('current corpus: '+self.corpus.targets[index])
		# 	print('Current Lang: ', corpus.targets[index])
		# 	loader=ColumnDataLoader(list(subcorpus)+list(corpus.dev_list[index]),eval_mini_batch_size,use_bert=student.use_bert, model = student, sort_data = not args.keep_order)
		# 	loader.assign_tags(student.tag_type,student.tag_dictionary)
		# 	train_eval_result, train_loss = student.evaluate(
		# 		loader,
		# 		embeddings_storage_mode='none',
		# 		out_path=Path('outputs/train.'+config.config['model_name']+'.'+tar_file_name+'.conllu'),
		# 	)
		# 	print('Current accuracy: ' + str(train_eval_result.main_score*100))
		# 	print(train_eval_result.detailed_results)
		print('Current Set: ', 'test')
		for index, subcorpus in enumerate(corpus.test_list):
			# log_line(log)
			# log.info('current corpus: '+self.corpus.targets[index])
			if len(subcorpus)==0:
				continue
			print('Current Lang: ', corpus.targets[index])
			loader=ColumnDataLoader(list(subcorpus),eval_mini_batch_size,use_bert=student.use_bert, model = student, sort_data = not args.keep_order, sentence_level_batch = config.config[trainer_name]['sentence_level_batch'] if 'sentence_level_batch' in config.config[trainer_name] else True)
			loader.assign_tags(student.tag_type,student.tag_dictionary)
			train_eval_result, train_loss = student.evaluate(
				loader,
				embeddings_storage_mode='none',
				out_path=Path('system_pred/test.'+config.config['model_name']+'.conllu'),
			)
			print('Current accuracy: ' + str(train_eval_result.main_score*100))
			print(train_eval_result.detailed_results)
	elif args.target_dir != '':
		if args.recur_parse:
			file_dirs=os.listdir(args.target_dir)
			for file_dir in file_dirs:
				tar_dir=os.path.join(args.target_dir,file_dir)
				if not os.path.isdir(tar_dir):
					continue
				if student.tag_type=='dependency':
					corpus=datasets.UniversalDependenciesCorpus(tar_dir,add_root=True,spliter=args.spliter)
				else:
					corpus=datasets.ColumnCorpus(tar_dir, column_format={0: 'text', 1:'ner'}, tag_to_bioes='ner')
				tar_file_name = tar_dir.split('/')[-1]
				print('Parsing the file: '+tar_file_name)
				write_name='outputs/train.'+config.config['model_name']+'.'+tar_file_name+'.conllu'
				print('Writing to file: '+write_name)
				loader=ColumnDataLoader(list(corpus.train),eval_mini_batch_size,use_bert=student.use_bert, model = student, sort_data = not args.keep_order, sentence_level_batch = config.config[trainer_name]['sentence_level_batch'] if 'sentence_level_batch' in config.config[trainer_name] else True)
				loader.assign_tags(student.tag_type,student.tag_dictionary)
				train_eval_result, train_loss = student.evaluate(loader,out_path=Path(write_name),embeddings_storage_mode="none",prediction_mode=True)
				if train_eval_result is not None:
					print('Current accuracy: ' + str(train_eval_result.main_score*100))
					print(train_eval_result.detailed_results)
		else:
			if student.tag_type=='dependency' or student.tag_type=='enhancedud':
				corpus=datasets.UniversalDependenciesCorpus(args.target_dir,add_root=True,spliter=args.spliter)
			else:
				corpus=datasets.ColumnCorpus(args.target_dir, column_format={0: 'text', 1:'ner'}, tag_to_bioes='ner')
			tar_file_name = str(Path(args.target_dir)).split('/')[-1]
			loader=ColumnDataLoader(list(corpus.train),eval_mini_batch_size,use_bert=student.use_bert, model = student, sort_data = not args.keep_order, sentence_level_batch = config.config[trainer_name]['sentence_level_batch'] if 'sentence_level_batch' in config.config[trainer_name] else True)
			loader.assign_tags(student.tag_type,student.tag_dictionary)
			train_eval_result, train_loss = student.evaluate(loader,out_path=Path('outputs/train.'+config.config['model_name']+'.'+tar_file_name+'.conllu'),embeddings_storage_mode="none",prediction_mode=True)
			if train_eval_result is not None:
				print('Current accuracy: ' + str(train_eval_result.main_score*100))
				print(train_eval_result.detailed_results)
	elif args.parse_test:
		loader=ColumnDataLoader(list(corpus.test),eval_mini_batch_size,use_bert=student.use_bert, model = student, sort_data = not args.keep_order, sentence_level_batch = config.config[trainer_name]['sentence_level_batch'] if 'sentence_level_batch' in config.config[trainer_name] else True)
		loader.assign_tags(student.tag_type,student.tag_dictionary)
		train_eval_result, train_loss = student.evaluate(loader,out_path=Path('system_pred/test.'+config.config['model_name']+'.conllu'),embeddings_storage_mode="none",prediction_mode=True)
		if train_eval_result is not None:
			print('Current accuracy: ' + str(train_eval_result.main_score*100))
			print(train_eval_result.detailed_results)
	else:
		loader=ColumnDataLoader(list(corpus.train),eval_mini_batch_size,use_bert=student.use_bert, model = student, sort_data = not args.keep_order, sentence_level_batch = config.config[trainer_name]['sentence_level_batch'] if 'sentence_level_batch' in config.config[trainer_name] else True)
		loader.assign_tags(student.tag_type,student.tag_dictionary)
		train_eval_result, train_loss = student.evaluate(loader,out_path=Path('outputs/train.'+config.config['model_name']+'.'+corpus.targets[0]+'.conllu'),embeddings_storage_mode="none",prediction_mode=True)
		if train_eval_result is not None:
			print('Current accuracy: ' + str(train_eval_result.main_score*100))
			print(train_eval_result.detailed_results)
else:
	getattr(trainer,'train')(**train_config)
# trainer.train(
#   config.get_target_path,
#   learning_rate=0.1,
#   mini_batch_size=32,
#   max_epochs=150
# )

