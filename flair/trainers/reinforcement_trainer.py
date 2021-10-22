"""
Fine-tune trainer: a trainer for finetuning BERT and able to be parallelized based on flair
Author: Xinyu Wang
Contact: wangxy1@shanghaitech.edu.cn
"""

from .distillation_trainer import *
from transformers import (
	AdamW,
	get_linear_schedule_with_warmup,
)
from flair.models.biaffine_attention import BiaffineAttention, BiaffineFunction
# from flair.models.dependency_model import generate_tree, convert_score_back
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
import random
import copy
from flair.parser.utils.alg import crf
import h5py
from flair.models.controller import EmbedController
import numpy as np 
import json

import gc

# def check_garbage():
#   for obj in gc.get_objects():
#       try:
#           if torch.is_tensor(obj):
#               pring(type(obj),obj.size())
#       except:
#           pass


def count_parameters(model):
	total_param = 0
	for name,param in model.named_parameters():
		num_param = np.prod(param.size())
		# print(name,num_param)
		total_param+=num_param
	return total_param

dependency_tasks={'enhancedud', 'dependency', 'srl', 'ner_dp'}
def get_inverse_square_root_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, fix_embedding_steps, steepness = 0.5, factor = 5, model_size=1, last_epoch=-1):
	""" Create a schedule with a learning rate that decreases linearly after
	linearly increasing during a warmup period.
	"""

	def lr_lambda(current_step):
		# step 0 ~ fix_embedding_steps: no modification
		# step fix_embedding_steps ~ num_warmup_steps + fix_embedding_steps: warmup embedding training
		# step num_warmup_steps + fix_embedding_steps ~ : square root decay
		if current_step < fix_embedding_steps:
			return 1
		elif current_step < num_warmup_steps + fix_embedding_steps:
			return float(current_step-fix_embedding_steps) / float(max(1, num_warmup_steps))
		step = max(current_step - num_warmup_steps - fix_embedding_steps, 1)
		return max(0.0, factor * (model_size ** (-0.5) * min(step ** (-steepness), step * num_warmup_steps ** (-steepness - 1))))

	return LambdaLR(optimizer, lr_lambda, last_epoch)

class ReinforcementTrainer(ModelDistiller):
	def __init__(
		self,
		model: flair.nn.Model,
		teachers, # None, for consistency with other trainers
		corpus: ListCorpus,
		optimizer = AdamW,
		controller_optimizer = Adam,
		controller_learning_rate: float = 0.1,
		epoch: int = 0,
		distill_mode = False,
		optimizer_state: dict = None,
		scheduler_state: dict = None,
		use_tensorboard: bool = False,
		language_resample = False,
		config = None,
		is_test: bool = False,
		direct_upsample_rate: int = -1,
		down_sample_amount: int = -1,
		sentence_level_batch: bool = False,
		dev_sample: bool = False,
		assign_doc_id: bool = False,
		train_with_doc: bool = False,
		pretrained_file_dict: dict = {},
		sentence_level_pretrained_data: bool = False,
		# **kwargs,
	):
		"""
		Initialize a model trainer
		:param model: The model that you want to train. The model should inherit from flair.nn.Model
		:param corpus: The dataset used to train the model, should be of type Corpus
		:param optimizer: The optimizer to use (Default AdamW for finetuning BERT)
		:param epoch: The starting epoch (normally 0 but could be higher if you continue training model)
		:param optimizer_state: Optimizer state (necessary if continue training from checkpoint)
		:param scheduler_state: Scheduler state (necessary if continue training from checkpoint)
		:param use_tensorboard: If True, writes out tensorboard information
		"""
		# if teachers is not None:
		#   assert len(teachers)==len(corpus.train_list), 'Training data and teachers should be the same length now!'
		self.model: flair.nn.Model = model
		self.controller: flair.nn.Model = EmbedController(num_actions=len(self.model.embeddings.embeddings), state_size = self.model.embeddings.embedding_length, **config['Controller'])
		self.model.use_rl = True
		if self.controller.model_structure is not None:
			self.model.use_embedding_masks = True
		self.model.embedding_selector = True
		self.corpus: ListCorpus = corpus
		num_languages = len(self.corpus.targets)
		self.controller_learning_rate=controller_learning_rate
		self.corpus2id = {x:i for i,x in enumerate(self.corpus.targets)}
		self.id2corpus = {i:x for i,x in enumerate(self.corpus.targets)}
		self.sentence_level_batch = sentence_level_batch
		if language_resample or direct_upsample_rate>0:
			sent_per_set=torch.FloatTensor([len(x) for x in self.corpus.train_list])
			total_sents=sent_per_set.sum()
			sent_each_dataset=sent_per_set/total_sents
			exp_sent_each_dataset=sent_each_dataset.pow(0.7)
			sent_sample_prob=exp_sent_each_dataset/exp_sent_each_dataset.sum()
		self.sentence_level_pretrained_data=sentence_level_pretrained_data

		if assign_doc_id:
			doc_sentence_dict = {}
			same_corpus_mapping = {'CONLL_06_GERMAN': 'CONLL_03_GERMAN_NEW',
			'CONLL_03_GERMAN_DP': 'CONLL_03_GERMAN_NEW',
			'CONLL_03_DP': 'CONLL_03_ENGLISH',
			'CONLL_03_DUTCH_DP': 'CONLL_03_DUTCH_NEW',
			'CONLL_03_SPANISH_DP': 'CONLL_03_SPANISH_NEW'}
			for corpus_id in range(len(self.corpus2id)):
				
				if self.corpus.targets[corpus_id] in same_corpus_mapping:
					corpus_name = same_corpus_mapping[self.corpus.targets[corpus_id]].lower()+'_'
				else:
					corpus_name = self.corpus.targets[corpus_id].lower()+'_'
				doc_sentence_dict = self.assign_documents(self.corpus.train_list[corpus_id], 'train_', doc_sentence_dict, corpus_name, train_with_doc)
				doc_sentence_dict = self.assign_documents(self.corpus.dev_list[corpus_id], 'dev_', doc_sentence_dict, corpus_name, train_with_doc)
				doc_sentence_dict = self.assign_documents(self.corpus.test_list[corpus_id], 'test_', doc_sentence_dict, corpus_name, train_with_doc)
				if train_with_doc:
					new_sentences=[]
					for sentid, sentence in enumerate(self.corpus.train_list[corpus_id]):
						if sentence[0].text=='-DOCSTART-':
							continue
						new_sentences.append(sentence)
					self.corpus.train_list[corpus_id].sentences = new_sentences.copy()
					self.corpus.train_list[corpus_id].reset_sentence_count

					new_sentences=[]
					for sentid, sentence in enumerate(self.corpus.dev_list[corpus_id]):
						if sentence[0].text=='-DOCSTART-':
							continue
						new_sentences.append(sentence)
					self.corpus.dev_list[corpus_id].sentences = new_sentences.copy()
					self.corpus.dev_list[corpus_id].reset_sentence_count

					new_sentences=[]
					for sentid, sentence in enumerate(self.corpus.test_list[corpus_id]):
						if sentence[0].text=='-DOCSTART-':
							continue
						new_sentences.append(sentence)
					self.corpus.test_list[corpus_id].sentences = new_sentences.copy()
					self.corpus.test_list[corpus_id].reset_sentence_count

			if train_with_doc:
				self.corpus._train: FlairDataset = ConcatDataset([data for data in self.corpus.train_list])
				self.corpus._dev: FlairDataset = ConcatDataset([data for data in self.corpus.dev_list])		
				self.corpus._test: FlairDataset = ConcatDataset([data for data in self.corpus.test_list])		
			# for key in pretrained_file_dict:
			# pdb.set_trace()
			for embedding in self.model.embeddings.embeddings:
				if embedding.name in pretrained_file_dict:
					self.assign_predicted_embeddings(doc_sentence_dict,embedding,pretrained_file_dict[embedding.name])

		for corpus_name in self.corpus2id:
			i = self.corpus2id[corpus_name]
			for sentence in self.corpus.train_list[i]:
				sentence.lang_id=i
			if len(self.corpus.dev_list)>i:
				for sentence in self.corpus.dev_list[i]:
					sentence.lang_id=i
			if len(self.corpus.test_list)>i:
				for sentence in self.corpus.test_list[i]:
					sentence.lang_id=i
			if language_resample:
				length = len(self.corpus.train_list[i])
				# idx = random.sample(range(length), int(sent_sample_prob[i] * total_sents))
				idx = torch.randint(length, (int(sent_sample_prob[i] * total_sents),))
				self.corpus.train_list[i].sentences = [self.corpus.train_list[i][x] for x in idx]
			if direct_upsample_rate>0:
				if len(self.corpus.train_list[i].sentences)<(sent_per_set.max()/direct_upsample_rate).item():
					res_sent=[]
					dev_res_sent=[]
					for sent_batch in range(direct_upsample_rate):
						res_sent+=copy.deepcopy(self.corpus.train_list[i].sentences)
						if config['train']['train_with_dev']:
							dev_res_sent+=copy.deepcopy(self.corpus.dev_list[i].sentences)
					self.corpus.train_list[i].sentences = res_sent
					self.corpus.train_list[i].reset_sentence_count
					if config['train']['train_with_dev']:
						self.corpus.dev_list[i].sentences = dev_res_sent
						self.corpus.dev_list[i].reset_sentence_count
			if down_sample_amount>0:
				if len(self.corpus.train_list[i].sentences)>down_sample_amount:
					self.corpus.train_list[i].sentences = self.corpus.train_list[i].sentences[:down_sample_amount]
					self.corpus.train_list[i].reset_sentence_count
					if config['train']['train_with_dev']:
						self.corpus.dev_list[i].sentences = self.corpus.dev_list[i].sentences[:down_sample_amount]
						self.corpus.dev_list[i].reset_sentence_count
					if dev_sample:
						self.corpus.dev_list[i].sentences = self.corpus.dev_list[i].sentences[:down_sample_amount]
						self.corpus.dev_list[i].reset_sentence_count
		if direct_upsample_rate>0 or down_sample_amount:
			self.corpus._train: FlairDataset = ConcatDataset([data for data in self.corpus.train_list])
			if config['train']['train_with_dev']:
				self.corpus._dev: FlairDataset = ConcatDataset([data for data in self.corpus.dev_list])
		print(self.corpus)

		# self.corpus = self.assign_pretrained_teacher_predictions(self.corpus,self.corpus_teacher,self.teachers)
		self.update_params_group=[]
		
		self.optimizer: torch.optim.Optimizer = optimizer
		if type(optimizer)==str:
			self.optimizer = getattr(torch.optim,optimizer)

		self.controller_optimizer: torch.optim.Optimizer = controller_optimizer
		if type(controller_optimizer) == str:
			self.controller_optimizer = getattr(torch.optim,controller_optimizer)



		self.epoch: int = epoch
		self.scheduler_state: dict = scheduler_state
		self.optimizer_state: dict = optimizer_state
		self.use_tensorboard: bool = use_tensorboard
		
		self.config = config
		self.use_bert = False
		self.bert_tokenizer = None
		for embedding in self.model.embeddings.embeddings:
			if 'bert' in embedding.__class__.__name__.lower():
				self.use_bert=True
				self.bert_tokenizer = embedding.tokenizer

		if hasattr(self.model,'remove_x') and self.model.remove_x:
			for corpus_id in range(len(self.corpus2id)):
				for sent_id, sentence in enumerate(self.corpus.train_list[corpus_id]):
					sentence.orig_sent=copy.deepcopy(sentence)
					words = [x.text for x in sentence.tokens]
					if '<EOS>' in words:
						eos_id = words.index('<EOS>')
						sentence.chunk_sentence(0,eos_id)
					else:
						pass
				for sent_id, sentence in enumerate(self.corpus.dev_list[corpus_id]):
					sentence.orig_sent=copy.deepcopy(sentence)
					words = [x.text for x in sentence.tokens]
					if '<EOS>' in words:
						eos_id = words.index('<EOS>')
						sentence.chunk_sentence(0,eos_id)
					else:
						pass
				for sent_id, sentence in enumerate(self.corpus.test_list[corpus_id]):
					sentence.orig_sent=copy.deepcopy(sentence)
					words = [x.text for x in sentence.tokens]
					if '<EOS>' in words:
						eos_id = words.index('<EOS>')
						sentence.chunk_sentence(0,eos_id)
					else:
						pass
			self.corpus._train: FlairDataset = ConcatDataset([data for data in self.corpus.train_list])
			self.corpus._dev: FlairDataset = ConcatDataset([data for data in self.corpus.dev_list])		
			self.corpus._test: FlairDataset = ConcatDataset([data for data in self.corpus.test_list])		



	def train(
		self,
		base_path: Union[Path, str],
		learning_rate: float = 5e-5,
		mini_batch_size: int = 32,
		eval_mini_batch_size: int = None,
		max_epochs: int = 100,
		max_episodes: int = 10,
		anneal_factor: float = 0.5,
		patience: int = 10,
		min_learning_rate: float = 5e-9,
		train_with_dev: bool = False,
		macro_avg: bool = True,
		monitor_train: bool = False,
		monitor_test: bool = False,
		embeddings_storage_mode: str = "cpu",
		checkpoint: bool = False,
		save_final_model: bool = True,
		anneal_with_restarts: bool = False,
		shuffle: bool = True,
		true_reshuffle: bool = False,
		param_selection_mode: bool = False,
		num_workers: int = 4,
		sampler=None,
		use_amp: bool = False,
		amp_opt_level: str = "O1",
		max_epochs_without_improvement = 30,
		warmup_steps: int = 0,
		use_warmup: bool = True,
		gradient_accumulation_steps: int = 1,
		lr_rate: int = 1,
		decay: float = 0.75,
		decay_steps: int = 5000,
		sort_data: bool = True,
		fine_tune_mode: bool = False,
		debug: bool = False,
		min_freq: int = -1,
		min_lemma_freq: int = -1,
		min_pos_freq: int = -1,
		rootschedule: bool = False,
		freezing: bool = False,
		log_reward: bool = False,
		sqrt_reward: bool = False,
		controller_momentum: float = 0.0,
		discount: float = 0.5,
		curriculum_file = None,
		random_search = False,
		continue_training = False,
		old_reward = False,
		**kwargs,
	) -> dict:

		"""
		Trains any class that implements the flair.nn.Model interface.
		:param base_path: Main path to which all output during training is logged and models are saved
		:param learning_rate: Initial learning rate
		:param mini_batch_size: Size of mini-batches during training
		:param eval_mini_batch_size: Size of mini-batches during evaluation
		:param max_epochs: Maximum number of epochs to train. Terminates training if this number is surpassed.
		:param anneal_factor: The factor by which the learning rate is annealed
		:param patience: Patience is the number of epochs with no improvement the Trainer waits
		 until annealing the learning rate
		:param min_learning_rate: If the learning rate falls below this threshold, training terminates
		:param train_with_dev: If True, training is performed using both train+dev data
		:param monitor_train: If True, training data is evaluated at end of each epoch
		:param monitor_test: If True, test data is evaluated at end of each epoch
		:param embeddings_storage_mode: One of 'none' (all embeddings are deleted and freshly recomputed),
		'cpu' (embeddings are stored on CPU) or 'gpu' (embeddings are stored on GPU)
		:param checkpoint: If True, a full checkpoint is saved at end of each epoch
		:param save_final_model: If True, final model is saved
		:param anneal_with_restarts: If True, the last best model is restored when annealing the learning rate
		:param shuffle: If True, data is shuffled during training
		:param param_selection_mode: If True, testing is performed against dev data. Use this mode when doing
		parameter selection.
		:param num_workers: Number of workers in your data loader.
		:param sampler: You can pass a data sampler here for special sampling of data.
		:param kwargs: Other arguments for the Optimizer
		:return:
		"""
		self.n_gpu = torch.cuda.device_count()
		default_learning_rate = learning_rate
		self.embeddings_storage_mode=embeddings_storage_mode
		self.mini_batch_size=mini_batch_size
		if self.use_tensorboard:
			try:
				from torch.utils.tensorboard import SummaryWriter

				writer = SummaryWriter()
			except:
				log_line(log)
				log.warning(
					"ATTENTION! PyTorch >= 1.1.0 and pillow are required for TensorBoard support!"
				)
				log_line(log)
				self.use_tensorboard = False
				pass

		if use_amp:
			if sys.version_info < (3, 0):
				raise RuntimeError("Apex currently only supports Python 3. Aborting.")
			if amp is None:
				raise RuntimeError(
					"Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
					"to enable mixed-precision training."
				)

		if eval_mini_batch_size is None:
			eval_mini_batch_size = mini_batch_size

		# cast string to Path
		if type(base_path) is str:
			base_path = Path(base_path)

		log_handler = add_file_handler(log, base_path / "training.log")
		
		log_line(log)
		log.info(f'Model: "{self.model}"')
		log_line(log)
		log.info(f'Corpus: "{self.corpus}"')
		log_line(log)
		log.info("Parameters:")
		log.info(f' - Optimizer: "{self.optimizer.__name__}"')
		log.info(f' - learning_rate: "{learning_rate}"')
		log.info(f' - mini_batch_size: "{mini_batch_size}"')
		log.info(f' - patience: "{patience}"')
		log.info(f' - anneal_factor: "{anneal_factor}"')
		log.info(f' - max_epochs: "{max_epochs}"')
		log.info(f' - shuffle: "{shuffle}"')
		log.info(f' - train_with_dev: "{train_with_dev}"')
		log.info(f' - word min_freq: "{min_freq}"')
		log_line(log)
		log.info(f'Model training base path: "{base_path}"')
		log_line(log)
		log.info(f"Device: {flair.device}")
		log_line(log)
		log.info(f"Embeddings storage mode: {embeddings_storage_mode}")

		# determine what splits (train, dev, test) to evaluate and log
		if monitor_train:
			assert 0, 'monitor_train is not supported now!'            
		# if train_with_dev:
		#   assert 0, 'train_with_dev is not supported now!'

		log_train = True if monitor_train else False
		log_test = (
			True
			if (not param_selection_mode and self.corpus.test and monitor_test)
			else False
		)
		log_dev = True if not train_with_dev else False

		# prepare loss logging file and set up header
		loss_txt = init_output_file(base_path, "loss.tsv")



		controller_optimizer: torch.optim.Optimizer = self.controller_optimizer(
			self.controller.parameters(), lr=self.controller_learning_rate, momentum = controller_momentum
		)
		
		if continue_training:
			# though this is not the final model of training, but we use this currently to save the space
			if (base_path / "best-model.pt").exists():
				self.model = self.model.load(base_path / "best-model.pt")
			self.controller = self.controller.load(base_path / "controller.pt")
			if (base_path/'controller_optimizer_state.pt').exists():
				controller_optimizer.load_state_dict(torch.load(base_path/'controller_optimizer_state.pt'))
			training_state = torch.load(base_path/'training_state.pt')
			start_episode = training_state['episode']
			self.best_action = training_state['best_action']
			self.action_dict = training_state['action_dict']
			baseline_score = training_state['baseline_score']
			# pdb.set_trace()
		else:
			start_episode=0
			self.action_dict = {}
			baseline_score=0
		# weight_extractor = WeightExtractor(base_path)
		# finetune_params = {name:param for name,param in self.model.named_parameters()}
		finetune_params=[param for name,param in self.model.named_parameters() if 'embedding' in name or name=='linear.weight' or name=='linear.bias']
		other_params=[param for name,param in self.model.named_parameters() if 'embedding' not in name and name !='linear.weight' and name !='linear.bias']
		# other_params = {name:param for name,param in self.model.named_parameters() if 'embeddings' not in name}
		

		# minimize training loss if training with dev data, else maximize dev score
		
		# start from here, the train data is a list now


		train_data = self.corpus.train_list
		if train_with_dev:
			train_data = [ConcatDataset([train, self.corpus.dev_list[index]]) for index, train in enumerate(self.corpus.train_list)]
		batch_loader=ColumnDataLoader(ConcatDataset(train_data),mini_batch_size,shuffle,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, model = self.model, sentence_level_batch = self.sentence_level_batch)
		batch_loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
		
		if not train_with_dev:
			if macro_avg:

				dev_loaders=[ColumnDataLoader(list(subcorpus),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, sort_data=sort_data, model = self.model, sentence_level_batch = self.sentence_level_batch) \
							 for subcorpus in self.corpus.dev_list]
				for loader in dev_loaders:
					loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)

				test_loaders=[ColumnDataLoader(list(subcorpus),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, sort_data=sort_data, model = self.model, sentence_level_batch = self.sentence_level_batch) \
							 for subcorpus in self.corpus.test_list]
				for loader in test_loaders:
					loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)

			else:
				dev_loader=ColumnDataLoader(list(self.corpus.dev),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, sort_data=sort_data, model = self.model, sentence_level_batch = self.sentence_level_batch)
				dev_loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
				test_loader=ColumnDataLoader(list(self.corpus.test),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, sort_data=sort_data, model = self.model, sentence_level_batch = self.sentence_level_batch)
				test_loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
		
		### Finetune Scheduler
		if freezing:
			for embedding in self.model.embeddings.embeddings:
				embedding.fine_tune = False
		dev_score_history = []
		dev_loss_history = []
		test_score_history = []
		test_loss_history = []
		train_loss_history = []
		if self.n_gpu > 1:
			self.model = torch.nn.DataParallel(self.model)

		
		score_list=[]
		
		name_list=sorted([x.name for x in self.model.embeddings.embeddings])
		# for faster quit training, use larger anneal factor to quitting
		min_learning_rate = learning_rate/1000
		
		curriculum=[]
		if curriculum_file is not None:
			with open(curriculum_file) as f:
				curriculum = json.loads(f.read())
		# pdb.set_trace()
		# self.model.embeddings.to('cpu')
		self.model.embeddings=self.model.embeddings.to('cpu')
		with torch.no_grad():
			if macro_avg:
				self.gpu_friendly_assign_embedding([batch_loader]+dev_loaders+test_loaders)
			else:
				self.gpu_friendly_assign_embedding([batch_loader,dev_loader,test_loader])
		# pdb.set_trace()
		
		try:
			for episode in range(start_episode,max_episodes):
				best_score=0
				learning_rate = default_learning_rate
				# reinitialize the optimizer and scheduler in training

				if len(self.update_params_group)>0:
					optimizer: torch.optim.Optimizer = self.optimizer(
						[{"params":other_params,"lr":learning_rate*lr_rate},
						{"params":self.update_params_group,"lr":learning_rate*lr_rate},
						{"params":finetune_params}
						],
						lr=learning_rate, **kwargs
					)
				else:
					optimizer: torch.optim.Optimizer = self.optimizer(
						[{"params":other_params,"lr":learning_rate*lr_rate},
						{"params":finetune_params}
						],
						lr=learning_rate, **kwargs
					)
				if self.optimizer_state is not None:
					optimizer.load_state_dict(self.optimizer_state)

				if use_amp:
					self.model, optimizer = amp.initialize(
						self.model, optimizer, opt_level=amp_opt_level
					)

				if not fine_tune_mode:
					if self.model.tag_type in dependency_tasks:
						scheduler = ExponentialLR(optimizer, decay**(1/decay_steps))
					else:
						anneal_mode = "min" if train_with_dev else "max"
						scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
							optimizer,
							factor=anneal_factor,
							patience=patience,
							mode=anneal_mode,
							verbose=True,
						)
				else:
					### Finetune Scheduler
					t_total = len(batch_loader) // gradient_accumulation_steps * max_epochs
					if rootschedule:
						warmup_steps = len(batch_loader)
						scheduler = get_inverse_square_root_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total, fix_embedding_steps = warmup_steps)
					else:
						if use_warmup:
							warmup_steps = len(batch_loader)
						scheduler = get_linear_schedule_with_warmup(
							optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
						)

				if self.scheduler_state is not None:
					scheduler.load_state_dict(self.scheduler_state) 
				




				log.info(
					f"================================== Start episode {episode + 1} =================================="
				)
				if self.controller.model_structure is not None:
					log.info("#### Current Training Action Distributions ####")
					self.assign_embedding_masks(batch_loader,sample=True, first_episode= episode == 0)
					log.info("#### Current Dev Action Distributions ####")
					for dev_loader in dev_loaders: self.assign_embedding_masks(dev_loader,sample=False, first_episode= episode == 0)
					log.info("#### Current Test Action Distributions ####")
					for test_loader in test_loaders: self.assign_embedding_masks(test_loader,sample=False, first_episode= episode == 0)
					print(name_list)
				else:
					state = self.model.get_state()
					action, log_prob = self.controller.sample(state)
					if episode == 0 and not random_search:
						log_prob = torch.log(torch.sigmoid(self.controller.get_value()))
						action = torch.ones_like(action)
						self.controller.previous_selection = action

					if curriculum_file is None:
						curriculum.append(action.cpu().tolist())
					else:
						action = torch.Tensor(curriculum[episode]).type_as(action)
					print(name_list)
					print(action)
					print(self.controller(None))
					self.model.selection=action
			
				previous_learning_rate = learning_rate
				training_order = None
				bad_epochs2=0
				for epoch in range(0 + self.epoch, max_epochs + self.epoch):
					log_line(log)

					# get new learning rate
					if self.model.use_crf:
						learning_rate = optimizer.param_groups[0]["lr"]
					else:
						for group in optimizer.param_groups:
							learning_rate = group["lr"]
					if freezing and epoch == 1+self.epoch and fine_tune_mode:
						for embedding in self.model.embeddings.embeddings:
							if 'flair' in embedding.__class__.__name__.lower():
								embedding.fine_tune = False
								continue
							embedding.fine_tune = True
					# reload last best model if annealing with restarts is enabled
					if (
						learning_rate != previous_learning_rate
						and anneal_with_restarts
						and (base_path / "best-model.pt").exists()
					):
						log.info("resetting to best model")
						self.model.load(base_path / "best-model.pt")

					previous_learning_rate = learning_rate

					# stop training if learning rate becomes too small
					if learning_rate < min_learning_rate and warmup_steps <= 0:
						log_line(log)
						log.info("learning rate too small - quitting training!")
						log_line(log)
						break
					if self.model.tag_type in dependency_tasks:
						if bad_epochs2>=max_epochs_without_improvement:
							log_line(log)
							log.info(str(bad_epochs2) + " epochs after improvement - quitting training!")
							log_line(log)
							break
					if shuffle:
						batch_loader.reshuffle()
					if true_reshuffle:
						
						batch_loader.true_reshuffle()
						
						batch_loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
						
					self.model.train()
					self.controller.train()
					# TODO: check teacher parameters fixed and with eval() mode

					train_loss: float = 0

					seen_batches = 0
					#total_number_of_batches = sum([len(loader) for loader in batch_loader])
					total_number_of_batches = len(batch_loader)

					modulo = max(1, int(total_number_of_batches / 10))

					# process mini-batches
					batch_time = 0
					total_sent=0
					
					
					for batch_no, student_input in enumerate(batch_loader):
						# for group in optimizer.param_groups:
						#   temp_lr = group["lr"]
						# log.info('lr: '+str(temp_lr))
						# print(self.language_weight.softmax(1))
						# print(self.biaffine.U)
						
						start_time = time.time()
						total_sent+=len(student_input)
						try:
							
							loss = self.model.forward_loss(student_input)

							if self.model.use_decoder_timer:
								decode_time=time.time() - self.model.time
							optimizer.zero_grad()
							if self.n_gpu>1:
								loss = loss.mean()  # mean() to average on multi-gpu parallel training
							# Backward
							if use_amp:
								with amp.scale_loss(loss, optimizer) as scaled_loss:
									scaled_loss.backward()
							else:
								loss.backward()
								pass
						except Exception:
							traceback.print_exc()
							pdb.set_trace()
						torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
						if len(self.update_params_group)>0:
							torch.nn.utils.clip_grad_norm_(self.update_params_group, 5.0)
						optimizer.step()
						if (fine_tune_mode or self.model.tag_type in dependency_tasks):
							scheduler.step()

						seen_batches += 1
						train_loss += loss.item()

						# depending on memory mode, embeddings are moved to CPU, GPU or deleted
						store_embeddings(student_input, embeddings_storage_mode)
						if embeddings_storage_mode == "none" and hasattr(student_input,'features'):
							del student_input.features
						batch_time += time.time() - start_time
						if batch_no % modulo == 0:
							# print less information
							# if self.model.use_decoder_timer:
							#   log.info(
							#       f"epoch {epoch + 1} - iter {batch_no}/{total_number_of_batches} - loss "
							#       f"{train_loss / seen_batches:.8f} - samples/sec: {total_sent / batch_time:.2f} - decode_sents/sec: {total_sent / decode_time:.2f}"
							#   )
								
							# else:
							#   log.info(
							#       f"epoch {epoch + 1} - iter {batch_no}/{total_number_of_batches} - loss "
							#       f"{train_loss / seen_batches:.8f} - samples/sec: {total_sent / batch_time:.2f}"
							#   )
							total_sent = 0
							batch_time = 0
							iteration = epoch * total_number_of_batches + batch_no
							# if not param_selection_mode:
							#   weight_extractor.extract_weights(
							#       self.model.state_dict(), iteration
							#   )
					train_loss /= seen_batches

					self.model.eval()

					log_line(log)
					log.info(
						f"EPISODE {episode+1}, EPOCH {epoch + 1} done: loss {train_loss:.4f} - lr {learning_rate}"
					)

					if self.use_tensorboard:
						writer.add_scalar("train_loss", train_loss, epoch + 1)

					# anneal against train loss if training with dev, otherwise anneal against dev score
					current_score = train_loss

					# evaluate on train / dev / test split depending on training settings
					result_line: str = ""

					if log_train:
						train_eval_result, train_loss = self.model.evaluate(
							batch_loader,
							embeddings_storage_mode=embeddings_storage_mode,
						)
						result_line += f"\t{train_eval_result.log_line}"

						# depending on memory mode, embeddings are moved to CPU, GPU or deleted
						store_embeddings(self.corpus.train, embeddings_storage_mode)
						if embeddings_storage_mode == "none" and hasattr(self.corpus.train,'features'):
							del self.corpus.train.features
					log.info(f"==================Evaluating development set==================") 
					if log_dev:
						if macro_avg:
							
							if type(self.corpus) is ListCorpus:
								result_dict={}
								loss_list=[]
								print_sent='\n'
								
								for index, loader in enumerate(dev_loaders):
									if len(loader) == 0:
										continue
									# log_line(log)
									# log.info('current corpus: '+self.corpus.targets[index])
									current_result, dev_loss = self.model.evaluate(
										loader,
										embeddings_storage_mode=embeddings_storage_mode,
									)
									result_dict[self.corpus.targets[index]]=current_result.main_score*100
									print_sent+=self.corpus.targets[index]+'\t'+f'{result_dict[self.corpus.targets[index]]:.2f}'+'\t'
									loss_list.append(dev_loss)
									# log.info(current_result.log_line)
									# log.info(current_result.detailed_results)
							else:
								assert 0, 'not defined!'
							mavg=sum(result_dict.values())/len(result_dict)
							log.info('Macro Average: '+f'{mavg:.2f}'+'\tMacro avg loss: ' + f'{((sum(loss_list)/len(loss_list)).item()):.2f}' +  print_sent)
							dev_score_history.append(mavg)
							dev_loss_history.append((sum(loss_list)/len(loss_list)).item())
							
							current_score = mavg
						else:
							dev_eval_result, dev_loss = self.model.evaluate(
								dev_loader,
								embeddings_storage_mode=embeddings_storage_mode,
							)
							result_line += f"\t{dev_loss}\t{dev_eval_result.log_line}"
							log.info(
								f"DEV : loss {dev_loss} - score {dev_eval_result.main_score}"
							)
							# calculate scores using dev data if available
							# append dev score to score history
							dev_score_history.append(dev_eval_result.main_score)
							dev_loss_history.append(dev_loss)

							current_score = dev_eval_result.main_score
						# depending on memory mode, embeddings are moved to CPU, GPU or deleted
						store_embeddings(self.corpus.dev, embeddings_storage_mode)
						if embeddings_storage_mode == "none" and hasattr(self.corpus.dev,'features'):
							del self.corpus.dev.features
						if self.use_tensorboard:
							writer.add_scalar("dev_loss", dev_loss, epoch + 1)
							writer.add_scalar(
								"dev_score", dev_eval_result.main_score, epoch + 1
							)


					if current_score>=baseline_score:
						log.info(f"==================Evaluating test set==================")    
						if macro_avg:
							
							if type(self.corpus) is ListCorpus:
								result_dict={}
								loss_list=[]
								print_sent='\n'
								for index, loader in enumerate(test_loaders):
									# log_line(log)
									# log.info('current corpus: '+self.corpus.targets[index])
									if len(loader) == 0:
										continue
									current_result, test_loss = self.model.evaluate(
										loader,
										embeddings_storage_mode=embeddings_storage_mode,
									)
									result_dict[self.corpus.targets[index]]=current_result.main_score*100
									print_sent+=self.corpus.targets[index]+'\t'+f'{result_dict[self.corpus.targets[index]]:.2f}'+'\t'
									loss_list.append(test_loss)
									# log.info(current_result.log_line)
									# log.info(current_result.detailed_results)
							else:
								assert 0, 'not defined!'
							mavg=sum(result_dict.values())/len(result_dict)
							log.info('Test Average: '+f'{mavg:.2f}'+'\tTest avg loss: ' + f'{((sum(loss_list)/len(loss_list)).item()):.2f}' +  print_sent)
							test_score_history.append(mavg)
							test_loss_history.append((sum(loss_list)/len(loss_list)).item())
						else:
							test_eval_result, test_loss = self.model.evaluate(
								test_loader,
								embeddings_storage_mode=embeddings_storage_mode,
							)
							result_line += f"\t{test_loss}\t{test_eval_result.log_line}"
							log.info(
								f"test : loss {test_loss} - score {test_eval_result.main_score}"
							)
							# calculate scores using test data if available
							# append test score to score history
							test_score_history.append(test_eval_result.main_score)
							test_loss_history.append(test_loss)

						# depending on memory mode, embeddings are moved to CPU, GPU or deleted
						store_embeddings(self.corpus.test, embeddings_storage_mode)
						if embeddings_storage_mode == "none" and hasattr(self.corpus.test,'features'):
							del self.corpus.test.features
						if self.use_tensorboard:
							writer.add_scalar("test_loss", test_loss, epoch + 1)
							writer.add_scalar(
								"test_score", test_eval_result.main_score, epoch + 1
							)


					# determine learning rate annealing through scheduler
					if not fine_tune_mode and self.model.tag_type not in dependency_tasks:
						scheduler.step(current_score)
					if current_score>best_score:
						best_score=current_score
						bad_epochs2=0
					else:
						bad_epochs2+=1
					train_loss_history.append(train_loss)

					# determine bad epoch number
					try:
						bad_epochs = scheduler.num_bad_epochs
					except:
						bad_epochs = 0
					for group in optimizer.param_groups:
						new_learning_rate = group["lr"]
					if new_learning_rate != previous_learning_rate:
						bad_epochs = patience + 1

					# log bad epochs
					log.info(f"BAD EPOCHS (no improvement): {bad_epochs}")
					log.info(f"GLOBAL BAD EPOCHS (no improvement): {bad_epochs2}")

					# if checkpoint is enable, save model at each epoch
					# if checkpoint and not param_selection_mode:
					#   if self.n_gpu>1:
					#       self.model.module.save_checkpoint(
					#           base_path / "checkpoint.pt",
					#           optimizer.state_dict(),
					#           scheduler.state_dict(),
					#           epoch + 1,
					#           train_loss,
					#       )
					#   else:
					#       self.model.save_checkpoint(
					#           base_path / "checkpoint.pt",
					#           optimizer.state_dict(),
					#           scheduler.state_dict(),
					#           epoch + 1,
					#           train_loss,
					#       )

					# if we use dev data, remember best model based on dev evaluation score
					if (
						not train_with_dev
						and not param_selection_mode
						and current_score >= baseline_score
					):
						log.info(f"==================Saving the current overall best model: {current_score}==================") 
						if self.n_gpu>1:
							self.model.module.save(base_path / "best-model.pt")
						else:
							self.model.save(base_path / "best-model.pt")
							self.controller.save(base_path/ "controller.pt")
						baseline_score = current_score
						

				# if we do not use dev data for model selection, save final model
				# if save_final_model and not param_selection_mode:
				#   self.model.save(base_path / "final-model.pt")

				# pdb.set_trace()               
				log.info(
					f"================================== End episode {episode + 1} =================================="
				)
				# avoid back-propagation at the first iteration because reward is too large
				controller_optimizer.zero_grad()
				self.controller.zero_grad()
				if self.controller.model_structure is not None:
					if episode == 0:
						previous_best_score = best_score
						log.info(f"Setting baseline score to: {baseline_score}")
					else:
						base_reward = best_score - previous_best_score

						controller_loss = 0
						total_sent = 0
						if log_reward:
							base_reward = np.sign(base_reward)*np.log(np.abs(base_reward)+1)
						if sqrt_reward:
							base_reward = np.sign(base_reward)*np.sqrt(np.abs(base_reward))
						# pdb.set_trace()
						total_reward_at_each_position = torch.zeros(self.controller.num_actions).float().to(flair.device)
						for batch in batch_loader:

							action_change=torch.abs(batch.embedding_mask.to(flair.device)-batch.previous_embedding_mask.to(flair.device))
							reward = base_reward * (discount ** (action_change.sum(-1)-1))
							reward_at_each_position=reward[:,None]*action_change
							controller_loss+=-(batch.log_prob.to(flair.device)*reward_at_each_position).sum()
							total_sent+=len(batch)
							total_reward_at_each_position+=reward_at_each_position.sum(0)
						log.info(f"Current Reward at each position: {total_reward_at_each_position}")
						controller_loss/=total_sent
						controller_loss.backward()
						controller_optimizer.step()
					if best_score >= baseline_score:
						baseline_score = best_score
				else:
					if episode == 0:
						baseline_score = best_score
						log.info(f"Setting baseline score to: {baseline_score}")

						self.best_action = action
						self.controller.best_action = action
						log.info(f"Setting baseline action to: {self.best_action}")
					else:
						log.info(f"previous distributions: ")
						print(self.controller(None))
						# reward = best_score-baseline_score
						controller_loss = 0
						# pdb.set_trace()
						action_count = 0 
						average_reward = 0
						reward_at_each_position = torch.zeros_like(action)
						count_at_each_position = torch.zeros_like(action)
						if old_reward:
							# pdb.set_trace()
							reward = best_score - baseline_score
							reward_at_each_position += reward
						else:
							for prev_action in self.action_dict:
								reward = best_score - max(self.action_dict[prev_action]['scores'])
								prev_action = torch.Tensor(prev_action).type_as(action)
								if log_reward:
									reward = np.sign(reward)*np.log(np.abs(reward)+1)
								if sqrt_reward:
									reward = np.sign(reward)*np.sqrt(np.abs(reward))

								# reward* (discount^hamming_distance) to reduce the affect of long distance embeddings
								reward = reward * (discount ** (torch.abs(action-prev_action).sum()-1))
								average_reward += reward
								reward_at_each_position+=reward*torch.abs(action-prev_action)
								count_at_each_position+=torch.abs(action-prev_action)
								# controller_loss-=(log_prob*reward*torch.abs(action-prev_action)).sum()
								# remove the same action in the action_dict, since no reward
								if torch.abs(action-prev_action).sum() > 0:
									action_count+=1
						# controller_loss=controller_loss/action_count
						# pdb.set_trace()
						count_at_each_position[torch.where(count_at_each_position==0)]+=1
						controller_loss-=(log_prob*reward_at_each_position).sum()
						# controller_loss-=(log_prob*reward_at_each_position/count_at_each_position).sum()
						# only update the probability of embeddings that changes the selection compared to previous action
						# pdb.set_trace()
						# controller_loss = -(log_prob*reward*torch.abs(action-self.best_action)).sum()
						if random_search:
							log.info('================= Doing random search, stop updating the controller =================')
						else:
							controller_loss.backward()
							print("#=================")
							print(self.controller.selector)
							print(self.controller.selector.grad)
							# print(self.controller.selector - self.controller.selector.grad*self.controller_learning_rate)
							# pdb.set_trace()
							controller_optimizer.step()
							print(self.controller.selector)
							print("#=================")
							# pdb.set_trace()
						
						log.info(f"After distributions: ")
						print(self.controller(None))
						# pdb.set_trace()
						if best_score >= baseline_score:
							baseline_score = best_score
							self.best_action = action
							self.controller.best_action = action
							log.info(f"Setting baseline score to: {baseline_score}")
							log.info(f"Setting baseline action to: {self.best_action}") 

						log.info('=============================================')
						log.info(f"Current Action: {action}")
						log.info(f"Current best score: {best_score}")
						log.info(f"Current total Reward: {average_reward}")
						log.info(f"Current Reward at each position: {reward_at_each_position}")
						log.info('=============================================')
						log.info(f"Overall best Action: {self.best_action}")
						log.info(f"Overall best score: {baseline_score}")
						log.info(f"State dictionary: {self.action_dict}")
						log.info('=============================================')
						
					# pdb.set_trace()
					curr_action = tuple(action.cpu().tolist())
					if curr_action not in self.action_dict:
						self.action_dict[curr_action] = {}
						self.action_dict[curr_action]['counts']=0
						self.action_dict[curr_action]['scores']=[]
						# self.action_dict[curr_action]['scores'].append(best_score)
					self.action_dict[curr_action]['counts']+=1
					self.action_dict[curr_action]['scores'].append(best_score)

				training_state = {
								'episode':episode,
								'best_action':self.best_action if self.controller.model_structure is None else None,
								'baseline_score': baseline_score,
								'action_dict': self.action_dict,
								}
				torch.save(training_state,base_path/'training_state.pt')
				torch.save(controller_optimizer.state_dict(),base_path/'controller_optimizer_state.pt')
				# pdb.set_trace()

		except KeyboardInterrupt:
			log_line(log)
			log.info("Exiting from training early.")

			if self.use_tensorboard:
				writer.close()

			if not param_selection_mode:
				log.info("Saving model ...")
				self.model.save(base_path / "final-model.pt")
				log.info("Done.")
		# pdb.set_trace()
		if self.controller.model_structure is None:
			print(name_list)
			print(self.controller(state)>=0.5)

			for action in self.action_dict:
				self.action_dict[action]['average']=sum(self.action_dict[action]['scores'])/self.action_dict[action]['counts']
			log.info(f"Final State dictionary: {self.action_dict}")
			self.model.selection=self.best_action
			with open(base_path/"curriculum.json",'w') as f:
				f.write(json.dumps(curriculum))

		# test best model if test data is present
		if self.corpus.test:
			final_score = self.final_test(base_path, eval_mini_batch_size, num_workers)
		else:
			final_score = 0
			log.info("Test data not provided setting final score to 0")
		log.removeHandler(log_handler)
		if self.use_tensorboard:
			writer.close()
		if self.model.use_language_attention:
			if self.model.biaf_attention:
				print(language_weight.softmax(1))
			else:
				print(self.language_weight.softmax(1))
		return {
			"test_score": final_score,
			"dev_score_history": dev_score_history,
			"test_score_history": test_score_history,
			"train_loss_history": train_loss_history,
			"dev_loss_history": dev_loss_history,
			"test_loss_history": test_loss_history,
		}
	@property
	def interpolation(self):
		try:
			return self.config['interpolation']
		except:
			return 0.5
	@property
	def teacher_annealing(self):
		try:
			return self.config['teacher_annealing']
		except:
			return False
	@property
	def anneal_factor(self):
		try:
			return self.config['anneal_factor']
		except:
			return 2

	def assign_embedding_masks(self, data_loader, sample=False, first_episode=False):
		lang_dict = {}
		distr_dict = {}
		for batch_no, sentences in enumerate(data_loader):
			
			lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
			longest_token_sequence_in_batch: int = max(lengths)
			
			self.model.embeddings.embed(sentences)
			sentence_tensor = torch.cat([sentences.features[x].to(flair.device) for x in sorted(sentences.features.keys())],-1)
			mask=self.model.sequence_mask(torch.tensor(lengths),longest_token_sequence_in_batch).to(flair.device).type_as(sentence_tensor)

			# sum over all embeddings to get the sentence level features
			# sentence_feature = (sentence_tensor*mask).sum(-2)/mask.sum(-1)

			# given the sentence feature, calculate the embedding mask selection and log probability
			sentence_tensor = sentence_tensor.detach()
			if sample:
				selection, log_prob = self.controller.sample(sentence_tensor,mask)
				# pdb.set_trace()
				selection = selection.to('cpu')
				log_prob = log_prob.to('cpu')
				sentences.log_prob = log_prob
			else:
				prediction = self.controller(sentence_tensor,mask)
				selection = prediction >= 0.5
				for idx in range(len(selection)):
					if selection[idx].sum() == 0:
						# pdb.set_trace()
						selection[idx][torch.argmax(prediction[idx])]=1
						# m_temp = torch.distributions.Bernoulli(one_prob[idx])
						# selection[idx] = m_temp.sample()

				selection = selection.to('cpu')
			
			if first_episode:
				selection = torch.ones_like(selection)

			# for idx in range(len(selection)):
			#   if sentences[idx].lang_id == 0:
			#       selection[idx,0]=1
			#       selection[idx,1]=0
			#   if sentences[idx].lang_id == 1:
			#       selection[idx,0]=0
			#       selection[idx,1]=1

			if hasattr(sentences,'embedding_mask'):
				sentences.previous_embedding_mask = sentences.embedding_mask
			sentences.embedding_mask = selection
			# pdb.set_trace()
			distribution=self.controller(sentence_tensor,mask)
			for sent_id, sentence in enumerate(sentences):
				if hasattr(sentence,'embedding_mask'):
					sentence.previous_embedding_mask = selection[sent_id]
				sentence.embedding_mask = selection[sent_id]
				if sample:
					sentence.log_prob = log_prob[sent_id]

				if sentence.lang_id not in lang_dict:
					lang_dict[sentence.lang_id] = []
					distr_dict[sentence.lang_id] = []
				lang_dict[sentence.lang_id].append(selection[sent_id])
				distr_dict[sentence.lang_id].append(distribution[sent_id])
			

		# pdb.set_trace()
		for lang_id in lang_dict:
			print(self.id2corpus[lang_id], (sum(lang_dict[lang_id])/len(lang_dict[lang_id])).tolist())
			print(self.id2corpus[lang_id], (sum(distr_dict[lang_id])/len(distr_dict[lang_id])).tolist())
		return


	# def gpu_friendly_assign_embedding(self,loaders):
	#   # pdb.set_trace()
	#   for embedding in self.model.embeddings.embeddings:
	#       if ('WordEmbeddings' not in embedding.__class__.__name__ and 'Char' not in embedding.__class__.__name__ and 'Lemma' not in embedding.__class__.__name__ and 'POS' not in embedding.__class__.__name__) and not (hasattr(embedding,'fine_tune') and embedding.fine_tune):
	#           print(embedding.name, count_parameters(embedding))
	#           # 
	#           embedding.to(flair.device)
	#           for loader in loaders:
	#               for sentences in loader:
	#                   lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
	#                   longest_token_sequence_in_batch: int = max(lengths)
	#                   # if longest_token_sequence_in_batch>100:
	#                   #   pdb.set_trace()
	#                   embedding.embed(sentences)
	#                   store_embeddings(sentences, self.embeddings_storage_mode)
	#           embedding=embedding.to('cpu')
	#       else:
	#           embedding=embedding.to(flair.device)
	#   # torch.cuda.empty_cache()
	#   log.info("Finished Embeddings Assignments")
	#   return 
	# def assign_predicted_embeddings(self,doc_dict,embedding,file_name):
	#   # torch.cuda.empty_cache()
	#   lm_file = h5py.File(file_name, "r")
	#   for key in doc_dict:
	#       if key == 'start':
	#           for i, sentence in enumerate(doc_dict[key]):
	#               for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
	#                   word_embedding = torch.zeros(embedding.embedding_length).float()
	#                   word_embedding = torch.FloatTensor(word_embedding)

	#                   token.set_embedding(embedding.name, word_embedding)
	#           continue
	#       group = lm_file[key]
	#       num_sentences = len(list(group.keys()))
	#       sentences_emb = [group[str(i)][...] for i in range(num_sentences)]
	#       try: 
	#           assert len(doc_dict[key])==len(sentences_emb)
	#       except:
	#           pdb.set_trace()
	#       for i, sentence in enumerate(doc_dict[key]):
	#           for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
	#               word_embedding = sentences_emb[i][token_idx]
	#               word_embedding = torch.from_numpy(word_embedding).view(-1)

	#               token.set_embedding(embedding.name, word_embedding)
	#           store_embeddings([sentence], 'cpu')
	#       # for idx, sentence in enumerate(doc_dict[key]):
	#   log.info("Loaded predicted embeddings: "+file_name)
	#   return 
	
	def resort(self,loader,is_crf=False, is_posterior=False, is_token_att=False):
		for batch in loader:
			if is_posterior:
				try:
					posteriors=[x._teacher_posteriors for x in batch]
					posterior_lens=[len(x[0]) for x in posteriors]
					lens=posterior_lens.copy()
					targets=posteriors.copy()
				except:
					pdb.set_trace()
			if is_token_att:
				sentfeats=[x._teacher_sentfeats for x in batch]
				sentfeats_lens=[len(x[0]) for x in sentfeats]
			#     lens=sentfeats_lens.copy()
			#     targets=sentfeats.copy()
			if is_crf:
				targets=[x._teacher_target for x in batch]
				lens=[len(x[0]) for x in targets]
				if hasattr(self.model,'distill_rel') and self.model.distill_rel:
					rel_targets=[x._teacher_rel_target for x in batch]
			if not is_crf and not is_posterior:
				targets=[x._teacher_prediction for x in batch]
				if hasattr(self.model, 'distill_factorize') and self.model.distill_factorize:
					rel_targets=[x._teacher_rel_prediction for x in batch]
				lens=[len(x[0]) for x in targets]
			sent_lens=[len(x) for x in batch]

			if is_posterior:
				assert posterior_lens==lens, 'lengths of two targets not match'
			
			if max(lens)>min(lens) or max(sent_lens)!=max(lens) or (is_posterior and self.model.tag_type=='dependency'):
				# if max(sent_lens)!=max(lens):
				max_shape=max(sent_lens)
				for index, target in enumerate(targets):
					new_targets=[]
					new_rel_targets=[]
					new_posteriors=[]
					new_sentfeats=[]
					if is_posterior:
						post_vals=posteriors[index]
					if is_token_att:
						sentfeats_vals=sentfeats[index]
					for idx, val in enumerate(target):
						if self.model.tag_type=='dependency':
							if is_crf:
								shape=[max_shape]+list(val.shape[1:])
								new_target=torch.zeros(shape).type_as(val)
								new_target[:sent_lens[index]]=val[:sent_lens[index]]
								new_targets.append(new_target)
								if hasattr(self.model,'distill_rel') and self.model.distill_rel:
									cur_val = rel_targets[index][idx]
									rel_shape=[max_shape]+list(cur_val.shape[1:])
									new_rel_target=torch.zeros(rel_shape).type_as(cur_val)
									new_rel_target[:sent_lens[index]]=cur_val[:sent_lens[index]]
									new_rel_targets.append(new_rel_target)
							if not is_crf and not is_posterior:
								shape=[max_shape,max_shape]+list(val.shape[2:])
								new_target=torch.zeros(shape).type_as(val)
								new_target[:sent_lens[index],:sent_lens[index]]=val[:sent_lens[index],:sent_lens[index]]
								new_targets.append(new_target)
								if hasattr(self.model, 'distill_factorize') and self.model.distill_factorize:
									cur_val = rel_targets[index][idx]
									rel_shape=[max_shape,max_shape]+list(cur_val.shape[2:])
									new_rel_target=torch.zeros(rel_shape).type_as(cur_val)
									new_rel_target[:sent_lens[index],:sent_lens[index]]=cur_val[:sent_lens[index],:sent_lens[index]]
									new_rel_targets.append(new_rel_target)
							if is_posterior:
								post_val=post_vals[idx]
								# shape=[max_shape-1,max_shape-1] + list(post_val.shape[2:])
								shape=[max_shape,max_shape] + list(post_val.shape[2:])
								# if max_shape==8:
								#   pdb.set_trace()
								new_posterior=torch.zeros(shape).type_as(post_val)
								# remove the root token
								# new_posterior[:sent_lens[index]-1,:sent_lens[index]-1]=post_val[:sent_lens[index]-1,:sent_lens[index]-1]
								new_posterior[:sent_lens[index],:sent_lens[index]]=post_val[:sent_lens[index],:sent_lens[index]]
								new_posteriors.append(new_posterior)
						else:
							if is_crf or (not is_crf and not is_posterior):
								shape=[max_shape]+list(val.shape[1:])+list(val.shape[2:])
								new_target=torch.zeros(shape).type_as(val)
								new_target[:sent_lens[index]]=val[:sent_lens[index]]
								new_targets.append(new_target)
							if is_token_att:
								sentfeats_val=sentfeats_vals[idx]
								shape=[max_shape]+list(sentfeats_val.shape[1:])
								new_sentfeat=torch.zeros(shape).type_as(sentfeats_val)
								new_sentfeat[:sent_lens[index]]=sentfeats_val[:sent_lens[index]]
								new_sentfeats.append(new_sentfeat)
							if is_posterior:
								post_val=post_vals[idx]
								shape=[max_shape]+list(post_val.shape[1:])
								new_posterior=torch.zeros(shape).type_as(post_val)
								new_posterior[:sent_lens[index]]=post_val[:sent_lens[index]]
								new_posteriors.append(new_posterior)
							
					if is_crf:
						batch[index]._teacher_target=new_targets
						if hasattr(self.model,'distill_rel') and  self.model.distill_rel:
							batch[index]._teacher_rel_target=new_rel_targets
					if is_posterior:
						batch[index]._teacher_posteriors=new_posteriors
					if is_token_att:
						batch[index]._teacher_sentfeats=new_sentfeats
					if not is_crf and not is_posterior:
						if hasattr(self.model, 'distill_factorize') and self.model.distill_factorize:
							batch[index]._teacher_rel_prediction=new_rel_targets
						batch[index]._teacher_prediction=new_targets
			if hasattr(batch,'teacher_features'):
				if is_posterior:
					batch.teacher_features['posteriors']=torch.stack([sentence.get_teacher_posteriors() for sentence in batch],0).cpu()
					# lens=[len(x) for x in batch]
					# posteriors = batch.teacher_features['posteriors']
					# if max(lens) == posteriors.shape[-1]:
					#   pdb.set_trace()
				if (not is_crf and not is_posterior):
					batch.teacher_features['distributions'] = torch.stack([sentence.get_teacher_prediction() for sentence in batch],0).cpu()
					if hasattr(self.model, 'distill_factorize') and self.model.distill_factorize:
						batch.teacher_features['rel_distributions'] = torch.stack([sentence.get_teacher_rel_prediction() for sentence in batch],0).cpu()
				if is_crf:
					batch.teacher_features['topk']=torch.stack([sentence.get_teacher_target() for sentence in batch],0).cpu()
					if self.model.crf_attention or self.model.tag_type=='dependency':
						batch.teacher_features['weights']=torch.stack([sentence.get_teacher_weights() for sentence in batch],0).cpu()
					if hasattr(self.model,'distill_rel') and self.model.distill_rel:
						batch.teacher_features['topk_rels']=torch.stack([sentence.get_teacher_rel_target() for sentence in batch],0).cpu()
		return loader
		
	def final_test(
		self, base_path: Path, eval_mini_batch_size: int, num_workers: int = 8, overall_test: bool = True, quiet_mode: bool = False, nocrf: bool = False, predict_posterior: bool = False, debug: bool = False, keep_embedding: int = -1, sort_data=False, mst = False
	):

		log_line(log)
		

		self.model.eval()
		self.model.to('cpu')
		name_list=sorted([x.name for x in self.model.embeddings.embeddings])
		if quiet_mode:
			#blockPrint()
			log.disabled=True
		if (base_path / "best-model.pt").exists():
			self.model = self.model.load(base_path / "best-model.pt", device='cpu')
			log.info("Testing using best model ...")
		elif (base_path / "final-model.pt").exists():
			self.model = self.model.load(base_path / "final-model.pt", device='cpu')
			log.info("Testing using final model ...")
		try:
			if self.controller.model_structure is not None:
				self.controller = self.controller.load(base_path / "controller.pt")
				log.info("Testing using best controller ...")
			if self.controller.model_structure is None:
				training_state = torch.load(base_path/'training_state.pt')
				self.best_action = training_state['best_action']
				self.model.selection=self.best_action
			
				log.info(f"Setting embedding mask to the best action: {self.best_action}")
				print(name_list)
		except:
			pdb.set_trace()

		# Since there are a lot of embeddings, we keep these embeddings to cpu in order to avoid OOM
		for name, module in self.model.named_modules():
			if 'embeddings' in name or name == '':
				continue
			else:
				module.to(flair.device)
		parameters = [x for x in self.model.named_parameters()]
		for parameter in parameters:
			name = parameter[0]
			module = parameter[1]
			module.data.to(flair.device)
			if '.' not in name:
				if type(getattr(self.model, name))==torch.nn.parameter.Parameter:
					setattr(self.model, name, torch.nn.parameter.Parameter(getattr(self.model,name).to(flair.device)))

		# if hasattr(self.model,'transitions'):
		#   self.model.transitions = torch.nn.parameter.Parameter(self.model.transitions.to(flair.device))
		if mst == True:
			self.model.is_mst=mst
		for embedding in self.model.embeddings.embeddings:
			embedding.to('cpu')
		if debug:
			self.model.debug=True
			# if hasattr(self.model,'transitions'):
			#   self.model.transitions = torch.nn.Parameter(torch.randn(self.model.tagset_size, self.model.tagset_size).to(flair.device))
		else:
			self.model.debug=False
		if nocrf:
			self.model.use_crf=False
		if predict_posterior:
			self.model.predict_posterior=True
		if keep_embedding>-1:
			self.model.keep_embedding=keep_embedding
		if overall_test:
			loader=ColumnDataLoader(list(self.corpus.test),eval_mini_batch_size, use_bert=self.use_bert,tokenizer=self.bert_tokenizer, model = self.model, sentence_level_batch = self.sentence_level_batch, sort_data=sort_data)
			loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
			with torch.no_grad():
				self.gpu_friendly_assign_embedding([loader], selection = self.model.selection)
				if self.controller.model_structure is not None:
					self.assign_embedding_masks(loader,sample=False)
			test_results, test_loss = self.model.evaluate(
				loader,
				out_path=base_path / "test.tsv",
				embeddings_storage_mode="cpu",
				prediction_mode=True,
			)
			test_results: Result = test_results
			log.info(test_results.log_line)
			log.info(test_results.detailed_results)
			log_line(log)
		if quiet_mode:
			enablePrint()
			if overall_test:
				if keep_embedding>-1:
					embedding_name = sorted(loader[0].features.keys())[keep_embedding].split()
					embedding_name = '_'.join(embedding_name)
					if 'lm-' in embedding_name.lower():
						embedding_name = 'Flair'
					elif 'bert' in embedding_name.lower():
						embedding_name = 'MBERT'
					elif 'word' in embedding_name.lower():
						embedding_name = 'Word'
					elif 'char' in embedding_name.lower():
						embedding_name = 'char'
					print(embedding_name,end=' ')
				print('Average', end=' ')
				print(test_results.main_score, end=' ')

		# if we are training over multiple datasets, do evaluation for each
		if type(self.corpus) is MultiCorpus:
			for subcorpus in self.corpus.corpora:
				log_line(log)
				log.info('current corpus: '+subcorpus.name)
				loader=ColumnDataLoader(list(subcorpus.test),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, model = self.model, sentence_level_batch = self.sentence_level_batch, sort_data=sort_data)
				loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
				with torch.no_grad():
					self.gpu_friendly_assign_embedding([loader], selection = self.model.selection)
					if self.controller.model_structure is not None:
						self.assign_embedding_masks(loader,sample=False)
				current_result, test_loss = self.model.evaluate(
					loader,
					out_path=base_path / f"{subcorpus.name}-test.tsv",
					embeddings_storage_mode="none",
					prediction_mode=True,
				)
				log.info(current_result.log_line)
				log.info(current_result.detailed_results)
				if quiet_mode:
					if keep_embedding>-1:
						embedding_name = sorted(loader[0].features.keys())[keep_embedding].split()
						embedding_name = '_'.join(embedding_name)
						if 'lm-' in embedding_name.lower() or 'forward' in embedding_name.lower() or 'backward' in embedding_name.lower():
							embedding_name = 'Flair'
						elif 'bert' in embedding_name.lower():
							embedding_name = 'MBERT'
						elif 'word' in embedding_name.lower():
							embedding_name = 'Word'
						elif 'char' in embedding_name.lower():
							embedding_name = 'char'
						print(embedding_name,end=' ')
					print(subcorpus.name,end=' ')
					print(current_result.main_score,end=' ')

		elif type(self.corpus) is ListCorpus:
			for index,subcorpus in enumerate(self.corpus.test_list):
				log_line(log)
				log.info('current corpus: '+self.corpus.targets[index])
				loader=ColumnDataLoader(list(subcorpus),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, model = self.model, sentence_level_batch = self.sentence_level_batch, sort_data=sort_data)
				loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
				with torch.no_grad():
					self.gpu_friendly_assign_embedding([loader], selection = self.model.selection)
					if self.controller.model_structure is not None:
						self.assign_embedding_masks(loader,sample=False)
				current_result, test_loss = self.model.evaluate(
					loader,
					out_path=base_path / f"{self.corpus.targets[index]}-test.tsv",
					embeddings_storage_mode="none",
					prediction_mode=True,
				)
				log.info(current_result.log_line)
				log.info(current_result.detailed_results)
				if quiet_mode:
					if keep_embedding>-1:
						embedding_name = sorted(loader[0].features.keys())[keep_embedding].split()
						embedding_name = '_'.join(embedding_name)
						if 'lm-' in embedding_name.lower() or 'forward' in embedding_name.lower() or 'backward' in embedding_name.lower():
							embedding_name = 'Flair'
						elif 'bert' in embedding_name.lower():
							embedding_name = 'MBERT'
						elif 'word' in embedding_name.lower():
							embedding_name = 'Word'
						elif 'char' in embedding_name.lower():
							embedding_name = 'char'
						print(embedding_name,end=' ')
					print(self.corpus.targets[index],end=' ')
					print(current_result.main_score,end=' ')
		if keep_embedding<0:
			print()
		if overall_test:
			# get and return the final test score of best model
			final_score = test_results.main_score

			return final_score
		return 0
