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
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
import random
import copy
from flair.parser.utils.alg import crf
import h5py
import numpy as np
START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"

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

class ModelFinetuner(ModelDistiller):
	def __init__(
		self,
		model: flair.nn.Model,
		teachers: List[flair.nn.Model],
		corpus: ListCorpus,
		optimizer = AdamW,
		professors: List[flair.nn.Model] = [],
		epoch: int = 0,
		optimizer_state: dict = None,
		scheduler_state: dict = None,
		use_tensorboard: bool = False,
		distill_mode: bool = False,
		ensemble_distill_mode: bool = False,
		config = None,
		train_with_professor: bool = False,
		is_test: bool = False,
		language_resample: bool = False,
		direct_upsample_rate: int = -1,
		down_sample_amount: int = -1,
		sentence_level_batch: bool = False,
		clip_sentences: int = -1,
		remove_sentences: bool = False,
		assign_doc_id: bool = False,
		pretrained_file_dict: dict = {},
	):
		"""
		Initialize a model trainer
		:param model: The model that you want to train. The model should inherit from flair.nn.Model
		:param teachers: The teacher models for knowledge distillation. The model should inherit from flair.nn.Model
		:param corpus: The dataset used to train the model, should be of type Corpus
		:param optimizer: The optimizer to use (Default AdamW for finetuning BERT)
		:param epoch: The starting epoch (normally 0 but could be higher if you continue training model)
		:param optimizer_state: Optimizer state (necessary if continue training from checkpoint)
		:param scheduler_state: Scheduler state (necessary if continue training from checkpoint)
		:param use_tensorboard: If True, writes out tensorboard information
		:param sentence_level_batch: If True, count the batch size by the number of sentences, otherwise the number of tokens
		:param assign_doc_id: Set to True if using document-level embeddings
		:param pretrained_file_dict: The dictionary of predicted embeddings. Set to True if using document-level embeddings
		:param down_sample_amount: Downsample the training set
		"""
		# if teachers is not None:
		#   assert len(teachers)==len(corpus.train_list), 'Training data and teachers should be the same length now!'
		self.model: flair.nn.Model = model
		self.config = config
		self.corpus: ListCorpus = corpus
		num_languages = len(self.corpus.targets)
		self.corpus2id = {x:i for i,x in enumerate(self.corpus.targets)}
		self.sentence_level_batch = sentence_level_batch
		if language_resample or direct_upsample_rate>0:
			sent_per_set=torch.FloatTensor([len(x) for x in self.corpus.train_list])
			total_sents=sent_per_set.sum()
			sent_each_dataset=sent_per_set/total_sents
			exp_sent_each_dataset=sent_each_dataset.pow(0.7)
			sent_sample_prob=exp_sent_each_dataset/exp_sent_each_dataset.sum()

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
				doc_name = 'train_'
				doc_idx = -1
				for sentence in self.corpus.train_list[corpus_id]:
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
				for sentence in self.corpus.dev_list[corpus_id]:
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
				for sentence in self.corpus.test_list[corpus_id]:
					if '-DOCSTART-' in sentence[0].text:
						doc_idx+=1
						doc_key='start'
					else:
						doc_key=corpus_name+doc_name+str(doc_idx)
					if doc_key not in doc_sentence_dict:
						doc_sentence_dict[doc_key]=[]
					doc_sentence_dict[doc_key].append(sentence)
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
					if 'use_unlabeled_data' in config['train'] and config['train']['use_unlabeled_data']:
						if 'unlabel' not in corpus_name.lower():
							continue
					self.corpus.train_list[i].sentences = self.corpus.train_list[i].sentences[:down_sample_amount]
					self.corpus.train_list[i].reset_sentence_count
					if config['train']['train_with_dev']:
						self.corpus.dev_list[i].sentences = self.corpus.dev_list[i].sentences[:down_sample_amount]
						self.corpus.dev_list[i].reset_sentence_count
			if clip_sentences>-1:
				new_sentences=[]
				removed_count=0
				max_len = 0
				for sentence in self.corpus.train_list[i].sentences:
					subtoken_length = self.get_subtoken_length(sentence)
					if subtoken_length>max_len:
						max_len = subtoken_length
					if subtoken_length > clip_sentences:
						removed_count+=1
					else:
						new_sentences.append(sentence)
				self.corpus.train_list[i].sentences = new_sentences
				self.corpus.train_list[i].reset_sentence_count
				log.info(f"Longest subwords in the training set {max_len}")
				log.info(f"Removed {removed_count} sentences whose subwords are longer than {clip_sentences}")



		if direct_upsample_rate>0 or down_sample_amount:
			self.corpus._train: FlairDataset = ConcatDataset([data for data in self.corpus.train_list])
			if config['train']['train_with_dev']:
				self.corpus._dev: FlairDataset = ConcatDataset([data for data in self.corpus.dev_list])
		print(self.corpus)
		self.distill_mode = distill_mode

		if self.distill_mode:
			# self.corpus_mixed_train: ListCorpus = [CoupleDataset(student_set,self.corpus_teacher.train_list[index]) for index,student_set in enumerate(self.corpus.train_list)]
			self.teachers: List[flair.nn.Model] = teachers
			self.professors: List[flair.nn.Model] = professors
			if self.teachers is not None:
				for teacher in self.teachers: teacher.eval()
			for professor in self.professors: professor.eval()
			try:
				num_teachers = len(self.teachers)+int(len(self.professors)>0)
				self.num_teachers=num_teachers
			except:
				num_teachers = 0
				self.num_teachers=num_teachers
		# self.corpus = self.assign_pretrained_teacher_predictions(self.corpus,self.corpus_teacher,self.teachers)
		self.update_params_group=[]


		

		self.optimizer: torch.optim.Optimizer = optimizer
		if type(optimizer)==str:
			self.optimizer = getattr(torch.optim,optimizer)

		self.epoch: int = epoch
		self.scheduler_state: dict = scheduler_state
		self.optimizer_state: dict = optimizer_state
		self.use_tensorboard: bool = use_tensorboard
		
		
		self.use_bert = False
		self.bert_tokenizer = None
		for embedding in self.model.embeddings.embeddings:
			if 'bert' in embedding.__class__.__name__.lower():
				self.use_bert=True
				self.bert_tokenizer = embedding.tokenizer
		self.ensemble_distill_mode: bool = ensemble_distill_mode
		self.train_with_professor: bool = train_with_professor
		# if self.train_with_professor:
		#   assert len(self.professors) == len(self.corpus.train_list), 'Now only support same number of professors and corpus!'

	def train(
		self,
		base_path: Union[Path, str],
		learning_rate: float = 5e-5,
		mini_batch_size: int = 32,
		eval_mini_batch_size: int = None,
		max_epochs: int = 100,
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
		language_attention_warmup_and_fix: bool = False,
		language_attention_warmup: bool = False,
		language_attention_entropy: bool = False,
		train_language_attention_by_dev: bool = False,
		calc_teachers_target_loss: bool = False,
		entropy_loss_rate: float = 1,
		amp_opt_level: str = "O1",
		professor_interpolation = 0.5,
		best_k = 10,
		max_epochs_without_improvement = 100,
		gold_reward = False,
		warmup_steps: int = 0,
		use_warmup: bool = False,
		gradient_accumulation_steps: int = 1,
		lr_rate: int = 1,
		decay: float = 0.75,
		decay_steps: int = 5000,
		use_unlabeled_data: bool =False,
		sort_data: bool = True,
		fine_tune_mode: bool = False,
		debug: bool = False,
		min_freq: int = -1,
		min_lemma_freq: int = -1,
		min_pos_freq: int = -1,
		unlabeled_data_for_zeroshot: bool = False,
		rootschedule: bool = False,
		freezing: bool = False,
		save_finetuned_embedding: bool = False,
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
		min_learning_rate = learning_rate/1000
		self.gold_reward = gold_reward
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

		# weight_extractor = WeightExtractor(base_path)
		# finetune_params = {name:param for name,param in self.model.named_parameters()}
		finetune_params=[param for name,param in self.model.named_parameters() if 'embedding' in name or name=='linear.weight' or name=='linear.bias']
		other_params=[param for name,param in self.model.named_parameters() if 'embedding' not in name and name !='linear.weight' and name !='linear.bias']
		# other_params = {name:param for name,param in self.model.named_parameters() if 'embeddings' not in name}
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


		# minimize training loss if training with dev data, else maximize dev score
		
		# start from here, the train data is a list now
		train_data = self.corpus.train_list
		# if self.distill_mode:
		#   train_data_teacher = self.corpus_teacher.train_list
		# train_data = self.corpus_mixed
		# if training also uses dev data, include in training set
		if train_with_dev:
			train_data = [ConcatDataset([train, self.corpus.dev_list[index]]) for index, train in enumerate(self.corpus.train_list)]
			# if self.distill_mode:
			#   train_data_teacher = [ConcatDataset([train, self.corpus_teacher.dev_list[index]]) for index, train in enumerate(self.corpus_teacher.train_list)]
			# train_data = [ConcatDataset([train, self.corpus_mixed.dev_list[index]]) for index, train in self.corpus_mixed.train_list]
			# train_data_teacher = ConcatDataset([self.corpus_teacher.train, self.corpus_teacher.dev])
			# train_data = ConcatDataset([self.corpus_mixed.train, self.corpus_mixed.dev])
		if self.distill_mode:
			
			# coupled_train_data = [CoupleDataset(data,train_data_teacher[index]) for index, data in enumerate(train_data)]
			coupled_train_data = train_data
			# faster=True
			# if 'fast' in self.model.__class__.__name__.lower():
			#   faster=True
			# else:
			#   faster=False
			faster = False

			if self.train_with_professor:
				log.info(f"Predicting professor prediction")
				# train_data_teacher = self.corpus_teacher.train_list
				coupled_train_data=self.assign_pretrained_teacher_predictions(coupled_train_data,self.professors,is_professor=True,faster=faster)
				
				for professor in self.professors:
					del professor
				del self.professors
			if self.model.distill_crf or self.model.distill_posterior:
				train_data=self.assign_pretrained_teacher_targets(coupled_train_data,self.teachers,best_k=best_k)
			else:
				train_data=self.assign_pretrained_teacher_predictions(coupled_train_data,self.teachers,faster=faster)
				
			# if self.ensemble_distill_mode:
			#   log.info(f"Ensembled distillation mode")
			#   coupled_train_data = ConcatDataset(coupled_train_data)
			#   train_data=self.assign_ensembled_teacher_predictions(coupled_train_data,self.teachers)
			#   # coupled_train_data = []
			# else:
			#   train_data=self.assign_pretrained_teacher_predictions(coupled_train_data,self.teachers)
			#   #train_data=ConcatDataset(train_data)
			for teacher in self.teachers:
				del teacher
			del self.teachers
			batch_loader=ColumnDataLoader(train_data,mini_batch_size,shuffle,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, sort_data=sort_data, model = self.model, sentence_level_batch = self.sentence_level_batch)
		else:
			batch_loader=ColumnDataLoader(ConcatDataset(train_data),mini_batch_size,shuffle,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, sort_data=sort_data, model = self.model, sentence_level_batch = self.sentence_level_batch)
		batch_loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
		if self.distill_mode:
			batch_loader=self.resort(batch_loader,is_crf=self.model.distill_crf, is_posterior = self.model.distill_posterior, is_token_att = self.model.token_level_attention)
		if not train_with_dev:
			if macro_avg:
				dev_loaders=[ColumnDataLoader(list(subcorpus),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, sort_data=sort_data, model = self.model, sentence_level_batch = self.sentence_level_batch) \
							 for subcorpus in self.corpus.dev_list]
				for loader in dev_loaders:
					loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)

			else:
				dev_loader=ColumnDataLoader(list(self.corpus.dev),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, sort_data=sort_data, model = self.model, sentence_level_batch = self.sentence_level_batch)
				dev_loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
		test_loader=ColumnDataLoader(list(self.corpus.test),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, sort_data=sort_data, model = self.model, sentence_level_batch = self.sentence_level_batch)
		test_loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
		# if self.distill_mode:
		#   batch_loader.expand_teacher_predictions()
		# if sampler is not None:
		#   sampler = sampler(train_data)
		#   shuffle = False

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
			t_total = (len(batch_loader) // gradient_accumulation_steps + int((len(batch_loader) % gradient_accumulation_steps)>0)) * max_epochs
			if rootschedule:
				warmup_steps = (len(batch_loader) // gradient_accumulation_steps + int((len(batch_loader) % gradient_accumulation_steps)>0))
				scheduler = get_inverse_square_root_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total, fix_embedding_steps = warmup_steps)
			else:
				if use_warmup:
					warmup_steps = (len(batch_loader) // gradient_accumulation_steps + int((len(batch_loader) % gradient_accumulation_steps)>0))
				scheduler = get_linear_schedule_with_warmup(
					optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
				)

		if self.scheduler_state is not None:
			scheduler.load_state_dict(self.scheduler_state)
		### Finetune Scheduler
		if freezing:
			for embedding in self.model.embeddings.embeddings:
				embedding.fine_tune = False
		dev_score_history = []
		dev_loss_history = []
		train_loss_history = []

		# At any point you can hit Ctrl + C to break out of training early.
		best_score=0
		interpolation=1
		
		if fine_tune_mode:
			pass
		else:
			self.model.embeddings=self.model.embeddings.to('cpu')
			if log_test:
				loaders = [batch_loader,test_loader]
			else:
				loaders = [batch_loader]
			if not train_with_dev:
				if macro_avg:
					self.gpu_friendly_assign_embedding(loaders+dev_loaders)
				else:
					self.gpu_friendly_assign_embedding(loaders+[dev_loader])
			else:
				self.gpu_friendly_assign_embedding(loaders)
		try:
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
					if self.distill_mode:
						batch_loader=self.resort(batch_loader,is_crf=self.model.distill_crf, is_posterior = self.model.distill_posterior, is_token_att = self.model.token_level_attention)
				self.model.train()
				# TODO: check teacher parameters fixed and with eval() mode

				train_loss: float = 0

				seen_batches = 0
				#total_number_of_batches = sum([len(loader) for loader in batch_loader])
				total_number_of_batches = len(batch_loader)

				modulo = max(1, int(total_number_of_batches / 10))

				# process mini-batches
				batch_time = 0
				total_sent=0
				if self.distill_mode:
					if self.teacher_annealing:
						# interpolation=1
						interpolation=1-(((epoch-warmup_bias)*total_number_of_batches)/total_number_of_batches*self.anneal_factor)/100.0
						if interpolation<0:
							interpolation=0
					else:
						interpolation=self.interpolation


				log.info("Current loss interpolation: "+ str(interpolation))
				
				name_list=sorted([x.name for x in self.model.embeddings.embeddings])
				print(name_list)
				for batch_no, student_input in enumerate(batch_loader):
					# for group in optimizer.param_groups:
					#   temp_lr = group["lr"]
					# log.info('lr: '+str(temp_lr))
					if self.distill_mode:

						if self.teacher_annealing:
							interpolation=1-(((epoch-warmup_bias)*total_number_of_batches+batch_no)/total_number_of_batches*self.anneal_factor)/100.0
							if interpolation<0:
								interpolation=0
						else:
							interpolation=self.interpolation
						# log.info("Current loss interpolation: "+ str(interpolation))
					start_time = time.time()
					total_sent+=len(student_input)

					try:
						if self.distill_mode:
							
							
							loss = self.model.simple_forward_distillation_loss(student_input, interpolation = interpolation, train_with_professor=self.train_with_professor)
						else:
							loss = self.model.forward_loss(student_input)

						if self.model.use_decoder_timer:
							decode_time=time.time() - self.model.time
						
						# Backward
						if batch_no >= total_number_of_batches//gradient_accumulation_steps * gradient_accumulation_steps:
							# only accumulate the rest of batch
							loss = loss/(total_number_of_batches-total_number_of_batches//gradient_accumulation_steps * gradient_accumulation_steps)
						else:
							loss = loss/gradient_accumulation_steps
						if use_amp:
							with amp.scale_loss(loss, optimizer) as scaled_loss:
								scaled_loss.backward()
						else:
							loss.backward()
							pass
					except Exception:
						traceback.print_exc()
						pdb.set_trace()
					# pdb.set_trace()
					# print(self.model.linear.weight.sum())
					train_loss += loss.item()
					seen_batches += 1
					batch_time += time.time() - start_time
					if (batch_no+1)%gradient_accumulation_steps==0 or (batch_no == total_number_of_batches - 1):
						torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
						if len(self.update_params_group)>0:
							torch.nn.utils.clip_grad_norm_(self.update_params_group, 5.0)
						# print("update model")
						optimizer.step()
						self.model.zero_grad()

						# optimizer.zero_grad()
						if (fine_tune_mode or self.model.tag_type in dependency_tasks):
							scheduler.step()

					
					if batch_no % modulo == 0:
						if self.model.use_decoder_timer:
							log.info(
								f"epoch {epoch + 1} - iter {batch_no}/{total_number_of_batches} - loss "
								f"{train_loss / (seen_batches) * gradient_accumulation_steps:.8f} - samples/sec: {total_sent / batch_time:.2f} - decode_sents/sec: {total_sent / decode_time:.2f}"
							)
							
						else:
							log.info(
								f"epoch {epoch + 1} - iter {batch_no}/{total_number_of_batches} - loss "
								f"{train_loss / seen_batches* gradient_accumulation_steps:.8f} - samples/sec: {total_sent / batch_time:.2f}"
							)
						total_sent=0
						batch_time = 0
						iteration = epoch * total_number_of_batches + batch_no
						# if not param_selection_mode:
						#   weight_extractor.extract_weights(
						#       self.model.state_dict(), iteration
						#   )
					# depending on memory mode, embeddings are moved to CPU, GPU or deleted
					store_embeddings(student_input, embeddings_storage_mode)
					if self.distill_mode:
						store_teacher_predictions(student_input, embeddings_storage_mode)
				# if self.model.embedding_selector:
				#   print(sorted(student_input.features.keys()))
				#   print(self.model.selector)
				#   print(torch.argmax(self.model.selector,-1))
				train_loss /= seen_batches

				self.model.eval()

				log_line(log)
				log.info(
					f"EPOCH {epoch + 1} done: loss {train_loss:.4f} - lr {learning_rate}"
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
				log_line(log)
				if log_dev:
					if macro_avg:
						
						if type(self.corpus) is ListCorpus:
							result_dict={}
							loss_list=[]
							print_sent='\n'
							for index, loader in enumerate(dev_loaders):
								# log_line(log)
								# log.info('current corpus: '+self.corpus.targets[index])
								if len(loader) == 0:
									continue
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

					if self.use_tensorboard:
						writer.add_scalar("dev_loss", dev_loss, epoch + 1)
						writer.add_scalar(
							"dev_score", dev_eval_result.main_score, epoch + 1
						)
				log_line(log)
				if log_test:
					test_eval_result, test_loss = self.model.evaluate(
						test_loader,
						base_path / "test.tsv",
						embeddings_storage_mode=embeddings_storage_mode,
					)
					result_line += f"\t{test_loss}\t{test_eval_result.log_line}"
					log.info(
						f"TEST : loss {test_loss} - score {test_eval_result.main_score}"
					)

					# depending on memory mode, embeddings are moved to CPU, GPU or deleted
					store_embeddings(self.corpus.test, embeddings_storage_mode)

					if self.use_tensorboard:
						writer.add_scalar("test_loss", test_loss, epoch + 1)
						writer.add_scalar(
							"test_score", test_eval_result.main_score, epoch + 1
						)
					log.info(test_eval_result.log_line)
					log.info(test_eval_result.detailed_results)
					if type(self.corpus) is MultiCorpus:
						for subcorpus in self.corpus.corpora:
							log_line(log)
							log.info('current corpus: '+subcorpus.name)
							current_result, test_loss = self.model.evaluate(
								ColumnDataLoader(list(subcorpus.test),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, sort_data=sort_data, model = self.model, sentence_level_batch = self.sentence_level_batch),
								out_path=base_path / f"{subcorpus.name}-test.tsv",
								embeddings_storage_mode=embeddings_storage_mode,
							)
							log.info(current_result.log_line)
							log.info(current_result.detailed_results)
					elif type(self.corpus) is ListCorpus:
						for index,subcorpus in enumerate(self.corpus.test_list):
							log_line(log)
							log.info('current corpus: '+self.corpus.targets[index])
							current_result, test_loss = self.model.evaluate(
								ColumnDataLoader(list(subcorpus),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, sort_data=sort_data, model = self.model, sentence_level_batch = self.sentence_level_batch),
								out_path=base_path / f"{self.corpus.targets[index]}-test.tsv",
								embeddings_storage_mode=embeddings_storage_mode,
							)
							log.info(current_result.log_line)
							log.info(current_result.detailed_results)


				# determine learning rate annealing through scheduler
				if not fine_tune_mode and self.model.tag_type not in dependency_tasks:
					scheduler.step(current_score)
				
				if current_score>best_score:
					best_score=current_score
					bad_epochs2=0
				else:
					bad_epochs2+=1
				if train_with_dev:
					# train as the learning rate gradually drops
					bad_epochs2 = 0

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

				# output log file
				# with open(loss_txt, "a") as f:

				#   # make headers on first epoch
				#   if epoch == 0:
				#       f.write(
				#           f"EPOCH\tTIMESTAMP\tBAD_EPOCHS\tLEARNING_RATE\tTRAIN_LOSS"
				#       )

				#       if log_train:
				#           f.write(
				#               "\tTRAIN_"
				#               + "\tTRAIN_".join(
				#                   train_eval_result.log_header.split("\t")
				#               )
				#           )
				#       if log_dev:
				#           f.write(
				#               "\tDEV_LOSS\tDEV_"
				#               + "\tDEV_".join(dev_eval_result.log_header.split("\t"))
				#           )
				#       if log_test:
				#           f.write(
				#               "\tTEST_LOSS\tTEST_"
				#               + "\tTEST_".join(
				#                   test_eval_result.log_header.split("\t")
				#               )
				#           )

				#   f.write(
				#       f"\n{epoch}\t{datetime.datetime.now():%H:%M:%S}\t{bad_epochs}\t{learning_rate:.4f}\t{train_loss}"
				#   )
				#   f.write(result_line)

				# if checkpoint is enable, save model at each epoch
				if checkpoint and not param_selection_mode:
					
					self.model.save_checkpoint(
						base_path / "checkpoint.pt",
						optimizer.state_dict(),
						scheduler.state_dict(),
						epoch + 1,
						train_loss,
					)

				# if we use dev data, remember best model based on dev evaluation score
				if (
					not train_with_dev
					and not param_selection_mode
					and current_score == best_score
				):
					log.info(f"==================Saving the current best model: {current_score}==================") 
					self.model.save(base_path / "best-model.pt")
					if save_finetuned_embedding:
						# pdb.set_trace()
						log.info(f"==================Saving the best language model: {current_score}==================") 
						for embedding in self.model.embeddings.embeddings:
							if hasattr(embedding,'fine_tune') and embedding.fine_tune: 
								if not os.path.exists(base_path/embedding.name.split('/')[-1]):
									os.mkdir(base_path/embedding.name.split('/')[-1])
								embedding.tokenizer.save_pretrained(base_path/embedding.name.split('/')[-1])
								embedding.model.save_pretrained(base_path/embedding.name.split('/')[-1])
							# torch.save(embedding,base_path/(embedding.name.split('/')[-1]+'.bin'))

			# if we do not use dev data for model selection, save final model
			if save_final_model and not param_selection_mode:
				self.model.save(base_path / "final-model.pt")
				if save_finetuned_embedding and train_with_dev:
					# pdb.set_trace()
					log.info(f"==================Saving the best language model: {current_score}==================") 
					for embedding in self.model.embeddings.embeddings:
						if hasattr(embedding,'fine_tune') and embedding.fine_tune: 
							if not os.path.exists(base_path/embedding.name.split('/')[-1]):
								os.mkdir(base_path/embedding.name.split('/')[-1])
							embedding.tokenizer.save_pretrained(base_path/embedding.name.split('/')[-1])
							embedding.model.save_pretrained(base_path/embedding.name.split('/')[-1])
						# torch.save(embedding,base_path/(embedding.name.split('/')[-1]+'.bin'))
		except KeyboardInterrupt:
			log_line(log)
			log.info("Exiting from training early.")

			if self.use_tensorboard:
				writer.close()

			if not param_selection_mode:
				log.info("Saving model ...")
				self.model.save(base_path / "final-model.pt")
				log.info("Done.")

		# test best model if test data is present
		if self.corpus.test:
			final_score = self.final_test(base_path, eval_mini_batch_size, num_workers)
		else:
			final_score = 0
			log.info("Test data not provided setting final score to 0")

		# pdb.set_trace()
		log.removeHandler(log_handler)

		if self.use_tensorboard:
			writer.close()
		return {
			"test_score": final_score,
			"dev_score_history": dev_score_history,
			"train_loss_history": train_loss_history,
			"dev_loss_history": dev_loss_history,
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

	def assign_pretrained_teacher_predictions(self,coupled_train_data,teachers,is_professor=False,faster=False):
		if not is_professor:
			log.info('Distilling sentences...')
		else:
			log.info('Distilling professor sentences...')
		assert len(self.corpus.targets) == len(coupled_train_data), 'Coupled train data is not equal to target!'
		counter=0
		# res_input=[]
		use_bert=False
		for teacher in teachers:
			if teacher.use_bert:
				use_bert=True
				# break
		start_time=time.time()
		for teacher in teachers:
			teacher = teacher.to(flair.device)
			for index, train_data in enumerate(coupled_train_data):
				target = self.corpus.targets[index]
				if target not in teacher.targets:
					continue
				loader=ColumnDataLoader(list(train_data),self.mini_batch_size,grouped_data=False,use_bert=use_bert, model = teacher, sentence_level_batch = self.sentence_level_batch)
				loader.word_map = teacher.word_map
				loader.char_map = teacher.char_map
				if self.model.tag_dictionary.item2idx !=teacher.tag_dictionary.item2idx:
					pdb.set_trace()
					assert 0, "the tag_dictionaries of the teacher and student are not same"
				for batch in loader:
					counter+=len(batch)
					# student_input, teacher_input = zip(*batch)
					# student_input=list(student_input)
					teacher_input=list(batch)
					lengths1 = torch.Tensor([len(sentence.tokens) for sentence in teacher_input])
					# lengths2 = torch.Tensor([len(sentence.tokens) for sentence in student_input])
					# assert (lengths1==lengths2).all(), 'two batches are not equal!'
					max_len = max(lengths1)
					mask=self.model.sequence_mask(lengths1, max_len).unsqueeze(-1).cuda().float()
					with torch.no_grad():
						# assign tags for the dependency parsing
						teacher_input=loader.assign_tags(teacher.tag_type,teacher.tag_dictionary,teacher_input=teacher_input)
						teacher_input=teacher_input[0]
						if self.model.tag_type=='dependency':
							arc_scores, rel_scores=teacher.forward(teacher_input)
							if self.model.distill_arc:
								logits = arc_scores
							if hasattr(self.model,'distill_rel') and self.model.distill_rel:
								arc_probs = arc_scores.softmax(-1)
								rel_probs = rel_scores.softmax(-1)
								if self.model.distill_factorize:
									logits = arc_probs
								else:
									logits = arc_probs.unsqueeze(-1) * rel_probs
						else:
							logits=teacher.forward(teacher_input)
					if self.model.distill_prob:
						logits=F.softmax(logits,-1)
					if hasattr(teacher_input,'features'):
						teacher_input.features = {}
					for idx, sentence in enumerate(teacher_input):
						# if hasattr(sentence,'_teacher_target'):
						#   assert 0, 'The sentence has been filled with teacher target!'
						
						if not faster:
							if self.model.tag_type=="dependency":
								if self.model.distill_factorize:
									sentence.set_teacher_rel_prediction(rel_probs[idx][:len(sentence),:len(sentence),:], self.embeddings_storage_mode)
								sentence.set_teacher_prediction(logits[idx][:len(sentence),:len(sentence)], self.embeddings_storage_mode)
							else:
								sentence.set_teacher_prediction(logits[idx][:len(sentence)], self.embeddings_storage_mode)
						else:
							sentence.set_teacher_prediction(logits[idx]*mask[idx], self.embeddings_storage_mode)
						teacher_input[idx].clear_embeddings()
					del logits
					# res_input+=student_input
					# store_embeddings(teacher_input, "none")
					
				# del teacher
			teacher = teacher.to('cpu')
		end_time=time.time()
		print("Distilling Costs: ", f"{end_time-start_time:.2f}","senconds")
		res_input=[]
		for data in coupled_train_data:
			for sentence in data:
				res_input.append(sentence)
		if is_professor:
			log.info('Distilled '+str(counter)+' professor sentences')
			return coupled_train_data
		else:
			log.info('Distilled '+str(counter)+' sentences by '+str(len(teachers))+' models')
			return res_input

	def assign_pretrained_teacher_targets(self,coupled_train_data,teachers,best_k=10):
		log.info('Distilling sentences as targets...')
		assert len(self.corpus.targets) == len(coupled_train_data), 'Coupled train data is not equal to target!'
		counter=0
		use_bert=False
		for teacher in teachers:
			if teacher.use_bert:
				use_bert=True
		for teacherid, teacher in enumerate(teachers):
			teacher = teacher.to(flair.device)
			for index, train_data in enumerate(coupled_train_data):
				target = self.corpus.targets[index]
				if target not in teacher.targets:
					continue
				loader=ColumnDataLoader(list(train_data),self.mini_batch_size,grouped_data=False,use_bert=use_bert, model = teacher, sentence_level_batch = self.sentence_level_batch)
				loader.word_map = teacher.word_map
				loader.char_map = teacher.char_map
				if self.model.tag_dictionary.item2idx !=teacher.tag_dictionary.item2idx:
					# pdb.set_trace()
					assert 0, "the tag_dictionaries of the teacher and student are not same"
				for batch in loader:
					counter+=len(batch)
					# student_input, teacher_input = zip(*batch)
					# student_input=list(student_input)
					teacher_input=list(batch)
					lengths1 = torch.Tensor([len(sentence.tokens) for sentence in teacher_input])
					# lengths2 = torch.Tensor([len(sentence.tokens) for sentence in student_input])
					# assert (lengths1==lengths2).all(), 'two batches are not equal!'
					max_len = max(lengths1)
					mask=self.model.sequence_mask(lengths1, max_len).unsqueeze(-1).cuda().long()
					lengths1=lengths1.long()
					with torch.no_grad():
						teacher_input=loader.assign_tags(teacher.tag_type,teacher.tag_dictionary,teacher_input=teacher_input)
						teacher_input=teacher_input[0]
						if self.model.tag_type=='dependency':
							mask[:,0] = 0
							arc_scores, rel_scores=teacher.forward(teacher_input)
							logits = arc_scores
						else:
							logits=teacher.forward(teacher_input)
						if self.model.distill_crf:
							if self.model.tag_type=='dependency':
								if self.model.distill_rel:
									arc_probs = arc_scores.softmax(-1)
									rel_probs = rel_scores.softmax(-1)
									arc_rel_probs = arc_probs.unsqueeze(-1) * rel_probs
									arc_rel_scores = (arc_rel_probs+1e-100).log()
									dist=generate_tree(arc_rel_scores,mask.squeeze(-1).long(),is_mst=self.model.is_mst)
									decode_idx = dist.topk(best_k)
									decode_idx = decode_idx.permute(1,2,3,4,0)
									decode_idx = convert_score_back(decode_idx)
									# dependency, head
									arc_predictions = decode_idx.sum(-2).argmax(-2)
									rel_predictions = decode_idx.sum(-3).argmax(-2)
									decode_idx = arc_predictions
									# decode_idx = convert_score_back(arc_predictions.permute(1,2,0))
									# rel_predictions = convert_score_back(rel_predictions.permute(1,2,0))
								else:
									dist=generate_tree(arc_scores,mask.squeeze(-1).long(),is_mst=self.model.is_mst)
									decode_idx = dist.topk(best_k)
									decode_idx = decode_idx.permute(1,2,3,0)
									decode_idx = convert_score_back(decode_idx)
									decode_idx = decode_idx.argmax(-2)
								maximum_num_trees = dist.count
								# sentence_lens = sentence_lens ** 2
								# generate the top-k mask
								path_mask = torch.arange(best_k).expand(len(mask), best_k).type_as(mask) < maximum_num_trees.unsqueeze(1)
								if self.model.crf_attention:
									path_score = dist.kmax(best_k).transpose(0,1)
									path_score.masked_fill_(~path_mask.bool(), float('-inf'))
									path_score = path_score.softmax(-1)
								else:
									path_score = path_mask
							else:
								if self.gold_reward:
									for s_id, sentence in enumerate(batch):
										# get the tags in this sentence
										tag_idx: List[int] = [
											tag_dictionary.get_idx_for_item(token.get_tag(tag_type).value)
											for token in sentence
										]
										# add tags as tensor
										tag_template = torch.zeros(max_len,device='cpu')
										tag = torch.tensor(tag_idx, device='cpu')
										tag_template[:len(sentence)]=tag
								path_score, decode_idx=teacher._viterbi_decode_nbest(logits,mask,best_k)
								# test=(decode_idx- decode_idx[:,:,0][:,:,None]).abs().sum(1)[:,[1,2,3,4]]
								# sent_len=mask.sum([-1,-2])
								# if 0 in test:
								#   for i,line in enumerate(test):
								#       if 0 in line and sent_len[i]!=1:
								#           pdb.set_trace()
						if self.model.distill_posterior:
							if self.model.tag_type=='dependency':
								if self.model.distill_rel:
									assert 0
									arc_probs = arc_scores.softmax(-1)
									rel_probs = rel_scores.softmax(-1)
									arc_rel_probs = arc_probs.unsqueeze(-1) * rel_probs
									arc_rel_scores = (arc_rel_probs+1e-100).log()
									# arc_rel_scores.masked_fill_(~mask.unsqueeze(1).unsqueeze(1).bool(), float(-1e9))
									dist=generate_tree(arc_rel_scores,mask.squeeze(-1).long(),is_mst=self.model.is_mst)
								else:
									# calculate the marginal distribution
									forward_backward_score = crf(arc_scores, mask.squeeze(-1).bool())
									# dist=generate_tree(arc_scores,mask.squeeze(-1).long(),is_mst=self.model.is_mst)
								# forward_backward_score = dist.marginals
								# forward_backward_score = forward_backward_score.detach()
							else:
								# forward_var = self.model._forward_alg(logits, lengths1, distill_mode=True)
								# backward_var = self.model._backward_alg(logits, lengths1)
								logits[:,:,teacher.tag_dictionary.get_idx_for_item(STOP_TAG)]-=1e12
								logits[:,:,teacher.tag_dictionary.get_idx_for_item(START_TAG)]-=1e12
								logits[:,:,teacher.tag_dictionary.get_idx_for_item('<unk>')]-=1e12
								if not hasattr(teacher,'transitions'):
									forward_backward_score = logits
								else:
									forward_var = teacher._forward_alg(logits, lengths1, distill_mode=True)
									backward_var = teacher._backward_alg(logits, lengths1)
									forward_backward_score = (forward_var + backward_var) * mask.float()


									# pdb.set_trace()
									# temperature = 10
									# partition_score = teacher._forward_alg(logits,lengths1, T = temperature)
									# forward_var = teacher._forward_alg(logits, lengths1, distill_mode=True, T = temperature)
									# backward_var = teacher._backward_alg(logits, lengths1, T = temperature)
									# forward_backward_score2 = (forward_var + backward_var) * mask.float()
									# fw_bw_partition = forward_backward_score2.logsumexp(-1)
									# fw_bw_partition = fw_bw_partition * mask.squeeze(-1)
									# print(((fw_bw_partition - partition_score[:,None]) * mask.squeeze(-1)).abs().max())
									# print(max(lengths1))

									# (logits[:,0]/temperature+teacher.transitions[None,:,teacher.tag_dictionary.get_idx_for_item(START_TAG)]/temperature).logsumexp(-1)
									# (teacher.transitions[teacher.tag_dictionary.get_idx_for_item(STOP_TAG)]/temperature).logsumexp(-1)
									# backward_partition = teacher._backward_alg(logits,lengths1, distill_mode = False)
									# print(((forward_backward_score.logsumexp(-1)[:,0]-partition_score).abs()).max())
									# print((backward_partition-partition_score).abs().max())
									# print((((forward_backward_score.logsumexp(-1)[:,0]-partition_score).abs())>0).sum()/float(len(forward_backward_score)))
									# print((((backward_partition-partition_score).abs())>0).sum()/float(len(forward_backward_score)))
									# print(max(lengths1))
									# torch.logsumexp(forward_backward_score,-1)
									# temp_var = forward_var[range(forward_var.shape[0]), lengths1-1, :]
									# terminal_var = temp_var + teacher.transitions[teacher.tag_dictionary.get_idx_for_item(STOP_TAG)][None,:]
									# temp_var = backward_var[range(forward_var.shape[0]), 0, :]
									# terminal_var = temp_var + teacher.transitions[:,teacher.tag_dictionary.get_idx_for_item(START_TAG)][None,:]
									# forward_backward_score.logsumexp(-1)
						
					for idx, sentence in enumerate(teacher_input):
						# if hasattr(sentence,'_teacher_target'):
						#   assert 0, 'The sentence has been filled with teacher target!'
						if self.model.distill_crf:
							if self.model.crf_attention or self.model.tag_type=='dependency':
								sentence.set_teacher_weights(path_score[idx], self.embeddings_storage_mode)
							sentence.set_teacher_target(decode_idx[idx]*mask[idx], self.embeddings_storage_mode)
							if hasattr(self.model,'distill_rel') and self.model.distill_rel:
								sentence.set_teacher_rel_target(rel_predictions[idx]*mask[idx], self.embeddings_storage_mode)
						if self.model.distill_posterior:
							# sentence.set_teacher_posteriors(forward_backward_score[idx][:len(sentence)-1,:len(sentence)-1], self.embeddings_storage_mode)
							sentence.set_teacher_posteriors(forward_backward_score[idx], self.embeddings_storage_mode)
						teacher_input[idx].clear_embeddings()
					del logits

				# store_embeddings(teacher_input, "none")
			teacher = teacher.to('cpu')
			# del teacher

		log.info('Distilled '+str(counter)+' sentences')
		res_input=[]
		for data in coupled_train_data:
			for sentence in data:
				res_input.append(sentence)
		return res_input
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
			if (not is_crf and not is_posterior):
				targets=[x._teacher_prediction for x in batch]
				if hasattr(self.model, 'distill_factorize') and self.model.distill_factorize:
					rel_targets=[x._teacher_rel_prediction for x in batch]
				# pdb.set_trace()
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
					new_starts=[]
					new_ends=[]
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
								bias = 0
								# pdb.set_trace()
								
								# if max_shape - bias == 0:
								# pdb.set_trace()
								# if sent_lens[index] == 1:
								#   pdb.set_trace()
								post_val=post_vals[idx]
								shape=[max_shape-bias]+list(post_val.shape[1:])
								new_posterior=torch.zeros(shape).type_as(post_val)
								new_posterior[:sent_lens[index]-bias]=post_val[:sent_lens[index]-bias]
								new_posteriors.append(new_posterior)
							# pdb.set_trace()
							
							
					if is_crf:
						batch[index]._teacher_target=new_targets
						if hasattr(self.model,'distill_rel') and  self.model.distill_rel:
							batch[index]._teacher_rel_target=new_rel_targets
					if is_posterior:
						batch[index]._teacher_posteriors=new_posteriors
					
					if is_token_att:
						batch[index]._teacher_sentfeats=new_sentfeats
					if (not is_crf and not is_posterior):
						if hasattr(self.model, 'distill_factorize') and self.model.distill_factorize:
							batch[index]._teacher_rel_prediction=new_rel_targets
						batch[index]._teacher_prediction=new_targets
			if hasattr(batch,'teacher_features'):
				if is_posterior:
					try:
						batch.teacher_features['posteriors']=torch.stack([sentence.get_teacher_posteriors() for sentence in batch],0).cpu()
					except:
						pdb.set_trace()
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
	def compare_posterior(self, base_path: Path, eval_mini_batch_size: int, max_k=21, min_k=1):
		self.model.eval()
		if (base_path / "best-model.pt").exists():
			self.model = self.model.load(base_path / "best-model.pt")
			log.info("Testing using best model ...")
		elif (base_path / "final-model.pt").exists():
			self.model = self.model.load(base_path / "final-model.pt")
			log.info("Testing using final model ...")
		loader=ColumnDataLoader(list(self.corpus.test),eval_mini_batch_size, use_bert=self.use_bert,tokenizer=self.bert_tokenizer)
		loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
		XE=torch.zeros(len(range(min_k,max_k))).float().cuda()
		weighted_XE=torch.zeros(len(range(min_k,max_k))).float().cuda()
		total_tp=0
		total=0
		with torch.no_grad():
			total_length=0
			for batch in loader:
				total_length+=len(batch)
				lengths1 = torch.Tensor([len(sentence.tokens) for sentence in batch])
				

				max_len = max(lengths1)
				mask=self.model.sequence_mask(lengths1, max_len).unsqueeze(-1).cuda().long()
				lengths1=lengths1.long()
				batch_range=torch.arange(len(batch))
				logits=self.model.forward(batch)
				forward_var = self.model._forward_alg(logits, lengths1, distill_mode=True)
				backward_var = self.model._backward_alg(logits, lengths1)
				forward_backward_score = (forward_var + backward_var) * mask.float()
				fwbw_probability = F.softmax(forward_backward_score,dim=-1)
				total_tp+=((fwbw_probability.max(-1)[0]>0.98).type_as(mask)*mask.squeeze(-1)).sum().item()
				total+=mask.sum().item()
				# log_prob=torch.log(fwbw_probability+1e-12)

				# for current_idx,best_k in enumerate(range(min_k,max_k)):
				
				#     path_score, decode_idx=self.model._viterbi_decode_nbest(logits,mask,best_k)

				#     
					
				#     tag_distribution = torch.zeros(fwbw_probability.shape).type_as(fwbw_probability)
				#     weighted_tag_distribution = torch.zeros(fwbw_probability.shape).type_as(fwbw_probability)
				#     for k in range(best_k):
				#         for i in range(max_len.long().item()):
				#             tag_distribution[batch_range,i,decode_idx[:,i,k]]+=1
				#             weighted_tag_distribution[batch_range,i,decode_idx[:,i,k]]+=path_score[:,k]
				#     tag_distribution=tag_distribution/tag_distribution.sum(-1,keepdim=True)
				#     weighted_tag_distribution=weighted_tag_distribution/weighted_tag_distribution.sum(-1,keepdim=True)
					

				#     XE[current_idx]+=((log_prob*tag_distribution*mask.float()).sum(-1).sum(-1)/(mask.float().sum(-1).sum(-1)*tag_distribution.shape[-1])).sum()
				#     weighted_XE[current_idx]+=((log_prob*weighted_tag_distribution*mask.float()).sum(-1).sum(-1)/(mask.float().sum(-1).sum(-1)*weighted_tag_distribution.shape[-1])).sum()
				#     if best_k==min_k or best_k==max_k-1:
				#         pdb.set_trace()
					
			pdb.set_trace()
			print(total)
			print(total_tp)
			# print('XE: ',XE)
			# print('weighted_XE: ',weighted_XE)
			# print('total_length: ',total_length)
		
	def final_test(
		self, base_path: Path, eval_mini_batch_size: int, num_workers: int = 8, overall_test: bool = True, quiet_mode: bool = False, nocrf: bool = False, predict_posterior: bool = False, debug: bool = False, keep_embedding: int = -1, sort_data=False,
	):

		log_line(log)
		

		self.model.eval()
		
		if quiet_mode:
			#blockPrint()
			log.disabled=True
		if (base_path / "best-model.pt").exists():
			self.model = self.model.load(base_path / "best-model.pt")
			log.info("Testing using best model ...")
		elif (base_path / "final-model.pt").exists():
			self.model = self.model.load(base_path / "final-model.pt")
			log.info("Testing using final model ...")
		if debug:
			self.model.debug=True
			
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
				self.gpu_friendly_assign_embedding([loader])
			for x in sorted(loader[0].features.keys()):
				print(x)
			test_results, test_loss = self.model.evaluate(
				loader,
				out_path=base_path / "test.tsv",
				embeddings_storage_mode="cpu",
			)
			test_results: Result = test_results
			log.info(test_results.log_line)
			log.info(test_results.detailed_results)
			log_line(log)
			# if self.model.embedding_selector:
			#   print(sorted(loader[0].features.keys()))
			#   print(self.model.selector)
			#   print(torch.argmax(self.model.selector,-1))
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
					self.gpu_friendly_assign_embedding([loader])
				current_result, test_loss = self.model.evaluate(
					loader,
					out_path=base_path / f"{subcorpus.name}-test.tsv",
					embeddings_storage_mode="none",
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
					self.gpu_friendly_assign_embedding([loader])
				current_result, test_loss = self.model.evaluate(
					loader,
					out_path=base_path / f"{self.corpus.targets[index]}-test.tsv",
					embeddings_storage_mode="none",
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
		# if self.model.embedding_selector:
		#       print(sorted(loader[0].features.keys()))
		#       print(self.model.selector)
		#       print(torch.argmax(self.model.selector,-1))
		if keep_embedding<0:
			print()
		if overall_test:
			# get and return the final test score of best model
			final_score = test_results.main_score

			return final_score
		return 0
	def find_learning_rate(
		self,
		base_path: Union[Path, str],
		file_name: str = "learning_rate.tsv",
		start_learning_rate: float = 1e-7,
		end_learning_rate: float = 10,
		iterations: int = 200,
		mini_batch_size: int = 32,
		stop_early: bool = False,
		smoothing_factor: float = 0.98,
		**kwargs,
	) -> Path:
		best_loss = None
		moving_avg_loss = 0

		# cast string to Path
		if type(base_path) is str:
			base_path = Path(base_path)
		learning_rate_tsv = init_output_file(base_path, file_name)

		with open(learning_rate_tsv, "a") as f:
			f.write("ITERATION\tTIMESTAMP\tLEARNING_RATE\tTRAIN_LOSS\n")

		optimizer = self.optimizer(
			self.model.parameters(), lr=start_learning_rate, **kwargs
		)

		train_data = self.corpus.train

		scheduler = ExpAnnealLR(optimizer, end_learning_rate, iterations)

		model_state = self.model.state_dict()
		self.model.train()
		print('Batch Size: ', mini_batch_size)
		step = 0
		while step < iterations:
			batch_loader=ColumnDataLoader(list(train_data),mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer)            
			# batch_loader = DataLoader(
			#     train_data, batch_size=mini_batch_size, shuffle=True
			# )
			for batch in batch_loader:
				batch_loader.true_reshuffle()
				step += 1

				# forward pass
				loss = self.model.forward_loss(batch)

				# update optimizer and scheduler
				optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
				optimizer.step()
				scheduler.step(step)

				print(scheduler.get_lr())
				learning_rate = scheduler.get_lr()[0]

				loss_item = loss.item()
				if step == 1:
					best_loss = loss_item
				else:
					if smoothing_factor > 0:
						moving_avg_loss = (
							smoothing_factor * moving_avg_loss
							+ (1 - smoothing_factor) * loss_item
						)
						loss_item = moving_avg_loss / (
							1 - smoothing_factor ** (step + 1)
						)
					if loss_item < best_loss:
						best_loss = loss

				if step > iterations:
					break

				if stop_early and (loss_item > 4 * best_loss or torch.isnan(loss)):
					log_line(log)
					log.info("loss diverged - stopping early!")
					step = iterations
					break

				with open(str(learning_rate_tsv), "a") as f:
					f.write(
						f"{step}\t{datetime.datetime.now():%H:%M:%S}\t{learning_rate}\t{loss_item}\n"
					)

			self.model.load_state_dict(model_state)
			self.model.to(flair.device)

		log_line(log)
		log.info(f"learning rate finder finished - plot {learning_rate_tsv}")
		log_line(log)

		return Path(learning_rate_tsv)

	def get_subtoken_length(self,sentence):
		return len(self.model.embeddings.embeddings[0].tokenizer.tokenize(sentence.to_tokenized_string()))

	