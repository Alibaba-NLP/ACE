from .trainer import *
from flair.training_utils import store_teacher_predictions
from flair.list_data import ListCorpus
import math
import random
import pdb
import copy
from flair.datasets import CoupleDataset
from ..custom_data_loader import ColumnDataLoader
from torch.optim.adam import Adam
import torch.nn.functional as F
import traceback
import sys
import os
import numpy as np
import h5py
def get_corpus_lengths(train_data):
	return [len(corpus) for corpus in train_data]

def get_corpus_iterations(train_data, batch_size):
	corpus_lengths=get_corpus_lengths(train_data)
	return [math.ceil(corpus_length/float(batch_size)) for corpus_length in corpus_lengths]

def generate_training_order(train_data,batch_size,training_order=None):
	if training_order is None:
		corpus_iters=get_corpus_iterations(train_data,batch_size)
		training_order=[]
		for idx, iters in enumerate(corpus_iters):
			training_order=training_order+iters*[idx]
	random.shuffle(training_order)
	return training_order

# Disable
def blockPrint():
		sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
		sys.stdout = sys.__stdout__

def count_parameters(model):
	total_param = 0
	for name,param in model.named_parameters():
		num_param = np.prod(param.size())
		# print(name,num_param)
		total_param+=num_param
	return total_param



class ModelDistiller(ModelTrainer):
	def __init__(
		self,
		student: flair.nn.Model,
		teachers: List[flair.nn.Model],
		corpus: ListCorpus,
		optimizer: torch.optim.Optimizer = SGD,
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
	):
		"""
		Initialize a model trainer
		:param model: The model that you want to train. The model should inherit from flair.nn.Model
		:param corpus: The dataset used to train the model, should be of type Corpus
		:param optimizer: The optimizer to use (typically SGD or Adam)
		:param epoch: The starting epoch (normally 0 but could be higher if you continue training model)
		:param optimizer_state: Optimizer state (necessary if continue training from checkpoint)
		:param scheduler_state: Scheduler state (necessary if continue training from checkpoint)
		:param use_tensorboard: If True, writes out tensorboard information
		"""
		# if teachers is not None:
		#   assert len(teachers)==len(corpus.train_list), 'Training data and teachers should be the same length now!'
		self.model: flair.nn.Model = student
		self.corpus: ListCorpus = corpus
		self.distill_mode = distill_mode
		self.sentence_level_batch = sentence_level_batch

		if self.distill_mode:
			self.corpus_teacher: ListCorpus = copy.deepcopy(corpus)
			# self.corpus_mixed_train: ListCorpus = [CoupleDataset(student_set,self.corpus_teacher.train_list[index]) for index,student_set in enumerate(self.corpus.train_list)]
			self.teachers: List[flair.nn.Model] = teachers
			self.professors: List[flair.nn.Model] = professors
			if self.teachers is not None:
				for teacher in self.teachers: teacher.eval()
			for professor in self.professors: professor.eval()
		# self.corpus = self.assign_pretrained_teacher_predictions(self.corpus,self.corpus_teacher,self.teachers)
		if self.model.biaf_attention and not is_test:
			
			pass
			self.model.init_biaf(self.teachers[0].hidden_size, num_teachers=len(self.teachers)+int(len(self.professors)>0))
		self.optimizer: torch.optim.Optimizer = optimizer
		if type(optimizer)==str:
			self.optimizer = getattr(torch.optim,optimizer)

		self.epoch: int = epoch
		self.scheduler_state: dict = scheduler_state
		self.optimizer_state: dict = optimizer_state
		self.use_tensorboard: bool = use_tensorboard
		
		self.config = config
		self.use_bert = False
		for embedding in self.config['embeddings']:
			if 'bert' in embedding.lower():
				self.use_bert=True
		self.ensemble_distill_mode: bool = ensemble_distill_mode
		self.train_with_professor: bool = train_with_professor
		# if self.train_with_professor:
		#   assert len(self.professors) == len(self.corpus.train_list), 'Now only support same number of professors and corpus!'
	def train(
		self,
		base_path: Union[Path, str],
		learning_rate: float = 0.1,
		mini_batch_size: int = 32,
		eval_mini_batch_size: int = None,
		max_epochs: int = 100,
		anneal_factor: float = 0.5,
		patience: int = 3,
		min_learning_rate: float = 0.0001,
		train_with_dev: bool = False,
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
		train_teacher: bool = False,
		professor_interpolation = 0.5,
		best_k = 10,
		gold_reward = False,
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
		log.info(f' - learning_rate: "{learning_rate}"')
		log.info(f' - mini_batch_size: "{mini_batch_size}"')
		log.info(f' - patience: "{patience}"')
		log.info(f' - anneal_factor: "{anneal_factor}"')
		log.info(f' - max_epochs: "{max_epochs}"')
		log.info(f' - shuffle: "{shuffle}"')
		log.info(f' - train_with_dev: "{train_with_dev}"')
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

		optimizer: torch.optim.Optimizer = self.optimizer(
			self.model.parameters(), lr=learning_rate, **kwargs
		)
		if self.optimizer_state is not None:
			optimizer.load_state_dict(self.optimizer_state)

		if use_amp:
			self.model, optimizer = amp.initialize(
				self.model, optimizer, opt_level=amp_opt_level
			)

		# minimize training loss if training with dev data, else maximize dev score
		anneal_mode = "min" if train_with_dev else "max"

		scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
			optimizer,
			factor=anneal_factor,
			patience=patience,
			mode=anneal_mode,
			verbose=True,
		)

		if self.scheduler_state is not None:
			scheduler.load_state_dict(self.scheduler_state)

		# start from here, the train data is a list now
		train_data = self.corpus.train_list
		if self.distill_mode:
			train_data_teacher = self.corpus_teacher.train_list
		# train_data = self.corpus_mixed
		# if training also uses dev data, include in training set
		if train_with_dev:
			train_data = [ConcatDataset([train, self.corpus.dev_list[index]]) for index, train in enumerate(self.corpus.train_list)]
			if self.distill_mode:
				train_data_teacher = [ConcatDataset([train, self.corpus_teacher.dev_list[index]]) for index, train in enumerate(self.corpus_teacher.train_list)]
			# train_data = [ConcatDataset([train, self.corpus_mixed.dev_list[index]]) for index, train in self.corpus_mixed.train_list]
			# train_data_teacher = ConcatDataset([self.corpus_teacher.train, self.corpus_teacher.dev])
			# train_data = ConcatDataset([self.corpus_mixed.train, self.corpus_mixed.dev])
		if self.distill_mode:
			coupled_train_data = [CoupleDataset(data,train_data_teacher[index]) for index, data in enumerate(train_data)]
			if 'fast' in self.model.__class__.__name__.lower():
				faster=True
			else:
				faster=False

			if self.train_with_professor:
				log.info(f"Predicting professor prediction")
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
			del self.teachers, self.corpus_teacher  
			batch_loader=ColumnDataLoader(train_data,mini_batch_size,shuffle,use_bert=self.use_bert, model = self.model, sentence_level_batch = self.sentence_level_batch)
		else:
			batch_loader=ColumnDataLoader(ConcatDataset(train_data),mini_batch_size,shuffle,use_bert=self.use_bert, model = self.model, sentence_level_batch = self.sentence_level_batch)
		batch_loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
		if self.distill_mode:
			if faster:
				batch_loader=self.resort(batch_loader,is_crf=self.model.distill_crf, is_posterior = self.model.distill_posterior, is_token_att = self.model.token_level_attention)

		dev_loader=ColumnDataLoader(list(self.corpus.dev),eval_mini_batch_size,use_bert=self.use_bert, model = self.model, sentence_level_batch = self.sentence_level_batch)
		dev_loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
		test_loader=ColumnDataLoader(list(self.corpus.test),eval_mini_batch_size,use_bert=self.use_bert, model = self.model, sentence_level_batch = self.sentence_level_batch)
		test_loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
		# if self.distill_mode:
		#   batch_loader.expand_teacher_predictions()
		# if sampler is not None:
		#   sampler = sampler(train_data)
		#   shuffle = False

		dev_score_history = []
		dev_loss_history = []
		train_loss_history = []
		# At any point you can hit Ctrl + C to break out of training early.
		try:
			previous_learning_rate = learning_rate
			training_order = None
			for epoch in range(0 + self.epoch, max_epochs + self.epoch):
				log_line(log)

				# get new learning rate
				for group in optimizer.param_groups:
					learning_rate = group["lr"]

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
				if learning_rate < min_learning_rate:
					log_line(log)
					log.info("learning rate too small - quitting training!")
					log_line(log)
					break
				
				if shuffle:
					batch_loader.reshuffle()
				if true_reshuffle:
					
					batch_loader.true_reshuffle()
					if self.distill_mode:
						batch_loader=self.resort(batch_loader,is_crf=self.model.distill_crf, is_posterior = self.model.distill_posterior, is_token_att = self.model.token_level_attention)
					batch_loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
				self.model.train()
				# TODO: check teacher parameters fixed and with eval() mode

				train_loss: float = 0

				seen_batches = 0
				#total_number_of_batches = sum([len(loader) for loader in batch_loader])
				total_number_of_batches = len(batch_loader)

				modulo = max(1, int(total_number_of_batches / 10))

				# process mini-batches
				batch_time = 0
				if self.distill_mode:
					if self.teacher_annealing:
						interpolation=1-(epoch*self.anneal_factor)/100.0
						if interpolation<0:
							interpolation=0
					else:
						interpolation=self.interpolation
					log.info("Current loss interpolation: "+ str(interpolation))
				total_sent=0
				for batch_no, student_input in enumerate(batch_loader):
					start_time = time.time()
					total_sent+=len(student_input)
					
					try:
						if self.distill_mode:
							loss = self.model.simple_forward_distillation_loss(student_input, interpolation = interpolation, train_with_professor=self.train_with_professor, professor_interpolation = professor_interpolation)
						else:
							loss = self.model.forward_loss(student_input)
						if self.model.use_decoder_timer:
							decode_time=time.time() - self.model.time
						optimizer.zero_grad()
						# Backward
						if use_amp:
							with amp.scale_loss(loss, optimizer) as scaled_loss:
								scaled_loss.backward()
						else:
							loss.backward()
					except Exception:
						traceback.print_exc()
						pdb.set_trace()
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
					optimizer.step()

					seen_batches += 1
					train_loss += loss.item()

					# depending on memory mode, embeddings are moved to CPU, GPU or deleted
					store_embeddings(student_input, embeddings_storage_mode)
					if self.distill_mode:
						store_teacher_predictions(student_input, embeddings_storage_mode)

					batch_time += time.time() - start_time
					if batch_no % modulo == 0:
						if self.model.use_decoder_timer:
							log.info(
								f"epoch {epoch + 1} - iter {batch_no}/{total_number_of_batches} - loss "
								f"{train_loss / seen_batches:.8f} - samples/sec: {total_sent / batch_time:.2f} - decode_sents/sec: {total_sent / decode_time:.2f}"
							)
							
						else:
							log.info(
								f"epoch {epoch + 1} - iter {batch_no}/{total_number_of_batches} - loss "
								f"{train_loss / seen_batches:.8f} - samples/sec: {total_sent / batch_time:.2f}"
							)
						total_sent=0
						batch_time = 0
						iteration = epoch * total_number_of_batches + batch_no
						# if not param_selection_mode:
						# 	weight_extractor.extract_weights(
						# 		self.model.state_dict(), iteration
						# 	)

				train_loss /= seen_batches

				self.model.eval()

				log_line(log)
				log.info(
					f"EPOCH {epoch + 1} done: loss {train_loss:.4f} - lr {learning_rate:.4f}"
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

				if log_dev:
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
								ColumnDataLoader(list(subcorpus.test),eval_mini_batch_size,use_bert=self.use_bert, model = self.model, sentence_level_batch = self.sentence_level_batch),
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
								ColumnDataLoader(list(subcorpus),eval_mini_batch_size,use_bert=self.use_bert, model = self.model, sentence_level_batch = self.sentence_level_batch),
								out_path=base_path / f"{self.corpus.targets[index]}-test.tsv",
								embeddings_storage_mode=embeddings_storage_mode,
							)
							log.info(current_result.log_line)
							log.info(current_result.detailed_results)


				# determine learning rate annealing through scheduler
				scheduler.step(current_score)

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

				# output log file
				with open(loss_txt, "a") as f:

					# make headers on first epoch
					if epoch == 0:
						f.write(
							f"EPOCH\tTIMESTAMP\tBAD_EPOCHS\tLEARNING_RATE\tTRAIN_LOSS"
						)

						if log_train:
							f.write(
								"\tTRAIN_"
								+ "\tTRAIN_".join(
									train_eval_result.log_header.split("\t")
								)
							)
						if log_dev:
							f.write(
								"\tDEV_LOSS\tDEV_"
								+ "\tDEV_".join(dev_eval_result.log_header.split("\t"))
							)
						if log_test:
							f.write(
								"\tTEST_LOSS\tTEST_"
								+ "\tTEST_".join(
									test_eval_result.log_header.split("\t")
								)
							)

					f.write(
						f"\n{epoch}\t{datetime.datetime.now():%H:%M:%S}\t{bad_epochs}\t{learning_rate:.4f}\t{train_loss}"
					)
					f.write(result_line)

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
					and current_score == scheduler.best
				):
					self.model.save(base_path / "best-model.pt")

			# if we do not use dev data for model selection, save final model
			if save_final_model and not param_selection_mode:
				self.model.save(base_path / "final-model.pt")

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
		res_input=[]
		use_bert=False
		for teacher in teachers:
			if self.model.biaf_attention:
				teacher.biaf_attention=True
			if self.model.token_level_attention:
				teacher.token_level_attention=True
			if teacher.use_bert:
				use_bert=True
				# break
		for teacher in teachers:
			teacher = teacher.to(flair.device)
			for index, train_data in enumerate(coupled_train_data):
				target = self.corpus.targets[index]
				if target not in teacher.targets:
					continue
				loader=ColumnDataLoader(list(train_data),self.mini_batch_size,grouped_data=True,use_bert=use_bert, model = self.model, sentence_level_batch = self.sentence_level_batch)
				for batch in loader:
					counter+=len(batch)
					student_input, teacher_input = zip(*batch)
					student_input=list(student_input)
					teacher_input=list(teacher_input)
					lengths1 = torch.Tensor([len(sentence.tokens) for sentence in teacher_input])
					lengths2 = torch.Tensor([len(sentence.tokens) for sentence in student_input])
					assert (lengths1==lengths2).all(), 'two batches are not equal!'
					max_len = max(lengths1)
					mask=self.model.sequence_mask(lengths1, max_len).unsqueeze(-1).cuda().float()
					
					
					with torch.no_grad():
						logits=teacher.forward(teacher_input)
					if self.model.distill_prob:
						
						logits=F.softmax(logits,-1)
					for idx, sentence in enumerate(student_input):
						# if hasattr(sentence,'_teacher_target'):
						#   assert 0, 'The sentence has been filled with teacher target!'
						if self.model.biaf_attention:
							try:
								sentence.set_teacher_sentfeats(teacher.sent_feats[idx],self.embeddings_storage_mode)
							except:
								pdb.set_trace()
						if not faster:
							sentence.set_teacher_prediction(logits[idx][:len(sentence)], self.embeddings_storage_mode)
						else:
							sentence.set_teacher_prediction(logits[idx]*mask[idx], self.embeddings_storage_mode)
						teacher_input[idx].clear_embeddings()
					del logits
					# del teacher.sent_feats[idx]
					
					# store_embeddings(teacher_input, "none")
					
			teacher=teacher.to('cpu')

		if is_professor:
			log.info('Distilled '+str(counter)+' professor sentences')
			return coupled_train_data
		else:
			log.info('Distilled '+str(counter)+' sentences')
			return res_input

	def assign_pretrained_teacher_targets(self,coupled_train_data,teachers,best_k=10):
		log.info('Distilling sentences as targets...')
		assert len(self.corpus.targets) == len(coupled_train_data), 'Coupled train data is not equal to target!'
		counter=0
		res_input=[]
		use_bert=False
		for teacher in teachers:
			if teacher.use_bert:
				use_bert=True
		for teacher in teachers:
			teacher = teacher.to(flair.device)
			for index, train_data in enumerate(coupled_train_data):
				target = self.corpus.targets[index]
				if target not in teacher.targets:
					continue
				loader=ColumnDataLoader(list(train_data),self.mini_batch_size,grouped_data=True,use_bert=use_bert, model = self.model, sentence_level_batch = self.sentence_level_batch)
				for batch in loader:
					counter+=len(batch)
					student_input, teacher_input = zip(*batch)
					student_input=list(student_input)
					teacher_input=list(teacher_input)
					lengths1 = torch.Tensor([len(sentence.tokens) for sentence in teacher_input])
					lengths2 = torch.Tensor([len(sentence.tokens) for sentence in student_input])
					assert (lengths1==lengths2).all(), 'two batches are not equal!'
					max_len = max(lengths1)
					mask=self.model.sequence_mask(lengths1, max_len).unsqueeze(-1).cuda().long()
					lengths1=lengths1.long()
					
					with torch.no_grad():
						logits=teacher.forward(teacher_input)
						if self.model.distill_crf:
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
						if self.model.distill_posterior:
							
							forward_var = teacher._forward_alg(logits, lengths1, distill_mode=True)
							backward_var = teacher._backward_alg(logits, lengths1)
							forward_backward_score = (forward_var + backward_var) * mask.float()
							
						for idx, sentence in enumerate(student_input):
							# if hasattr(sentence,'_teacher_target'):
							#   assert 0, 'The sentence has been filled with teacher target!'
							if self.model.distill_crf:
								if self.model.crf_attention:
									
									sentence.set_teacher_weights(path_score[idx], self.embeddings_storage_mode)                            
								sentence.set_teacher_target(decode_idx[idx]*mask[idx], self.embeddings_storage_mode)
							if self.model.distill_posterior:
								sentence.set_teacher_posteriors(forward_backward_score[idx], self.embeddings_storage_mode)
							teacher_input[idx].clear_embeddings()
						del logits
					res_input+=student_input
					# store_embeddings(teacher_input, "none")
			teacher=teacher.to('cpu')	
			# del teacher
		log.info('Distilled '+str(counter)+' sentences')
		res_input=[]
		for data in coupled_train_data:
			for sentence in data:
				res_input.append(sentence[0])
		return res_input
		# log.info('Distilled '+str(counter)+' sentences')
		# return res_input
	def resort(self,loader,is_crf=False, is_posterior=False, is_token_att=False):
		for batch in loader.data:
			if is_posterior:
				posteriors=[x._teacher_posteriors for x in batch]
				posterior_lens=[len(x[0]) for x in posteriors]
				lens=posterior_lens.copy()
				targets=posteriors.copy()
			if is_token_att:
				sentfeats=[x._teacher_sentfeats for x in batch]
				sentfeats_lens=[len(x[0]) for x in sentfeats]
			#     lens=sentfeats_lens.copy()
			#     targets=sentfeats.copy()
			if is_crf:
				targets=[x._teacher_target for x in batch]
				lens=[len(x[0]) for x in targets]
			
			if not is_crf and not is_posterior:
				targets=[x._teacher_prediction for x in batch]
				lens=[len(x[0]) for x in targets]
			sent_lens=[len(x) for x in batch]
			if is_posterior:
				
				assert posterior_lens==lens, 'lengths of two targets not match'
			if max(lens)>min(lens) or max(sent_lens)!=max(lens):
				# if max(sent_lens)!=max(lens):
				#   pdb.set_trace()
				max_shape=max(sent_lens)
				for index, target in enumerate(targets):
					new_targets=[]
					new_posteriors=[]
					new_sentfeats=[]
					if is_posterior:
						post_vals=posteriors[index]
					if is_token_att:
						sentfeats_vals=sentfeats[index]
					for idx, val in enumerate(target):
						
						if is_crf or (not is_crf and not is_posterior):
							shape=[max_shape]+list(val.shape[1:])
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
					if is_posterior:
						batch[index]._teacher_posteriors=new_posteriors
					if is_token_att:
						batch[index]._teacher_sentfeats=new_sentfeats
					if not is_crf and not is_posterior:
						batch[index]._teacher_prediction=new_targets

		return loader
	def compare_posterior(self, base_path: Path, eval_mini_batch_size: int, max_k=21, min_k=1):
		self.model.eval()
		if (base_path / "best-model.pt").exists():
			self.model = self.model.load(base_path / "best-model.pt")
			log.info("Testing using best model ...")
		elif (base_path / "final-model.pt").exists():
			self.model = self.model.load(base_path / "final-model.pt")
			log.info("Testing using final model ...")
		loader=ColumnDataLoader(list(self.corpus.test),eval_mini_batch_size, use_bert=self.use_bert, model = self.model, sentence_level_batch = self.sentence_level_batch)
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
		self, base_path: Path, eval_mini_batch_size: int, num_workers: int = 8, overall_test: bool = True, quiet_mode: bool = False, nocrf: bool = False, predict_posterior: bool = False, sort_data=False,
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
		if nocrf:
			self.model.use_crf=False
		if predict_posterior:
			self.model.predict_posterior=True
		if overall_test:
			loader=ColumnDataLoader(list(self.corpus.test),eval_mini_batch_size, use_bert=self.use_bert, model = self.model, sentence_level_batch = self.sentence_level_batch)
			loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
			test_results, test_loss = self.model.evaluate(
				loader,
				out_path=base_path / "test.tsv",
				embeddings_storage_mode="none",
			)
			test_results: Result = test_results
			log.info(test_results.log_line)
			log.info(test_results.detailed_results)
			log_line(log)
		if quiet_mode:
			enablePrint()
			print('Average', end=' ')
			print(test_results.main_score, end=' ')
		# if we are training over multiple datasets, do evaluation for each
		if type(self.corpus) is MultiCorpus:
			for subcorpus in self.corpus.corpora:
				log_line(log)
				log.info('current corpus: '+subcorpus.name)
				loader=ColumnDataLoader(list(subcorpus.test),eval_mini_batch_size,use_bert=self.use_bert, model = self.model, sentence_level_batch = self.sentence_level_batch)
				loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
				current_result, test_loss = self.model.evaluate(
					loader,
					out_path=base_path / f"{subcorpus.name}-test.tsv",
					embeddings_storage_mode="none",
				)
				log.info(current_result.log_line)
				log.info(current_result.detailed_results)
				if quiet_mode:
					print(subcorpus.name,end=' ')
					print(current_result.main_score,end=' ')

		elif type(self.corpus) is ListCorpus:
			for index,subcorpus in enumerate(self.corpus.test_list):
				log_line(log)
				log.info('current corpus: '+self.corpus.targets[index])
				loader=ColumnDataLoader(list(subcorpus),eval_mini_batch_size,use_bert=self.use_bert, model = self.model, sentence_level_batch = self.sentence_level_batch)
				loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
				current_result, test_loss = self.model.evaluate(
					loader,
					out_path=base_path / f"{self.corpus.targets[index]}-test.tsv",
					embeddings_storage_mode="none",
				)
				log.info(current_result.log_line)
				log.info(current_result.detailed_results)
				if quiet_mode:
					print(self.corpus.targets[index],end=' ')
					print(current_result.main_score,end=' ')
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
			batch_loader=ColumnDataLoader(list(train_data),mini_batch_size,use_bert=self.use_bert, model = self.model, sentence_level_batch = self.sentence_level_batch)
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
	def gpu_friendly_assign_embedding(self,loaders, selection = None):
		# pdb.set_trace()
		# torch.cuda.empty_cache()
		embedlist = sorted([(embedding.name, embedding) for embedding in self.model.embeddings.embeddings], key = lambda x: x[0])
		# for embedding in self.model.embeddings.embeddings:
		for idx, embedding_tuple in enumerate(embedlist):
			embedding = embedding_tuple[1]
			
			if ('WordEmbeddings' != embedding.__class__.__name__ and 'FastWordEmbeddings' != embedding.__class__.__name__ and 'Char' not in embedding.__class__.__name__ and 'Lemma' not in embedding.__class__.__name__ and 'POS' not in embedding.__class__.__name__) and not (hasattr(embedding,'fine_tune') and embedding.fine_tune):
				# pdb.set_trace()

				log.info(f"{embedding.name} {count_parameters(embedding)}")
				# 
				if embedding.__class__.__name__ == 'TransformerWordEmbeddings':
					log.info(f"{embedding.pooling_operation}")
				if selection is not None:
					if selection[idx] == 0:
						log.info(f"{embedding.name} is not selected, Skipping")
						continue
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
				for loader in loaders:
					for sentences in loader:
						lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
						longest_token_sequence_in_batch: int = max(lengths)
						# if longest_token_sequence_in_batch>100:
						#   pdb.set_trace()
						embedding.embed(sentences)
						store_embeddings(sentences, self.embeddings_storage_mode)
				embedding=embedding.to('cpu')
				if 'elmo' in embedding.name:
					embedding.ee.elmo_bilm.to('cpu')
			else:
				embedding=embedding.to(flair.device)
		# torch.cuda.empty_cache()
		log.info("Finished Embeddings Assignments")
		return 
	def assign_predicted_embeddings(self,doc_dict,embedding,file_name):
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
			# for idx, sentence in enumerate(doc_dict[key]):
		log.info("Loaded predicted embeddings: "+file_name)
		return 
