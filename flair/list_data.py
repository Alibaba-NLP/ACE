from .data import *
class ListCorpus(Corpus):
	def __init__(
		self,
		train: List[FlairDataset],
		dev: List[FlairDataset],
		test: List[FlairDataset],
		name: str = "listcorpus",
		targets: list = [],
	):
		# In this Corpus, we set train list to be our target to train, we keep self._train the same as the Class Corpus as the counting and preprocessing is needed
		self.train_list: List[FlairDataset] = train
		self.dev_list: List[FlairDataset] = dev
		self.test_list: List[FlairDataset] = test
		self._train: FlairDataset = ConcatDataset([data for data in train])
		self._dev: FlairDataset = ConcatDataset([data for data in dev])
		self._test: FlairDataset = ConcatDataset([data for data in test])
		self.name: str = name
		self.targets = targets