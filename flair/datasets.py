import os, csv
from abc import abstractmethod

from torch.utils.data import Dataset, random_split
from typing import List, Dict, Union
import re
import logging
from pathlib import Path

import torch.utils.data.dataloader
from torch.utils.data.dataset import Subset, ConcatDataset

import flair
from flair.data import Sentence, Corpus, Token, FlairDataset
from flair.file_utils import cached_path
import pdb

log = logging.getLogger("flair")


class ColumnCorpus(Corpus):
    def __init__(
        self,
        data_folder: Union[str, Path],
        column_format: Dict[int, str],
        train_file=None,
        test_file=None,
        dev_file=None,
        tag_to_bioes=None,
        comment_symbol: str = None,
        in_memory: bool = True,
    ):
        """
        Instantiates a Corpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.

        :param data_folder: base folder with the task data
        :param column_format: a map specifying the column format
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param tag_to_bioes: whether to convert to BIOES tagging scheme
        :param comment_symbol: if set, lines that begin with this symbol are treated as comments
        :param in_memory: If set to True, the dataset is kept in memory as Sentence objects, otherwise does disk reads
        :return: a Corpus with annotated train, dev and test data
        """

        if type(data_folder) == str:
            data_folder: Path = Path(data_folder)

        if train_file is not None:
            train_file = data_folder / train_file
        if test_file is not None:
            test_file = data_folder / test_file
        if dev_file is not None:
            dev_file = data_folder / dev_file

        # automatically identify train / test / dev files
        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if file_name.endswith(".gz"):
                    continue
                if "train" in file_name:
                    train_file = file
                if "dev" in file_name:
                    dev_file = file
                if "testa" in file_name:
                    dev_file = file
                if "testb" in file_name:
                    test_file = file

            # if no test file is found, take any file with 'test' in name
            if test_file is None:
                for file in data_folder.iterdir():
                    file_name = file.name
                    if file_name.endswith(".gz"):
                        continue
                    if "test" in file_name:
                        test_file = file

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(train_file))
        log.info("Dev: {}".format(dev_file))
        log.info("Test: {}".format(test_file))

        # get train data
        train = ColumnDataset(
            train_file,
            column_format,
            tag_to_bioes,
            comment_symbol=comment_symbol,
            in_memory=in_memory,
        )

        # read in test file if exists, otherwise sample 10% of train data as test dataset
        if test_file is not None:
            test = ColumnDataset(
                test_file,
                column_format,
                tag_to_bioes,
                comment_symbol=comment_symbol,
                in_memory=in_memory,
            )
        else:
            train_length = len(train)
            test_size: int = round(train_length / 10)
            splits = random_split(train, [train_length - test_size, test_size])
            train = splits[0]
            test = splits[1]

        # read in dev file if exists, otherwise sample 10% of train data as dev dataset
        if dev_file is not None:
            dev = ColumnDataset(
                dev_file,
                column_format,
                tag_to_bioes,
                comment_symbol=comment_symbol,
                in_memory=in_memory,
            )
        else:
            train_length = len(train)
            dev_size: int = round(train_length / 10)
            splits = random_split(train, [train_length - dev_size, dev_size])
            train = splits[0]
            dev = splits[1]

        super(ColumnCorpus, self).__init__(train, dev, test, name=data_folder.name)


class UniversalDependenciesCorpus(Corpus):
    def __init__(
        self,
        data_folder: Union[str, Path],
        train_file=None,
        test_file=None,
        dev_file=None,
        in_memory: bool = True,
        add_root: bool = False,
        spliter = '\t',
    ):
        """
        Instantiates a Corpus from CoNLL-U column-formatted task data such as the UD corpora

        :param data_folder: base folder with the task data
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param in_memory: If set to True, keeps full dataset in memory, otherwise does disk reads
        :return: a Corpus with annotated train, dev and test data
        """
        if type(data_folder) == str:
            data_folder: Path = Path(data_folder)

        # automatically identify train / test / dev files
        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if "train" in file_name:
                    train_file = file
                if "test" in file_name:
                    test_file = file
                if "dev" in file_name:
                    dev_file = file
                if "testa" in file_name:
                    dev_file = file
                if "testb" in file_name:
                    test_file = file

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(train_file))
        log.info("Test: {}".format(test_file))
        log.info("Dev: {}".format(dev_file))

        # get train data
        train = UniversalDependenciesDataset(train_file, in_memory=in_memory, add_root=add_root, spliter=spliter)

        # get test data
        test = UniversalDependenciesDataset(test_file, in_memory=in_memory, add_root=add_root, spliter=spliter)

        # get dev data
        dev = UniversalDependenciesDataset(dev_file, in_memory=in_memory, add_root=add_root, spliter=spliter)

        super(UniversalDependenciesCorpus, self).__init__(
            train, dev, test, name=data_folder.name
        )

class UD(UniversalDependenciesCorpus):
    def __init__(self, treebank: str, base_path: Union[str, Path] = None, in_memory: bool = True, add_root: bool = False):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name / treebank

        # if not os.path.exists(data_folder):
        #     os.call("git clone https://github.com/UniversalDependencies/"+treebank+" "+data_folder+"/"+treebank)
        

        # # # download data if necessary
        # # web_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Tamil-TTB/master"
        # # cached_path(f"{web_path}/ta_ttb-ud-dev.conllu", Path("datasets") / dataset_name)
        # # cached_path(
        # #     f"{web_path}/ta_ttb-ud-test.conllu", Path("datasets") / dataset_name
        # # )
        # # cached_path(
        # #     f"{web_path}/ta_ttb-ud-train.conllu", Path("datasets") / dataset_name
        # )

        super(UD, self).__init__(data_folder, in_memory=in_memory, add_root=add_root)

# class SRL(UniversalDependenciesCorpus):
#     def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True, lang='en', add_root: bool = True):

#         if type(base_path) == str:
#             base_path: Path = Path(base_path)

#         # this dataset name
#         dataset_name = self.__class__.__name__.lower()

#         # default dataset folder is the cache root
#         if not base_path:
#             base_path = Path(flair.cache_root) / "datasets"
#         data_folder = base_path / dataset_name / Path(lang)

#         # if not os.path.exists(data_folder):
#         #     os.call("git clone https://github.com/UniversalDependencies/"+treebank+" "+data_folder+"/"+treebank)
        

#         # # # download data if necessary
#         # # web_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Tamil-TTB/master"
#         # # cached_path(f"{web_path}/ta_ttb-ud-dev.conllu", Path("datasets") / dataset_name)
#         # # cached_path(
#         # #     f"{web_path}/ta_ttb-ud-test.conllu", Path("datasets") / dataset_name
#         # # )
#         # # cached_path(
#         # #     f"{web_path}/ta_ttb-ud-train.conllu", Path("datasets") / dataset_name
#         # )

#         super(SRL, self).__init__(data_folder, in_memory=in_memory, add_root=add_root)

class SRL(Corpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True, train_file=None, test_file=None, dev_file=None, lang='en'):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / 'srl' / Path(lang)

        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if "train" in file_name:
                    train_file = file
                if "test" in file_name:
                    test_file = file
                if "dev" in file_name:
                    dev_file = file
                if "testa" in file_name:
                    dev_file = file
                if "testb" in file_name:
                    test_file = file
            if test_file is None:
                test_file = dev_file

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(train_file))
        log.info("Test: {}".format(test_file))
        log.info("Dev: {}".format(dev_file))

        # get train data
        train = UniversalDependenciesDataset(train_file, in_memory=in_memory, add_root=True)

        # get test data
        test = UniversalDependenciesDataset(test_file, in_memory=in_memory, add_root=True)

        # get dev data
        dev = UniversalDependenciesDataset(dev_file, in_memory=in_memory, add_root=True)

        super(SRL, self).__init__(
            train, dev, test, name='srl-'+lang
        )


class UD_PROJ(UniversalDependenciesCorpus):
    def __init__(self, treebank: str, base_path: Union[str, Path] = None, in_memory: bool = True, add_root: bool = False):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name / treebank

        # if not os.path.exists(data_folder):
        #     os.call("git clone https://github.com/UniversalDependencies/"+treebank+" "+data_folder+"/"+treebank)
        

        # # # download data if necessary
        # # web_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Tamil-TTB/master"
        # # cached_path(f"{web_path}/ta_ttb-ud-dev.conllu", Path("datasets") / dataset_name)
        # # cached_path(
        # #     f"{web_path}/ta_ttb-ud-test.conllu", Path("datasets") / dataset_name
        # # )
        # # cached_path(
        # #     f"{web_path}/ta_ttb-ud-train.conllu", Path("datasets") / dataset_name
        # )

        super(UD_PROJ, self).__init__(data_folder, in_memory=in_memory, add_root=add_root)

class PTB(Corpus):
    def __init__(self, treebank: str = None, base_path: Union[str, Path] = None, in_memory: bool = True, add_root: bool = True, tag_to_bioes=None):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets" / "ptb_3.3.0_modified"
        data_folder = base_path
        # data_folder = '../Parser-v4/data/SemEval15/ptb_3.3.0'
        # if not os.path.exists(data_folder):
        #     os.call("git clone https://github.com/UniversalDependencies/"+treebank+" "+data_folder+"/"+treebank)
        

        # # # download data if necessary
        # # web_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Tamil-TTB/master"
        # # cached_path(f"{web_path}/ta_ttb-ud-dev.conllu", Path("datasets") / dataset_name)
        # # cached_path(
        # #     f"{web_path}/ta_ttb-ud-test.conllu", Path("datasets") / dataset_name
        # # )
        # # cached_path(
        # #     f"{web_path}/ta_ttb-ud-train.conllu", Path("datasets") / dataset_name
        # )
        log.info("Reading data from {}".format(data_folder))
        # log.info("Train: {}".format(data_folder/'train_modified_projective.conllu'))
        log.info("Train: {}".format(data_folder/'train_modified.conllu'))
        log.info("Test: {}".format(data_folder/'test.conllu'))
        log.info("Dev: {}".format(data_folder/'dev.conllu'))

        # train = UniversalDependenciesDataset(data_folder/'train_modified_projective.conllu', in_memory=in_memory, add_root=True)
        train = UniversalDependenciesDataset(data_folder/'train_modified.conllu', in_memory=in_memory, add_root=True)
        # train = UniversalDependenciesDataset(Path('test2.conll'), in_memory=in_memory, add_root=True)

        # get test data
        test = UniversalDependenciesDataset(data_folder/'test.conllu', in_memory=in_memory, add_root=True)
        # test = UniversalDependenciesDataset(Path('test2.conll'), in_memory=in_memory, add_root=True)
        # get dev data
        dev = UniversalDependenciesDataset(data_folder/'dev.conllu', in_memory=in_memory, add_root=True)
        # dev = UniversalDependenciesDataset(Path('test2.conll'), in_memory=in_memory, add_root=True)

        super(PTB, self).__init__(
            train, dev, test, name=treebank
        )


class WSJ_POS(Corpus):
    def __init__(self, treebank: str = None, base_path: Union[str, Path] = None, in_memory: bool = True, add_root: bool = True, tag_to_bioes=None):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets" / "wsj_pos"
        data_folder = base_path
        # data_folder = '../Parser-v4/data/SemEval15/ptb_3.3.0'
        # if not os.path.exists(data_folder):
        #     os.call("git clone https://github.com/UniversalDependencies/"+treebank+" "+data_folder+"/"+treebank)
        

        # # # download data if necessary
        # # web_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Tamil-TTB/master"
        # # cached_path(f"{web_path}/ta_ttb-ud-dev.conllu", Path("datasets") / dataset_name)
        # # cached_path(
        # #     f"{web_path}/ta_ttb-ud-test.conllu", Path("datasets") / dataset_name
        # # )
        # # cached_path(
        # #     f"{web_path}/ta_ttb-ud-train.conllu", Path("datasets") / dataset_name
        # )
        log.info("Reading data from {}".format(data_folder))
        # log.info("Train: {}".format(data_folder/'train_modified_projective.conllu'))
        log.info("Train: {}".format(data_folder/'train.conllu'))
        log.info("Test: {}".format(data_folder/'test.conllu'))
        log.info("Dev: {}".format(data_folder/'dev.conllu'))

        # train = UniversalDependenciesDataset(data_folder/'train_modified_projective.conllu', in_memory=in_memory, add_root=True)
        train = UniversalDependenciesDataset(data_folder/'train.conllu', in_memory=in_memory, add_root=True)
        # train = UniversalDependenciesDataset(Path('test2.conll'), in_memory=in_memory, add_root=True)

        # get test data
        test = UniversalDependenciesDataset(data_folder/'test.conllu', in_memory=in_memory, add_root=True)
        # test = UniversalDependenciesDataset(Path('test2.conll'), in_memory=in_memory, add_root=True)
        # get dev data
        dev = UniversalDependenciesDataset(data_folder/'dev.conllu', in_memory=in_memory, add_root=True)
        # dev = UniversalDependenciesDataset(Path('test2.conll'), in_memory=in_memory, add_root=True)

        super(WSJ_POS, self).__init__(
            train, dev, test, name=treebank
        )



class CTB(Corpus):
    def __init__(self, treebank: str = None, base_path: Union[str, Path] = None, in_memory: bool = True, add_root: bool = True, tag_to_bioes=None):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets" / "CTB5_YM"
        data_folder = base_path
        # data_folder = '../Parser-v4/data/SemEval15/ptb_3.3.0'
        # if not os.path.exists(data_folder):
        #     os.call("git clone https://github.com/UniversalDependencies/"+treebank+" "+data_folder+"/"+treebank)
        

        # # # download data if necessary
        # # web_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Tamil-TTB/master"
        # # cached_path(f"{web_path}/ta_ttb-ud-dev.conllu", Path("datasets") / dataset_name)
        # # cached_path(
        # #     f"{web_path}/ta_ttb-ud-test.conllu", Path("datasets") / dataset_name
        # # )
        # # cached_path(
        # #     f"{web_path}/ta_ttb-ud-train.conllu", Path("datasets") / dataset_name
        # )

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(data_folder/'CTB5.1-train.gp_modified.conll'))
        log.info("Test: {}".format(data_folder/'CTB5.1-test.gp_modified.conll'))
        log.info("Dev: {}".format(data_folder/'CTB5.1-devel.gp_modified.conll'))

        train = UniversalDependenciesDataset(data_folder/'CTB5.1-train.gp_modified.conll', in_memory=in_memory, add_root=True)

        # get test data
        test = UniversalDependenciesDataset(data_folder/'CTB5.1-test.gp_modified.conll', in_memory=in_memory, add_root=True)

        # get dev data
        dev = UniversalDependenciesDataset(data_folder/'CTB5.1-devel.gp_modified.conll', in_memory=in_memory, add_root=True)

        super(CTB, self).__init__(
            train, dev, test, name=treebank
        )


class ENHANCEDUD(Corpus):
    def __init__(self, treebank: str, base_path: Union[str, Path] = None, in_memory: bool = True, train_file=None, test_file=None, dev_file=None,):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / 'enhanced_ud' / treebank

        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if "train" in file_name:
                    train_file = file
                if "test" in file_name:
                    test_file = file
                if "dev" in file_name:
                    dev_file = file
                if "testa" in file_name:
                    dev_file = file
                if "testb" in file_name:
                    test_file = file
            if test_file is None:
                test_file = dev_file

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(train_file))
        log.info("Test: {}".format(test_file))
        log.info("Dev: {}".format(dev_file))

        # get train data
        train = UniversalDependenciesDataset(train_file, in_memory=in_memory, add_root=True)

        # get test data
        test = UniversalDependenciesDataset(test_file, in_memory=in_memory, add_root=True)

        # get dev data
        dev = UniversalDependenciesDataset(dev_file, in_memory=in_memory, add_root=True)

        super(ENHANCEDUD, self).__init__(
            train, dev, test, name=treebank
        )

class UNREL_ENHANCEDUD(Corpus):
    def __init__(self, treebank: str, base_path: Union[str, Path] = None, in_memory: bool = True, train_file=None, test_file=None, dev_file=None,):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / 'unrel_enhanced_ud' / treebank

        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if "train" in file_name:
                    train_file = file
                if "test" in file_name:
                    test_file = file
                if "dev" in file_name:
                    dev_file = file
                if "testa" in file_name:
                    dev_file = file
                if "testb" in file_name:
                    test_file = file
            if test_file is None:
                test_file = dev_file

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(train_file))
        log.info("Test: {}".format(test_file))
        log.info("Dev: {}".format(dev_file))

        # get train data
        train = UniversalDependenciesDataset(train_file, in_memory=in_memory, add_root=True)

        # get test data
        test = UniversalDependenciesDataset(test_file, in_memory=in_memory, add_root=True)

        # get dev data
        dev = UniversalDependenciesDataset(dev_file, in_memory=in_memory, add_root=True)

        super(UNREL_ENHANCEDUD, self).__init__(
            train, dev, test, name=treebank
        )


class SDP(Corpus):
    def __init__(self, treebank: str, base_path: Union[str, Path] = None, in_memory: bool = True, train_file=None, test_file=None, dev_file=None,):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / 'sdp' / treebank

        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if "train" in file_name:
                    train_file = file
                if "test" in file_name:
                    test_file = file
                if "dev" in file_name:
                    dev_file = file
                if "testa" in file_name:
                    dev_file = file
                if "testb" in file_name:
                    test_file = file
            if test_file is None:
                test_file = dev_file

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(train_file))
        log.info("Test: {}".format(test_file))
        log.info("Dev: {}".format(dev_file))

        # get train data
        train = UniversalDependenciesDataset(train_file, in_memory=in_memory, add_root=True)

        # get test data
        test = UniversalDependenciesDataset(test_file, in_memory=in_memory, add_root=True)

        # get dev data
        dev = UniversalDependenciesDataset(dev_file, in_memory=in_memory, add_root=True)

        super(SDP, self).__init__(
            train, dev, test, name=treebank
        )



class ClassificationCorpus(Corpus):
    def __init__(
        self,
        data_folder: Union[str, Path],
        train_file=None,
        test_file=None,
        dev_file=None,
        use_tokenizer: bool = True,
        max_tokens_per_doc: int = -1,
        max_chars_per_doc: int = -1,
        in_memory: bool = False,
    ):
        """
        Instantiates a Corpus from text classification-formatted task data

        :param data_folder: base folder with the task data
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param use_tokenizer: If True, tokenizes the dataset, otherwise uses whitespace tokenization
        :param max_tokens_per_doc: If set, truncates each Sentence to a maximum number of Tokens
        :param max_chars_per_doc: If set, truncates each Sentence to a maximum number of chars
        :param in_memory: If True, keeps dataset as Sentences in memory, otherwise only keeps strings
        :return: a Corpus with annotated train, dev and test data
        """

        if type(data_folder) == str:
            data_folder: Path = Path(data_folder)

        if train_file is not None:
            train_file = data_folder / train_file
        if test_file is not None:
            test_file = data_folder / test_file
        if dev_file is not None:
            dev_file = data_folder / dev_file

        # automatically identify train / test / dev files
        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if "train" in file_name:
                    train_file = file
                if "test" in file_name:
                    test_file = file
                if "dev" in file_name:
                    dev_file = file
                if "testa" in file_name:
                    dev_file = file
                if "testb" in file_name:
                    test_file = file

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(train_file))
        log.info("Dev: {}".format(dev_file))
        log.info("Test: {}".format(test_file))

        train: Dataset = ClassificationDataset(
            train_file,
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
        )
        test: Dataset = ClassificationDataset(
            test_file,
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
        )

        if dev_file is not None:
            dev: Dataset = ClassificationDataset(
                dev_file,
                use_tokenizer=use_tokenizer,
                max_tokens_per_doc=max_tokens_per_doc,
                max_chars_per_doc=max_chars_per_doc,
                in_memory=in_memory,
            )
        else:
            train_length = len(train)
            dev_size: int = round(train_length / 10)
            splits = random_split(train, [train_length - dev_size, dev_size])
            train = splits[0]
            dev = splits[1]

        super(ClassificationCorpus, self).__init__(
            train, dev, test, name=data_folder.name
        )


class CSVClassificationCorpus(Corpus):
    def __init__(
        self,
        data_folder: Union[str, Path],
        column_name_map: Dict[int, str],
        train_file=None,
        test_file=None,
        dev_file=None,
        use_tokenizer: bool = True,
        max_tokens_per_doc=-1,
        max_chars_per_doc=-1,
        in_memory: bool = False,
        skip_header: bool = False,
        **fmtparams,
    ):
        """
        Instantiates a Corpus for text classification from CSV column formatted data

        :param data_folder: base folder with the task data
        :param column_name_map: a column name map that indicates which column is text and which the label(s)
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param max_tokens_per_doc: If set, truncates each Sentence to a maximum number of Tokens
        :param max_chars_per_doc: If set, truncates each Sentence to a maximum number of chars
        :param use_tokenizer: If True, tokenizes the dataset, otherwise uses whitespace tokenization
        :param in_memory: If True, keeps dataset as Sentences in memory, otherwise only keeps strings
        :param fmtparams: additional parameters for the CSV file reader
        :return: a Corpus with annotated train, dev and test data
        """

        if type(data_folder) == str:
            data_folder: Path = Path(data_folder)

        if train_file is not None:
            train_file = data_folder / train_file
        if test_file is not None:
            test_file = data_folder / test_file
        if dev_file is not None:
            dev_file = data_folder / dev_file

        # automatically identify train / test / dev files
        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if "train" in file_name:
                    train_file = file
                if "test" in file_name:
                    test_file = file
                if "dev" in file_name:
                    dev_file = file
                if "testa" in file_name:
                    dev_file = file
                if "testb" in file_name:
                    test_file = file

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(train_file))
        log.info("Dev: {}".format(dev_file))
        log.info("Test: {}".format(test_file))

        train: Dataset = CSVClassificationDataset(
            train_file,
            column_name_map,
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            skip_header=skip_header,
            **fmtparams,
        )

        if test_file is not None:
            test: Dataset = CSVClassificationDataset(
                test_file,
                column_name_map,
                use_tokenizer=use_tokenizer,
                max_tokens_per_doc=max_tokens_per_doc,
                max_chars_per_doc=max_chars_per_doc,
                in_memory=in_memory,
                skip_header=skip_header,
                **fmtparams,
            )
        else:
            train_length = len(train)
            test_size: int = round(train_length / 10)
            splits = random_split(train, [train_length - test_size, test_size])
            train = splits[0]
            test = splits[1]

        if dev_file is not None:
            dev: Dataset = CSVClassificationDataset(
                dev_file,
                column_name_map,
                use_tokenizer=use_tokenizer,
                max_tokens_per_doc=max_tokens_per_doc,
                max_chars_per_doc=max_chars_per_doc,
                in_memory=in_memory,
                skip_header=skip_header,
                **fmtparams,
            )
        else:
            train_length = len(train)
            dev_size: int = round(train_length / 10)
            splits = random_split(train, [train_length - dev_size, dev_size])
            train = splits[0]
            dev = splits[1]

        super(CSVClassificationCorpus, self).__init__(
            train, dev, test, name=data_folder.name
        )


class SentenceDataset(FlairDataset):
    """
    A simple Dataset object to wrap a List of Sentence
    """

    def __init__(self, sentences: Union[Sentence, List[Sentence]]):
        """
        Instantiate SentenceDataset
        :param sentences: Sentence or List of Sentence that make up SentenceDataset
        """
        # cast to list if necessary
        if type(sentences) == Sentence:
            sentences = [sentences]
        self.sentences = sentences

    @abstractmethod
    def is_in_memory(self) -> bool:
        return True

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index: int = 0) -> Sentence:
        return self.sentences[index]


class ColumnDataset(FlairDataset):
    def __init__(
        self,
        path_to_column_file: Path,
        column_name_map: Dict[int, str],
        tag_to_bioes: str = None,
        comment_symbol: str = None,
        in_memory: bool = True,
    ):
        """
        Instantiates a column dataset (typically used for sequence labeling or word-level prediction).

        :param path_to_column_file: path to the file with the column-formatted data
        :param column_name_map: a map specifying the column format
        :param tag_to_bioes: whether to convert to BIOES tagging scheme
        :param comment_symbol: if set, lines that begin with this symbol are treated as comments
        :param in_memory: If set to True, the dataset is kept in memory as Sentence objects, otherwise does disk reads
        """
        assert path_to_column_file.exists()
        self.path_to_column_file = path_to_column_file
        self.tag_to_bioes = tag_to_bioes
        self.column_name_map = column_name_map
        self.comment_symbol = comment_symbol

        # store either Sentence objects in memory, or only file offsets
        self.in_memory = in_memory
        if self.in_memory:
            self.sentences: List[Sentence] = []
        else:
            self.indices: List[int] = []

        self.total_sentence_count: int = 0

        # most data sets have the token text in the first column, if not, pass 'text' as column
        self.text_column: int = 0
        for column in self.column_name_map:
            if column_name_map[column] == "text":
                self.text_column = column

        # determine encoding of text file
        encoding = "utf-8"
        try:
            lines: List[str] = open(str(path_to_column_file), encoding="utf-8").read(
                10
            ).strip().split("\n")
        except:
            log.info(
                'UTF-8 can\'t read: {} ... using "latin-1" instead.'.format(
                    path_to_column_file
                )
            )
            encoding = "latin1"

        sentence: Sentence = Sentence()
        with open(str(self.path_to_column_file), encoding=encoding) as f:

            line = f.readline()
            position = 0

            while line:

                if self.comment_symbol is not None and line.startswith(comment_symbol):
                    line = f.readline()
                    continue

                if line.isspace():
                    if len(sentence) > 0:
                        sentence.infer_space_after()
                        if self.in_memory:
                            if self.tag_to_bioes is not None:
                                sentence.convert_tag_scheme(
                                    tag_type=self.tag_to_bioes, target_scheme="iobes"
                                )
                            self.sentences.append(sentence)
                        else:
                            self.indices.append(position)
                            position = f.tell()
                        self.total_sentence_count += 1
                    sentence: Sentence = Sentence()

                else:
                    fields: List[str] = re.split("\s+", line)
                    token = Token(fields[self.text_column])
                    for column in column_name_map:
                        if len(fields) > column:
                            if column != self.text_column:
                                token.add_tag(
                                    self.column_name_map[column], fields[column]
                                )

                    sentence.add_token(token)

                line = f.readline()

        if len(sentence.tokens) > 0:
            sentence.infer_space_after()
            if self.in_memory:
                self.sentences.append(sentence)
            else:
                self.indices.append(position)
            self.total_sentence_count += 1
    @property
    def reset_sentence_count(self):
        self.total_sentence_count = len(self.sentences)
        return
    
    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:
        if self.in_memory:
            sentence = self.sentences[index]

        else:
            with open(str(self.path_to_column_file), encoding="utf-8") as file:
                file.seek(self.indices[index])
                line = file.readline()
                sentence: Sentence = Sentence()
                while line:
                    if self.comment_symbol is not None and line.startswith("#"):
                        line = file.readline()
                        continue

                    if line.strip().replace("﻿", "") == "":
                        if len(sentence) > 0:
                            sentence.infer_space_after()

                            if self.tag_to_bioes is not None:
                                sentence.convert_tag_scheme(
                                    tag_type=self.tag_to_bioes, target_scheme="iobes"
                                )
                            break
                    else:
                        fields: List[str] = re.split("\s+", line)
                        token = Token(fields[self.text_column])
                        for column in self.column_name_map:
                            if len(fields) > column:
                                if column != self.text_column:
                                    token.add_tag(
                                        self.column_name_map[column], fields[column]
                                    )

                        sentence.add_token(token)
                    line = file.readline()

        return sentence

class UniversalDependenciesDataset(FlairDataset):
    def __init__(self, path_to_conll_file: Path, in_memory: bool = True, add_root=False, root_tag='<ROOT>',spliter='\t'):
        """
        Instantiates a column dataset in CoNLL-U format.

        :param path_to_conll_file: Path to the CoNLL-U formatted file
        :param in_memory: If set to True, keeps full dataset in memory, otherwise does disk reads
        """
        assert path_to_conll_file.exists()

        self.in_memory = in_memory
        self.path_to_conll_file = path_to_conll_file
        self.total_sentence_count: int = 0

        if self.in_memory:
            self.sentences: List[Sentence] = []
        else:
            self.indices: List[int] = []

        with open(str(self.path_to_conll_file), encoding="utf-8") as file:

            line = file.readline()
            line_count=0
            position = 0
            sentence: Sentence = Sentence()
            if add_root:
                token = Token(root_tag, head_id=int(0))
                token.add_tag("lemma", str('_'))
                token.add_tag("upos", str('_'))
                token.add_tag("pos", str('_'))
                token.add_tag("dependency", str('root'))
                token.add_tag("enhancedud", str('0:root'))
                token.add_tag("srl", str('0:root'))
                token.lemma = token.tags['lemma']._value
                token.upos = token.tags['upos']._value
                token.pos = token.tags['pos']._value
                sentence.add_token(token)
            while line:
                line_count+=1
                line = line.strip()
                fields: List[str] = re.split(spliter+"+", line)
                # if 'unlabel' in str(path_to_conll_file) and line_count>92352:
                #     pdb.set_trace()
                if line == "":
                    if (len(sentence)==1 and sentence[0].text==root_tag):
                        pdb.set_trace()
                    if len(sentence) > 0 and not (len(sentence)==1 and sentence[0].text==root_tag):
                        # pdb.set_trace()
                        self.total_sentence_count += 1
                        if self.in_memory:
                            self.sentences.append(sentence)
                        else:
                            self.indices.append(position)
                            position = file.tell()
                    sentence: Sentence = Sentence()
                    if add_root:
                        token = Token(root_tag, head_id=int(0))
                        token.add_tag("lemma", str('_'))
                        token.add_tag("upos", str('_'))
                        token.add_tag("pos", str('_'))
                        token.add_tag("dependency", str('root'))
                        token.add_tag("enhancedud", str('0:root'))
                        token.add_tag("srl", str('0:root'))
                        token.lemma = token.tags['lemma']._value
                        token.upos = token.tags['upos']._value
                        token.pos = token.tags['pos']._value
                        sentence.add_token(token)
                elif line.startswith("#"):
                    line = file.readline()
                    continue
                elif "." in fields[0] and (len(fields) == 10 or len(fields) == 3):
                    line = file.readline()
                    continue
                elif "-" in fields[0] and (len(fields) == 10 or len(fields) == 3):
                    line = file.readline()
                    continue
                elif len(fields)==2:
                    # reading the raw text
                    token = Token(fields[0])
                    sentence.add_token(token)
                elif len(fields)==3:
                    token = Token(fields[1])
                    # pdb.set_trace()
                    sentence.add_token(token)
                else:
                    token = Token(fields[1], head_id=int(fields[6]))
                    token.add_tag("lemma", str(fields[2]))
                    token.add_tag("upos", str(fields[3]))
                    token.add_tag("pos", str(fields[4]))
                    token.add_tag("dependency", str(fields[7]))
                    token.add_tag("enhancedud", str(fields[8]))
                    token.add_tag("srl", str(fields[8]))

                    for morph in str(fields[5]).split("|"):
                        if not "=" in morph:
                            continue
                        token.add_tag(morph.split("=")[0].lower(), morph.split("=")[1])

                    if len(fields) > 10 and str(fields[10]) == "Y":
                        token.add_tag("frame", str(fields[11]))
                    token.lemma = token.tags['lemma']._value
                    token.upos = token.tags['upos']._value
                    token.pos = token.tags['pos']._value
                    sentence.add_token(token)

                line = file.readline()
            if len(sentence.tokens) > 0 and not (len(sentence)==1 and sentence[0].text==root_tag):
                self.total_sentence_count += 1
                if self.in_memory:
                    self.sentences.append(sentence)
                else:
                    self.indices.append(position)

    def is_in_memory(self) -> bool:
        return self.in_memory
    @property
    def reset_sentence_count(self):
        self.total_sentence_count = len(self.sentences)
        return
    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:

        if self.in_memory:
            sentence = self.sentences[index]
        else:
            with open(str(self.path_to_conll_file), encoding="utf-8") as file:
                file.seek(self.indices[index])
                line = file.readline()
                sentence: Sentence = Sentence()
                while line:

                    line = line.strip()
                    fields: List[str] = re.split("\t+", line)
                    if line == "":
                        if len(sentence) > 0:
                            break

                    elif line.startswith("#"):
                        line = file.readline()
                        continue
                    elif "." in fields[0]:
                        line = file.readline()
                        continue
                    elif "-" in fields[0]:
                        line = file.readline()
                        continue
                    else:
                        token = Token(fields[1], head_id=int(fields[6]))
                        token.add_tag("lemma", str(fields[2]))
                        token.add_tag("upos", str(fields[3]))
                        token.add_tag("pos", str(fields[4]))
                        token.add_tag("dependency", str(fields[7]))

                        for morph in str(fields[5]).split("|"):
                            if not "=" in morph:
                                continue
                            token.add_tag(
                                morph.split("=")[0].lower(), morph.split("=")[1]
                            )

                        if len(fields) > 10 and str(fields[10]) == "Y":
                            token.add_tag("frame", str(fields[11]))
                        token.lemma = token.tags['lemma']._value
                        token.upos = token.tags['upos']._value
                        token.pos = token.tags['pos']._value
                        sentence.add_token(token)

                    line = file.readline()
        return sentence


class CSVClassificationDataset(FlairDataset):
    def __init__(
        self,
        path_to_file: Union[str, Path],
        column_name_map: Dict[int, str],
        max_tokens_per_doc: int = -1,
        max_chars_per_doc: int = -1,
        use_tokenizer=True,
        in_memory: bool = True,
        skip_header: bool = False,
        **fmtparams,
    ):
        """
        Instantiates a Dataset for text classification from CSV column formatted data

        :param path_to_file: path to the file with the CSV data
        :param column_name_map: a column name map that indicates which column is text and which the label(s)
        :param max_tokens_per_doc: If set, truncates each Sentence to a maximum number of Tokens
        :param max_chars_per_doc: If set, truncates each Sentence to a maximum number of chars
        :param use_tokenizer: If True, tokenizes the dataset, otherwise uses whitespace tokenization
        :param in_memory: If True, keeps dataset as Sentences in memory, otherwise only keeps strings
        :param skip_header: If True, skips first line because it is header
        :param fmtparams: additional parameters for the CSV file reader
        :return: a Corpus with annotated train, dev and test data
        """

        if type(path_to_file) == str:
            path_to_file: Path = Path(path_to_file)

        assert path_to_file.exists()

        # variables
        self.path_to_file = path_to_file
        self.in_memory = in_memory
        self.use_tokenizer = use_tokenizer
        self.column_name_map = column_name_map
        self.max_tokens_per_doc = max_tokens_per_doc
        self.max_chars_per_doc = max_chars_per_doc

        # different handling of in_memory data than streaming data
        if self.in_memory:
            self.sentences = []
        else:
            self.raw_data = []

        self.total_sentence_count: int = 0

        # most data sets have the token text in the first column, if not, pass 'text' as column
        self.text_columns: List[int] = []
        for column in column_name_map:
            if column_name_map[column] == "text":
                self.text_columns.append(column)

        with open(self.path_to_file) as csv_file:

            csv_reader = csv.reader(csv_file, **fmtparams)

            if skip_header:
                next(csv_reader, None)  # skip the headers

            for row in csv_reader:

                # test if format is OK
                wrong_format = False
                for text_column in self.text_columns:
                    if text_column >= len(row):
                        wrong_format = True

                if wrong_format:
                    continue

                # test if at least one label given
                has_label = False
                for column in self.column_name_map:
                    if self.column_name_map[column].startswith("label") and row[column]:
                        has_label = True
                        break

                if not has_label:
                    continue

                if self.in_memory:

                    text = " ".join(
                        [row[text_column] for text_column in self.text_columns]
                    )

                    if self.max_chars_per_doc > 0:
                        text = text[: self.max_chars_per_doc]

                    sentence = Sentence(text, use_tokenizer=self.use_tokenizer)

                    for column in self.column_name_map:
                        if (
                            self.column_name_map[column].startswith("label")
                            and row[column]
                        ):
                            sentence.add_label(row[column])

                    if (
                        len(sentence) > self.max_tokens_per_doc
                        and self.max_tokens_per_doc > 0
                    ):
                        sentence.tokens = sentence.tokens[: self.max_tokens_per_doc]
                    self.sentences.append(sentence)

                else:
                    self.raw_data.append(row)

                self.total_sentence_count += 1

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:
        if self.in_memory:
            return self.sentences[index]
        else:
            row = self.raw_data[index]

            text = " ".join([row[text_column] for text_column in self.text_columns])

            if self.max_chars_per_doc > 0:
                text = text[: self.max_chars_per_doc]

            sentence = Sentence(text, use_tokenizer=self.use_tokenizer)
            for column in self.column_name_map:
                if self.column_name_map[column].startswith("label") and row[column]:
                    sentence.add_label(row[column])

            if len(sentence) > self.max_tokens_per_doc and self.max_tokens_per_doc > 0:
                sentence.tokens = sentence.tokens[: self.max_tokens_per_doc]

            return sentence


class ClassificationDataset(FlairDataset):
    def __init__(
        self,
        path_to_file: Union[str, Path],
        max_tokens_per_doc=-1,
        max_chars_per_doc=-1,
        use_tokenizer=True,
        in_memory: bool = True,
    ):
        """
        Reads a data file for text classification. The file should contain one document/text per line.
        The line should have the following format:
        __label__<class_name> <text>
        If you have a multi class task, you can have as many labels as you want at the beginning of the line, e.g.,
        __label__<class_name_1> __label__<class_name_2> <text>
        :param path_to_file: the path to the data file
        :param max_tokens_per_doc: Takes at most this amount of tokens per document. If set to -1 all documents are taken as is.
        :param max_tokens_per_doc: If set, truncates each Sentence to a maximum number of Tokens
        :param max_chars_per_doc: If set, truncates each Sentence to a maximum number of chars
        :param use_tokenizer: If True, tokenizes the dataset, otherwise uses whitespace tokenization
        :param in_memory: If True, keeps dataset as Sentences in memory, otherwise only keeps strings
        :return: list of sentences
        """
        if type(path_to_file) == str:
            path_to_file: Path = Path(path_to_file)

        assert path_to_file.exists()

        self.label_prefix = "__label__"

        self.in_memory = in_memory
        self.use_tokenizer = use_tokenizer

        if self.in_memory:
            self.sentences = []
        else:
            self.indices = []

        self.total_sentence_count: int = 0
        self.max_chars_per_doc = max_chars_per_doc
        self.max_tokens_per_doc = max_tokens_per_doc

        self.path_to_file = path_to_file

        with open(str(path_to_file), encoding="utf-8") as f:
            line = f.readline()
            position = 0
            while line:
                if "__label__" not in line or " " not in line:
                    position = f.tell()
                    line = f.readline()
                    continue

                if self.in_memory:
                    sentence = self._parse_line_to_sentence(
                        line, self.label_prefix, use_tokenizer
                    )
                    if sentence is not None and len(sentence.tokens) > 0:
                        self.sentences.append(sentence)
                        self.total_sentence_count += 1
                else:
                    self.indices.append(position)
                    self.total_sentence_count += 1

                position = f.tell()
                line = f.readline()

    def _parse_line_to_sentence(
        self, line: str, label_prefix: str, use_tokenizer: bool = True
    ):
        words = line.split()

        labels = []
        l_len = 0

        for i in range(len(words)):
            if words[i].startswith(label_prefix):
                l_len += len(words[i]) + 1
                label = words[i].replace(label_prefix, "")
                labels.append(label)
            else:
                break

        text = line[l_len:].strip()

        if self.max_chars_per_doc > 0:
            text = text[: self.max_chars_per_doc]

        if text and labels:
            sentence = Sentence(text, labels=labels, use_tokenizer=use_tokenizer)

            if (
                sentence is not None
                and len(sentence) > self.max_tokens_per_doc
                and self.max_tokens_per_doc > 0
            ):
                sentence.tokens = sentence.tokens[: self.max_tokens_per_doc]

            return sentence
        return None

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:
        if self.in_memory:
            return self.sentences[index]
        else:

            with open(str(self.path_to_file), encoding="utf-8") as file:
                file.seek(self.indices[index])
                line = file.readline()
                sentence = self._parse_line_to_sentence(
                    line, self.label_prefix, self.use_tokenizer
                )
                return sentence



class TWITTER(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = None,
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "upos"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name
        
        super(TWITTER, self).__init__(
            data_folder, columns, tag_to_bioes=None, in_memory=in_memory
        )

class TWITTER_NEW(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = None,
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "upos"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name
        
        super(TWITTER_NEW, self).__init__(
            data_folder, columns, tag_to_bioes=None, in_memory=in_memory
        )


class ARK(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = None,
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "upos"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name
        
        super(ARK, self).__init__(
            data_folder, columns, tag_to_bioes=None, in_memory=in_memory
        )



class RITTER(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = None,
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "upos"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name
        
        super(RITTER, self).__init__(
            data_folder, columns, tag_to_bioes=None, in_memory=in_memory
        )


class RITTER_NEW(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = None,
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "upos"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name
        
        super(RITTER_NEW, self).__init__(
            data_folder, columns, tag_to_bioes=None, in_memory=in_memory
        )

class TWEEBANK(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        super(TWEEBANK, self).__init__(data_folder, in_memory=in_memory)

class TWEEBANK_NEW(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        super(TWEEBANK_NEW, self).__init__(data_folder, in_memory=in_memory)

class CONLL_03(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "chunk", 3: "ner"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'ACHTUNG: CoNLL-03 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://www.clips.uantwerpen.be/conll2003/ner/"'
            )
            log.warning("-" * 100)

        super(CONLL_03, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class CONLL_03_NEW(CONLL_03):
    def __init__(self,**kwargs):
        super(CONLL_03_NEW, self).__init__(
            **kwargs
        )

class CONLL_03_ENGLISH(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "chunk", 3: "ner"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'ACHTUNG: CoNLL-03 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://www.clips.uantwerpen.be/conll2003/ner/"'
            )
            log.warning("-" * 100)

        super(CONLL_03_ENGLISH, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class CONLL_03_ENGLISH_DOC(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "chunk", 3: "ner"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'ACHTUNG: CoNLL-03 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://www.clips.uantwerpen.be/conll2003/ner/"'
            )
            log.warning("-" * 100)

        super(CONLL_03_ENGLISH_DOC, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class CONLL_03_ENGLISH_CASED(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "chunk", 3: "ner"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'ACHTUNG: CoNLL-03 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://www.clips.uantwerpen.be/conll2003/ner/"'
            )
            log.warning("-" * 100)

        super(CONLL_03_ENGLISH_CASED, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )

class CONLL_03_ENGLISH_DOC_CASED(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "chunk", 3: "ner"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'ACHTUNG: CoNLL-03 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://www.clips.uantwerpen.be/conll2003/ner/"'
            )
            log.warning("-" * 100)

        super(CONLL_03_ENGLISH_DOC_CASED, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class CONLL_03_VIETNAMESE(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "chunk",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "chunk"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'ACHTUNG: CoNLL-03 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://www.clips.uantwerpen.be/conll2003/ner/"'
            )
            log.warning("-" * 100)

        super(CONLL_03_VIETNAMESE, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class CHUNK_CONLL_03_VIETNAMESE(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "chunk",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "chunk"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'ACHTUNG: CoNLL-03 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://www.clips.uantwerpen.be/conll2003/ner/"'
            )
            log.warning("-" * 100)

        super(CHUNK_CONLL_03_VIETNAMESE, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class CONLL_03_GERMAN(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus for German. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'lemma', 'pos' or 'np' to predict
        word lemmas, POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "lemma", 2: "pos", 3: "chunk", 4: "ner"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'ACHTUNG: CoNLL-03 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://www.clips.uantwerpen.be/conll2003/ner/"'
            )
            log.warning("-" * 100)

        super(CONLL_03_GERMAN, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class CONLL_03_GERMAN_NEW(CONLL_03_GERMAN):
    def __init__(self,**kwargs):
        super(CONLL_03_GERMAN_NEW, self).__init__(
            **kwargs
        )



class CONLL_06_GERMAN(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus for German. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'lemma', 'pos' or 'np' to predict
        word lemmas, POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'ACHTUNG: CoNLL-03 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://www.clips.uantwerpen.be/conll2003/ner/"'
            )
            log.warning("-" * 100)

        super(CONLL_06_GERMAN, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )

class CONLL_03_DUTCH(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus for Dutch. The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' to predict
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        conll_02_path = "https://www.clips.uantwerpen.be/conll2002/ner/data/"
        cached_path(f"{conll_02_path}ned.testa", Path("datasets") / dataset_name)
        cached_path(f"{conll_02_path}ned.testb", Path("datasets") / dataset_name)
        cached_path(f"{conll_02_path}ned.train", Path("datasets") / dataset_name)

        super(CONLL_03_DUTCH, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )

class CONLL_03_DUTCH_NEW(CONLL_03_DUTCH):
    def __init__(self,**kwargs):
        super(CONLL_03_DUTCH_NEW, self).__init__(
            **kwargs
        )



class CONLL_03_SPANISH(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus for Spanish. The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, should not be changed
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        conll_02_path = "https://www.clips.uantwerpen.be/conll2002/ner/data/"
        cached_path(f"{conll_02_path}esp.testa", Path("datasets") / dataset_name)
        cached_path(f"{conll_02_path}esp.testb", Path("datasets") / dataset_name)
        cached_path(f"{conll_02_path}esp.train", Path("datasets") / dataset_name)

        super(CONLL_03_SPANISH, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )

class CONLL_03_SPANISH_NEW(CONLL_03_SPANISH):
    def __init__(self,**kwargs):
        super(CONLL_03_SPANISH_NEW, self).__init__(
            **kwargs
        )



#------------------------------------------------------------
#for NER as dp
class CONLL_03_DP(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner_dp",
        in_memory: bool = True,
        ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        #------------------
        columns = {0: "text", 1: "pos", 2: "chunk", 3: "ner_dp"}
        self.columns = columns
        # this dataset name
        #---------------
        dataset_name = 'conll_03_english'

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'ACHTUNG: CoNLL-03 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://www.clips.uantwerpen.be/conll2003/ner/"'
            )
            log.warning("-" * 100)

        super(CONLL_03_DP, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
            )



#------------------------KD-------------------------------------------
class CONLL_03_GERMAN_DP(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner_dp",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus for German. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'lemma', 'pos' or 'np' to predict
        word lemmas, POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "lemma", 2: "pos", 3: "chunk", 4: "ner_dp"}
        self.columns = columns
        # this dataset name
        # dataset_name = self.__class__.__name__.lower()
        dataset_name = 'conll_03_german_new'

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'ACHTUNG: CoNLL-03 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://www.clips.uantwerpen.be/conll2003/ner/"'
            )
            log.warning("-" * 100)

        super(CONLL_03_GERMAN_DP, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class CONLL_06_GERMAN_DP(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner_dp",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus for German. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'lemma', 'pos' or 'np' to predict
        word lemmas, POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner_dp"}
        self.columns = columns
        # this dataset name
        dataset_name = 'conll_06_german'

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'ACHTUNG: CoNLL-03 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://www.clips.uantwerpen.be/conll2003/ner/"'
            )
            log.warning("-" * 100)

        super(CONLL_06_GERMAN_DP, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )

class CONLL_03_DUTCH_DP(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner_dp",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus for Dutch. The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' to predict
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner_dp"}
        self.columns = columns
        # this dataset name
        # dataset_name = self.__class__.__name__.lower()
        dataset_name = 'conll_03_dutch_new'

        # default dataset folder is the cache root
        
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        conll_02_path = "https://www.clips.uantwerpen.be/conll2002/ner/data/"
        # pdb.set_trace()
        cached_path(f"{conll_02_path}ned.testa", Path("datasets") / dataset_name)
        cached_path(f"{conll_02_path}ned.testb", Path("datasets") / dataset_name)
        cached_path(f"{conll_02_path}ned.train", Path("datasets") / dataset_name)

        super(CONLL_03_DUTCH_DP, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class CONLL_03_SPANISH_DP(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner_dp",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus for Spanish. The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, should not be changed
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner_dp"}
        self.columns = columns
        # this dataset name
        # dataset_name = self.__class__.__name__.lower()
        dataset_name = 'conll_03_spanish_new'

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        conll_02_path = "https://www.clips.uantwerpen.be/conll2002/ner/data/"
        cached_path(f"{conll_02_path}esp.testa", Path("datasets") / dataset_name)
        cached_path(f"{conll_02_path}esp.testb", Path("datasets") / dataset_name)
        cached_path(f"{conll_02_path}esp.train", Path("datasets") / dataset_name)

        super(CONLL_03_SPANISH_DP, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )




class CONLL_03_IND(CONLL_03):
    pass

class CONLL_03_GERMAN_IND(CONLL_03_GERMAN):
    pass

class CONLL_03_DUTCH_IND(CONLL_03_DUTCH):
    pass

class CONLL_03_SPANISH_IND(CONLL_03_SPANISH):
    pass

class CONLL_03_TOY(CONLL_03):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        super(CONLL_03_TOY, self).__init__(
            base_path, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )

class CONLL_03_GERMAN_TOY(CONLL_03_GERMAN):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        super(CONLL_03_GERMAN_TOY, self).__init__(
            base_path, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )

class CONLL_03_FAKE(CONLL_03):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        super(CONLL_03_FAKE, self).__init__(
            base_path, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )

class CONLL_03_GERMAN_FAKE(CONLL_03_GERMAN):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        super(CONLL_03_GERMAN_FAKE, self).__init__(
            base_path, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )

class CONLL_03_SPANISH_FAKE(CONLL_03_SPANISH):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        super(CONLL_03_SPANISH_FAKE, self).__init__(
            base_path, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )

class CONLL_03_DUTCH_FAKE(CONLL_03_DUTCH):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        super(CONLL_03_DUTCH_FAKE, self).__init__(
            base_path, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )

class CHUNK_CONLL_03_ENGLISH(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "chunk",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-2000 corpus for English chunking.
        The first time you call this constructor it will automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: 'np' by default, should not be changed, but you can set 'pos' instead to predict POS tags
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 2: "chunk"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        super(CHUNK_CONLL_03_ENGLISH, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )

class PANX(ColumnCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True, lang='en',tag_to_bioes='ner'):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()
        columns = {0: "text", 1: "ner"}
        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / Path('panxdataset')/ Path(lang)

        super(PANX, self).__init__(data_folder, columns, in_memory=in_memory,
                                                train_file=data_folder/('wikiann-'+lang+'_train.bio'),
                                                test_file=data_folder/('wikiann-'+lang+'_test.bio'),
                                                dev_file=data_folder/('wikiann-'+lang+'_dev.bio'),
                                                tag_to_bioes=tag_to_bioes,
                                                )


class PANX_DP(ColumnCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True, lang='en',tag_to_bioes='ner_dp'):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        # dataset_name = self.__class__.__name__.lower()
        dataset_name = 'panx'
        columns = {0: "text", 1: "ner_dp"}
        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / Path('panxdataset')/ Path(lang)

        super(PANX_DP, self).__init__(data_folder, columns, in_memory=in_memory,
                                                train_file=data_folder/('wikiann-'+lang+'_train.bio'),
                                                test_file=data_folder/('wikiann-'+lang+'_test.bio'),
                                                dev_file=data_folder/('wikiann-'+lang+'_dev.bio'),
                                                tag_to_bioes=tag_to_bioes,
                                                )

class ATIS(ColumnCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True, lang='en',tag_to_bioes='atis'):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()
        columns = {0: "text", 1: "atis"}
        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / Path('atis')/ Path(lang)

        super(ATIS, self).__init__(data_folder, columns, in_memory=in_memory,
                                                tag_to_bioes=tag_to_bioes,
                                                )

class COMMNER(ColumnCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True, lang='en',tag_to_bioes='ner'):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()
        columns = {0: "text", 1: "ner"}
        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / Path('commner')/ Path(lang)

        super(COMMNER, self).__init__(data_folder, columns, in_memory=in_memory,
                                                train_file=data_folder/(lang+'.train.conll.bio'),
                                                test_file=data_folder/(lang+'.test.conll.bio'),
                                                dev_file=data_folder/(lang+'.dev.conll.bio'),
                                                tag_to_bioes=tag_to_bioes,
                                                )


class FRQUERY(ColumnCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True, lang='fr.annotated.all.clean.conll',tag_to_bioes='ner'):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()
        columns = {0: "text", 1: "ner"}
        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / Path('frquery')

        super(FRQUERY, self).__init__(data_folder, columns, in_memory=in_memory,
                                                train_file=data_folder/(lang+'.train'),
                                                test_file=data_folder/(lang+'.test'),
                                                dev_file=data_folder/(lang+'.dev'),
                                                tag_to_bioes=tag_to_bioes,
                                                )

class ICBU(ColumnCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True, lang='all.csv',tag_to_bioes='ner'):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()
        columns = {0: "text", 1: "ner"}
        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / Path('icbu')

        super(ICBU, self).__init__(data_folder, columns, in_memory=in_memory,
                                                # train_file=data_folder/(lang+'.train'),
                                                # test_file=data_folder/(lang+'.test'),
                                                # dev_file=data_folder/(lang+'.dev'),
                                                train_file=data_folder/('train.txt'),
                                                test_file=data_folder/('test.txt'),
                                                dev_file=data_folder/('dev.txt'),
                                                tag_to_bioes=tag_to_bioes,
                                                )


class ONTONOTE_ENG(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        super(ONTONOTE_ENG, self).__init__(
            data_folder, columns, tag_to_bioes=None, in_memory=in_memory
        )


class UNLABEL(ColumnCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True, modelname='', lang='en',tag_to_bioes='ner', columns = {0: "text", 1: "ner"}, extra=None):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()
        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        columns = {0: "text", 1: "gold_ner", 2:"ner", 3:"score"}
        data_folder = base_path / Path('unlabeled_data')
        super(UNLABEL, self).__init__(data_folder, columns, in_memory=in_memory,
                                                train_file=data_folder/('train.'+modelname+'.'+lang+'.conllu') if extra is None else data_folder/('train.'+modelname+'.'+lang+'.'+extra+'.conllu'),
                                                test_file=data_folder/('empty_testb.conllu'),
                                                dev_file=data_folder/('empty_testa.conllu'),
                                                tag_to_bioes=None,
                                                )

class UNLABEL_DEPENDENCY(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True, modelname='', lang='en', extra=None):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()
        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / Path('unlabeled_data')

        super(UNLABEL_DEPENDENCY, self).__init__(data_folder, in_memory=in_memory,
                                                train_file=data_folder/('train.'+modelname+'.'+lang+'.conllu') if extra is None else data_folder/('train.'+modelname+'.'+lang+'.'+extra+'.conllu'),
                                                test_file=data_folder/('empty_testb.conllu'),
                                                dev_file=data_folder/('empty_testa.conllu'),
                                                add_root=True,
                                                )


class MIXED_NER(ColumnCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True, lang='en',tag_to_bioes='ner'):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()
        columns = {0: "text", 1: "ner"}
        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / Path('mixed_ner')/ Path(lang)

        super(MIXED_NER, self).__init__(data_folder, columns, in_memory=in_memory,
                                                train_file=data_folder/('wikiann-'+lang+'_train.bio'),
                                                test_file=data_folder/('wikiann-'+lang+'_test.bio'),
                                                dev_file=data_folder/('wikiann-'+lang+'_dev.bio'),
                                                tag_to_bioes=tag_to_bioes,
                                                )


class LOW10_NER(ColumnCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True, lang='en',tag_to_bioes='ner'):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()
        columns = {0: "text", 1: "ner"}
        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / Path('low10_ner')/ Path(lang)

        super(LOW10_NER, self).__init__(data_folder, columns, in_memory=in_memory,
                                                train_file=data_folder/('wikiann-'+lang+'_train.bio'),
                                                test_file=data_folder/('wikiann-'+lang+'_test.bio'),
                                                dev_file=data_folder/('wikiann-'+lang+'_dev.bio'),
                                                tag_to_bioes=tag_to_bioes,
                                                )



class PANXPRED(ColumnCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True, lang='en',tag_to_bioes='ner'):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()
        columns = {0: "text", 1: "ner"}
        # columns = {0: "text", 1: "gold_ner", 2:"ner", 3:"score"}
        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / Path('unlabeledpanx')/ Path(lang)

        super(PANXPRED, self).__init__(data_folder, columns, in_memory=in_memory,
                                                train_file=data_folder/('wikiann-'+lang+'_train.bio'),
                                                test_file=base_path/Path('panxdataset')/ Path(lang) /('wikiann-'+lang+'_test.bio'),
                                                dev_file=base_path/Path('panxdataset')/ Path(lang) /('wikiann-'+lang+'_dev.bio'),
                                                tag_to_bioes=None,
                                                )

class PANXPRED2(ColumnCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True, lang='en',tag_to_bioes='ner'):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()
        columns = {0: "text", 1: "ner"}
        # columns = {0: "text", 1: "gold_ner", 2:"ner", 3:"score"}
        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / Path('unlabeledpanx2')/ Path(lang)

        super(PANXPRED2, self).__init__(data_folder, columns, in_memory=in_memory,
                                                train_file=data_folder/('wikiann-'+lang+'_train.bio'),
                                                test_file=base_path/Path('panxdataset')/ Path(lang) /('wikiann-'+lang+'_test.bio'),
                                                dev_file=base_path/Path('panxdataset')/ Path(lang) /('wikiann-'+lang+'_dev.bio'),
                                                tag_to_bioes=None,
                                                )


class SEMEVAL16(ColumnCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True, lang='en',tag_to_bioes='ner'):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()
        columns = {0: "text", 1: "ast"}
        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / Path('semeval-2016')
        lc_to_lang = {'tr':'Turkish','es':'Spanish','nl':'Dutch','en':'English','ru':'Russian'}
        language = lc_to_lang[lang]
        train_file = Path('train/' + language+'_semeval2016_restaurants_train.bio')
        dev_file = Path('train/' + language+'_semeval2016_restaurants_dev.bio')
        test_file = Path('test/' + lang.upper()+'_REST_SB1_TEST.xml.gold.bio')
        # pdb.set_trace()
        super(SEMEVAL16, self).__init__(data_folder, columns, in_memory=in_memory,
                                                train_file=data_folder/train_file,
                                                test_file=data_folder/test_file,
                                                dev_file=data_folder/dev_file,
                                                tag_to_bioes=tag_to_bioes,
                                                )


class SEMEVAL14_LAPTOP(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ast"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        super(SEMEVAL14_LAPTOP, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class SEMEVAL14_RESTAURANT(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ast"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        super(SEMEVAL14_RESTAURANT, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class SEMEVAL15_RESTAURANT(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ast"}
        self.columns = columns
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        super(SEMEVAL15_RESTAURANT, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class CALCS(ColumnCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True, lang='en',tag_to_bioes='ner'):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()
        columns = {0: "text", 1: "cs"}
        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        if lang=='en' or lang=='es':
            target_path=Path('CALCS_ENG_SPA')
        elif lang=='ar' or lang=='eg':
            target_path=Path('CALCS_MSA_EGY')
        elif lang=='hi':
            columns = {0: "text",1: "lang", 2: "ner"}
            target_path=Path('CALCS_ENG_HIN')

        
        data_folder = base_path / target_path
        # lc_to_lang = {'tr':'Turkish','es':'Spanish','nl':'Dutch','en':'English','ru':'Russian'}
        # language = lc_to_lang[lang]
        # pdb.set_trace()
        super(CALCS, self).__init__(data_folder, columns, in_memory=in_memory,
                                                train_file=data_folder/'calcs_train.conll',
                                                test_file=data_folder/'calcs_test.conll',
                                                dev_file=data_folder/'calcs_dev.conll',
                                                tag_to_bioes=tag_to_bioes,
                                                )





class CONLL_2000(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "chunk",
        in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-2000 corpus for English chunking.
        The first time you call this constructor it will automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: 'np' by default, should not be changed, but you can set 'pos' instead to predict POS tags
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "chunk"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        conll_2000_path = "https://www.clips.uantwerpen.be/conll2000/chunking/"
        data_file = Path(flair.cache_root) / "datasets" / dataset_name / "train.txt"
        if not data_file.is_file():
            cached_path(
                f"{conll_2000_path}train.txt.gz", Path("datasets") / dataset_name
            )
            cached_path(
                f"{conll_2000_path}test.txt.gz", Path("datasets") / dataset_name
            )
            import gzip, shutil

            with gzip.open(
                Path(flair.cache_root) / "datasets" / dataset_name / "train.txt.gz",
                "rb",
            ) as f_in:
                with open(
                    Path(flair.cache_root) / "datasets" / dataset_name / "train.txt",
                    "wb",
                ) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            with gzip.open(
                Path(flair.cache_root) / "datasets" / dataset_name / "test.txt.gz", "rb"
            ) as f_in:
                with open(
                    Path(flair.cache_root) / "datasets" / dataset_name / "test.txt",
                    "wb",
                ) as f_out:
                    shutil.copyfileobj(f_in, f_out)

        super(CONLL_2000, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class GERMEVAL(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        """
        Initialize the GermEval NER corpus for German. This is only possible if you've manually downloaded it to your
        machine. Obtain the corpus from https://sites.google.com/site/germeval2014ner/home/ and put it into some folder.
        Then point the base_path parameter in the constructor to this folder
        :param base_path: Path to the GermEval corpus on your machine
        :param tag_to_bioes: 'ner' by default, should not be changed.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {1: "text", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'ACHTUNG: GermEval-14 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://sites.google.com/site/germeval2014ner/home/"'
            )
            log.warning("-" * 100)
        super(GERMEVAL, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            comment_symbol="#",
            in_memory=in_memory,
        )


class IMDB(ClassificationCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = False):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        imdb_acl_path = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        data_path = Path(flair.cache_root) / "datasets" / dataset_name
        data_file = data_path / "train.txt"
        if not data_file.is_file():
            cached_path(imdb_acl_path, Path("datasets") / dataset_name)
            import tarfile

            with tarfile.open(
                Path(flair.cache_root)
                / "datasets"
                / dataset_name
                / "aclImdb_v1.tar.gz",
                "r:gz",
            ) as f_in:
                datasets = ["train", "test"]
                labels = ["pos", "neg"]

                for label in labels:
                    for dataset in datasets:
                        f_in.extractall(
                            data_path,
                            members=[
                                m
                                for m in f_in.getmembers()
                                if f"{dataset}/{label}" in m.name
                            ],
                        )
                        with open(f"{data_path}/{dataset}.txt", "at") as f_p:
                            current_path = data_path / "aclImdb" / dataset / label
                            for file_name in current_path.iterdir():
                                if file_name.is_file() and file_name.name.endswith(
                                    ".txt"
                                ):
                                    f_p.write(
                                        f"__label__{label} "
                                        + file_name.open("rt", encoding="utf-8").read()
                                        + "\n"
                                    )

        super(IMDB, self).__init__(
            data_folder, use_tokenizer=False, in_memory=in_memory
        )


class NEWSGROUPS(ClassificationCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = False):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        twenty_newsgroups_path = (
            "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
        )
        data_path = Path(flair.cache_root) / "datasets" / dataset_name
        data_file = data_path / "20news-bydate-train.txt"
        if not data_file.is_file():
            cached_path(
                twenty_newsgroups_path, Path("datasets") / dataset_name / "original"
            )

            import tarfile

            with tarfile.open(
                Path(flair.cache_root)
                / "datasets"
                / dataset_name
                / "original"
                / "20news-bydate.tar.gz",
                "r:gz",
            ) as f_in:
                datasets = ["20news-bydate-test", "20news-bydate-train"]
                labels = [
                    "alt.atheism",
                    "comp.graphics",
                    "comp.os.ms-windows.misc",
                    "comp.sys.ibm.pc.hardware",
                    "comp.sys.mac.hardware",
                    "comp.windows.x",
                    "misc.forsale",
                    "rec.autos",
                    "rec.motorcycles",
                    "rec.sport.baseball",
                    "rec.sport.hockey",
                    "sci.crypt",
                    "sci.electronics",
                    "sci.med",
                    "sci.space",
                    "soc.religion.christian",
                    "talk.politics.guns",
                    "talk.politics.mideast",
                    "talk.politics.misc",
                    "talk.religion.misc",
                ]

                for label in labels:
                    for dataset in datasets:
                        f_in.extractall(
                            data_path / "original",
                            members=[
                                m
                                for m in f_in.getmembers()
                                if f"{dataset}/{label}" in m.name
                            ],
                        )
                        with open(f"{data_path}/{dataset}.txt", "at") as f_p:
                            current_path = data_path / "original" / dataset / label
                            for file_name in current_path.iterdir():
                                if file_name.is_file():
                                    f_p.write(
                                        f"__label__{label} "
                                        + file_name.open("rt", encoding="latin1")
                                        .read()
                                        .replace("\n", " <n> ")
                                        + "\n"
                                    )

        super(NEWSGROUPS, self).__init__(
            data_folder, use_tokenizer=False, in_memory=in_memory
        )


class NER_BASQUE(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ner_basque_path = "http://ixa2.si.ehu.eus/eiec/"
        data_path = Path(flair.cache_root) / "datasets" / dataset_name
        data_file = data_path / "named_ent_eu.train"
        if not data_file.is_file():
            cached_path(
                f"{ner_basque_path}/eiec_v1.0.tgz", Path("datasets") / dataset_name
            )
            import tarfile, shutil

            with tarfile.open(
                Path(flair.cache_root) / "datasets" / dataset_name / "eiec_v1.0.tgz",
                "r:gz",
            ) as f_in:
                corpus_files = (
                    "eiec_v1.0/named_ent_eu.train",
                    "eiec_v1.0/named_ent_eu.test",
                )
                for corpus_file in corpus_files:
                    f_in.extract(corpus_file, data_path)
                    shutil.move(f"{data_path}/{corpus_file}", data_path)

        super(NER_BASQUE, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class TREC_50(ClassificationCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        trec_path = "https://cogcomp.seas.upenn.edu/Data/QA/QC/"

        original_filenames = ["train_5500.label", "TREC_10.label"]
        new_filenames = ["train.txt", "test.txt"]
        for original_filename in original_filenames:
            cached_path(
                f"{trec_path}{original_filename}",
                Path("datasets") / dataset_name / "original",
            )

        data_file = data_folder / new_filenames[0]

        if not data_file.is_file():
            for original_filename, new_filename in zip(
                original_filenames, new_filenames
            ):
                with open(
                    data_folder / "original" / original_filename,
                    "rt",
                    encoding="latin1",
                ) as open_fp:
                    with open(
                        data_folder / new_filename, "wt", encoding="utf-8"
                    ) as write_fp:
                        for line in open_fp:
                            line = line.rstrip()
                            fields = line.split()
                            old_label = fields[0]
                            question = " ".join(fields[1:])

                            # Create flair compatible labels
                            # TREC-6 : NUM:dist -> __label__NUM
                            # TREC-50: NUM:dist -> __label__NUM:dist
                            new_label = "__label__"
                            new_label += old_label

                            write_fp.write(f"{new_label} {question}\n")

        super(TREC_50, self).__init__(
            data_folder, use_tokenizer=False, in_memory=in_memory
        )


class TREC_6(ClassificationCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        trec_path = "https://cogcomp.seas.upenn.edu/Data/QA/QC/"

        original_filenames = ["train_5500.label", "TREC_10.label"]
        new_filenames = ["train.txt", "test.txt"]
        for original_filename in original_filenames:
            cached_path(
                f"{trec_path}{original_filename}",
                Path("datasets") / dataset_name / "original",
            )

        data_file = data_folder / new_filenames[0]

        if not data_file.is_file():
            for original_filename, new_filename in zip(
                original_filenames, new_filenames
            ):
                with open(
                    data_folder / "original" / original_filename,
                    "rt",
                    encoding="latin1",
                ) as open_fp:
                    with open(
                        data_folder / new_filename, "wt", encoding="utf-8"
                    ) as write_fp:
                        for line in open_fp:
                            line = line.rstrip()
                            fields = line.split()
                            old_label = fields[0]
                            question = " ".join(fields[1:])

                            # Create flair compatible labels
                            # TREC-6 : NUM:dist -> __label__NUM
                            # TREC-50: NUM:dist -> __label__NUM:dist
                            new_label = "__label__"
                            new_label += old_label.split(":")[0]

                            write_fp.write(f"{new_label} {question}\n")

        super(TREC_6, self).__init__(
            data_folder, use_tokenizer=False, in_memory=in_memory
        )


class UD_ENGLISH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master"
        cached_path(f"{web_path}/en_ewt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{web_path}/en_ewt-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{web_path}/en_ewt-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_ENGLISH, self).__init__(data_folder, in_memory=in_memory)


class UD_TAMIL(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Tamil-TTB/master"
        cached_path(f"{web_path}/ta_ttb-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{web_path}/ta_ttb-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{web_path}/ta_ttb-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_TAMIL, self).__init__(data_folder, in_memory=in_memory)

class UD_GERMAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master"
        cached_path(f"{ud_path}/de_gsd-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/de_gsd-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/de_gsd-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_GERMAN, self).__init__(data_folder, in_memory=in_memory)


class UD_GERMAN_HDT(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = False):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = (
            "https://raw.githubusercontent.com/UniversalDependencies/UD_German-HDT/dev"
        )
        cached_path(f"{ud_path}/de_hdt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/de_hdt-ud-test.conllu", Path("datasets") / dataset_name)

        train_filenames = [
            "de_hdt-ud-train-a-1.conllu",
            "de_hdt-ud-train-a-2.conllu",
            "de_hdt-ud-train-b-1.conllu",
            "de_hdt-ud-train-b-2.conllu",
        ]

        for train_file in train_filenames:
            cached_path(
                f"{ud_path}/{train_file}", Path("datasets") / dataset_name / "original"
            )

        data_path = Path(flair.cache_root) / "datasets" / dataset_name

        new_train_file: Path = data_path / "de_hdt-ud-train-all.conllu"

        if not new_train_file.is_file():
            with open(new_train_file, "wt") as f_out:
                for train_filename in train_filenames:
                    with open(data_path / "original" / train_filename, "rt") as f_in:
                        f_out.write(f_in.read())

        super(UD_GERMAN_HDT, self).__init__(data_folder, in_memory=in_memory)


class UD_DUTCH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Dutch-Alpino/master"
        cached_path(
            f"{ud_path}/nl_alpino-ud-dev.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/nl_alpino-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/nl_alpino-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_DUTCH, self).__init__(data_folder, in_memory=in_memory)


class UD_FRENCH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_French-GSD/master"
        cached_path(f"{ud_path}/fr_gsd-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/fr_gsd-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/fr_gsd-ud-train.conllu", Path("datasets") / dataset_name
        )
        super(UD_FRENCH, self).__init__(data_folder, in_memory=in_memory)


class UD_ITALIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master"
        cached_path(f"{ud_path}/it_isdt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/it_isdt-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/it_isdt-ud-train.conllu", Path("datasets") / dataset_name
        )
        super(UD_ITALIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_SPANISH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-GSD/master"
        cached_path(f"{ud_path}/es_gsd-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/es_gsd-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/es_gsd-ud-train.conllu", Path("datasets") / dataset_name
        )
        super(UD_SPANISH, self).__init__(data_folder, in_memory=in_memory)


class UD_PORTUGUESE(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Portuguese-Bosque/master"
        cached_path(
            f"{ud_path}/pt_bosque-ud-dev.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/pt_bosque-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/pt_bosque-ud-train.conllu", Path("datasets") / dataset_name
        )
        super(UD_PORTUGUESE, self).__init__(data_folder, in_memory=in_memory)


class UD_ROMANIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Romanian-RRT/master"
        cached_path(f"{ud_path}/ro_rrt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/ro_rrt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/ro_rrt-ud-train.conllu", Path("datasets") / dataset_name
        )
        super(UD_ROMANIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_CATALAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Catalan-AnCora/master"
        cached_path(
            f"{ud_path}/ca_ancora-ud-dev.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/ca_ancora-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/ca_ancora-ud-train.conllu", Path("datasets") / dataset_name
        )
        super(UD_CATALAN, self).__init__(data_folder, in_memory=in_memory)


class UD_POLISH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Polish-LFG/master"
        cached_path(f"{ud_path}/pl_lfg-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/pl_lfg-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/pl_lfg-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_POLISH, self).__init__(data_folder, in_memory=in_memory)


class UD_CZECH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = False):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Czech-PDT/master"
        cached_path(f"{ud_path}/cs_pdt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/cs_pdt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/cs_pdt-ud-train-c.conllu",
            Path("datasets") / dataset_name / "original",
        )
        cached_path(
            f"{ud_path}/cs_pdt-ud-train-l.conllu",
            Path("datasets") / dataset_name / "original",
        )
        cached_path(
            f"{ud_path}/cs_pdt-ud-train-m.conllu",
            Path("datasets") / dataset_name / "original",
        )
        cached_path(
            f"{ud_path}/cs_pdt-ud-train-v.conllu",
            Path("datasets") / dataset_name / "original",
        )
        data_path = Path(flair.cache_root) / "datasets" / dataset_name

        train_filenames = [
            "cs_pdt-ud-train-c.conllu",
            "cs_pdt-ud-train-l.conllu",
            "cs_pdt-ud-train-m.conllu",
            "cs_pdt-ud-train-v.conllu",
        ]

        new_train_file: Path = data_path / "cs_pdt-ud-train-all.conllu"

        if not new_train_file.is_file():
            with open(new_train_file, "wt") as f_out:
                for train_filename in train_filenames:
                    with open(data_path / "original" / train_filename, "rt") as f_in:
                        f_out.write(f_in.read())
        super(UD_CZECH, self).__init__(data_folder, in_memory=in_memory)


class UD_SLOVAK(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Slovak-SNK/master"
        cached_path(f"{ud_path}/sk_snk-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/sk_snk-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/sk_snk-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_SLOVAK, self).__init__(data_folder, in_memory=in_memory)


class UD_SWEDISH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Swedish-Talbanken/master"
        cached_path(
            f"{ud_path}/sv_talbanken-ud-dev.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/sv_talbanken-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/sv_talbanken-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_SWEDISH, self).__init__(data_folder, in_memory=in_memory)


class UD_DANISH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Danish-DDT/master"
        cached_path(f"{ud_path}/da_ddt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/da_ddt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/da_ddt-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_DANISH, self).__init__(data_folder, in_memory=in_memory)


class UD_NORWEGIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Norwegian-Bokmaal/master"
        cached_path(
            f"{ud_path}/no_bokmaal-ud-dev.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/no_bokmaal-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/no_bokmaal-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_NORWEGIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_FINNISH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Finnish-TDT/master"
        cached_path(f"{ud_path}/fi_tdt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/fi_tdt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/fi_tdt-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_FINNISH, self).__init__(data_folder, in_memory=in_memory)


class UD_SLOVENIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Slovenian-SSJ/master"
        cached_path(f"{ud_path}/sl_ssj-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/sl_ssj-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/sl_ssj-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_SLOVENIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_CROATIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Croatian-SET/master"
        cached_path(f"{ud_path}/hr_set-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/hr_set-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/hr_set-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_CROATIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_SERBIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Serbian-SET/master"
        cached_path(f"{ud_path}/sr_set-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/sr_set-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/sr_set-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_SERBIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_BULGARIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Bulgarian-BTB/master"
        cached_path(f"{ud_path}/bg_btb-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/bg_btb-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/bg_btb-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_BULGARIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_ARABIC(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Arabic-PADT/master"
        cached_path(f"{ud_path}/ar_padt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/ar_padt-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/ar_padt-ud-train.conllu", Path("datasets") / dataset_name
        )
        super(UD_ARABIC, self).__init__(data_folder, in_memory=in_memory)


class UD_HEBREW(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Hebrew-HTB/master"
        cached_path(f"{ud_path}/he_htb-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/he_htb-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/he_htb-ud-train.conllu", Path("datasets") / dataset_name
        )
        super(UD_HEBREW, self).__init__(data_folder, in_memory=in_memory)


class UD_TURKISH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/master"
        cached_path(f"{ud_path}/tr_imst-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/tr_imst-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/tr_imst-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_TURKISH, self).__init__(data_folder, in_memory=in_memory)


class UD_PERSIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Persian-Seraji/master"
        cached_path(
            f"{ud_path}/fa_seraji-ud-dev.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/fa_seraji-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/fa_seraji-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_PERSIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_RUSSIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-SynTagRus/master"
        cached_path(
            f"{ud_path}/ru_syntagrus-ud-dev.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/ru_syntagrus-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/ru_syntagrus-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_RUSSIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_HINDI(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Hindi-HDTB/master"
        cached_path(f"{ud_path}/hi_hdtb-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/hi_hdtb-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/hi_hdtb-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_HINDI, self).__init__(data_folder, in_memory=in_memory)


class UD_INDONESIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Indonesian-GSD/master"
        cached_path(f"{ud_path}/id_gsd-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/id_gsd-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/id_gsd-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_INDONESIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_JAPANESE(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Japanese-GSD/master"
        cached_path(f"{ud_path}/ja_gsd-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/ja_gsd-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/ja_gsd-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_JAPANESE, self).__init__(data_folder, in_memory=in_memory)


class UD_CHINESE(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-GSD/master"
        cached_path(f"{ud_path}/zh_gsd-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/zh_gsd-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/zh_gsd-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_CHINESE, self).__init__(data_folder, in_memory=in_memory)


class UD_KOREAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Korean-Kaist/master"
        cached_path(
            f"{ud_path}/ko_kaist-ud-dev.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/ko_kaist-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/ko_kaist-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_KOREAN, self).__init__(data_folder, in_memory=in_memory)


class UD_BASQUE(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Basque-BDT/master"
        cached_path(f"{ud_path}/eu_bdt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/eu_bdt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/eu_bdt-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_BASQUE, self).__init__(data_folder, in_memory=in_memory)


def _download_wassa_if_not_there(emotion, data_folder, dataset_name):
    for split in ["train", "dev", "test"]:

        data_file = data_folder / f"{emotion}-{split}.txt"

        if not data_file.is_file():

            if split == "train":
                url = f"http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/{emotion}-ratings-0to1.train.txt"
            if split == "dev":
                url = f"http://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/{emotion}-ratings-0to1.dev.gold.txt"
            if split == "test":
                url = f"http://saifmohammad.com/WebDocs/EmoInt%20Test%20Gold%20Data/{emotion}-ratings-0to1.test.gold.txt"

            path = cached_path(url, Path("datasets") / dataset_name)

            with open(path, "r") as f:
                with open(data_file, "w") as out:
                    next(f)
                    for line in f:
                        fields = line.split("\t")
                        out.write(f"__label__{fields[3].rstrip()} {fields[1]}\n")

            os.remove(path)


class WASSA_ANGER(ClassificationCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = False):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wassa_if_not_there("anger", data_folder, dataset_name)

        super(WASSA_ANGER, self).__init__(
            data_folder, use_tokenizer=False, in_memory=in_memory
        )


class WASSA_FEAR(ClassificationCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = False):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wassa_if_not_there("fear", data_folder, dataset_name)

        super(WASSA_FEAR, self).__init__(
            data_folder, use_tokenizer=False, in_memory=in_memory
        )


class WASSA_JOY(ClassificationCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = False):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wassa_if_not_there("joy", data_folder, dataset_name)

        super(WASSA_JOY, self).__init__(
            data_folder, use_tokenizer=False, in_memory=in_memory
        )


class WASSA_SADNESS(ClassificationCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = False):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wassa_if_not_there("sadness", data_folder, dataset_name)

        super(WASSA_SADNESS, self).__init__(
            data_folder, use_tokenizer=False, in_memory=in_memory
        )


def _download_wikiner(language_code: str, dataset_name: str):
    # download data if necessary
    wikiner_path = (
        "https://raw.githubusercontent.com/dice-group/FOX/master/input/Wikiner/"
    )
    lc = language_code

    data_file = (
        Path(flair.cache_root)
        / "datasets"
        / dataset_name
        / f"aij-wikiner-{lc}-wp3.train"
    )
    if not data_file.is_file():

        cached_path(
            f"{wikiner_path}aij-wikiner-{lc}-wp3.bz2", Path("datasets") / dataset_name
        )
        import bz2, shutil

        # unpack and write out in CoNLL column-like format
        bz_file = bz2.BZ2File(
            Path(flair.cache_root)
            / "datasets"
            / dataset_name
            / f"aij-wikiner-{lc}-wp3.bz2",
            "rb",
        )
        with bz_file as f, open(
            Path(flair.cache_root)
            / "datasets"
            / dataset_name
            / f"aij-wikiner-{lc}-wp3.train",
            "w",
        ) as out:
            for line in f:
                line = line.decode("utf-8")
                words = line.split(" ")
                for word in words:
                    out.write("\t".join(word.split("|")) + "\n")


class WIKINER_ENGLISH(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = False,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wikiner("en", dataset_name)

        super(WIKINER_ENGLISH, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class WIKINER_GERMAN(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = False,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wikiner("de", dataset_name)

        super(WIKINER_GERMAN, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class WIKINER_DUTCH(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = False,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wikiner("nl", dataset_name)

        super(WIKINER_DUTCH, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class WIKINER_FRENCH(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = False,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wikiner("fr", dataset_name)

        super(WIKINER_FRENCH, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class WIKINER_ITALIAN(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = False,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wikiner("it", dataset_name)

        super(WIKINER_ITALIAN, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class WIKINER_SPANISH(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = False,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wikiner("es", dataset_name)

        super(WIKINER_SPANISH, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class WIKINER_PORTUGUESE(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = False,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wikiner("pt", dataset_name)

        super(WIKINER_PORTUGUESE, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class WIKINER_POLISH(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = False,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wikiner("pl", dataset_name)

        super(WIKINER_POLISH, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class WIKINER_RUSSIAN(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = False,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wikiner("ru", dataset_name)

        super(WIKINER_RUSSIAN, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class WNUT_17(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        wnut_path = "https://noisy-text.github.io/2017/files/"
        cached_path(f"{wnut_path}wnut17train.conll", Path("datasets") / dataset_name)
        cached_path(f"{wnut_path}emerging.dev.conll", Path("datasets") / dataset_name)
        cached_path(
            f"{wnut_path}emerging.test.annotated", Path("datasets") / dataset_name
        )

        super(WNUT_17, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class DataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=4,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):

        # in certain cases, multi-CPU data loading makes no sense and slows
        # everything down. For this reason, we detect if a dataset is in-memory:
        # if so, num_workers is set to 0 for faster processing
        flair_dataset = dataset
        while True:
            if type(flair_dataset) is Subset:
                flair_dataset = flair_dataset.dataset
            elif type(flair_dataset) is ConcatDataset:
                flair_dataset = flair_dataset.datasets[0]
            else:
                break

        if type(flair_dataset) is list:
            num_workers = 0
        elif isinstance(flair_dataset, FlairDataset) and flair_dataset.is_in_memory():
            num_workers = 0

        super(DataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=list,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )

class CoupleDataset:
    """docstring for CoupleDataset"""
    def __init__(self, corpus1, corpus2):
        self.corpus1=corpus1
        self.corpus2=corpus2
    def __len__(self):
        return len(self.corpus1)
    def __getitem__(self, index):
        return self.corpus1[index],self.corpus2[index]
    def is_in_memory(self) -> bool:
        return self.corpus1.in_memory
