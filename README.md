
# ACE

The code is for our ACL-IJCNLP 2021 paper: [Automated Concatenation of Embeddings for Structured Prediction](https://arxiv.org/abs/2010.05006)

ACE is a framework for automatically searching a good embedding concatenation for structured prediction tasks and achieving state-of-the-art accuracy. The code is based on [flair version 0.4.3](https://github.com/flairNLP/flair) with a lot of modifications.

## News
 - **2021-07-19**: New versions of document-level SOTA NER models are released, see [Instructions for Reproducing Results](#Instructions-for-Reproducing-Results) for more details.

## Comparison with State-of-the-Art

| Task | Language | Dataset | ACE | Previous best |
| -------------------------------  | ---  | ----------- | ---------------- | ------------- |
| Named Entity Recognition |English | CoNLL 03 (document-level)   |  **94.6** (F1)  | *94.3 [(Yamada et al., 2020)](https://arxiv.org/pdf/2010.01057.pdf)* |
| Named Entity Recognition |German | CoNLL 03 (document-level)   |  **88.3** (F1) | *86.4 [(Yu et al., 2020)](https://arxiv.org/pdf/2005.07150.pdf)* |
| Named Entity Recognition |German | CoNLL 03 (06 Revision) (document-level)   |  **91.7** (F1)   | *90.3 [(Yu et al., 2020)](https://arxiv.org/pdf/2005.07150.pdf)* |
| Named Entity Recognition |Dutch | CoNLL 02 (document-level)   |  **95.7** (F1) | *93.7 [(Yu et al., 2020)](https://arxiv.org/pdf/2005.07150.pdf)* |
| Named Entity Recognition |Spanish | CoNLL 02 (document-level)   |  **95.9** (F1)  | *90.3 [(Yu et al., 2020)](https://arxiv.org/pdf/2005.07150.pdf)* |
| Named Entity Recognition |English | CoNLL 03 (sentence-level)   |  **93.6** (F1)  | *93.5 [(Baevski et al., 2019)](https://arxiv.org/pdf/1903.07785v1.pdf)* |
| Named Entity Recognition |German | CoNLL 03 (sentence-level)   |  **87.0** (F1) | *86.4 [(Yu et al., 2020)](https://arxiv.org/pdf/2005.07150.pdf)* |
| Named Entity Recognition |German | CoNLL 03 (06 Revision) (sentence-level)   |  **90.5** (F1)   | *90.3 [(Yu et al., 2020)](https://arxiv.org/pdf/2005.07150.pdf)* |
| Named Entity Recognition |Dutch | CoNLL 02 (sentence-level)   |  **94.6** (F1) | *93.7 [(Yu et al., 2020)](https://arxiv.org/pdf/2005.07150.pdf)* |
| Named Entity Recognition |Spanish | CoNLL 02 (sentence-level)   |  **91.7** (F1)  | *90.3 [(Yu et al., 2020)](https://arxiv.org/pdf/2005.07150.pdf)* |
| POS Tagging |English | Ritter's |  **93.4** (Acc)  | *90.1 [(Nguyen et al., 2020)](https://arxiv.org/pdf/2005.10200.pdf)* |
| POS Tagging |English | Ark |  **94.4** (Acc)  | *94.1 [(Nguyen et al., 2020)](https://arxiv.org/pdf/2005.10200.pdf)* |
| POS Tagging |English | TweeBank v2 |  **95.8** (Acc)  | *95.2 [(Nguyen et al., 2020)](https://arxiv.org/pdf/2005.10200.pdf)* |
| Aspect Extraction |English | SemEval 2014 Laptop |  **87.4** (F1)  | *84.3 [(Xu et al., 2019)](https://arxiv.org/pdf/1904.02232.pdf)* |
| Aspect Extraction |English | SemEval 2014 Restaurant |  **92.0** (F1)  | *87.1 [(Wei et al., 2020)](https://www.aclweb.org/anthology/2020.acl-main.339/)* |
| Aspect Extraction |English | SemEval 2015 Restaurant |  **80.3** (F1)  | *72.7 [(Wei et al., 2020)](https://www.aclweb.org/anthology/2020.acl-main.339/)* |
| Aspect Extraction |English | SemEval 2016 Restaurant |  **81.3** (F1)  | *78.0 [(Xu et al., 2019)](https://arxiv.org/pdf/1904.02232.pdf)* |
| Dependency Parsing | English | PTB    |  **95.7** (LAS)  | *95.3 [(Wang et al., 2020)](https://arxiv.org/pdf/2010.05003.pdf)* |
| Semantic Dependency Parsing | English | DM ID   |  **95.6** (LF1)  | *94.4 [(Fernández-González and Gómez-Rodríguez, 2020)](https://www.aclweb.org/anthology/2020.acl-main.629/)* |
| Semantic Dependency Parsing | English | DM OOD   |  **92.6** (LF1)  | *91.0 [(Fernández-González and Gómez-Rodríguez, 2020)](https://www.aclweb.org/anthology/2020.acl-main.629/)* |
| Semantic Dependency Parsing | English | PAS ID   |  **95.8** (LF1)  | *95.1 [(Fernández-González and Gómez-Rodríguez, 2020)](https://www.aclweb.org/anthology/2020.acl-main.629/)* |
| Semantic Dependency Parsing | English | PAS OOD   |  **94.6** (LF1)  | *93.4 [(Fernández-González and Gómez-Rodríguez, 2020)](https://www.aclweb.org/anthology/2020.acl-main.629/)* |
| Semantic Dependency Parsing | English | PSD ID   |  **83.8** (LF1)  | *82.6 [(Fernández-González and Gómez-Rodríguez, 2020)](https://www.aclweb.org/anthology/2020.acl-main.629/)* |
| Semantic Dependency Parsing | English | PSD OOD   |  **83.4** (LF1)  | *82.0 [(Fernández-González and Gómez-Rodríguez, 2020)](https://www.aclweb.org/anthology/2020.acl-main.629/)* |

## Guide

- [Requirements](#requirements)
- [Pretrained Models](#Pretrained-Models)
  - [Instructions for Reproducing Results](#Instructions-for-Reproducing-Results)
    <!-- - [Named Entity Recognition](#Named_Entity_Recognition) -->
- [Download Embeddings](#Download-Embeddings)
  <!-- - [Dump Fine-tuned Embeddings in the Pretrained Models ](#Dump-Fine-tuned-Embeddings-in-the-Pretrained-Models) -->
- [Training](#training)
  - [Training ACE Models](#training-ace-models)
  - [Train on Your Own Dataset](#Train-on-Your-Own-Dataset)
  - [Set the Embeddings](#Set-the-Embeddings)
  - [(Optional) Fine-tune Transformer-based Embeddings](#Optional-Fine-tune-Transformer-based-Embeddings)
  - [(Optional) Extract Document Features for BERT Embeddings](#Optional-Extract-Document-Features-for-BERT-Embeddings)
- [Parse files](#parse-files)
- [Config File](#Config-File)
- [TODO](#todo)
- [Citing Us](#Citing-Us)
- [Contact](#contact)

## Requirements
The project is based on PyTorch 1.1+ and Python 3.6+. To run our code, install:

```
pip install -r requirements.txt
```

The following requirements should be satisfied:
* [transformers](https://github.com/huggingface/transformers): **3.0.0** 


## Download Embeddings

In our code, most of the embeddings can be downloaded automatically (except ELMo for non-English languages). You can also download the embeddings manually. The embeddings we used in the paper can be downloaded here:

| Name | Link | 
| -------------------------------  | ---|
|GloVe | [nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove)|
|fastText | [github.com/facebookresearch/fastText](https://github.com/facebookresearch/fastText)|
|ELMo | [github.com/allenai/allennlp](https://github.com/allenai/allennlp)|
|ELMo (Other languages) | [github.com/TalSchuster/CrossLingualContextualEmb](https://github.com/TalSchuster/CrossLingualContextualEmb)|
|BERT | [huggingface.co/bert-base-cased](https://huggingface.co/bert-base-cased)|
|M-BERT | [huggingface.co/bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)|
|BERT (Dutch) | [huggingface.co/wietsedv/bert-base-dutch-cased](https://huggingface.co/wietsedv/bert-base-dutch-cased)|
|BERT (German) | [huggingface.co/bert-base-german-dbmdz-cased](https://huggingface.co/bert-base-german-dbmdz-cased)|
|BERT (Spanish) | [huggingface.co/dccuchile/bert-base-spanish-wwm-cased](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)|
|BERT (Turkish) | [huggingface.co/dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased)|
|XLM-R | [huggingface.co/xlm-roberta-large](https://huggingface.co/xlm-roberta-large)|
|XLM-R (CoNLL 02 Dutch) | [huggingface.co/xlm-roberta-large-finetuned-conll02-dutch](https://huggingface.co/xlm-roberta-large-finetuned-conll02-dutch)|
|XLM-R (CoNLL 02 Spanish) | [huggingface.co/xlm-roberta-large-finetuned-conll02-spanish](https://huggingface.co/xlm-roberta-large-finetuned-conll02-spanish)|
|XLM-R (CoNLL 03 English) | [huggingface.co/xlm-roberta-large-finetuned-conll03-english](https://huggingface.co/xlm-roberta-large-finetuned-conll03-english)|
|XLM-R (CoNLL 03 German) | [huggingface.co/xlm-roberta-large-finetuned-conll03-german](https://huggingface.co/xlm-roberta-large-finetuned-conll03-german)|
|XLNet | [huggingface.co/xlnet-large-cased](https://huggingface.co/xlnet-large-cased)|

After the embeddings are downloaded, you need to set the path of embeddings in the config file manually. For example in `config/conll_03_english.yaml`:
```
TransformerWordEmbeddings-1:
    model: your/embedding/path 
    layers: -1,-2,-3,-4
    pooling_operation: mean
```


## Pretrained Models
We provide pretrained models for Named Entity Recognition (Sentence-/Document-Level) and Dependency Parsing (PTB) on [OneDrive](https://1drv.ms/u/s!Am53YNAPSsodg810cf4qsCBCMW9PAg?e=qmGMgG). You can find the corresponding config file in `config/`. For the zip files named with `doc*.zip`, you need to extract document-level embeddings at first. Please check [(Optional) Extract Document Features for BERT Embeddings](#Optional-Extract-Document-Features-for-BERT-Embeddings).

* Download models
* `unzip` the zip file
* Put the directory in the `resources/taggers/`

To check the accuracy of the model, run:
```
CUDA_VISIBLE_DEVICES=0 python train.py --config config/conll_03_english.yaml --test
```
where `--config $config_file` is setting the configureation file. 
Here we take CoNLL 2003 English NER as an example. The `$config_file` is `config/conll_03_english.yaml`. 

<!-- ### Model list:

|File Name| Task | Language | Dataset | ACE | 
| -------------------------------  | ---  | ----------- | ----------------  |----------------  |
| Named Entity Recognition |English | CoNLL 03 (sentence-level)  |  **93.64** (F1)  |
| Named Entity Recognition |English | CoNLL 03 (document-level)  |  **93.14** (F1)  |
| Named Entity Recognition |German | CoNLL 03 (document-level) |  **88.04** (F1)  |
| Named Entity Recognition |German | CoNLL 03 (06 Revision) (document-level)  |  **91.38** (F1)   |
| Named Entity Recognition |Dutch | CoNLL 02 (document-level) |  **95.54** (F1) | 
| Named Entity Recognition |Spanish | CoNLL 02 (document-level)   |  **95.58** (F1) (+doc) | 
 -->
<!-- ### Dump Fine-tuned Embeddings in the Pretrained Models 

TODO -->

### Instructions for Reproducing Results

Currently, we give an instruction for reproducing the results of our NER models is in [named_entity_recognition.md](resources/docs/named_entity_recognition.md). Other tasks can simply follow the guide of [named_entity_recognition.md](resources/docs/named_entity_recognition.md) to reproduce the results. 

<!-- #### Named Entity Recognition
Instructions: [named_entity_recognition.md](resources/docs/named_entity_recognition.md) -->

---

## Training

### Training ACE Models

To train the model, run:

```
CUDA_VISIBLE_DEVICES=0 python train.py --config $config_file
```

### Train on Your Own Dataset

To set the dataset manully, you can set the dataset in the `$confile_file` by:

**Sequence Labeling**:
```
targets: ner
ner:
  Corpus: ColumnCorpus-1
  ColumnCorpus-1: 
    data_folder: datasets/conll_03_new
    column_format:
      0: text
      1: pos
      2: chunk
      3: ner
    tag_to_bioes: ner
  tag_dictionary: resources/taggers/your_ner_tags.pkl
```

**Parsing**:
```
targets: dependency
dependency:
  Corpus: UniversalDependenciesCorpus-1
  UniversalDependenciesCorpus-1:
    data_folder: datasets/ptb
    add_root: True
  tag_dictionary: resources/taggers/your_parsing_tags.pkl
```

The `tag_dictionary` is a path to the tag dictionary for the task. If the path does not exist, the code will generate a tag dictionary at the path automatically. The dataset format is: `Corpus: $CorpusClassName-$id`, where `$id` is the name of datasets (anything you like). You can train multiple datasets jointly. For example:
```
Corpus: ColumnCorpus-1:ColumnCorpus-2:ColumnCorpus-3

ColumnCorpus-1:
  data_folder: ...
  column_format: ...
  tag_to_bioes: ...

ColumnCorpus-2:
  data_folder: ...
  column_format: ...
  tag_to_bioes: ...

ColumnCorpus-3:
  data_folder: ...
  column_format: ...
  tag_to_bioes: ...
```

Please refer to [Config File](#Config-File) for more details.

### Set the Embeddings

You need to modifiy the embedding paths in the `$config_file` to change the embeddings for concatenation. For example, you need to add `bert-large-cased` in the `config/conll_03_english.yaml`
```
embeddings:
  TransformerWordEmbeddings-0:
    layers: '-1'
    pooling_operation: first
    model: xlm-roberta-large-finetuned-conll03-english

  TransformerWordEmbeddings-1:
    model: bert-base-cased
    layers: -1,-2,-3,-4
    pooling_operation: mean

  TransformerWordEmbeddings-2:
    model: bert-base-multilingual-cased
    layers: -1,-2,-3,-4
    pooling_operation: mean

  TransformerWordEmbeddings-3: # New embeddings
    model: bert-large-cased
    layers: -1,-2,-3,-4
    pooling_operation: mean
  ...
```


### (Optional) Fine-tune Transformer-based Embeddings

To archieve state-of-the-art accuracy, one optional approach is fine-tuning the transformer-based embeddings over the task. We use fine-tuned embeddings in [huggingface](huggingface.co/models) for NER tasks while embeddings in other tasks are fine-tuned by ourselves. Then take the embeddings as an embedding candidate of ACE. Taking fine-tuning BERT model on PTB parsing as an example, run:

```
CUDA_VISIBLE_DEVICES=0 python train.py --config config/en-bert-finetune-ptb.yaml
```

After the model is fine-tuned, you will find a tuned BERT model at 
```
ls resources/taggers/en-bert_10epoch_0.5inter_2000batch_0.00005lr_20lrrate_ptb_monolingual_nocrf_fast_warmup_freezing_beta_weightdecay_finetune_saving_nodev_dependency16/bert-base-cased
```


Then, replace `bert-base-cased` with `resources/taggers/en-bert_10epoch_0.5inter_2000batch_0.00005lr_20lrrate_ptb_monolingual_nocrf_fast_warmup_freezing_beta_weightdecay_finetune_saving_nodev_dependency16/bert-base-cased` in the `$config_file` of the ACE model (for example, `config/ptb_parsing_model.yaml`).

The config `config/en-bert-finetune-ptb.yaml` can be applied to fine-tuning other embeddings in **parsing tasks**. Here is an example config for fine-tuning NER (**sequence labeling tasks**): `config/en-bert-finetune-ner.yaml`


### (Optional) Extract Document Features for BERT Embeddings

To archieve state-of-the-art accuracy of **NER**, one optional approach is extracting the document-level features from the BERT embeddings (for RoBERTa, XLM-R and XLNET, we feed the model with the whole document, if you are interested in this part, see [embeddings.py](flair/embeddings.py#L3533-L3749)). Then take the features as an embedding candidate of ACE. We follow the embedding extraction approach of [Yu et al., 2020](https://arxiv.org/pdf/2005.07150.pdf). We use the sentences with a single word `-DOCSTART-` to split the documents. For CoNLL 2002 Spanish, there is not `-DOCSTART-` sentences. Therefore, we add a `-DOCSTART-` sentence for every 25 sentences. For CoNLL 2002 Dutch, the `-DOCSTART-` is in the first sentence of the document, please split the `-DOCSTART-` token into a single sentence. For example:

```
-DOCSTART- -DOCSTART- O

De Art O
tekst N O
van Prep O
het Art O
arrest N O
is V O
nog Adv O
niet Adv O
schriftelijk Adj O
beschikbaar Adj O
maar Conj O
het Art O
bericht N O
werd V O
alvast Adv O
bekendgemaakt V O
door Prep O
een Art O
communicatiebureau N O
dat Conj O
Floralux N B-ORG
inhuurde V O
. Punc O

...
```


Taking English BERT model on CoNLL English NER as an example, run:

```
CUDA_VISIBLE_DEVICES=0 python extract_features.py --config config/en-bert-extract.yaml --batch_size 32 
```

## Parse files

If you want to parse a certain file, add `train` in the file name and put the file in a certain `$dir` (for example, `parse_file_dir/train.your_file_name`). Run:

```
CUDA_VISIBLE_DEVICES=0 python train.py --config $config_file --parse --target_dir $dir --keep_order
```

The format of the file should be `column_format={0: 'text', 1:'ner'}` for sequence labeling or you can modifiy line 232 in `train.py`. The parsed results will be in `outputs/`.
Note that you may need to preprocess your file with the dummy tags for prediction, please check this [issue](https://github.com/Alibaba-NLP/ACE/issues/12) for more details.

## Config File

The config files are based on yaml format.

* `targets`: The target task
  * `ner`: named entity recognition
  * `upos`: part-of-speech tagging
  * `chunk`: chunking
  * `ast`: abstract extraction
  * `dependency`: dependency parsing
  * `enhancedud`: semantic dependency parsing/enhanced universal dependency parsing
* `ner`: An example for the `targets`. If `targets: ner`, then the code will read the values with the key of `ner`.
  * `Corpus`: The training corpora for the model, use `:` to split different corpora.
  * `tag_dictionary`: A path to the tag dictionary for the task. If the path does not exist, the code will generate a tag dictionary at the path automatically.
* `target_dir`: Save directory.
* `model_name`: The trained models will be save in `$target_dir/$model_name`.
* `model`: The model to train, depending on the task.
  * `FastSequenceTagger`: Sequence labeling model. The values are the parameters.
  * `SemanticDependencyParser`: Syntactic/semantic dependency parsing model. The values are the parameters.
* `embeddings`: The embeddings for the model, each key is the class name of the embedding and the values of the key are the parameters, see `flair/embeddings.py` for more details. For each embedding, use `$classname-$id` to represent the class. For example, if you want to use BERT and M-BERT for a single model, you can name: `TransformerWordEmbeddings-0`, `TransformerWordEmbeddings-1`.
* `trainer`: The trainer class.
  * `ModelFinetuner`: The trainer for fine-tuning embeddings or simply train a task model without ACE.
  * `ReinforcementTrainer`: The trainer for training ACE.
* `train`: the parameters for the `train` function in `trainer` (for example, `ReinforcementTrainer.train()`).

## TODO
* Knowledge Distillation with ACE: [Wang et al., 2020](https://github.com/Alibaba-NLP/MultilangStructureKD)

## Citing Us
If you feel the code helpful, please cite:
```
@article{wang2020automated,
  title={Automated Concatenation of Embeddings for Structured Prediction},
  author={Wang, Xinyu and Jiang, Yong and Bach, Nguyen and Wang, Tao and Huang, Zhongqiang and Huang, Fei and Tu, Kewei},
  journal={arXiv preprint arXiv:2010.05006},
  year={2020}
}
```

## Contact 

Feel free to email your questions or comments to issues or to [Xinyu Wang](http://wangxinyu0922.github.io/).

