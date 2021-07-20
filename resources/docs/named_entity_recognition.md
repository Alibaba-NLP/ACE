# Named Entity Recognition (NER)
Here is the guide about how to run NER models in our code. Before the guide starts, please ensure the [essential packages](https://github.com/Alibaba-NLP/ACE#requirements) to run the code have been installed. We share our NER datasets at [data.zip](https://1drv.ms/u/s!Am53YNAPSsodg88m3q41pypnfHL8zg?e=2UNuGw).

**Sentence-level** training and evaluation approach has been widely applied to a lot of work while recent work finds that feeding the whole document into the transformer-based model can significantly improve the accuracy of NER models (**document-level**). Both kinds of training and evaluation approaches are practical and essential in different scenarios. As a result, in this part, we will discuss how to run both kinds of models.

## CoNLL 2003 English

### Sentence-level Model
Download the [conll_en_ner_model.zip](https://1drv.ms/u/s!Am53YNAPSsodg811rn6NoFbA2eFcww?e=sh7oJd) at [OneDrive](https://1drv.ms/u/s!Am53YNAPSsodg810NxHQcrJpcNIOig?e=FRsJNR). Unzip the file and move the unzipped repository to `resources/taggers`. 

#### Evaluation 
To evaluate the accuracy of the model, run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/conll_03_english.yaml --test
```

#### Training 

##### Finetune Embeddings
To train the model by yourself, firstly, training finetuned transformer-based models (i.e. BERT and M-BERT) are required. Train the following models:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/mbert-en-ner-finetune.yaml
CUDA_VISIBLE_DEVICES=0 python train.py --config config/bert-en-ner-finetune.yaml
```
You can find the trained models and its finetuned embeddings in `resources/taggers`. Modify the embedding path in `config/conll_03_english.yaml` with your trained models. For example:
```yaml
  TransformerWordEmbeddings-1:
    layers: '-1'
    model: resources/taggers/en-bert_10epoch_32batch_0.00005lr_10000lrrate_en_monolingual_nocrf_fast_relearn_sentbatch_sentloss_finetune_saving_nodev_newner3/bert-base-cased
    pooling_operation: first
  TransformerWordEmbeddings-2:
    layers: '-1'
    model: resources/taggers/multi-bert_10epoch_32batch_0.00005lr_10000lrrate_en_monolingual_nocrf_fast_relearn_sentbatch_sentloss_finetune_saving_nodev_newner3/bert-base-multilingual-cased
    pooling_operation: first
```

##### Train the Model
Finally, run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/conll_03_english.yaml
```

#### Prediction 
To use the model predict on your own file, run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/conll_03_english.yaml --parse --target_dir $dir --keep_order
```
Note that you may need to preprocess your file with the dummy tags, please check this [issue](https://github.com/Alibaba-NLP/ACE/issues/12) for more details.

---

### Document-level Model
Download the [doc_ner_best.zip](https://1drv.ms/u/s!Am53YNAPSsodg9AuDfr4SYEfx27eVQ?e=QwEBQo) at [OneDrive](https://1drv.ms/u/s!Am53YNAPSsodg810NxHQcrJpcNIOig?e=FRsJNR). Unzip the file and move the unzipped repository to `resources/taggers`. 

The model needs the pre-extracted document-level features of `bert-base-cased`, `bert-large-cased` and `bert-base-multilingual-cased`. Pre-extracted features: [bert-base-cased.hdf5](https://1drv.ms/u/s!Am53YNAPSsodg9cb8ugrKUdK3Ra_fQ?e=whaLV0), [bert-large-cased.hdf5](https://1drv.ms/u/s!Am53YNAPSsodg9cYy2-kf7trqxIHaQ?e=vIfjMZ) and [bert-base-multilingual-cased.hdf5](https://1drv.ms/u/s!Am53YNAPSsodg9ccJzhEa1qMDjtBDw?e=eFNYf7) (Note that [bert-base-multilingual-cased.hdf5](https://1drv.ms/u/s!Am53YNAPSsodg9ccJzhEa1qMDjtBDw?e=eFNYf7) contains the pre-extracted of all languages of CoNLL datasets, so you do not need to download it again if you already downloaded it.). If you want to extract the features by yourself, see [(Optional) Extract Document Features](https://github.com/Alibaba-NLP/ACE#optional-extract-document-features-for-bert-models) for the guide to extract the document-level features. 

#### Evaluation 
To evaluate the accuracy of the model, run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/doc_ner_best.yaml --test
```

#### Training 

##### Finetune Document-Level Embeddings
To train the model by yourself, firstly, training finetuned transformer-based models (i.e. RoBERTa, XLM-R, XLNET) are required. Train the following models:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/xlnet-doc-en-ner-finetune.yaml
CUDA_VISIBLE_DEVICES=0 python train.py --config config/xlmr-doc-en-ner-finetune.yaml
CUDA_VISIBLE_DEVICES=0 python train.py --config config/roberta-doc-en-ner-finetune.yaml
```
You can find the trained models and its finetuned embeddings in `resources/taggers`. Modify the embedding path in `config/doc_ner_bert.yaml` with your trained models. For example:
```yaml
  TransformerWordEmbeddings-0:
    layers: '-1'
    model: resources/taggers/xlnet-first-docv2_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_eng_monolingual_nocrf_fast_norelearn_sentbatch_sentloss_finetune_nodev_saving_ner4/xlnet-large-cased
    pooling_operation: first
    v2_doc: true
  TransformerWordEmbeddings-1:
    layers: '-1'
    model: resources/taggers/xlmr-first-docv2_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_eng_monolingual_nocrf_fast_norelearn_sentbatch_sentloss_finetune_nodev_saving_ner3/xlm-roberta-large
    pooling_operation: first
    v2_doc: true
  TransformerWordEmbeddings-2:
    layers: '-1'
    model: resources/taggers/en-xlmr-first-docv2_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_eng_monolingual_nocrf_fast_norelearn_sentbatch_sentloss_finetune_nodev_saving_ner5/roberta-large
    pooling_operation: first
    v2_doc: true
```

##### Extract Document-Level Features
Please follow the instruction above.

##### Train the Model
Finally, run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/doc_ner_best.yaml
```

#### Prediction 
To use the model predict on your own file, run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/doc_ner_best.yaml --parse --target_dir $dir --keep_order
```
Note that:
 - You may need to preprocess your file with the dummy tags for prediction, please check this [issue](https://github.com/Alibaba-NLP/ACE/issues/12) for more details.
 - You need to pre-extract document-level features of `bert-base-cased`, `bert-large-cased` and `bert-base-multilingual-cased` embeddings. Please follow [(Optional) Extract Document Features](https://github.com/Alibaba-NLP/ACE#optional-extract-document-features-for-bert-models).


## CoNLL 2003 German
CoNLL German dataset contains version 2003 and a revised version of 2006. Currently, we release 2003 models for sentence-level and document-level, 2006 model for document-level.

### Sentence-level Model
Download the [conll_03_de_model.zip](https://1drv.ms/u/s!Am53YNAPSsodg9cUbajQqiP0gvq_UQ?e=az1vdG) at [OneDrive](https://1drv.ms/u/s!Am53YNAPSsodg810NxHQcrJpcNIOig?e=FRsJNR). Unzip the file and move the unzipped repository to `resources/taggers`. 

#### Evaluation 
To evaluate the accuracy of the model, run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/conll_03_de_model.yaml --test
```

#### Training 

##### Finetune Embeddings
To train the model by yourself, firstly, training finetuned transformer-based models (i.e. BERT and M-BERT) are required. Train the following models:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/mbert-de-03-ner-finetune.yaml
CUDA_VISIBLE_DEVICES=0 python train.py --config config/bert-de-03-ner-finetune.yaml
```
You can find the trained models and its finetuned embeddings in `resources/taggers`. Modify the embedding path in `config/conll_03_de_model.yaml` with your trained models. For example:
```yaml
  TransformerWordEmbeddings-1:
    layers: '-1'
    model: resources/taggers/de-bert_10epoch_32batch_0.00005lr_10000lrrate_de_monolingual_nocrf_fast_relearn_sentbatch_sentloss_finetune_saving_nodev_newner3/bert-base-german-cased
    pooling_operation: first
  TransformerWordEmbeddings-2:
    layers: '-1'
    model: resources/taggers/multi-bert_10epoch_32batch_0.00005lr_10000lrrate_de_monolingual_nocrf_fast_relearn_sentbatch_sentloss_finetune_saving_nodev_newner3/bert-base-multilingual-cased
    pooling_operation: first
```

##### Train the Model
Finally, run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/conll_03_de_model.yaml
```

#### Prediction 
To use the model predict on your own file, run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/conll_03_de_model.yaml --parse --target_dir $dir --keep_order
```
Note that you may need to preprocess your file with the dummy tags, please check this [issue](https://github.com/Alibaba-NLP/ACE/issues/12) for more details.

---

### Document-level Model
Download the [doc_ner_de_03_best.zip](https://1drv.ms/u/s!Am53YNAPSsodg9cVGGamr-oucXuMdA?e=B9hmN6) and [doc_ner_de_06_best.zip](https://1drv.ms/u/s!Am53YNAPSsodg9cdAmAm3NlblIihtQ?e=HN1jHP) at [OneDrive](https://1drv.ms/u/s!Am53YNAPSsodg810NxHQcrJpcNIOig?e=FRsJNR). Unzip the file and move the unzipped repository to `resources/taggers`. 

Both of the models need the pre-extracted document-level features of `bert-base-german-dbmdz-cased` and `bert-base-multilingual-cased`. Pre-extracted features: [bert-base-german-dbmdz-cased.hdf5](https://1drv.ms/u/s!Am53YNAPSsodg9cXXwqPsONUgpedHw?e=pD58wC) and [bert-base-multilingual-cased.hdf5](https://1drv.ms/u/s!Am53YNAPSsodg9ccJzhEa1qMDjtBDw?e=eFNYf7) (Note that [bert-base-multilingual-cased.hdf5](https://1drv.ms/u/s!Am53YNAPSsodg9ccJzhEa1qMDjtBDw?e=eFNYf7) contains the pre-extracted of all languages of CoNLL datasets, so you do not need to download it again if you already downloaded it.). If you want to extract the features by yourself, see [(Optional) Extract Document Features](https://github.com/Alibaba-NLP/ACE#optional-extract-document-features-for-bert-models) for the guide to extract the document-level features. 

#### Evaluation 
To evaluate the accuracy of the model, run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/doc_ner_de_03_best.yaml --test # for version 2003
CUDA_VISIBLE_DEVICES=0 python train.py --config config/doc_ner_de_06_best.yaml --test # for version 2006
```

#### Training 
##### Finetune Document-Level Embeddings
To train the model by yourself, firstly, training finetuned transformer-based models (i.e. XLM-R) are required. Train the following models:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/xlmr-doc-de-03-ner-finetune.yaml # for version 2003
CUDA_VISIBLE_DEVICES=0 python train.py --config config/xlmr-doc-de-06-ner-finetune.yaml # for version 2006
```
You can find the trained models and its finetuned embeddings in `resources/taggers`. Modify the embedding path in `config/doc_ner_de_03_best.yaml` or `config/doc_ner_de_06_best.yaml` with your trained models.

##### Extract Document-Level Features
Please follow the instruction above.

##### Train the Model
Finally, run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/doc_ner_de_03_best.yaml # for version 2003
CUDA_VISIBLE_DEVICES=0 python train.py --config config/doc_ner_de_06_best.yaml # for version 2006
```

#### Prediction 
To use the model predict on your own file, run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/doc_ner_de_03_best.yaml --parse --target_dir $dir --keep_order # for version 2003
CUDA_VISIBLE_DEVICES=0 python train.py --config config/doc_ner_de_06_best.yaml --parse --target_dir $dir --keep_order # for version 2006
```
Note that:
 - You may need to preprocess your file with the dummy tags for prediction, please check this [issue](https://github.com/Alibaba-NLP/ACE/issues/12) for more details.
 - You need to pre-extract document-level features of `bert-base-german-dbmdz-cased` and `bert-base-multilingual-cased` embeddings. Please follow [(Optional) Extract Document Features](https://github.com/Alibaba-NLP/ACE#optional-extract-document-features-for-bert-models).


## CoNLL 2002 Dutch
Currently, we release the model for document-level.

### Document-level Model
Download the [doc_ner_nl_best.zip](https://1drv.ms/u/s!Am53YNAPSsodg9cWsBM-32n5wdcOSQ?e=mIUzDN) at [OneDrive](https://1drv.ms/u/s!Am53YNAPSsodg810NxHQcrJpcNIOig?e=FRsJNR). Unzip the file and move the unzipped repository to `resources/taggers`. 

Both of the models need the pre-extracted document-level features of `bert-base-dutch-cased-finetuned-conll2002-ner` and `bert-base-multilingual-cased`. Pre-extracted features: [bert-base-dutch-cased-finetuned-conll2002-ner.hdf5](https://1drv.ms/u/s!Am53YNAPSsodg9cark1_snG4kRDQ9w?e=cNb6pm) and [bert-base-multilingual-cased.hdf5](https://1drv.ms/u/s!Am53YNAPSsodg9ccJzhEa1qMDjtBDw?e=eFNYf7) (Note that [bert-base-multilingual-cased.hdf5](https://1drv.ms/u/s!Am53YNAPSsodg9ccJzhEa1qMDjtBDw?e=eFNYf7) contains the pre-extracted of all languages of CoNLL datasets, so you do not need to download it again if you already downloaded it.). If you want to extract the features by yourself, see [(Optional) Extract Document Features](https://github.com/Alibaba-NLP/ACE#optional-extract-document-features-for-bert-models) for the guide to extract the document-level features. 

#### Evaluation 
To evaluate the accuracy of the model, run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/doc_ner_nl_best.yaml --test
```

#### Training 

##### Finetune Document-Level Embeddings
To train the model by yourself, firstly, training finetuned transformer-based models (i.e. XLM-R) are required. Train the following model:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/xlmr-doc-nl-ner-finetune.yaml 
```
You can find the trained models and its finetuned embeddings in `resources/taggers`. Modify the embedding path in `config/doc_ner_nl_best.yaml` with your trained models.

##### Extract Document-Level Features
Please follow the instruction above.

##### Train the Model
Finally, run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/doc_ner_nl_best.yaml
```

#### Prediction 
To use the model predict on your own file, run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/doc_ner_nl_best.yaml --parse --target_dir $dir --keep_order
```
Note that:
 - You may need to preprocess your file with the dummy tags for prediction, please check this [issue](https://github.com/Alibaba-NLP/ACE/issues/12) for more details.
 - You need to pre-extract document-level features of `bert-base-dutch-cased-finetuned-conll2002-ner` and `bert-base-multilingual-cased` embeddings. Please follow [(Optional) Extract Document Features](https://github.com/Alibaba-NLP/ACE#optional-extract-document-features-for-bert-models).


## CoNLL 2002 Spanish
Currently, we release the model for document-level.

### Document-level Model
Download the [doc_ner_es_best.zip](https://1drv.ms/u/s!Am53YNAPSsodg9cTi-HkIfGWHK_DPA?e=qD5BPK) at [OneDrive](https://1drv.ms/u/s!Am53YNAPSsodg810NxHQcrJpcNIOig?e=FRsJNR). Unzip the file and move the unzipped repository to `resources/taggers`. 

Both of the models need the pre-extracted document-level features of `bert-spanish-cased-finetuned-ner` and `bert-base-multilingual-cased`. Pre-extracted features: [bert-spanish-cased-finetuned-ner.hdf5](https://1drv.ms/u/s!Am53YNAPSsodg9cark1_snG4kRDQ9w?e=cNb6pm) and [bert-base-multilingual-cased.hdf5](https://1drv.ms/u/s!Am53YNAPSsodg9ccJzhEa1qMDjtBDw?e=eFNYf7) (Note that [bert-base-multilingual-cased.hdf5](https://1drv.ms/u/s!Am53YNAPSsodg9ccJzhEa1qMDjtBDw?e=eFNYf7) contains the pre-extracted of all languages of CoNLL datasets, so you do not need to download it again if you already downloaded it.). If you want to extract the features by yourself, see [(Optional) Extract Document Features](https://github.com/Alibaba-NLP/ACE#optional-extract-document-features-for-bert-models) for the guide to extract the document-level features. 

#### Evaluation 
To evaluate the accuracy of the model, run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/doc_ner_es_best.yaml --test
```

#### Training

##### Finetune Document-Level Embeddings
To train the model by yourself, firstly, training finetuned transformer-based models (i.e. XLM-R) are required. Train the following model:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/xlmr-doc-es-ner-finetune.yaml 
```
You can find the trained models and its finetuned embeddings in `resources/taggers`. Modify the embedding path in `config/doc_ner_es_best.yaml` with your trained models.

##### Extract Document-Level Features
Please follow the instruction above.

##### Train the Model
Finally, run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/doc_ner_es_best.yaml
```

#### Prediction 
To use the model predict on your own file, run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/doc_ner_es_best.yaml --parse --target_dir $dir --keep_order
```
Note that:
 - You may need to preprocess your file with the dummy tags for prediction, please check this [issue](https://github.com/Alibaba-NLP/ACE/issues/12) for more details.
 - You need to pre-extract document-level features of `bert-spanish-cased-finetuned-ner` and `bert-base-multilingual-cased` embeddings. Please follow [(Optional) Extract Document Features](https://github.com/Alibaba-NLP/ACE#optional-extract-document-features-for-bert-models).
