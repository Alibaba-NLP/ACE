# Tutorial 9: Training your own Flair Embeddings

Flair Embeddings are the secret sauce in Flair, allowing us to achieve state-of-the-art accuracies across a
range of NLP tasks.
This tutorial shows you how to train your own Flair embeddings, which may come in handy if you want to apply Flair
to new languages or domains.


## Preparing a Text Corpus

Language models are trained with plain text. In the case of character LMs, we train them to predict the next character
in a sequence of characters.
To train your own model, you first need to identify a suitably large corpus. In our experiments, we used corpora that
have about 1 billion words.

You need to split your corpus into train, validation and test portions.
Our trainer class assumes that there is a folder for the corpus in which there is a 'test.txt' and a 'valid.txt' with
test and validation data.
Importantly, there is also a folder called 'train' that contains the training data in splits.
For instance, the billion word corpus is split into 100 parts.
The splits are necessary if all the data does not fit into memory, in which case the trainer randomly iterates through
all splits.

So, the folder structure must look like this:

```
corpus/
corpus/train/
corpus/train/train_split_1
corpus/train/train_split_2
corpus/train/...
corpus/train/train_split_X
corpus/test.txt
corpus/valid.txt
```


## Training the Language Model

Once you have this folder structure, simply point the `LanguageModelTrainer` class to it to start learning a model.

```python
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

# are you training a forward or backward LM?
is_forward_lm = True

# load the default character dictionary
dictionary: Dictionary = Dictionary.load('chars')

# get your corpus, process forward and at the character level
corpus = TextCorpus('/path/to/your/corpus',
                    dictionary,
                    is_forward_lm,
                    character_level=True)

# instantiate your language model, set hidden size and number of layers
language_model = LanguageModel(dictionary,
                               is_forward_lm,
                               hidden_size=128,
                               nlayers=1)

# train your language model
trainer = LanguageModelTrainer(language_model, corpus)

trainer.train('resources/taggers/language_model',
              sequence_length=10,
              mini_batch_size=10,
              max_epochs=10)
```

The parameters in this script are very small. We got good results with a hidden size of 1024 or 2048, a sequence length
of 250, and a mini-batch size of 100.
Depending on your resources, you can try training large models, but beware that you need a very powerful GPU and a lot
of time to train a model (we train for > 1 week).



## Using the LM as Embeddings

Once you have the LM trained, using it as embeddings is easy. Just load the model into the `FlairEmbeddings` class and
use as you would any other embedding in Flair:

```python
sentence = Sentence('I love Berlin')

# init embeddings from your trained LM
char_lm_embeddings = FlairEmbeddings('resources/taggers/language_model/best-lm.pt')

# embed sentence
char_lm_embeddings.embed(sentence)
```

Done!


## Non-Latin Alphabets

If you train embeddings for a language that uses a non-latin alphabet such as Arabic or Japanese, you need to create your own character dictionary first. You can do this with the following code snippet: 

```python

# make an empty character dictionary
from flair.data import Dictionary
char_dictionary: Dictionary = Dictionary()

# counter object
import collections
counter = collections.Counter()

processed = 0

import glob
files = glob.glob('/path/to/your/corpus/files/*.*')

print(files)
for file in files:
    print(file)

    with open(file, 'r', encoding='utf-8') as f:
        tokens = 0
        for line in f:

            processed += 1            
            chars = list(line)
            tokens += len(chars)

            # Add chars to the dictionary
            counter.update(chars)

            # comment this line in to speed things up (if the corpus is too large)
            # if tokens > 50000000: break

    # break

total_count = 0
for letter, count in counter.most_common():
    total_count += count

print(total_count)
print(processed)

sum = 0
idx = 0
for letter, count in counter.most_common():
    sum += count
    percentile = (sum / total_count)

    # comment this line in to use only top X percentile of chars, otherwise filter later
    # if percentile < 0.00001: break

    char_dictionary.add_item(letter)
    idx += 1
    print('%d\t%s\t%7d\t%7d\t%f' % (idx, letter, count, sum, percentile))

print(char_dictionary.item2idx)

import pickle
with open('/path/to/your_char_mappings', 'wb') as f:
    mappings = {
        'idx2item': char_dictionary.idx2item,
        'item2idx': char_dictionary.item2idx
    }
    pickle.dump(mappings, f)
```

You can then use this dictionary instead of the default one in your code for training the language model: 

```python
import pickle
dictionary = Dictionary.load_from_file('/path/to/your_char_mappings')
```

## Parameters

You might to play around with some of the learning parameters in the `LanguageModelTrainer`.
For instance, we generally find that an initial learning rate of 20, and an annealing factor of 4 is pretty good for
most corpora.
You might also want to modify the 'patience' value of the learning rate scheduler. We currently have it at 25, meaning
that if the training loss does not improve for 25 splits, it decreases the learning rate.


## Fine-Tuning an Existing LM

Sometimes it makes sense to fine-tune an existing language model instead of training from scratch. For instance, if you have a general LM for English and you would like to fine-tune for a specific domain. 

To fine tune a `LanguageModel`, you only need to load an existing `LanguageModel` instead of instantiating a new one. The rest of the training code remains the same as above:

```python
from flair.data import Dictionary
from flair.embeddings import FlairEmbeddings
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus


# instantiate an existing LM, such as one from the FlairEmbeddings
language_model = FlairEmbeddings('news-forward').lm

# are you fine-tuning a forward or backward LM?
is_forward_lm = language_model.is_forward_lm

# get the dictionary from the existing language model
dictionary: Dictionary = language_model.dictionary

# get your corpus, process forward and at the character level
corpus = TextCorpus('path/to/your/corpus',
                    dictionary,
                    is_forward_lm,
                    character_level=True)

# use the model trainer to fine-tune this model on your corpus
trainer = LanguageModelTrainer(language_model, corpus)

trainer.train('resources/taggers/language_model',
              sequence_length=100,
              mini_batch_size=100,
              learning_rate=20,
              patience=10,
              checkpoint=True)
```              
              
Note that when you fine-tune, you must use the same character dictionary as before and copy the direction (forward/backward).


## Consider Contributing your LM

If you train a good LM for a language or domain we don't yet have in Flair, consider contacting us! We would be happy
to integrate more LMs into the library so that other people can use them!



