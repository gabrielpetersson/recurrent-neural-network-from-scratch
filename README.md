# Recurrent Neural Network Generator

RnnGen is a generative natural language processing program using Word2Vec. 
Everything is built from scratch for learning purposes, using numpy for machine learning math.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install RnnGen.

```bash
pip install rnngen
```

## Usage

```python
import rnngen
from rnngen import datasets

# Processes the dataset "SIMPSON" to directory "processed_simpson".
rnngen.process(datasets.SIMPSON, 'processed_simpson')

# Uses the processed text "processed_simpson" and trains the model with word2vec on it.
rnngen.generate('processed_simpson')
```
### Word2Vec
Word2Vec is integrated in rnngen.generate, and it creates word embeddings based on the text. For good quality, the model should train for at least 24 hours (On a mediocre school computer, may go way faster with a better processor)

### rnngen.generate
Given a sentence 'the mouse caught a', the model will try to predict the next word. Hopefully it gives the word "mouse" a high probability, but if not, the model will change parameters so that next time a similar sentence appear, it will predict something similar. This is then repeated for a few hours.

## Knowledge prerequisites for understanding
### Word2Vec:
Instead of sparse vector representations of a word, you can have a dense representation. These are called embeddings. </br>
Link for more info: https://en.wikipedia.org/wiki/Word2vec
#### Common words
Embedding: A vector of size 300 (default) that represent a word. Embeddings in plural is a matrix (vocabulary size, embedding size)
</br></br>

### Recurrent neural network:
Instead of normal neural networks, recurrent neural networks can be used to look not just at the current input, but at inputs and probabilities in the past, and is therefore good for forecasting or creating generative models that
depends on earlier inputs, such as text. You need to have the past context to keep on writing.</br>
Link for more info: https://medium.com/explore-artificial-intelligence/an-introduction-to-recurrent-neural-networks-72c97bf0912

#### Common words
Backpropagation look back (bp_look_back): The number of words the model will look back at while training. If set to 5, the model
will look at the last 4 words and use them to predict the next one.
</br></br>

## Advanced Usage
All parameters can be tuned and can be changed in setupvariables.py

```python
import rnngen
from rnngen import datasets

# You can access already processed datasets: datasets.SIMPSON_PROCESSED or datasets.BOOKS_PROCESSED
# SIMPSON_PROCESSED contains 300 episode manuscripts of simpsons and BOOKS_PROCESSED contains 20 books.
rnngen.generate(datasets.SIMPSON_PROCESSED)

# text dir(string): Directory to processed text you want to use
# word2vec_setting(string): 'new' creates new word embeddings, 'load' loads an already trained one
# emb_dir(string): Where embeddings will be saved or loaded depending on word2vec_setting
# use_word2vec(Boolean): If True it will use word2vec. If false, it uses sparse word vectors. [0, 0, 1]
# Default settings:
rnngen.generate(text_dir, word2vec_setting='new', emb_dir='embeddings.npy', use_word2vec=True)
```
## Important parameters in setupvariables.py

### parameters_setup:
Is responsible for parameters when training Rnn, and is essential to tune for good results.

### preprocess_setup:
'WORD_THRESHOLD'  removes all sentences that contains word that appear less than 'WORD_THRESHOLD' times.
                  It is defaulted at 40 because of huge datasets, but if you try on your own smaller dataset,
                  this needs to be lowered.
                  </br></br>
                  'TRAINING_TYPE' is defaulted to 'words' but can be set to 'letters'. When set to 'letters', the generative
                  model will divide the words into letters instead, and predict letter for letter. If 'letters', do not
                  forget to increase 'BP_LOOK_BACK'. If it is set to 4, it will only be able to look 4 letters back in
                  time and can therefore not create any complex words. Also set 'MIN_WORDS' to around 20 to prevent short sentences.

### word2vec_params_setup:
Contains all word2vec training parameters. Can be tuned so training takes shorter time and more optimal
                       learning rate decay, but it can also increase quality on word embeddings.

### More info about each variable is in the setupvariables.py file.

## Dependencies
numpy</br>
matplotlib</br>
sklearn (only for cosine distance)
