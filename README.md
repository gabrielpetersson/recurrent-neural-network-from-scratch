# Recurrent Neural Network Generator

RnnGen is a generative natural language processing program using Word2Vec. 
Everything is built from scratch for learning purposes, using numpy for machine learning math.</br>
With RnnGen, you can use an already implemented or your own set of text, create word vector embeddings of it and</br>
train your own word generator!

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install RnnGen.

```bash
pip install rnngen
```
## Dependencies
numpy=<1.15.4</br>
matplotlib=<3.0.2 </br>
sklearn=<0.0 (only for cosine similarity)

## Usage

```python
import rnngen  # Imports package
from rnngen import datasets  # Imports dataset directories

# Will start training the model with default settings and embeddings using word2vec. 
# Only datasets.SIMPSONS_PROCESSED will work if using default embeddings_dir parameter.
rnngen.generate(datasets.SIMPSONS_PROCESSED)
```


## Advanced Usage
All parameters should be tuned in setupvariables.py for optimal results.

```python
import rnngen
from rnngen import datasets

# To use your own data, you will need a txt file at least the size of a book, 
# and some variables might need to get changed. Read setupvariables.py for more info.

# rnngen.pre_process will process your txt file into a clean txt file.
rnngen.pre_process('dir_to_your_file', 'dir_to_save_file')

# You can access already processed datasets: datasets.SIMPSONS_PROCESSED and datasets.BOOKS_PROCESSED.
# SIMPSONS_PROCESSED contains 300 episode manuscripts of The Simpsons and BOOKS_PROCESSED contains 25 books.

# To train Word2Vec, pass in a directory to a processed file and directory where to save the word embeddings.
# The term Word Embeddings are explained in 'Knowledge Prerequisites' below.
rnngen.word2vec('dir_to_processed_file', 'dir_to_save_embeddings')

# You can access already trained embeddings in datasets.SIMPSONS_EMBEDDINGS_300 or 100. Yet to come BOOKS_EMBEDDINGS

# To train model, pass in directory to the processed file(must be same as in word2vec), and 
# specify the word2vec embeddings in embeddings_dir. 
# If you set use_word2vec=False, sparse vectors will be used instead. (which are slow and boring)
# The default embeddings_dir is SIMPSONS_EMBEDDINGS_300, and only works with SIMPSONS_PROCESSED
rnngen.generate('dir_to_processed_file',  use_word2vec=True, embeddings_dir='dir_to_embeddings')

#Example of full usage using datasets.BOOKS (which is './rnngen/resources/datasets/books.txt')
import rnngen
from rnngen import datasets as d

rnngen.pre_process(d.BOOKS, 'processed_text')
rnngen.word2vec('processed_text', 'word_embeddings')
rnngen.generate('processed_text', embeddings_dir='word_embeddings') # Must use SAME processed_text as in word2vec

```

## Outputs

### Word2Vec

```
Loss: 0.2342 [1.46, 0.4549, 0.3594, 0.3191, 0.256, 0.2449]  # Current loss followed by earlier losses
Iter: 600000 of 646860  # Current iteration
Epoch: 9 of 10  # Current epoch
he | she:  0.8017 # he and she are 2 tested word embeddings, followed by a high cosine similarity. (Similarity of words)
almost | tv:  0.0279 # almost and tv has way lower cosine similarity than he and she, therefore low similarity.
problem | window:  0.1334 # Sometimes interchangeable, 'i have a problem/window', therefore medium cosine similarity.
```
These word similarities are trained over time and are nonsense in the start.

### Test of embeddings
When testing embeddings after a training session with rnngen.word2vec(), test_embeddings will test 10 of the embeddings with 5 other each, so the user can see if it has created good embeddings. The highest similarity words should be interchangable with the tested word.
```
These words have the highest cosine similarity (most similar) to "great".
great 1.0  # The embedding of great is exactly the same as itself (no suprise)
good 0.762  # The embedding of great is very similar to the embedding of 'good'. They are nearly always interchangable
terrible 0.567  # Even though terrible has the opposite meaning, both can still be used at the same places.
wonderful 0.553  
perfect 0.456

# More interchangeable examples
These words have the highest cosine similarity (most similar) to "he".
he 1.0
she 0.935
youve 0.865
ive 0.832
weve 0.831

These words have the highest cosine similarity (most similar) to "cat".
cat 1.0
dog 0.752
clown 0.486  # Although very different words, they are both objects, and therefore have high similarity.
bus 0.454
husband 0.423

```

### Generated Sentences
When training this recurrent network we want it to generate sentences for us. It will every 10 seconds(if default settings) predict a new row of words, and improve with time. After a time (a few days for good quality), a few legitimate sentences will start popping up.
```
loss: 4.9 #START# seeing everybody eating ruined you and me # Loss and after that is the predicted text. Loss will go down with time
          #START# im sorry , nice and think opening  
          
loss: 6.82 #START# relax pretty cheese guy  # This one is predicted with a probability distribution
           #START# im what school say  # This one predicts independently, and is greedy, taking the word with highest probability 
```

## All callable modules

### rnngen.word2vec('dir_to_processed_text', 'dir_to_save_embeddings')
Word2Vec can be trained with rnngen.word2vec('some_processed_text', 'save_embeddings_dir'), and it creates word embeddings based on the text. For good quality, the model should train for at least 24 hours (On a mediocre school computer). 
Already trained embeddings are: datasets.SIMPSONS_EMBEDDINGS

### rnngen.generate('dir_to_processed_text', use_word2vec=True, embeddings_dir=datasets.SIMPSONS_EMBEDDINGS)
Given a sentence 'the mouse caught a', the model will try to predict the next word. Hopefully it gives the word "mouse" a high probability, but if not, the model will change parameters so that next time a similar sentence appear, it will predict something similar. This is then repeated for a few hours, until it generates legitimate sentences.

### rnngen.pre_process('dir_to_text', 'dir_to_save_processed_text')
Cleans sentences from trash, and creates high quality easy to work with text.</br>
Already processed texts are: datasets.SIMPSONS_PROCESSED and datasets.BOOKS_PROCESSED

## Knowledge prerequisites for understanding
### Word2Vec:
Instead of sparse vector representations of a word, you can have a dense representation. These are called embeddings. </br>
Link for more info: https://en.wikipedia.org/wiki/Word2vec
#### Common words
Embedding: A vector of size 300 (default) that represent a word. Embeddings in plural is a matrix (vocabulary size, embedding size)</br>
Cosine Similarity: Is used for testing embeddings. The cosine similarity between two word vectors is the semantic difference.
A cosine similarity of 1 means interchangable words, and a cosine similarity of -1 means completely different words. The words 'he' and 'she' should have close to 1.
</br></br>

### Recurrent neural network:
Instead of normal neural networks, recurrent neural networks can be used to look not just at the current input, but at inputs and probabilities in the past, and is therefore good for forecasting or creating generative models that
depends on earlier inputs, such as text. You need to have the past context to keep on writing.</br>
Link for more info: https://medium.com/explore-artificial-intelligence/an-introduction-to-recurrent-neural-networks-72c97bf0912

#### Common words
Backpropagation look back (bp_look_back): The number of words the model will look back at while training. If set to 5, the model
will look at the last 4 words and use them to predict the next one.</br>
Hidden state: The special part with recurrent networks is the hidden state, which saves information from past training and uses the information to hopefully generate legitimate sentences.
</br>

## Important parameters in setupvariables.py

### parameters_setup:
Is responsible for parameters when training Rnn, and is essential to tune for good results.

### preprocess_setup:
##### 'WORD_THRESHOLD'
Removes all sentences that contains word that appear less than 'WORD_THRESHOLD' times.
It is defaulted at 40 because of huge datasets, but if you try on your own smaller dataset,
this needs to be lowered.
</br>
##### 'TRAINING_TYPE'
Is defaulted to 'words' but can be set to 'letters'. When set to 'letters', the generative
model will divide the words into letters instead, and predict letter for letter. If 'letters', do not
forget to increase 'BP_LOOK_BACK'. If it is set to 4, it will only be able to look 4 letters back in
time and can therefore not create any complex words. Also set 'MIN_WORDS' to around 20 to prevent short sentences.

### word2vec_params_setup:
Contains all word2vec training parameters. Can be tuned so training takes shorter time and more optimal
learning rate decay, but it can also increase quality on word embeddings.

### More info about each variable is in the setupvariables.py file.



## Conclusion and what I have learned
Word2Vec is a very interesting form of representing words, and by experimenting with them, there is a lot to be learned. For example, if you would take the embedding of king, subtract man and add woman, the embedding would become very similar to queen. (Can be done with more sophisticated embeddings for example google) </br>

Recurrent neural networks do find some correlations in texts, for example knows when to stop and when to put a comma, and how some word groups appear after another, but the recurrent algorithm with hidden states is not powerful enough to create longer meaningful sentences based on other text. But I'm still amazed that an the algorithms can make sense of spoken english and even create some of its own, even of limited quality.</br>

To do the backpropagation with numpy in the recurrent neural network has really helped me to understand the mecahnics of backpropagation and how to deploy it on different algorithms.

## TODO/What could be done in the future
The next obvious step is to evolve the vanilla recurrent algorithm into a GRU or LSTM, which are similar algorithms but that has much better memory of what have happened in the past, which means longer meaningful sentences can be generated. If we also combine this with a attention mechaninism (which essentially is something that reminds the algorithm of the subject it generates text on) we could create long texts and even articles, which is already made today. https://digiday.com/media/washington-posts-robot-reporter-published-500-articles-last-year/</br></br>

Other improvements that are of diminishing returns are: Increase text data, training all models for longer with increased nodes and test different processing teqchniques.

