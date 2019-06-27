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
Python 3.6.7
numpy=<1.15.4</br>
matplotlib=<3.0.2 </br>
sklearn=<0.0 (only for cosine similarity)

## Usage

```python
import rnngen  # Imports package
from rnngen import datasets  # Imports dataset directories

# Will start training the model with default settings and pretrained embeddings. 
# Only datasets.SIMPSONS_PROCESSED will work if using default embeddings_dir parameter.
rnngen.generate(datasets.SIMPSONS_PROCESSED)
```


## Advanced Usage
All parameters should be tuned in setupvariables.py for optimal results. To use your own data, you will need a txt file at least the size of a book, and some variables might need to get changed. Read setupvariables.py for more info.</br>


### Preprocessing
Preprocessing of data makes the text clean and workable.</br>
pre_process takes a text directory and a directory to to save processed text.
```python
import rnngen

rnngen.pre_process('dir_file_to_process', 'dir_to_save_processed_text')
```
```python
```
#### Expected output:
```
Preprocessing 'dir_file_to_process' to directory 'dir_to_save_processed_text'
0 of 14000
10000 of 14000
...
Preprocessing done.
```
You can now access your processed file and use it to train word2vec and generator. (The same processed text MUST be used to generate words and to train word2vec.) </br>
### Word2Vec training
Word2Vec uses a processed text to train word embeddings, The term Word Embeddings are explained in 'Knowledge Prerequisites' below.</br>
Already trained and ready to use datasets can be used with datasets.SIMPSONS_PROCESSED (400 episodes of simpsons) and datasets.BOOKS_PROCESSED(25 books).</br>
Word2Vec takes a processed text and a directory to to save embeddings.

```python
rnngen.word2vec('dir_to_processed_file', 'dir_to_save_embeddings')
```
or to use already processed data:
```python
rnngen.word2vec(datasets.SIMPSONS_PROCESSED, 'dir_to_save_embeddings')
```
</br>
#### Expected output:
While training, word2vec will continuously verbose loss, earlier losses to keep track, iteration and word2vec cosine similarity (explained in Understanding Prerequities). The cosine similarity will take 2 random words and see how similar they are, and for us humans to judge the quality of the embeddings. He and She is always verbosed and should be as close to 1 as possible.
```
Loss: 0.2342 [1.46, 0.4549, 0.3594, 0.3191, 0.256, 0.2449]  # Current loss followed by earlier losses
Iter: 600000 of 646860  # Current iteration
Epoch: 9 of 10  # Current epoch
he | she:  0.8017 # 2 tested word embeddings, followed by a high word similarity.
almost | tv:  0.0279 # These are two very different words, therefore low similarity.
problem | window:  0.1334 # Can often be used interchangeably, therefore medium cosine similarity.
```
After training is done, test_embeddings will be called. It will take 10 random words, and print out the 5 most similar to them.</br>
The metric here is also cosine similarity. If these look correct/similar, your embeddings are probably good.
```
These words have the highest cosine similarity (most similar) to "great".
great 1.0  # The embedding of great is exactly the same as itself (no suprise)
good 0.762  # The embedding of great is very similar to the embedding of 'good'. 
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
### Generator training
The generator generates sentences using word embeddings. It takes a processed text and directory to embeddings. You can also choose to train using sparse vectors, by setting word2vec to False. </br>If embeddings_dir is not specified to something, it will use datasets.SIMPSONS_EMBEDDINGS_300 as default, and then only datasets.SIMPSONS_PROCESSED can be used as text.</br></br>
Default: rnngen.generate('dir_to_processed_file',  use_word2vec=True, embeddings_dir=datasets.SIMPSONS_EMBEDDINGS_300)</br>
The pretrained embeddings are: datasets.SIMPSONS_EMEBDDINGS_300 and SIMPSONS_EMBEDDINGS_100 (300 respectively 100 embedding_size)</br></br>The model will have to train for a few days do give good results.
```python
import rnngen
from rnngen import datasets

rnngen.generate('dir_to_processed_text',  use_word2vec=True, embeddings_dir=datasets.SIMPSONS_EMBEDDINGS_300)
```

#### Expected output
It will verbose every 10 seconds a loss and two predicted text. The upper text is chosen using a probability distribution and the below one is independently predicted using a greedy 'take the one with highest probability'.
```
loss: 6.82 #START# relax pretty cheese guy 
           #START# im what school say  
           
loss: 4.9 #START# seeing everybody eating ruined you and me 
          #START# im sorry , nice and think opening  
```

</br>
### Example of full usage
```python
#Example using datasets.BOOKS (which is './rnngen/resources/datasets/books.txt')
import rnngen
from rnngen import datasets

rnngen.pre_process(datasets.BOOKS, 'processed_text')
rnngen.word2vec('processed_text', 'word_embeddings')
rnngen.generate('processed_text', embeddings_dir='word_embeddings') # Must use SAME processed_text as in word2vec

```
</br>

## Knowledge prerequisites for understanding
### Word2Vec:
Instead of sparse vector representations of a word, you can have a dense representation. These are called embeddings. </br>
Link for more info: https://en.wikipedia.org/wiki/Word2vec
#### Common words
Embedding: A vector of size 300 (default) that represent a word. Embeddings in plural is a matrix (vocabulary size, embedding size)</br></br>
Cosine Similarity: Is used for testing embeddings. The cosine similarity between two word vectors is the semantic difference.
A cosine similarity of 1 means interchangable words, and a cosine similarity of -1 means completely different words. The words 'he' and 'she' should have close to 1.
</br></br>

### Recurrent neural network:
Instead of normal neural networks, recurrent neural networks can be used to look not just at the current input, but at inputs and probabilities in the past, and is therefore good for forecasting or creating generative models that
depends on earlier inputs, such as text. You need to have the past context to keep on writing.</br>
Link for more info: https://medium.com/explore-artificial-intelligence/an-introduction-to-recurrent-neural-networks-72c97bf0912

#### Common words
Backpropagation look back (bp_look_back): The number of words the model will look back at while training. If set to 5, the model
will look at the last 4 words and use them to predict the next one.</br></br>
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

