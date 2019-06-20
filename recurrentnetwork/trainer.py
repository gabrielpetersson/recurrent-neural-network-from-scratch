import time
import numpy as np

from rnngen.misc.tools import id_word
from rnngen.processing.text_to_id import text_to_id
from rnngen.recurrentnetwork.rnn_one_sequence import rnn_trainer
from rnngen.predict.generator import generate_words
from rnngen.word2vec.word2vec import word2vec_trainer
from rnngen.word2vec.embedding_testing import test_embeddings
from rnngen.setupvariables import preprocess_setup, test_emb_setup, \
                               parameters_setup, predict_setup, \
                               word2vec_params_setup

class Generator:

    def __init__(self, text_dir, word2vec_setting='new',
                 emb_dir='embeddings.npy', use_word2vec=True):
        """
        :param text_dir: Directory to text/dataset used in training
        :param word2vec_setting: If 'load' loads previously trained word embeddings
                                 If 'new' creates new word embeddings.
        :param emb_dir: The embeddings are saved/loaded in emb_dir
        :param use_word2vec: If True, use word2vec.
                             If False, use sparse vectors [0, 0, 1]
        """
        self.emb_dir = emb_dir
        self.use_word2vec = use_word2vec
        self.word2vec_setting = word2vec_setting

        # Loads the data.
        # Creates ID vectors, DICTS and removes sentences with rarely occurring words.
        self.data, self.dicts, self.sentences_lengths = load_data(text_dir, preprocess_setup)
        self.vocab_size = len(self.dicts['word_to_id'])

        # Initializes train params
        self.batches = parameters_setup['BATCHES']
        self.nodes = parameters_setup['NODES']
        self.bp_look_back = parameters_setup['BP_LOOK_BACK']
        self.LEARNING_RATE = parameters_setup['LEARNING_RATE']
        self.seconds_between_predict = predict_setup['SECONDS_BETWEEN_PREDICT']

        # Word2vec training
        self.embedding_size = word2vec_params_setup['EMBEDDING_SIZE']
        self.training_type = preprocess_setup['TRAINING_TYPE']

        # Assign additional params
        self.word2vec_params_setup = word2vec_params_setup
        self.word2vec_params_setup['EMB_DIR'] = emb_dir
        self.word2vec_params_setup['WORD2VEC_SETTING'] = word2vec_setting
        self.predict_setup = predict_setup
        self.predict_setup['TRAINING_TYPE'] = self.training_type
        self.predict_setup['NODES'] = self.nodes
        self.predict_setup['USE_WORD2VEC'] = self.use_word2vec

        # Sets input size
        if use_word2vec:
            # Trains embeddings
            self.embeddings = self.word2vec_training()
            input_size = self.embedding_size

        else:
            input_size = self.vocab_size

        # Weights
        self.input_hidden = np.random.randn(input_size, self.nodes) / 100
        self.hidden_hidden = np.random.randn(self.nodes, self.nodes) / 100
        self.hidden_output = np.random.randn(self.nodes, self.vocab_size) / 100
        self.bh = np.zeros((1, self.nodes))
        self.by = np.zeros((1, self.vocab_size))
        self.current_state = np.zeros((1, self.nodes))

        # Temporary weights
        self.t_hidden_output = np.zeros_like(self.hidden_output)
        self.t_hidden_hidden = np.zeros_like(self.hidden_hidden)
        self.t_input_hidden = np.zeros_like(self.input_hidden)
        self.t_by = np.zeros_like(self.by)
        self.t_bh = np.zeros_like(self.bh)

        # Memory for Ada
        self.mhidden_output = np.zeros_like(self.hidden_output)
        self.mhidden_hidden = np.zeros_like(self.hidden_hidden)
        self.minput_hidden = np.zeros_like(self.input_hidden)
        self.mby = np.zeros_like(self.by)
        self.mbh = np.zeros_like(self.bh)

        # Restricts random factors while training.
        np.random.seed(10)

        self.training()

    def word2vec_training(self):
        """
        Loads or creates new word embeddings depending on word2vec_setting.
        :return: word embeddings
        """
        print('Extract of Training Data:')
        print([id_word(sentence, self.dicts['word_to_id']) for sentence in self.data[:3]])

        # Loads already trained embeddings
        if self.word2vec_setting == 'load':
            embeddings = np.load(self.emb_dir)
            if test_emb_setup['USE_TEST_EMBEDDINGS']:
                test_embeddings(embeddings, self.dicts['id_to_word'],
                                self.dicts['word_to_id'], test_emb_setup)

        # Trains new embeddings
        elif self.word2vec_setting == 'new':
            embeddings = word2vec_trainer(self.data, self.word2vec_params_setup,
                                          self.dicts, test_emb_setup)

        else:
            raise Exception('Invalid "word2vec_setting". Use "new" or "load"')

        print('Embeddings Shape:', embeddings.shape, ' Vocab size by embeddings')

        return embeddings

    def accumulate_derivatives(self, delta_weights):
        # Accumulates all deltas through x amount of batches.
        delta_hidden_output = delta_weights['HO']
        delta_hidden_hidden = delta_weights['HH']
        delta_input_hidden = delta_weights['IH']
        delta_by = delta_weights['BY']
        delta_dbh = delta_weights['BH']

        self.t_hidden_output += delta_hidden_output
        self.t_hidden_hidden += delta_hidden_hidden
        self.t_input_hidden += delta_input_hidden
        self.t_by += delta_by
        self.t_bh += delta_dbh

    def training(self):
        print('\nStarting to train recurrent model...')
        total_loss = 0              # Counts loss before normalization of loss
        current_sequence = 0        # One sequence is one sentence. Counts sequences
        last_sequence = 1           # Helps normalize loss
        current_time = time.time()  # Loss will be printed every X seconds. Starts at current_time

        # Infinite Epochs. Will train until turned off.
        while True:
            x = self.data[current_sequence][:-1]                        # Xs
            y = self.data[current_sequence][1:]                         # Ys
            sentence_length = self.sentences_lengths[current_sequence]  # Length of sentence

            # Checks if needed to start over from start
            if current_sequence > len(self.data) - 10:
                current_sequence = 0
                print('---------All processing trained. Starting from beginning again.----------')

            # self.by = np.zeros_like(self.by)
            # self.bh = np.zeros_like(self.bh)

            # Training of model, returns deltas
            delta_weights, loss = rnn_trainer(
                x, y,
                self.input_hidden, self.hidden_hidden,
                self.hidden_output, self.bh,
                self.by, self.nodes,
                sentence_length, self.dicts,
                self.bp_look_back, self.use_word2vec,
                self.embeddings)

            self.accumulate_derivatives(delta_weights)   # Collects all deltas if batches
            total_loss += loss/sentence_length  # Calculates loss w.r.t. length of processing

            # Updates weights every self.batches sequences.
            if current_sequence % self.batches == 0:
                # Alters weights with help of Adagrad. m is prefix for memory
                for weights, d_weights, mem in zip(
                        [self.hidden_output, self.hidden_hidden, self.input_hidden, self.by, self.bh],
                        [self.t_hidden_output, self.t_hidden_hidden, self.t_input_hidden, self.t_by, self.t_bh],
                        [self.mhidden_output, self.mhidden_hidden, self.minput_hidden, self.mby, self.mbh]):
                    mem += d_weights * d_weights
                    weights += -self.LEARNING_RATE * d_weights / np.sqrt(mem + 0.00000001)

                # Standard SGD. Can be uncommented.
                # hidden_output -= (t_hidden_output) / batches
                # hidden_hidden -= (t_hidden_hidden) / batches
                # input_hidden -= (t_input_hidden) / batches
                # by -= (Tby) / batches
                # bh -= (Tbh) / batches

                # Reset temp gradients
                self.t_hidden_output = 0
                self.t_hidden_hidden = 0
                self.t_input_hidden = 0
                self.t_by = 0
                self.t_bh = 0

            # Checks if X seconds has passed by. If True, print loss
            if current_time + self.seconds_between_predict < time.time():
                current_time = time.time()
                # self.by = np.zeros_like(self.by)
                # self.bh = np.zeros_like(self.bh)

                # Predicts words
                created = generate_words(
                                        self.dicts['word_to_id']['#START#'],
                                        self.input_hidden, self.hidden_hidden,
                                        self.bh, self.by, self.hidden_output,
                                        self.dicts, self.embeddings,
                                        self.predict_setup)

                # Prints and resets loss
                print('\nloss: ' + str(round(total_loss / self.batches
                                           / (current_sequence - last_sequence), 2)), created)
                self.LEARNING_RATE = self.LEARNING_RATE * 0.995  # Decay learning rate
                last_sequence = current_sequence
                total_loss = 0

            current_sequence += 1  # One sequence has passed


def load_data(pickle_dir, prepro_setup):
    # Loads and makes data trainable
    processed_text, dic, reversed_dic, sentences_lengths = text_to_id(
        pickle_dir, prepro_setup)
    # Creates a dict of word dictionaries
    dicts = {'word_to_id': dic, 'id_to_word': reversed_dic}

    return processed_text, dicts, sentences_lengths