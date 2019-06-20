from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
from rnngen.word2vec.embedding_testing import test_embeddings


def get_data(data, window_size=1):
    """
    Creates input training data by looking at words in context.

    If window_size=2 in the sentence "the black cat took all my cookies.
    If we are to find the context words to cat, these will be: the, black, took, all.
    These are xs and the y for all of them is cat.

    :param data: All text to train word2vec on
    :param window_size: How many words to look on beside the chosen one
    :return: training data, one y for every x.
    """
    xs = []
    ys = []
    for sentence in data:
        for ind in range(len(sentence)):
            indexes = list(range(max(ind-window_size, 0), max(ind, 0))) \
                      + list(range(min(ind+1, len(sentence)), min(ind + window_size+1, len(sentence))))
            for y in indexes:
                xs.append(sentence[ind])
                ys.append(sentence[y])

    return np.array(xs), np.array(ys)


def forward_propagation(x_batch, weights):
    # x_batch: batches x emb
    # y_batch: batches x vocab_size
    # weights: emb x vocab_size
    z = np.dot(x_batch, weights)
    softmax = np.exp(z) / (np.sum(np.exp(z)))

    return softmax


def back_propagation(y_batch, softmax, weights, x_batch):
    # Softmax:       64, 4036
    # Weights:       50, 4036
    # Batch Emb:     64, 50
    # d_embeddings:  4036, 50

    d_softmax = softmax - y_batch
    d_weights = np.dot(x_batch.T, d_softmax)
    d_embeddings = np.dot(d_softmax, weights.T)

    return d_weights, d_embeddings


def calculate_loss(softmax, y):

    m = softmax.shape[1]
    cost = -(1 / m) * np.sum(np.sum(y * np.log(softmax)))

    return cost


def one_hot(ys, unique_words):
    # Backprop needs ys in sparse vectors in order to function
    m = len(ys)
    one_hot_y = np.zeros((m, unique_words))
    for row in range(m):
        one_hot_y[row][int(ys[row])] = 1
    return one_hot_y


def get_words_to_check(dicts, test_words, random_words):
    words_to_check = []
    words_to_check += test_words
    print(test_words)
    vocab_size = len(dicts['word_to_id'])
    for _ in range(random_words):
        try:
            random_words1 = dicts['id_to_word'][random.randint(0, vocab_size-4)]
            random_words2 = dicts['id_to_word'][random.randint(0, vocab_size-4)]
            words_to_check.append((random_words1, random_words2))
        except KeyError:
            pass
    print(words_to_check)
    return words_to_check


def print_word_similarity(words_to_check, word_emb, dicts):
    checked_words = []
    for word1, word2 in words_to_check:
        try:
            sim = cosine_similarity(word_emb[dicts['word_to_id'][word1]].reshape(1, -1),
                                    word_emb[dicts['word_to_id'][word2]].reshape(1, -1))[0][0]
            print(f'{word1} | {word2}:   {round(sim, 4)}')
        except KeyError:
            pass
    print('')

    return checked_words


def get_batches(x, y, batches, word_emb, unique_words, iteration):

    x_indexes = x[batches * iteration:batches * iteration + batches]
    x_batch = word_emb[x_indexes.astype(int), :]
    y_batch = one_hot(y[batches * iteration:batches * iteration + batches], unique_words)

    return x_indexes, x_batch, y_batch


def count_print_loss(loss, losses, iters_before_reset, m, batches, epoch, epochs, iteration):

    loss = round(loss / iters_before_reset * 100, 4)
    losses.append(loss)
    print_losses = losses
    if len(losses) > 40:
        print_losses = losses[::3]
    if len(losses) > 200:
        print_losses = losses[::10]
    if len(losses) > 600:
        print_losses = losses[::20]
    if len(losses) > 1200:
        print_losses = losses[::60]
    print('Loss: {}'.format(loss), print_losses)
    print('Iter: {} of {}\nEpoch: {} of {}'.format(iteration, m // batches, epoch + 1, epochs))
    return losses


def word2vec_trainer(data, word2vec_setup, dicts, test_emb_setup):
    print('Word2Vec trainer starting...')
    batches = word2vec_setup['BATCHES']
    embedding_size = word2vec_setup['EMBEDDING_SIZE']
    epochs = word2vec_setup['EPOCHS']
    emb_dir = word2vec_setup['EMB_DIR']
    lr = word2vec_setup['LEARNING_RATE']
    short = word2vec_setup['SHORT_MODE']
    iters_before_decrease = word2vec_setup['ITERATIONS_BEFORE_LR_DECREASE']
    lr_decrease = word2vec_setup['LR_DECREASE']

    use_test_embeddings = test_emb_setup['USE_TEST_EMBEDDINGS']
    verbose_cosine_distance = test_emb_setup['VERBOSE_COSINE']
    test_words = test_emb_setup['TESTING_WORDS']

    unique_words = len(dicts['word_to_id'])
    word_emb = np.random.randn(unique_words, embedding_size)
    weights = np.random.randn(embedding_size, unique_words)
    xs, ys = get_data(data)
    m = len(xs)
    losses = []  # saves accumulated losses
    all_losses = []  # saves all individual losses so it can be printed in matplot
    accumulated_loss = 0  # Accumulates losses before each LR_DECREASE

    assert ys.shape == xs.shape
    loss = 0
    print('Word2vec is up running.\n')
    for epoch in range(epochs):
        for iteration in range(m//batches-3):
            x_indexes, x_batch, y_batch = get_batches(xs, ys, batches, word_emb, unique_words, iteration)
            softmax = forward_propagation(x_batch, weights)
            loss = calculate_loss(softmax, y_batch)
            accumulated_loss += loss
            d_weights, d_word_embeddings = back_propagation(y_batch, softmax, weights, x_batch)

            if iteration % iters_before_decrease == 0 and iteration != 0:
                losses = count_print_loss(accumulated_loss, losses, iters_before_decrease, m, batches, epoch, epochs, iteration)
                accumulated_loss = 0
                lr = lr * lr_decrease
                if verbose_cosine_distance:
                    words_to_check = get_words_to_check(dicts, test_words, random_words=4)
                    print_word_similarity(words_to_check, word_emb, dicts)
                if iteration > 30000 or epoch >= 1:
                    np.save(emb_dir, word_emb)

            if short and iteration > 5:   # Only for developing, easier checking so things work out.
                break

            weights = weights - d_weights * lr / len(x_batch) / batches
            word_emb[x_indexes.astype(int), :] = word_emb[x_indexes.astype(int), :] \
                - (d_word_embeddings * lr / len(x_batch)) / batches

            all_losses.append(loss)
    np.save(emb_dir, word_emb)

    if use_test_embeddings:
        test_embeddings(word_emb, dicts['id_to_word'], dicts['word_to_id'], test_emb_setup)

    plt.plot(all_losses)
    plt.show()

    print('\nOBS: Word vectors must be separately trained for every new change. '
          '\nThey cannot be used for another set of text'
          ' or with another set of preprocessing parameters.\n')
    return word_emb
