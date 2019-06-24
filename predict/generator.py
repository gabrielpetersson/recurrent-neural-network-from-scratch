import numpy as np


def generate_words(seed, input_hidden, hidden_hidden, bh, by, hidden_output, dicts, emb, predict_setup):
    """
    :param seed: The first word to predict on
    :param input_hidden: Weight
    :param hidden_hidden: Weight
    :param bh: Weight
    :param by: Weight
    :param hidden_output: Weight
    :param dicts: Dictionaries in a dict
    :param emb: embeddings
    :param predict_setup: setup variables
    :print: Prints a generated sentence string of predicted words.
    """
    # Initiate variables and constants
    samples = predict_setup['SAMPLES']
    training_type = predict_setup['TRAINING_TYPE']
    nodes = predict_setup['NODES']
    use_word2vec = predict_setup['USE_WORD2VEC']
    vocab_size = len(dicts['id_to_word'])

    # Embeddings and sparse vector has different inputs
    if training_type == 'words':
        if use_word2vec:
            x = emb[seed]

        else:
            x = np.zeros((vocab_size,))
            x[seed] = 1
        x2 = x

        predicted_ids = []
        predicted_ids2 = []
        h = np.zeros((1, nodes))
        h2 = h
        # Generates SAMPLES amount of words
        for t in range(samples):
            # Forward propagation
            h = np.tanh(np.dot(x, input_hidden) + np.dot(hidden_hidden, h[0]) + bh)
            y = np.dot(h, hidden_output) + by

            h2 = np.tanh(np.dot(x2, input_hidden) + np.dot(hidden_hidden, h2[0]) + bh)
            y2 = np.dot(h2, hidden_output) + by

            # p2 shows what is the word given highest probability
            p = np.exp(y) / np.sum(np.exp(y))
            p2 = np.argmax(y2)

            # Chooses a word based on probability
            predicted_id = np.random.choice(range(vocab_size), p=p.ravel())
            predicted_id2 = p2

            if use_word2vec:
                x = emb[predicted_id]
                x2 = emb[predicted_id2]
            else:
                x = np.zeros((1, vocab_size))
                x[0][predicted_id] = 1
                x2 = np.zeros((1, vocab_size))
                x2[0][predicted_id2] = 1

            predicted_ids.append(predicted_id)
            predicted_ids2.append(predicted_id2)

            # If it generates a #END# tag, print the generated sentence
            if predicted_id == dicts['word_to_id']['#END#']:
                return dicts['id_to_word'][seed] + ' ' \
                    + ' '.join([dicts['id_to_word'][x] for x in predicted_ids]) \
                    + '\n           ' + dicts['id_to_word'][seed] + ' ' \
                    + ' '.join([dicts['id_to_word'][x] for x in predicted_ids2])

        # Returns generated text if max number of samples has been reached.
        return dicts['id_to_word'][seed] + ' ' + ' '.join([dicts['id_to_word'][x] for x in predicted_ids]) \
            + '\n            ' + dicts['id_to_word'][seed] + ' ' \
            + ' '.join([dicts['id_to_word'][x] for x in predicted_ids2[:-1]])

    # Same as words but with letters. Everything is the same unless return statement, I am yet to merge these 2
    elif training_type == 'letters':
        if use_word2vec:
            x = emb[seed]
        else:
            x = np.zeros((vocab_size,))
            x[seed] = 1

        predicted_ids = []
        h = np.zeros((1, nodes))

        for t in range(samples):
            h = np.tanh(np.dot(x, input_hidden) + np.dot(hidden_hidden, h[0])+bh)
            y = np.dot(h, hidden_output) + by
            p = np.exp(y)/np.sum(np.exp(y))
            predicted_id = np.random.choice(range(vocab_size), p=p.ravel())

            if use_word2vec:
                x = emb[predicted_id]
            else:
                x = np.zeros((1, vocab_size))
                x[0][predicted_id] = 1

            predicted_ids.append(predicted_id)
            if predicted_id == dicts['word_to_id']['#END#']:
                return dicts['id_to_word'][seed] + ' ' + ''.join([dicts['id_to_word'][x] for x in predicted_ids[:-1]])

        return dicts['id_to_word'][seed] + ''.join([dicts['id_to_word'][x] for x in predicted_ids])


    else:
        print('Incorrect selection of TRAINING_TYPE')
