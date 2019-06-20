import numpy as np


def rnn_trainer(xs, ys, input_hidden, hidden_hidden, hidden_output, bh, by, nodes,
                sentence_length, dicts, bp_look_back, use_word2vec, emb):
    """

    :param xs: Train xs
    :param ys: Train ys
    :param input_hidden: Weight
    :param hidden_hidden: Weight
    :param hidden_output: Weight
    :param bh: Weight
    :param by: Weight
    :param nodes: Number of nodes
    :param sentence_length: Length of sentence
    :param dicts: Dictionaries word to id and id to word
    :param bp_look_back: How many steps back backprop takes
    :param use_word2vec: If word2vec is used
    :param emb: Embeddings
    :return: Returns derivatives and loss
    """
    vocab_size = len(dicts['word_to_id'])
    loss = 0
    assert len(xs) == len(ys)

    # Dicts to be filled up with sequences of guesses, outputs and hidden states
    output = {}
    probabilities = {}
    hs = dict()
    hs[-1] = np.zeros((1, nodes))
    current_state = hs[-1]
    used_xs = {}
    x = None

    # Derivatives
    delta_input_hidden, delta_hidden_hidden, delta_hidden_output, dby, dbh = np.zeros_like(
        input_hidden), np.zeros_like(hidden_hidden), np.zeros_like(hidden_output), np.zeros_like(by), np.zeros_like(bh)

    # Forward propagation
    for seq in range(sentence_length - 1):
        if use_word2vec:
            x = emb[xs[seq]]
            used_xs[seq] = xs[seq]
        else:
            x = np.zeros(vocab_size)
            used_xs[seq] = xs[seq]
            x[used_xs[seq]] = 1

        y = ys[seq]
        inner_hidden = np.dot(x, input_hidden) + np.dot(current_state, hidden_hidden)
        current_state = np.tanh(inner_hidden + bh)
        hs[seq] = current_state

        # Calculate loss and probs
        output[seq] = np.dot(current_state, hidden_output) + by
        probabilities[seq] = np.exp(output[seq])/np.sum(np.exp(output[seq]))
        loss += -np.log(probabilities[seq][0, y])
    # Back propagation, calculates derivatives wrt weights to minimize loss.
    for seq in reversed(range(sentence_length - 1)):
        # Derivative for this step
        y = ys[seq]
        d_output = np.copy(probabilities[seq])
        d_output[0, y] -= 1
        dby += d_output
        delta_hidden_output += np.outer(hs[seq].T, d_output)
        d_hidden_state = np.dot(d_output, hidden_output.T)
        d_inner_hidden_state = d_hidden_state * (1 - (hs[seq] ** 2))

        # Back propagation through time. Goes back bp_look_back steps
        for bptt_step in range(max(0, seq - bp_look_back), seq + 1)[::-1]:
            dbh += d_inner_hidden_state
            delta_hidden_hidden += np.outer(d_inner_hidden_state, hs[bptt_step - 1])
            if use_word2vec:
                delta_input_hidden += np.dot(np.expand_dims(x, 1), d_inner_hidden_state)
            else:
                delta_input_hidden[used_xs[bptt_step], :] += d_inner_hidden_state[0]
            # Update delta for next step
            d_inner_hidden_state = np.dot(hs[bptt_step], hidden_hidden.T) * (1 - hs[bptt_step - 1] ** 2)

        # Cuts derivatives to restrict exploding gradient.
        for params in [delta_hidden_hidden, delta_input_hidden, dby, dbh, delta_hidden_output]:
            np.clip(params, -3, 3, out=params)
    # Returns derivatives
    return {'HO': delta_hidden_output, 'HH': delta_hidden_hidden, 'IH': delta_input_hidden, 'BY': dby, 'BH': dbh}, loss
