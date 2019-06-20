import random
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def test_embeddings(embeddings, id_to_word, word_to_id, test_emb_setup):
    """
    :param embeddings: Word embeddings
    :param id_to_word: Dictionary from id to word
    :param word_to_id: Dictionary from word to id
    :param test_emb_setup: Setup variables
    :prints: A chosen amount of words and their closest cosine distance similarities
    """
    vocab_size = embeddings.shape[0]
    common_word_indexes = []
    try:
        common_word_indexes = [word_to_id['he'], word_to_id['you'],
                               word_to_id['cat'], word_to_id['great']]
    except KeyError:
        pass

    indexes_to_test = [random.randint(0, vocab_size - 1) for _ in
                       range(test_emb_setup['NUM_TEST_EMBEDDINGS'])] + common_word_indexes
    for similar in indexes_to_test:
        ranking = []
        ranking_ind = []
        ranking1 = []
        for ind in range(vocab_size):
            answer1 = abs(sum(embeddings[similar] - embeddings[ind]))
            answer = cosine_similarity(embeddings[similar].reshape(1, -1), embeddings[ind].reshape(1, -1))[0][0]
            ranking.append(answer)
            ranking_ind.append(ind)
            ranking1.append(answer1)
        ranks1 = np.argsort(ranking1)[:5][::-1]
        ranks = np.argsort(ranking)[-5:]
        print('\nThese words are similar to "{}".'.format(id_to_word[similar]))
        for i, j in zip(ranks, ranks1):
            print(id_to_word[i], id_to_word[j])
            pass
