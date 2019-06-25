import numpy as np


def word_emb_encode(embeddings, ids, dicts, min_max):

    num_emb = embeddings.shape[1]
    m = len(ids)

    data_word_embedded = np.zeros((m, min_max['max_sentence'], num_emb))
    for nr, row_of_ids in enumerate(ids):
        temp = np.zeros((min_max['max_sentence'], num_emb))

        for num, one_id in enumerate(row_of_ids):
            if num == min_max['max_sentence'] - 1 or one_id == dicts['word_to_id']['#PAD#']:
                break

            temp[num] = embeddings[int(one_id)]

        data_word_embedded[nr] = temp

    return data_word_embedded
