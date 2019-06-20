import numpy as np


def text_to_id(filename, setup):
    """
    Replaces words with IDs and divides text into sentences.
    Also removes sentences that contain rarely used words.

    :param filename: What dir to open text file
    :param setup: setup variables
    :return: returns matrix with IDs of words, sentence lengths and dicts.
    """
    print('\nText to ID starting...')
    min_words = setup['MIN_WORDS']
    threshold = setup['WORD_THRESHOLD']
    letter_words = setup['TRAINING_TYPE']
    i = 0

    if letter_words == 'words':
        sentences = open(filename, 'r').read().split('.')

        # Control number of words in sentences.
        seg = []
        data = []
        for i, sentence in enumerate(sentences):
            splitted_sentence = sentence.split(' ')
            if len(splitted_sentence) > min_words:
                data.append(splitted_sentence[1:] + ['.'])
                seg.extend(splitted_sentence)

        # Create dict
        seg = [x for x in seg if x != '']
        word_dict = {}
        count_dict = {}
        unique_words = list(set(seg))

        for j, i in enumerate(unique_words):
            if i != '':
                word_dict[i] = j
                count_dict[i] = 0

        # Count occurrences
        for word in seg:
            count_dict[word] += 1

        # Remove low frequency words
        for item, value in count_dict.items():
            if value < threshold:
                word_dict.pop(item, None)

        # Create final word dict.
        final_word_dict = {}
        for i, key in enumerate(word_dict.keys()):
            final_word_dict[key] = i

        # Add words to simplify training
        words_to_add = ['#END#', '.', '#START#', '#UNK#']
        for word in words_to_add:
            i += 1
            final_word_dict[word] = i

        del seg

        # Creates a matrix where rows are sentences and columns are words.
        sentence_lengths = []
        all_sentences = []
        for sentence_nr, sentence in enumerate(data):
            one_sentence = []
            unk = False
            for word_nr, word in enumerate(sentence):

                if word in final_word_dict:
                    one_sentence.append(final_word_dict[word])
                elif word != '':
                    unk = True

            if not unk:
                one_sentence = [final_word_dict['#START#']] + one_sentence[:-1] \
                               + [final_word_dict['.']] + [final_word_dict['#END#']]
                sentence_lengths.append(len(one_sentence))
                all_sentences.append(one_sentence)

        # Make count dict a list counts so we can weigh words when training
        rev_dic = {value: letter for letter, value in final_word_dict.items()}

        print('Unique words:', len(final_word_dict))
        print('Number of sentences:', len(all_sentences))
        print('Processing Done.\n')
        return np.array(all_sentences), final_word_dict, rev_dic, sentence_lengths

    if letter_words == 'letters':
        sentences = open(filename, 'r').read().split('.')

        # Creates DICT
        letter_dic = {}
        nr = 0
        for sentence in sentences:
            for char in sentence[1:]:
                if char not in letter_dic:
                    letter_dic[char] = nr
                    nr += 1

        # Creates matrix with IDs instead of words.
        sentence_lengths = []
        all_sentences = []
        for sentence_nr, sentence in enumerate(sentences):
            one_sentence = []
            for char_nr, char in enumerate(sentence):
                if char in letter_dic:
                    one_sentence.append(letter_dic[char])

            one_sentence = [letter_dic['#START#']] + one_sentence[1:-1] + [letter_dic['.']] + [letter_dic['#END#']]
            sentence_lengths.append(len(one_sentence))
            all_sentences.append(one_sentence)

        rev_dic = {value: letter for letter, value in letter_dic.items()}
        return np.array(all_sentences), letter_dic, rev_dic, sentence_lengths
