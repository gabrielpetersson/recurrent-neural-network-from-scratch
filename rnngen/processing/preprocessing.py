import re


class ProcessData:
    """
    Takes a text USE_TEXT_DIR, cleanses and makes it easier to work with, and saves it to SAVE_TEXT_DIR
    """

    def __init__(self, used_text_dir, save_text_dir):
        print(f'\nPreprocessing "{used_text_dir}" to directory "{save_text_dir}".\n')
        self.save_pickle_dir = save_text_dir
        self.use_text_dir = used_text_dir
        self.create_processed_data()

    def check_for_web_words(self, word):
        if '@' in word:
            return '#USER#'

        if '#' in word:
            return word[1:]

        if 'http' in word:
            return '#WEBPAGE#'

        if word.isdigit():
            return '#NUMBER#'

        if 'haha' in word:
            return 'hahaha'
        else:
            return word

    def check_for_punctuation(self, word):
        '''
        Takes a word and returns the word separated from eventual punctuations.

        :param word: A word
        :return: A word and punctuations, if any.
        '''
        punctuation = None
        ending_punc = []
        extra_words = []

        # List of legit punctuations
        punctuations = [',']
        for punctuation in punctuations:
            if punctuation in word:
                ending_punc += [punctuation]

        if '-' in word:
            words = word.split('-')
            extra_words += [words[1]]
            word = words[0]

        return word.replace(punctuation, ''), ending_punc, extra_words

    def remove_strange_characters(self, word):

        word = word.lower()
        legit_chars = [x for x in "0123456789abcdefghijklmnopqrswtuvxyz-!,?."]

        for char in word:
            if char not in legit_chars:
                word = word.replace(char, '')

        word = word.rstrip(' ').lstrip(' ').strip('\n').lower()
        word = str(''.join(char for char in word if not char.isdigit()))
        word = word.replace("''", '').replace('ooh', 'oh')

        return word

    def create_processed_data(self):
        all_text = ''
        last_word = ''
        text = open(self.use_text_dir, encoding='utf8')
        text = text.read().replace('â', ' ')
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        splitted_text = text.replace('\n', ' ').replace('!', '.').replace('?', '.').split('.')
        total_lines = len(splitted_text)
        for line_num, line in enumerate(splitted_text):
            words = []

            for i, word in enumerate(line.split(' ')):
                word = self.remove_strange_characters(word)
                word, ending_punc, extra_words = self.check_for_punctuation(word)
                word = self.check_for_web_words(word)

                if (word != '') and (not word.isspace()) and (word != last_word):
                    words.append(word)

                    if extra_words:
                        words += extra_words

                    if ending_punc:
                        words[-1] = str(words[-1] + ' ' + ending_punc[0])

                last_word = word

            if (words != '') and (words != ' ') and ('screencaps' not in words) and ('scripts' not in words) and (
                    'episodenext' not in words):
                all_text += ' '.join(words) + ' . '
            if line_num % 10000 == 0:
                print(f'{line_num} of {total_lines}')
        outfile = open(self.save_pickle_dir, 'w', encoding='utf8')
        outfile.write(all_text)
        outfile.close()
        print('Preprocessing done. \n\n')
