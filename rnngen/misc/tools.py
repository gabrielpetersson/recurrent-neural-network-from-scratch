import numpy as np


def vec_word(word_vecs, dic, dim=2, rev=False):
    if rev:
        dic = {value: letter for letter, value in dic.items()}
    if dim == 1:
        res = dic[np.argmax(word_vecs)]
        return res
    if dim == 2:
        res = ''
        for letter in word_vecs:
            res = res + dic[np.argmax(letter)] + ' '
        return res
    if dim == 3:
        res = ''
        for letter in word_vecs:
            for let in letter:
                res = res + dic[np.argmax(let)] + ' '
            res = res + '\n'
        return res


def id_word(letters, dic, dim=2):
    dic = {value: letter for letter, value in dic.items()}
    if dim == 1:
        res = dic[letters]
        return res
    if dim == 2:
        res = ''
        for letter in letters:
            res = res + dic[letter] + ' '
        return res
    if dim == 3:
        res = ''
        for letter in letters:
            for let in letter:
                res = res + dic[let]
            res = res + '\n\n'
        return res


def word_id(letters, dic, dim=2):
    if dim == 1:
        res = dic[letters]
        return res
    if dim == 2:
        res = ''
        for letter in letters:
            res = res + dic[letter]
        return res
    if dim == 3:
        res = ''
        for letter in letters:
            for let in letter:
                res = res + dic[let]
            res = res + '\n\n'
        return res
