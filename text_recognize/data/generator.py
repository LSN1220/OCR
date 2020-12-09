# -*- coding: utf-8 -*-
import os
import sys
import threading
import numpy as np
from keras import backend as K
from keras.callbacks import Callback
sys.path.append("..")
import crnn_config as cfg
from utils import paint_text

lock = threading.Lock()


def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    ret = []
    assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val
    assert stop_ind <= len_val

    a = list(range(stop_ind))
    np.random.shuffle(a)
    a += list(range(stop_ind, len_val))
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
        else:
            raise TypeError('`shuffle_mats_or_lists` only supports '
                            'numpy.array and list objects.')
    return ret


def text_to_labels(text, unichar_list):
    res = []
    for char in text:
        res.append(unichar_list.index(char))
    if text == '':
        res.append(len(unichar_list))
    return res


class TextImageGenerator(Callback):
    def __init__(self, batch_size,
                 img_w, img_h, downsample_factor, val_split,
                 absolute_max_string_len=128):
        self.build_id = 0
        self.batch_size = batch_size
        self.img_w = img_w
        self.img_h = img_h
        self.downsample_factor = downsample_factor
        self.val_split = val_split
        self.absolute_max_string_len = absolute_max_string_len
        self.n_class = 0
        self.chars = []
        with open(cfg.uni_chars_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 空格在最后一位
                char = line.replace('\n', '').replace('\r', '')
                self.chars.append(char)
        self.n_class = len(self.chars)

    def get_output_size(self):
        return len(self.chars) + 1

    def random_text(self, word_len_list, dict_index):
        """
        选择随机长度,从打乱的字典中切片一段word
        Args:
            word_len_list:
            dict_index:

        Returns:

        """
        tlen = np.random.choice(word_len_list)
        # TODO
        dict_index = dict_index + tlen
        word = ''.join(self.random_chars_list[dict_index: dict_index + tlen])
        return word, dict_index

    def build_word_list(self, build_id, num_words, max_string_len=None,
                        gentypr=0, random_ratio=4):
        """

        Args:
            build_id:
            num_words: 生成词组个数
            max_string_len: 词组的组大长度
            gentypr: 0为随机生成，1从预料中提取，2随机+语料提取
            random_ratio:

        Returns:

        """
        assert max_string_len <= self.absolute_max_string_len
        assert num_words % self.batch_size == 0
        assert (self.val_split * num_words) % self.batch_size == 0
        if lock.acquire():
            self.build_id = build_id
            self.num_words = num_words
            self.string_list = [''] * self.num_words
            tmp_string_list = []
            self.Y_data = np.ones([self.num_words, self.absolute_max_string_len]) * -1
            tmp_x_text = []
            self.Y_len = [0] * self.num_words
            word_len_list = range(2, max_string_len + 1)  # [2, 16]

            self.random_chars_list = self.chars.copy()
            np.random.shuffle(self.random_chars_list)

            dict_index = 0
            while 1:
                if len(tmp_string_list) == int(self.num_words):
                    break
                if dict_index >= len(self.random_chars_list):
                    np.random.shuffle(self.random_chars_list)
                    dict_index = 0
                word = ''
                if gentypr == 0:  # all random
                    word, dict_index = self.random_text(word_len_list, dict_index)
                elif gentypr == 1:  # from dict
                    # word = np.random.choice(self.chars)
                    pass
                elif gentypr == 2:  # random or dict
                    # if np.random.choice(range(0, 10)) < random_ratio:
                    #     word, dict_index = self.random_text(word_len_list, dict_index)
                    # else:
                    #     word = np.random.choice(self.chars)
                    pass

                if len(word) > max_string_len - 2:
                    word = word[0: max_string_len - 2]
                if word.find('\t') >= 0:
                    word = word.replace('\t', '')
                tmp_string_list.append(word)

            if len(tmp_string_list) != self.num_words:
                raise IOError('Could not pull enough word from supplied monogram and bigram files. ')
            # 交叉打乱
            self.string_list[::2] = tmp_string_list[:self.num_words // 2]
            self.string_list[1::2] = tmp_string_list[self.num_words // 2:]

            for i, word in enumerate(self.string_list):
                if len(word) == 0:
                    self.Y_len[i] = 1
                    self.Y_data[i, 0:1] = self.n_class
                else:
                    self.Y_len[i] = len(word)
                    self.Y_data[i, 0: len(word)] = text_to_labels(word, self.chars)
                tmp_x_text.append(word)
            self.X_text = tmp_x_text
            self.Y_len = np.expand_dims(np.array(self.Y_len), 1)

            self.cur_val_index = self.val_split
            self.cur_train_index = 0
            lock.release()

    def get_batch(self, index, size, train):
        if lock.acquire():
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([size, 1, self.img_w, self.img_h])
            else:
                X_data = np.ones([size, self.img_w, self.img_h, 1])
            labels = np.ones([size, self.absolute_max_string_len])
            input_length = np.zeros([size, 1])
            label_length = np.zeros([size, 1])
            source_str = []
            for i in range(0, size):
                if train and i > size - 4:  # 训练集包含部分空白图片
                    if K.image_data_format() == 'channels_first':
                        X_data[i, 0, 0: self.img_w, :] = self.paint_func('').T
                    else:
                        X_data[i, 0: self.img_w, :, 0] = self.paint_func('').T
                    labels[i, 0] = self.n_class
                    input_length[i] = self.img_w // self.downsample_factor - 2
                    label_length[i] = 1
                    source_str.append('')
                else:
                    if K.image_data_format() == 'channels_first':
                        X_data[i, 0, 0: self.img_w, :] = self.paint_func(
                            self.X_text[index + i]).T
                    else:
                        X_data[i, 0: self.img_w, :, 0] = self.paint_func(
                            self.X_text[index + i]).T
                    labels[i, :] = self.Y_data[index + i]
                    input_length[i] = self.img_w // self.downsample_factor - 2
                    label_length[i] = self.Y_len[index + i]
                    source_str.append(self.X_text[index + i])
            inputs = {'the_input': X_data,
                      'the_labels': labels,
                      'input_length': input_length,
                      'label_length': label_length
                      }
            outputs = {'ctc': np.zeros([size]),
                       'source_str': source_str
                       }
            lock.release()
            return inputs, outputs

    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index, self.batch_size, train=True)
            self.cur_train_index += self.batch_size
            if self.cur_train_index >= self.val_split:
                self.cur_train_index = self.cur_train_index % 32
                (self.X_text, self.Y_data, self.Y_len) = shuffle_mats_or_lists(
                    [self.X_text, self.Y_data, self.Y_len], self.val_split)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.cur_val_index, self.batch_size, train=False)
            self.cur_val_index += self.batch_size
            if self.cur_val_index >= self.num_words:
                self.cur_val_index = self.val_split + self.cur_val_index % 32
            yield ret

    def on_train_begin(self, logs=None):
        self.build_word_list(0, 32000, 16)
        self.paint_func = lambda text: paint_text(
            text, self.img_w, self.img_h)

    def on_epoch_begin(self, epoch, logs=None):
        if (epoch >= 30) and self.build_id != 1:  # and (epoch < 60)
            self.build_word_list(1, 64000, 16)
        # if (epoch >= 60) and (epoch < 120) and self.build_id != 2:
        #     self.build_word_list(2, 128000, 32, 1)
        # if (epoch >= 120) and (epoch < 160) and self.build_id != 3:
        #     self.build_word_list(3, 256000, 16, 1)
        # if (epoch >= 160) and (epoch < 250) and self.build_id != 4:
        #     self.build_word_list(4, 256000, 32, 2)
        # if (epoch >= 250) and (epoch < 300) and self.build_id != 5:
        #     self.build_word_list(5, 256000, 16, 2)
        # if (epoch >= 300) and (epoch < 400) and self.build_id != 6:
        #     self.build_word_list(6, 256000, 32, 1)
        # if (epoch >= 400) and (epoch < 500) and self.build_id != 7:
        #     self.build_word_list(7, 256000, 16, 1)
        # if (epoch >= 500) and (epoch < 700) and self.build_id != 8:
        #     self.build_word_list(8, 256000, 32, 2)
        # if (epoch >= 700) and self.build_id != 9:
        #     self.build_word_list(9, 256000, 32, 1)
