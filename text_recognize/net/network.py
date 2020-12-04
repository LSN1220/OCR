# -*- coding: utf-8 -*-
import os
import sys
from PIL import Image
import keras.backend as K
import numpy as np
from keras.layers import Flatten, BatchNormalization, Permute, TimeDistributed, Dense, Bidirectional, Reshape, LSTM
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Lambda
from keras.models import Model
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
from keras.constraints import non_neg

img_h = 32
root_path = os.path.join(
    os.path.abspath(os.path.dirname(__file__)).split('ocr')[0], 'ocr')
char_file = os.path.join(
    root_path, 'text_recognize/net/japan_dict.txt')  # 'char_std_5990.txt'
char = ''
with open(char_file, encoding='utf-8') as f:
    for ch in f.readlines():
        ch = ch.strip('\r\n')
        char = char + ch
n_class = len(char) + 1


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def crnn_network(input_shape, output_size,
                 max_string_len, lr):
    input = Input(shape=input_shape, name='the_input')
    # CNN
    inner = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   kernel_constraint=non_neg(), name='conv1')(input)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)
    inner = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)
    inner = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   name='conv3')(inner)
    inner = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   name='conv4')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)
    inner = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   name='conv5')(inner)
    K.set_learning_phase(1)
    inner = BatchNormalization(axis=-1)(inner)
    inner = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   name='conv6')(inner)
    inner = BatchNormalization(axis=-1)(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)
    inner = Conv2D(512, (2, 2), padding='same', activation='relu', kernel_initializer='he_normal',
                   name='conv7')(inner)
    # m的输出维度为(h, w, c) -> (1, w/4, 512) 转换 (w, b, c) = (seq_len, batch, input_size)
    # m = Permute((2, 1, 3), name='permute')(m)
    # m = TimeDistributed(Flatten(), name='timedistrib')(m)

    conv_to_rnn_dims = (input_shape[0] // 4, (input_shape[1] // 4) * 128)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)
    inner = Dense(256, activation='relu', name='dense1')(inner)

    # RNN
    m = Bidirectional(LSTM(512, return_sequences=True), name='blstm1')(inner)
    m = Dense(512, name='blstm1_out', activation='linear')(m)
    m = Bidirectional(LSTM(512, return_sequences=True), name='blstm2')(m)
    y_pred = Dense(output_size, name='blstm2_out', activation='softmax')(m)

    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.summary()

    labels = Input(name='the_labels', shape=[max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd, metrics=['accuracy'])
    # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer="adadelta", metrics=['accuracy'])
    # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd, metrics=['accuracy'])

    # model.summary()

    return model, basemodel, input, y_pred


def predict(img_path, model):
    """
    输入图片，输出keras模型的识别结果
    :param img_path:
    :return:
    """
    img = Image.open(img_path)
    img = img.convert('L')

    scale = img.size[1] * 1.0 / 32
    w = int(img.size[0] / scale)
    img = img.resize((w, 32), Image.BILINEAR)
    img = np.array(img).astype(np.float32) / 255.0 - 0.5
    X = img.reshape((32, w, 1))
    X = np.array([X])

    y_pred = model.predict(X)

    y_pred = y_pred[:, :, :]

    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :]
    out_s = u''.join([char[x] for x in out[0]])

    return out_s
