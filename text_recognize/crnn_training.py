# -*- coding: utf-8 -*-
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
import numpy as np
import argparse
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from PIL import Image
from utils import VizCallback, Config
from net.network import crnn_network
from data.generator import TextImageGenerator

# root_path = os.path.join(
#     os.path.abspath(os.path.dirname(__file__)).split('ocr')[0], 'ocr')
# save_weigths = os.path.join(root_path, 'text_recognize/model/crnn_weights-{epoch:02d}.hdf5')
# save_tensorboard = os.path.join(root_path, 'text_recognize/model/log')
# pre_train_weigths = ''
# train_label_file = 'D:/GIT/github/data/my_crnn_data/jpn/train.txt'
# test_label_file = 'D:/GIT/github/data/my_crnn_data/jpn/test.txt'
# root_image_path = 'D:/GIT/github/data/my_crnn_data/jpn/'


def train(config, start_epoch, stop_epoch, img_w, lr, batch_size=32, load_weight=None):
    img_h = 64
    words_per_epoch = 16000
    val_split = 0.2
    val_words = int(words_per_epoch * val_split)
    downsample_factor = 4

    input_shape = (img_w, img_h, 1)

    img_gen = TextImageGenerator(config=config,
                                 batch_size=batch_size,
                                 img_w=img_w,
                                 img_h=img_h,
                                 downsample_factor=downsample_factor,
                                 val_split=words_per_epoch - val_words
                                 )

    model, basemodel, input_data, y_pred = crnn_network(
        input_shape,
        img_gen.get_output_size(),
        img_gen.absolute_max_string_len,
        lr=lr
    )
    test_func = K.function([input_data], [y_pred])

    viz_cb = VizCallback(config, test_func, img_gen.next_val(), img_gen.chars)

    tensorboard = TensorBoard(log_dir=os.path.join(config.job_output_path, 'logs'),
                              histogram_freq=0, write_graph=True, embeddings_freq=0)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0,
                               patience=100, verbose=0, mode='auto')
    callbacks_list = [tensorboard, viz_cb, early_stop, img_gen]

    if start_epoch > 0:
        weight_file = os.path.join(viz_cb.models_dir, 'crnn_ocr%04d.hdf5' % (start_epoch))
        model.load_weights(weight_file)
    elif load_weight is not None:
        weight_file = load_weight
        model.load_weights(weight_file)

    model.fit_generator(generator=img_gen.next_train(),
                        steps_per_epoch=(words_per_epoch - val_words) // batch_size,
                        epochs=stop_epoch,
                        validation_data=img_gen.next_val(),
                        validation_steps=val_words // batch_size,
                        callbacks=callbacks_list,
                        initial_epoch=start_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-name',
                        required=False,
                        type=str,
                        default='job_random_jp',
                        help='Job unique name')
    parser.add_argument('--job-dir',
                        required=False,
                        type=str,
                        default='D:/GIT/purvar/ocr/text_recognize',
                        help='GCS or local dir to write checkpoints and export model')
    parser.add_argument('--train-datas',
                        required=False,
                        type=str,
                        default='data',
                        help='train datas: gcs file *.zip or *.gz or local folder name in job_dir')
    parser.add_argument('--load-weights',
                        required=False,
                        default=None,
                        type=str,
                        help='weights filepath: load weights continue to train')
    # parser.add_argument('--gpu-num',
    #                     required=False,
    #                     default=1,
    #                     type=int,
    #                     help='GPU number')
    parser.add_argument('--base-lr',
                        required=False,
                        default=0.02,
                        type=float,
                        help='base learning_rate')
    parser.add_argument('--epochs',
                        required=False,
                        default=1000,
                        type=int,
                        help='default epochs is 1000')
    parse_args, unknown = parser.parse_known_args()

    config = Config(parse_args.job_name,
                    parse_args.job_dir,
                    parse_args.train_datas,
                    "train_by_fonts")
    train(config, 0, parse_args.epochs, 512, parse_args.base_lr,
          batch_size=32, load_weight=parse_args.load_weights)

    K.clear_session()
