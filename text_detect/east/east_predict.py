# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import ImageDraw
import PIL.Image as PI
from wand.image import Image, Color
import tensorflow as tf
import keras.backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import east_config as cfg
from net.east_network import east_network
from net.nms import nms


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict_quad(model, img, pixel_threshold=0.9, quiet=False, img_name=None):
    max_train_img_size = cfg.max_pred_img_size
    max_train_img_h, max_train_img_w = 0, 0
    if isinstance(max_train_img_size, int):
        max_train_img_h, max_train_img_w = max_train_img_size, max_train_img_size
    elif isinstance(max_train_img_size, tuple):
        max_train_img_h, max_train_img_w = max_train_img_size
    d_wight, d_height = max_train_img_w, max_train_img_h
    img = img.resize((d_wight, d_height), PI.BILINEAR).convert('RGB')
    img = image.img_to_array(img)

    num_img = 1
    img_all = np.zeros((num_img, d_height, d_wight, 3))
    img_all[0] = img

    img_ori = preprocess_input(img, mode='tf')  # suit tf tensor

    x = np.zeros((num_img, d_height, d_wight, 3))
    x[0] = img_ori

    y_pred = model.predict(x)

    text_recs_all = []
    text_recs_len = []
    for n in range(num_img):
        # (sample, rows, cols, 7_points_pred)
        y = y_pred[n]
        y[:, :, :3] = sigmoid(y[:, :, :3])
        cond = np.greater_equal(y[:, :, 0], pixel_threshold)
        activation_pixels = np.where(cond)  # fixme 返回元祖tuple类型 a[0]保存了纵坐标 a[1]保存横坐标
        quad_scores, quad_after_nms = nms(y, activation_pixels)

        text_recs = []
        x[n] = np.uint8(x[n])
        with image.array_to_img(img_all[n]) as im:
            # fixme 注意：拿去CRNN识别的是缩放后的图像
            scale_ratio_w = 1
            scale_ratio_h = 1

            quad_im = im.copy()
            draw = ImageDraw.Draw(im)
            for i, j in zip(activation_pixels[0], activation_pixels[1]):
                px = (j + 0.5) * cfg.downsample_factor
                py = (i + 0.5) * cfg.downsample_factor
                line_width, line_color = 1, 'blue'
                if y[i, j, 1] >= cfg.side_vertex_pixel_threshold:
                    if y[i, j, 2] < cfg.trunc_threshold:
                        line_width, line_color = 2, 'yellow'
                    elif y[i, j, 2] >= 1 - cfg.trunc_threshold:
                        line_width, line_color = 2, 'green'
                draw.line([(px - 0.5 * cfg.downsample_factor, py - 0.5 * cfg.downsample_factor),
                           (px + 0.5 * cfg.downsample_factor, py - 0.5 * cfg.downsample_factor),
                           (px + 0.5 * cfg.downsample_factor, py + 0.5 * cfg.downsample_factor),
                           (px - 0.5 * cfg.downsample_factor, py + 0.5 * cfg.downsample_factor),
                           (px - 0.5 * cfg.downsample_factor, py - 0.5 * cfg.downsample_factor)],
                          width=line_width, fill=line_color)
            if img_name:
                im.save(image_save_path + 'quad/' + img_name + '_%d_.jpg' % n)

            quad_draw = ImageDraw.Draw(quad_im)
            for score, geo, s in zip(quad_scores, quad_after_nms,
                                     range(len(quad_scores))):
                if np.amin(score) > 0:
                    quad_draw.line([tuple(geo[0]),
                                    tuple(geo[1]),
                                    tuple(geo[2]),
                                    tuple(geo[3]),
                                    tuple(geo[0])], width=2, fill='blue')

                    rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                    text_rec = np.reshape(rescaled_geo, (8,)).tolist()
                    text_recs.append(text_rec)
                elif not quiet:
                    print('quad invalid with vertex num less then 4.')

            if img_name:
                quad_im.save(image_save_path + 'predict/' + img_name + '_%d_.jpg' % n)

        for t in range(len(text_recs)):
            text_recs_all.append(text_recs[t])

        text_recs_len.append(len(text_recs))

    return text_recs_all, text_recs_len, img_all


if __name__ == '__main__':
    root_path = os.path.join(os.path.abspath(os.path.dirname(__file__)).split('ocr')[0], 'ocr')
    # ----get_sample----
    sample_dir = os.path.join(root_path, 'sample/')
    sample_list = os.listdir(sample_dir)
    image_save_path = os.path.join(root_path, 'east_output/')
    if not os.path.exists(image_save_path):
        os.mkdir(image_save_path)
    # ----pre_model----
    pre_model_weight = os.path.join(root_path, 'text_detect/east/pre_model/east_model_weights_1.h5')

    with tf.Session() as sess:
        K.set_session(sess)
        model, _, _ = east_network()
        model.load_weights(os.path.abspath(pre_model_weight))
        for i, sample in enumerate(sample_list):
            img_path = os.path.join(sample_dir, sample)
            if sample.endswith('.gitkeep'):
                continue
            if sample[-4:] == '.pdf':
                with Image(filename=img_path, resolution=(200, 200)) as img:
                    img.alpha_channel = False
                    img.background_color = Color('white')  # Set the background color
                    img = PI.fromarray(np.array(img), 'RGB')
            else:
                img = PI.open(img_path).convert('RGB')
            print('-----------------传入图片的img.size:', img.size)
            predict_quad(model, img, pixel_threshold=0.9, img_name=sample)
    K.clear_session()
