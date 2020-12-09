# coding=utf8
import os
import math
import itertools
import random
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from scipy import ndimage
from keras.callbacks import Callback
import crnn_config as cfg


def speckle(img):
    # print('use speckle')
    severity = np.random.uniform(0, 0.4)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck


def draw_shadow(img_draw, txt_x, txt_y, letter, font,
                fill=(255, 255, 255), shadow_width=3):
    if shadow_width == 0:
        return
    for i_x in range(0 - shadow_width, shadow_width+1):
        for i_y in range(0 - shadow_width, shadow_width+1):
            # 在图片上贴字
            img_draw.text((txt_x+i_x, txt_y+i_y), letter, fill, font=font)


# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.
def decode_batch(test_func, word_batch, all_string_list):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        # len(all_string_list)-1 is space, len(all_string_list) is CTC blank char
        outstr = ''
        for c in out_best:
            if c < len(all_string_list):
                outstr += all_string_list[c]
        ret.append(outstr)
    return ret


# fonts_dir = 'fonts'
# bg_dir = 'bg'
# char_dir = 'chars'
# evl_dir = 'evl_data'
# models_dir = 'chekcpoints'
#
#
# class Config():
#     def __init__(self, job_name, job_dir, train_datas, train_type):
#         uni_chars_file = 'jpn.txt'
#         # word_dict_file = 'jpn_word_dict.txt'
#
#         self.job_name = job_name
#         self.job_dir = job_dir
#         self.train_type = train_type  # 'train_by_fonts' 'train_by_images'
#         # self.only_price = only_price == 1
#
#         # job_dir is gcs_path, copy files to local
#         self.train_dir = os.path.abspath(job_dir)
#         self.tran_datas_dir = os.path.join(self.train_dir, train_datas)
#         self.job_output_path = os.path.join(self.train_dir, job_name)
#
#         # init train datas
#         if train_type == 'train_by_fonts':
#             # set font abs path
#             fonts_dir_abs = os.path.join(self.tran_datas_dir, fonts_dir)
#             char_dir_abs = os.path.join(self.tran_datas_dir, char_dir)
#             bg_dir_abs = os.path.join(self.tran_datas_dir, bg_dir)
#
#             self.fonts_files = [os.path.join(fonts_dir_abs, fonts_item) for fonts_item in os.listdir(fonts_dir_abs)]
#             self.uni_chars_file = os.path.join(char_dir_abs, uni_chars_file)
#             self.bg_files = [os.path.join(bg_dir_abs, bg_item) for bg_item in os.listdir(bg_dir_abs)]
#
#         elif train_type == 'train_by_images':
#             self.train_dir = job_dir


def random_crop_png(jpg_file, crop_range=(0.45, 0.99)):
    image_data = Image.open(jpg_file)
    crop_proportion = np.random.uniform(crop_range[0], crop_range[1], 2)
    width = image_data.width*crop_proportion[0]
    height = image_data.height*crop_proportion[1]

    crop_origin_x = random.randint(0, int(image_data.width - width))
    crop_origin_y = random.randint(0, int(image_data.height - height))

    crop_box = (crop_origin_x,
                crop_origin_y,
                crop_origin_x + width,
                crop_origin_y + height)
    image = image_data.crop(crop_box).convert('RGB')
    return image


def get_bg(w, h):
    bg_file_abs = np.random.choice(cfg.bg_files)
    img = random_crop_png(bg_file_abs)
    img = img.resize((w, h))
    return img


def paint_text(text, w, h):
    use_bg = np.random.choice(range(0, 10)) > 4
    if text is None or text == '':
        surface = Image.new('RGB', (w, h), (255, 255, 255))
        if use_bg:
            surface = get_bg(w, h)
    else:
        text_len = len(text)
        padding_size_w = np.random.choice(range(1, 6))
        padding_size_h = np.random.choice(range(1, 6))
        noise_w = np.random.choice(range(-10, 11))
        noise_h = np.random.choice(range(-10, 11))

        surface_w = w + noise_w
        surface_h = h + noise_h
        inner_w = surface_w - padding_size_w * 2
        inner_h = surface_h - padding_size_h * 2
        fontsize = min([math.floor(inner_w / text_len), inner_h])
        fontsize = random.randint(min(fontsize, 16), max(fontsize, 16))
        surface_w = text_len * fontsize - random.randint(0, int((fontsize * 0.6)))
        surface_h = int(fontsize * (random.random() * 0.25 + 0.95))
        surface = Image.new('RGB', (surface_w, surface_h), (255, 255, 255))
        if use_bg:
            surface = get_bg(
                surface_w + np.random.choice([0, padding_size_w, padding_size_w * 2]),
                surface_h + np.random.choice([0, padding_size_h, padding_size_h * 2]))
            fill_color = (np.random.choice([0, 50, 100, 150, 200, 255]), 0, 0)
        else:
            fill_color = (np.random.choice([0, 50, 100, 150, 200, 255]), 0, 0)

        if fontsize > 80:
            fontsize = np.random.choice([32, 46, 74, 80])
            fontsize = random.randint(16, max(fontsize, 16))
        font_file = np.random.choice(cfg.fonts_files)
        font = ImageFont.truetype(font_file, int(fontsize))
        # 在bg上贴字
        draw_brush = ImageDraw.Draw(surface)

        rect_w = fontsize * text_len
        rect_h = fontsize
        pos_x = math.floor((surface_w - rect_w) / 2)
        pos_y = math.floor((surface_h - rect_h) / 2)

        draw_shadow(draw_brush, pos_x, pos_y, text, font)
        draw_brush.text((pos_x, pos_y), text, fill=fill_color, font=font)

    # 随机反色
    if np.random.choice([True, False]):
        surface = ImageOps.invert(surface)

    # 随机缩窄
    rand_w = surface.size[0]
    if np.random.choice([True, False, False]):
        rang = [0.6, 1]
        rand_w = int(surface.size[0] * (random.random() * (rang[1] - rang[0]) + rang[0]))
        surface = surface.resize((rand_w, surface.size[1]), Image.ADAPTIVE)

    scale_w = w / surface.size[0]
    scale_h = h / surface.size[1]
    if scale_w > scale_h:
        surface = surface.resize((int(surface.size[0] * scale_h), h), Image.ADAPTIVE)
    else:
        surface = surface.resize((w, int(surface.size[1] * scale_w)), Image.ADAPTIVE)

    # 归一化
    a = np.array(surface.convert('L'))
    a = a.astype(np.float32) / 255
    # a.shape=(h,w)

    # 高斯滤波
    if np.random.choice([True, False]):
        a = speckle(a)

    # 统一图片尺寸
    result = np.ones((h, w))
    pad_w = math.floor((w-surface.size[0]) / 2)
    pad_h = math.floor((h-surface.size[1]) / 2)
    result[pad_h: pad_h + surface.size[1], pad_w: pad_w + surface.size[0]] = a
    return result


class VizCallback(Callback):
    def __init__(self, test_func, text_img_gen, all_string_list, num_display_words=6):
        self.test_func = test_func
        self.best = np.Inf
        self.job_dir = cfg.job_dir
        self.job_name = cfg.job_name
        self.models_dir = os.path.join(cfg.job_output_path, cfg.models_dir)
        self.evl_dir = os.path.join(cfg.job_output_path, cfg.evl_dir)
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        self.all_string_list = all_string_list
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_loss')
        if np.less(current, self.best):
            model_name = 'crnn_ocr-%04d-%.2f.hdf5' % (epoch+1, current)
            model_path = os.path.join(self.models_dir, model_name)
            print('Epoch %05d: val_loss improved from %0.5f to %0.5f,'
                  ' saving model to %s'
                  % (epoch + 1, self.best, current, model_path))
            self.best = current
            self.model.save_weights(model_path)

        word_batch, out_put = next(self.text_img_gen)
        res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words], self.all_string_list)

        for i in range(self.num_display_words):
            # if K.image_data_format() == 'channels_last': tensorflow is channels last
            the_input = word_batch['the_input'][i, :, :, 0]

            print('\nTruth = \'%s\',Decoded = \'%s\'' % (out_put['source_str'][i], res[i]))
            eval_data_dir = os.path.join(self.evl_dir, str(epoch+1))
            if os.path.exists(eval_data_dir) is False:
                os.makedirs(eval_data_dir)
            img = Image.fromarray(the_input.T*255)
            img = img.convert('RGB')
            img_name = str(i)+'.jpeg'
            evl_img_path = os.path.join(eval_data_dir, img_name)
            img.save(evl_img_path, 'JPEG')


if __name__ == '__main__':
    from text_recognize.data.generator import TextImageGenerator

    img_h = 64
    img_w = 512
    words_per_epoch = 100
    val_split = 0.2
    val_words = int(words_per_epoch * (val_split))
    downsample_factor = 4

    # Network parameters
    batch_size = 20
    img_gen = TextImageGenerator(batch_size=batch_size,
                                 img_w=img_w,
                                 img_h=img_h,
                                 downsample_factor=downsample_factor,
                                 val_split=words_per_epoch - val_words
                                 )
    img_gen.build_word_list(0, 200, 16)
    all_words_list = img_gen.string_list[:200]

    texts = []
    for i, word in enumerate(all_words_list):
        texts.append(word+'\n\r')
        a = paint_text(word, 512, 64)
        # print(a.shape,a.T.shape,np.array([a.T,a.T,a.T]).T.shape)
        # img=Image.fromarray(np.array([a.T,a.T,a.T]).T)
        img = Image.fromarray(a * 255)
        img = img.convert('RGB')
        # print(str(i)+'test_paint.jpeg')
        img.save(cfg.job_dir + '/data/temp/'+str(i) + 'test_paint.jpeg', 'jpeg')

    with open('texts.txt', 'w', encoding="utf-8") as f:
        f.writelines(texts)
