import os
import sys
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
sys.path.append('..')
import east_config as cfg


def gen(max_img_h, max_img_w, downsample_factor, batch_size, is_val=False):
    img_h, img_w = max_img_h, max_img_w
    x = np.zeros((batch_size, img_h, img_w, 3), dtype=np.float32)
    pixel_num_h = img_h // downsample_factor   # fixme 以4*4个像素点为一格 预测值score
    pixel_num_w = img_w // downsample_factor
    y = np.zeros((batch_size, pixel_num_h, pixel_num_w, 7), dtype=np.float32)
    if is_val:
        with open(os.path.join(cfg.data_dir, cfg.val_fname), 'r') as f_val:
            f_list = f_val.readlines()
            num_data = len(f_list)
    else:
        with open(os.path.join(cfg.data_dir, cfg.train_fname), 'r') as f_train:
            f_list = f_train.readlines()
            num_data = len(f_list)

    while True:
        for i in range(batch_size):
            # random gen an image name
            random_img = np.random.choice(f_list)
            img_filename = str(random_img).strip().split(',')[0]
            # load img and img anno
            img_path = os.path.join(cfg.data_dir,
                                    cfg.train_image_dir_name,
                                    img_filename)
            img = image.load_img(img_path)
            img = image.img_to_array(img)
            x[i] = preprocess_input(img, mode='tf')
            gt_file = os.path.join(cfg.data_dir,
                                   cfg.train_label_dir_name,
                                   img_filename[:-4] + '_gt.npy')
            y[i] = np.load(gt_file)
        yield x, y