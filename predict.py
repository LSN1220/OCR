# -*- coding: utf-8 -*-
import os
from text_detect.east.net.network import East
from text_detect.east.east_predict import predict_quad
from text_detect.layout_analysis.layout_analysis_predict import main
from text_recognize.net.network import crnn_network
from text_recognize.crnn_predict import predict_text
from keras.preprocessing import image

import numpy as np
import PIL.Image as PI
from wand.image import Image, Color

root_path = os.path.join(os.path.abspath(os.path.dirname(__file__)).split('ocr')[0], 'ocr')
east_pre_model = os.path.join(root_path, 'text_detect/east/pre_model/east_model_weights.h5')
crnn_pre_model = os.path.join(root_path, 'text_recognize/pre_model/crnn_weights_50.hdf5')
detect_method = 'layout_analysis'  # 'east', 'layout_analysis'

if __name__ == '__main__':
    # todo east model predict
    east = East()
    east_model = east.east_network()
    east_model.load_weights(east_pre_model)

    # todo crnn model predict
    model, crnn_model = crnn_network()
    crnn_model.load_weights(crnn_pre_model)

    sample_dir = os.path.join(root_path, 'sample/')
    sample_list = os.listdir(sample_dir)
    print('num_sample:', len(sample_list))

    save_dir = os.path.join(root_path, 'crnn_output')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i, sample in enumerate(sample_list):
        images = []
        img_path = os.path.join(sample_dir, sample)
        if sample[-4:] == '.pdf':
            with Image(filename=img_path, resolution=(200, 200)) as imgs:
                num_page = len(imgs.sequence)
                for i in range(num_page):
                    img = Image(image=imgs.sequence[i])
                    img.alpha_channel = False
                    img.background_color = Color('white')  # Set the background color
                    img = PI.fromarray(np.array(img), 'RGB')
                    im_name = sample[:-4] + '_p' + str(i)
                    images.append((img, im_name))
        else:
            im_name = sample[:-4]
            img = PI.open(img_path).convert('RGB')
            images = [(img, im_name)]
        for img, im_name in images:
            if detect_method == 'east':
                text_recs_all, text_recs_len, img_all = predict_quad(east_model, img, img_name=im_name)
                if len(text_recs_all) > 0:
                    texts = predict_text(crnn_model, text_recs_all, text_recs_len, img_all, img_name=im_name)
                    # for s in range(len(texts)):
                    #     print("result ：%s" % texts[s])
                    with open(os.path.join(save_dir, 'east/' + im_name + '_text.txt'), 'w', encoding='utf-8') as f:
                        for s in range(len(texts)):
                            f.write(texts[s] + '\n')
            elif detect_method == 'layout_analysis':
                text_recs_all, text_recs_len, img_all = main(img, im_name, max_img_size=2048)
                if len(text_recs_all) > 0:
                    texts = predict_text(crnn_model, text_recs_all, text_recs_len, img_all, img_name=im_name)
                    # for s in range(len(texts)):
                    #     print("result ：%s" % texts[s])
                    with open(os.path.join(save_dir, 'layout_analysis/' + im_name + '_text.txt'), 'w', encoding='utf-8') as f:
                        for s in range(len(texts)):
                            f.write(texts[s] + '\n')
