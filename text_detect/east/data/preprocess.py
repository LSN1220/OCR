import sys
import numpy as np
from PIL import Image, ImageDraw
import os
import random
from tqdm import tqdm
sys.path.append('..')
import east_config as cfg

epsilon = 1e-4


def shrink(xy_list, shrink_ratio=0.2):  # shrink_ratio = 0.2
    if shrink_ratio == 0.0:
        return xy_list, xy_list
    diff_1to3 = xy_list[:3, :] - xy_list[1:4, :]
    diff_4 = xy_list[3:4, :] - xy_list[0:1, :]
    diff = np.concatenate((diff_1to3, diff_4), axis=0)
    dis = np.sqrt(np.sum(np.square(diff), axis=-1))
    # determine which are long or short edges
    long_edge = int(np.argmax(np.sum(np.reshape(dis, (2, 2)), axis=0)))
    short_edge = 1 - long_edge
    # cal r length array
    r = [np.minimum(dis[i], dis[(i + 1) % 4]) for i in range(4)]
    # cal theta array
    diff_abs = np.abs(diff)
    diff_abs[:, 0] += epsilon
    theta = np.arctan(diff_abs[:, 1] / diff_abs[:, 0])
    # shrink two long edges
    temp_new_xy_list = np.copy(xy_list)
    shrink_edge(xy_list, temp_new_xy_list, long_edge, r, theta, shrink_ratio)
    shrink_edge(xy_list, temp_new_xy_list, long_edge + 2, r, theta, shrink_ratio)
    # shrink two short edges
    new_xy_list = np.copy(temp_new_xy_list)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge, r, theta, shrink_ratio)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge + 2, r, theta, shrink_ratio)
    return temp_new_xy_list, new_xy_list, long_edge


def shrink_edge(xy_list, new_xy_list, edge, r, theta, shrink_ratio=0.2):
    if shrink_ratio == 0.0:
        return
    start_point = edge
    end_point = (edge + 1) % 4
    long_start_sign_x = np.sign(
        xy_list[end_point, 0] - xy_list[start_point, 0])
    new_xy_list[start_point, 0] = \
        xy_list[start_point, 0] + \
        long_start_sign_x * shrink_ratio * r[start_point] * np.cos(theta[start_point])
    long_start_sign_y = np.sign(
        xy_list[end_point, 1] - xy_list[start_point, 1])
    new_xy_list[start_point, 1] = \
        xy_list[start_point, 1] + \
        long_start_sign_y * shrink_ratio * r[start_point] * np.sin(theta[start_point])
    # long edge one, end point
    long_end_sign_x = -1 * long_start_sign_x
    new_xy_list[end_point, 0] = \
        xy_list[end_point, 0] + \
        long_end_sign_x * shrink_ratio * r[end_point] * np.cos(theta[start_point])
    long_end_sign_y = -1 * long_start_sign_y
    new_xy_list[end_point, 1] = \
        xy_list[end_point, 1] + \
        long_end_sign_y * shrink_ratio * r[end_point] * np.sin(theta[start_point])


def point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
    """
        :判断(px,py)在quad_xy_list表示的文本框内
        :b[0] = (x1-x0)*(py-y0) - (y1-y0)*(px-x0)
    """
    if (p_min[0] <= px <= p_max[0]) and (p_min[1] <= py <= p_max[1]):
        xy_list = np.zeros((4, 2))
        xy_list[:3, :] = quad_xy_list[1:4, :] - quad_xy_list[:3, :]
        xy_list[3] = quad_xy_list[0, :] - quad_xy_list[3, :]
        yx_list = np.zeros((4, 2))
        yx_list[:, :] = quad_xy_list[:, -1:-3:-1]
        a = xy_list * ([py, px] - yx_list)
        b = a[:, 0] - a[:, 1]
        if np.amin(b) >= 0 or np.amax(b) <= 0:
            return True
        else:
            return False
    else:
        return False


def point_inside_of_nth_quad(px, py, xy_list, shrink_1, long_edge):
    nth = -1
    vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
          [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
    for ith in range(2):
        quad_xy_list = np.concatenate((
            np.reshape(xy_list[vs[long_edge][ith][0]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][1]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][2]], (1, 2)),
            np.reshape(xy_list[vs[long_edge][ith][3]], (1, 2))), axis=0)
        p_min = np.amin(quad_xy_list, axis=0)
        p_max = np.amax(quad_xy_list, axis=0)
        if point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
            if nth == -1:
                nth = ith
            else:
                nth = -1
                break
    return nth


def reorder_vertexes(xy_list):  # (4, 2)
    reorder_xy_list = np.zeros_like(xy_list)
    ordered = np.argsort(xy_list, axis=0)
    xmin1_index = ordered[0, 0]
    xmin2_index = ordered[1, 0]
    if xy_list[xmin1_index, 0] == xy_list[xmin2_index, 0]:
        if xy_list[xmin1_index, 1] <= xy_list[xmin2_index, 1]:
            reorder_xy_list[0] = xy_list[xmin1_index]
            first_v = xmin1_index
        else:
            reorder_xy_list[0] = xy_list[xmin2_index]
            first_v = xmin2_index
    else:
        reorder_xy_list[0] = xy_list[xmin1_index]
        first_v = xmin1_index
    # connect the first point to others, the third point on the other side of the line with the middle slope
    others = list(range(4))
    others.remove(first_v)
    k = np.zeros((len(others),))
    for index, i in zip(others, range(len(others))):
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) / (
                xy_list[index, 0] - xy_list[first_v, 0] + epsilon)
    k_mid = np.argsort(k)[1]
    third_v = others[k_mid]
    reorder_xy_list[2] = xy_list[third_v]
    # determine the second point which on the bigger side of the middle line
    others.remove(third_v)
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]
    second_v, fourth_v = 0, 0
    for index, i in zip(others, range(len(others))):
        # delta = y - (k * x + b)
        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:
            second_v = index
        else:
            fourth_v = index
    reorder_xy_list[1] = xy_list[second_v]
    reorder_xy_list[3] = xy_list[fourth_v]
    # compare slope of 13 and 24, determine the final order
    k13 = k[k_mid]
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (
            xy_list[second_v, 0] - xy_list[fourth_v, 0] + epsilon)
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    return reorder_xy_list


# def resize_image(im, max_img_size=cfg.max_train_img_size):
#     im_width = np.minimum(im.width, max_img_size)
#     if im_width == max_img_size < im.width:     # 起到and的作用
#         im_height = int((im_width / im.width) * im.height)
#     else:
#         im_height = im.height
#
#     o_height = np.minimum(im_height, max_img_size)
#     if o_height == max_img_size < im_height:
#         o_width = int((o_height / im_height) * im_width)
#     else:
#         o_width = im_width
#
#     # fixme 最多裁剪31个pixel 是否影响边缘效果
#     d_wight = o_width - (o_width % 32)
#     d_height = o_height - (o_height % 32)
#     return d_wight, d_height


def preprocess():
    max_train_img_size = cfg.max_train_img_size
    max_train_img_h = 0
    max_train_img_w = 0
    if isinstance(max_train_img_size, int):
        max_train_img_h, max_train_img_w = max_train_img_size, max_train_img_size
    elif isinstance(max_train_img_size, tuple):
        max_train_img_h, max_train_img_w = max_train_img_size

    data_dir = cfg.data_dir  # 'D:/GIT/github/data/icpr_text/train_1000/'
    origin_image_dir = os.path.join(data_dir, cfg.origin_image_dir_name)  # 'image_1000/'
    origin_txt_dir = os.path.join(data_dir, cfg.origin_txt_dir_name)  # 'txt_1000/'
    train_image_dir = os.path.join(data_dir, cfg.train_image_dir_name)  # 'images_3T736/'
    train_label_dir = os.path.join(data_dir, cfg.train_label_dir_name)  # 'labels_3T736/'
    if not os.path.exists(train_image_dir):
        os.mkdir(train_image_dir)
    if not os.path.exists(train_label_dir):
        os.mkdir(train_label_dir)

    draw_gt_quad = False
    o_img_list = os.listdir(origin_image_dir)
    print('found %d origin images.' % len(o_img_list))
    train_val_set = []
    for o_img_fname, _ in zip(o_img_list, tqdm(range(len(o_img_list)))):
        with Image.open(os.path.join(origin_image_dir, o_img_fname)) as im:
            # d_wight, d_height = resize_image(im)
            d_wight, d_height = max_train_img_w, max_train_img_h  # 736
            scale_ratio_w = d_wight / im.width
            scale_ratio_h = d_height / im.height
            im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')

            # show_gt_im = im.copy()
            # draw = ImageDraw.Draw(show_gt_im)

            with open(os.path.join(origin_txt_dir, o_img_fname[:-4] + '.txt'), 'r', encoding="UTF-8") as f:
                print("img file: ", o_img_fname[:-4])
                anno_list = f.readlines()
            xy_list_array = np.zeros((len(anno_list), 4, 2))
            gt = np.zeros((d_height // cfg.downsample_factor, d_wight // cfg.downsample_factor, 7))
            for anno, i in zip(anno_list, range(len(anno_list))):
                anno_colums = anno.strip().split(',')
                anno_array = np.array(anno_colums)
                xy_list = np.reshape(anno_array[:8].astype(float), (4, 2))
                xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w
                xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h
                xy_list = reorder_vertexes(xy_list)  # 重排点顺序
                xy_list_array[i] = xy_list
                # 缩小文本框
                _, shrink_xy_list, _ = shrink(xy_list, 0.2)  # shrink_ratio = 0.2
                shrink_1, _, long_edge = shrink(xy_list, 0.6)  # shrink_side_ratio = 0.6

                # draw.line([tuple(xy_list[0]), tuple(xy_list[1]),
                #            tuple(xy_list[2]), tuple(xy_list[3]),
                #            tuple(xy_list[0])
                #            ],
                #           width=2, fill='green')
                # draw.line([tuple(shrink_xy_list[0]),
                #            tuple(shrink_xy_list[1]),
                #            tuple(shrink_xy_list[2]),
                #            tuple(shrink_xy_list[3]),
                #            tuple(shrink_xy_list[0])
                #            ],
                #           width=2, fill='blue')
                # vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
                #       [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
                # for q_th in range(2):
                #     draw.line([tuple(xy_list[vs[long_edge][q_th][0]]),
                #                tuple(shrink_1[vs[long_edge][q_th][1]]),
                #                tuple(shrink_1[vs[long_edge][q_th][2]]),
                #                tuple(xy_list[vs[long_edge][q_th][3]]),
                #                tuple(xy_list[vs[long_edge][q_th][4]])],
                #               width=3, fill='yellow')
                # show_gt_im.show()
                p_min = np.amin(shrink_xy_list, axis=0)
                p_max = np.amax(shrink_xy_list, axis=0)
                ji_min = (p_min / cfg.downsample_factor - 0.5).astype(int) - 1
                ji_max = (p_max / cfg.downsample_factor - 0.5).astype(int) + 3
                imin = np.maximum(0, ji_min[1])
                imax = np.minimum(d_height // cfg.downsample_factor, ji_max[1])
                jmin = np.maximum(0, ji_min[0])
                jmax = np.minimum(d_wight // cfg.downsample_factor, ji_max[0])
                # 对每个在文本框内的像素打标签
                for i in range(imin, imax):
                    for j in range(jmin, jmax):
                        px = (j + 0.5) * cfg.downsample_factor
                        py = (i + 0.5) * cfg.downsample_factor
                        if point_inside_of_quad(px, py, shrink_xy_list, p_min, p_max):
                            gt[i, j, 0] = 1
                            # line_width, line_color = 1, 'red'
                            ith = point_inside_of_nth_quad(px, py,
                                                           xy_list,
                                                           shrink_1,
                                                           long_edge)
                            vs = [[[3, 0], [1, 2]], [[0, 1], [2, 3]]]
                            if ith in range(2):
                                gt[i, j, 1] = 1
                                # if ith == 0:
                                #     line_width, line_color = 2, 'yellow'
                                # else:
                                #     line_width, line_color = 2, 'green'
                                gt[i, j, 2:3] = ith
                                gt[i, j, 3:5] = \
                                    xy_list[vs[long_edge][ith][0]] - [px, py]
                                gt[i, j, 5:] = \
                                    xy_list[vs[long_edge][ith][1]] - [px, py]
            #                 draw.line([(px - 0.5 * cfg.downsample_factor,
            #                             py - 0.5 * cfg.downsample_factor),
            #                            (px + 0.5 * cfg.downsample_factor,
            #                             py - 0.5 * cfg.downsample_factor),
            #                            (px + 0.5 * cfg.downsample_factor,
            #                             py + 0.5 * cfg.downsample_factor),
            #                            (px - 0.5 * cfg.downsample_factor,
            #                             py + 0.5 * cfg.downsample_factor),
            #                            (px - 0.5 * cfg.downsample_factor,
            #                             py - 0.5 * cfg.downsample_factor)],
            #                           width=line_width, fill=line_color)
            # show_gt_im.show()
            im.save(os.path.join(train_image_dir, o_img_fname))
            np.save(os.path.join(
                train_label_dir,
                o_img_fname[:-4] + '_gt.npy'),
                gt)
            train_val_set.append('{},{},{}\n'.format(o_img_fname,
                                                     d_wight,
                                                     d_height))

    train_img_list = os.listdir(train_image_dir)
    print('found %d train images.' % len(train_img_list))
    train_label_list = os.listdir(train_label_dir)
    print('found %d train labels.' % len(train_label_list))

    random.shuffle(train_val_set)
    val_count = int(cfg.valid_split_ratio * len(train_val_set))  # 0.1
    with open(os.path.join(data_dir, cfg.val_fname), 'w') as f_val:
        f_val.writelines(train_val_set[:val_count])
    with open(os.path.join(data_dir, cfg.train_fname), 'w') as f_train:
        f_train.writelines(train_val_set[val_count:])


if __name__ == '__main__':
    preprocess()
