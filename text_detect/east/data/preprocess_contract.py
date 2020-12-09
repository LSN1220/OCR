import os
import sys
from tqdm import tqdm
import random
import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
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


def reorder_vertexes(xy_list):
    reorder_xy_list = np.zeros_like(xy_list)
    ordered = np.argsort(xy_list, axis=0)
    xmin1_index = ordered[0, 0]
    xmin2_index = ordered[1, 0]
    # 最左的点为第一个点,有两个最左选最靠上的一个.
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
    # 其余点与第一点的连线斜率中,值居中的点为第三点.
    others = list(range(4))
    others.remove(first_v)
    k = np.zeros((len(others),))
    for i, index in enumerate(others):
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) / (
                xy_list[index, 0] - xy_list[first_v, 0] + epsilon)
    k_mid = np.argsort(k)[1]
    third_v = others[k_mid]
    reorder_xy_list[2] = xy_list[third_v]
    # 在1-3中线之上的为第二点,另一个为第四点.
    others.remove(third_v)
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]
    second_v, fourth_v = 0, 0
    for i, index in enumerate(others):
        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:
            second_v = index
        else:
            fourth_v = index
    reorder_xy_list[1] = xy_list[second_v]
    reorder_xy_list[3] = xy_list[fourth_v]
    # 可能存在第一个点不是最左上的点,通过比较对角线斜率判断是否顺时针移动调整.
    k13 = k[k_mid]
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (
            xy_list[second_v, 0] - xy_list[fourth_v, 0] + epsilon)
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    return reorder_xy_list


def preprocess():
    max_train_img_size = cfg.max_train_img_size
    max_train_img_h = 0
    max_train_img_w = 0
    if isinstance(max_train_img_size, int):
        max_train_img_h, max_train_img_w = max_train_img_size, max_train_img_size
    elif isinstance(max_train_img_size, tuple):
        max_train_img_h, max_train_img_w = max_train_img_size

    origin_file_path = os.path.join(cfg.data_dir, cfg.origin_image_dir_name)
    train_image_dir = os.path.join(cfg.data_dir, cfg.train_image_dir_name)
    train_label_dir = os.path.join(cfg.data_dir, cfg.train_label_dir_name)
    if not os.path.exists(train_image_dir):
        os.mkdir(train_image_dir)
    if not os.path.exists(train_label_dir):
        os.mkdir(train_label_dir)
    dir_list = os.listdir(origin_file_path)
    # png_list = [file_name for file_name in dir_list if file_name.endswith('.png')]
    xml_list = [file_name for file_name in dir_list if file_name.endswith('.xml')]
    tr_val_set = []
    for xml, _ in zip(xml_list, tqdm(range(len(xml_list)))):
        path = os.path.join(origin_file_path, xml)
        dom_tree = ET.parse(path)
        root_node = dom_tree.getroot()
        image_name = root_node.attrib['image_path']
        image_path = os.path.join(origin_file_path, image_name)
        height = int(root_node.attrib['height'])
        width = int(root_node.attrib['width'])
        if height < width:  # 过滤横版合同
            continue
        if os.path.exists(os.path.join(train_label_dir, image_name[:-4] + '_gt.npy')):
            continue
        with Image.open(image_path) as im:
            d_wight, d_height = max_train_img_w, max_train_img_h
            scale_ratio_w = d_wight / width
            scale_ratio_h = d_height / height
            im = im.resize((d_wight, d_height), Image.CUBIC).convert('RGB')
            #
            # show_im_1 = im.copy()
            # show_im_2 = im.copy()
            # show_im_1.resize((d_wight, d_height), Image.BILINEAR).convert('RGB').show()
            # show_im_2.resize((d_wight, d_height), Image.CUBIC).convert('RGB').show()
            # draw = ImageDraw.Draw(show_gt_im)
            texts = root_node.findall("text")
            xy_list_array = np.zeros((len(texts), 4, 2))
            gt = np.zeros((d_height // cfg.downsample_factor, d_wight // cfg.downsample_factor, 7))
            for ind, text in enumerate(texts):
                y = int(text.attrib['top'])
                x = int(text.attrib['left'])
                w = int(text.attrib['width'])
                h = int(text.attrib['height'])
                anno_list = [x, y, x, y+h, x+w, y+h, x+w, y]  # 逆时针采点
                anno_array = np.array(anno_list)
                xy_list = np.reshape(anno_array.astype(float), (4, 2))
                xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w
                xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h
                xy_list = reorder_vertexes(xy_list)
                xy_list_array[ind] = xy_list
                # 训练标签
                # 缩小文本框
                _, shrink_xy_list, _ = shrink(xy_list, shrink_ratio=0.2)  # shrink_ratio = 0.2
                shrink_1, _, long_edge = shrink(xy_list, shrink_ratio=0.6)  # shrink_side_ratio = 0.6
                #
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
                # 计算像素采样范围
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
                            # draw.line([(px - 0.5 * pixel_size,
                            #             py - 0.5 * pixel_size),
                            #            (px + 0.5 * pixel_size,
                            #             py - 0.5 * pixel_size),
                            #            (px + 0.5 * pixel_size,
                            #             py + 0.5 * pixel_size),
                            #            (px - 0.5 * pixel_size,
                            #             py + 0.5 * pixel_size),
                            #            (px - 0.5 * pixel_size,
                            #             py - 0.5 * pixel_size)],
                            #           width=line_width, fill=line_color)
        # show_gt_im.show()
        im.save(os.path.join(train_image_dir, image_name))
        # np.save(os.path.join(train_label_dir, image_name[:-4] + '.npy'),
        #         xy_list_array)
        np.save(os.path.join(train_label_dir, image_name[:-4] + '_gt.npy'),
                gt)
        tr_val_set.append('{},{},{}\n'.format(image_name,
                                              d_wight,
                                              d_height))
    random.shuffle(tr_val_set)
    val_num = int(cfg.valid_split_ratio * len(tr_val_set))
    with open(os.path.join(cfg.data_dir, 'val.txt'), 'w') as f_val:
        f_val.writelines(tr_val_set[:val_num])
    with open(os.path.join(cfg.data_dir, 'train.txt'), 'w') as f_tr:
        f_tr.writelines(tr_val_set[val_num:])


if __name__ == "__main__":
    preprocess()
