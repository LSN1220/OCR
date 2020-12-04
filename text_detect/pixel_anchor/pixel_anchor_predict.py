import os
import sys
import math
import numpy as np
import torch
from torchvision import transforms
from PIL import ImageDraw
import PIL.Image as PI
from wand.image import Image, Color
from text_detect.pixel_anchor.model.backbone import PixelAnchornet
from text_detect.pixel_anchor.utils.anchorencoder import DataEncoder
from text_detect.pixel_anchor.utils.anchor_nms_poly import non_max_suppression_poly

root_path = os.path.join(os.path.abspath(os.path.dirname(__file__)).split('ocr')[0], 'ocr')
# ----get_sample----
sample_dir = os.path.join(root_path, 'sample/')
sample_list = os.listdir(sample_dir)
# ----output_save----
image_save_path = os.path.join(root_path, 'pixel_anchor_output/')
if not os.path.exists(image_save_path):
    os.mkdir(image_save_path)
# ----pre_model----
pre_model_weight = os.path.join(root_path, 'text_detect/pixel_anchor/pre_model/model_epoch_400.pth')


# def resize_img(img):
#     w, h = img.size
#     resize_w = w
#     resize_h = h
#
#     resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
#     resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
#     img = img.resize((resize_w, resize_h), PI.BILINEAR)
#     ratio_h = resize_h / h
#     ratio_w = resize_w / w
#     return img, ratio_h, ratio_w
def resize_image(im, max_img_size=640):
    im_width = np.minimum(im.width, max_img_size)
    if im_width == max_img_size < im.width:     # 起到and的作用
        im_height = int((im_width / im.width) * im.height)
    else:
        im_height = im.height

    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width
    # fixme 最多裁剪31个pixel 是否影响边缘效果
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height


def get_rotate_mat(theta):  # 得到某点按照某弧度旋转后的所需要相乘的旋转矩阵
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def is_valid_poly(res, score_shape, scale):
    cnt = 0
    for i in range(res.shape[1]):
        if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or \
                res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
            cnt += 1
    return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
    polys = []  # 返回的最终的预测框
    index = []  # 返回的最终预测框的序列id
    valid_pos *= scale  # 得到原图上有效点的坐标
    d = valid_geo[:4, :]  # 4 x N   #得到有效的每个像素点与预测框之间的距离
    angle = valid_geo[4, :]  # N,   #得到每个像素点的预测框角度
    for i in range(valid_pos.shape[0]):  # 循环每个有效的像素点
        x = valid_pos[i, 0]  # 每个有效像素点的x坐标
        y = valid_pos[i, 1]  # --------------y坐标
        y_min = y - d[0, i]  # 像素点对应预测框 ymin
        y_max = y + d[1, i]  # ------------- ymax
        x_min = x - d[2, i]  # ------------- xmin
        x_max = x + d[3, i]  # ------------- xmax
        rotate_mat = get_rotate_mat(-angle[i])  # 每个像素点对应的旋转角度矩阵

        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x  # 每个像素点对应的预测框4个点x的坐标与像素点坐标x间的距离
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y  # 每个像素点对应的预测框4个点y的坐标与像素点坐标y间的距离
        coordidates = np.concatenate((temp_x, temp_y), axis=0)  # 2*4
        res = np.dot(rotate_mat, coordidates)  # 得到旋转后的偏差值
        res[0, :] += x
        res[1, :] += y
        # 最后得到的res为旋转后的预测框坐标
        if is_valid_poly(res, score_shape, scale):  # 判断是否为有效的框
            index.append(i)
            polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1], res[0, 2], res[1, 2], res[0, 3], res[1, 3]])
    return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2, if_eval=True):
    score = score[0, :, :]  # 去掉通道1的维度
    xy_text = np.argwhere(score > score_thresh)  # n x 2, format is [r, c] #
    # 得到score大于置信度阈值的坐标 (n,2),n为n个像素点
    if xy_text.size == 0:
        return None
    xy_text = xy_text[np.argsort(xy_text[:, 0])]  # 将置信度阈值大于一定值的坐标，
    # 按照每个点的行坐标进行排序后的得到的xy_text中的从小到大的行索引，
    valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]# 将y,x 转换为x,y
    # 有效的坐标点
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n #经过阈值筛选后的有效geo 5*n n为有效像素点的个数。
    polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape)  # 得到最终的预测框集合，以及在valid_pos中的id序号

    if polys_restored.size == 0:
        return None
    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)  #
    boxes[:, :8] = polys_restored  # 装最终所有预测框4个点的信息
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]  # 对应预测框的置信度值
    # if if_eval:
    #     return boxes
    # boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)  #  经过NMS返回最终的预测框
    #
    # print('-----------------pixel中未经过NMS后的boxes.shape---------------:',boxes.shape)
    return boxes


def plot_boxes(img, boxes):
    if boxes is None:
        print('boxes is none')
        return img

    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0, 0, 255))
    return img


def adjust_ratio(boxes, ratio_w, ratio_h):
    if boxes is None or boxes.size == 0:
        return None
    boxes[:, [0, 2, 4, 6]] /= ratio_w
    boxes[:, [1, 3, 5, 7]] /= ratio_h
    return np.around(boxes)


def pixle_anchor_detect(img, model, NMS_choice='', NMS_thresh=0.1, img_save_pths='', img_size=640):
    orignal_img = img.copy()
    width, heigth = img.size[0], img.size[1]
    d_wight, d_height = resize_image(img, img_size)
    ratio_w = d_wight / img.width
    ratio_h = d_height / img.height
    img = img.resize((d_wight, d_height), PI.BILINEAR).convert('RGB')
    # t = transforms.Compose([transforms.ToTensor(),
    #                         transforms.Normalize(mean=(0.5, 0.5, 0.5),
    #                                              std=(0.5, 0.5, 0.5))])
    # img_input = t(img).unsqueeze(0).to(device)
    transform = transforms.Compose([
        transforms.Resize((d_wight, d_height), interpolation=2),
        transforms.ToTensor()
    ])
    x = transform(img)
    x = torch.unsqueeze(x, 0)  # 增加一个维度
    # -------------------------pixel_detect部分----------------
    with torch.no_grad():
        score, geo, attention_map, pre_location, pre_class = model(x)

    pixel_boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy(), if_eval=True)
    # 设置为eval为true的时候，都不经过各自的nms

    print('----------------------pixel_boxes.shape:', pixel_boxes.shape)  # (n, 9)

    # ---------------------anchor_detect部分---------------
    decoder = DataEncoder()
    anchor_boxes, labels, scores = decoder.decode(pre_location.data.squeeze(0), pre_class.data.squeeze(0),
                                                  img.size, if_eval=True)
    # 得到的为经过阈值筛选后的，并没有经过NMS
    if pixel_boxes is None and anchor_boxes is None:
        return pixel_boxes
    if pixel_boxes is None and anchor_boxes is not None:
        score = np.expand_dims(scores, axis=1)  # 将置信度升维
        # anchor_boxes=anchor_boxes.reshape(-1,8) #改变anchor_boxes的形状，与置信度拼接在一起
        total_boxes = np.hstack((anchor_boxes, score))  # 拼接在一起
    if pixel_boxes is not None and anchor_boxes is None:
        total_boxes = pixel_boxes
    if pixel_boxes is not None and anchor_boxes is not None:
        # new_anchor_boxes=np.expand_dims(anchor_boxes,axis=0)
        score = np.expand_dims(scores, axis=1)  # 将置信度升维
        # anchor_boxes=anchor_boxes.reshape(-1,8) #改变anchor_boxes的形状，与置信度拼接在一起
        anchor_total_boxes = np.hstack((anchor_boxes, score))  # 拼接在一起
        anchor_total_boxes[:, 8] += 0.5  # 提升从anchor部分中出来的置信度设置为
        total_boxes = np.vstack((anchor_total_boxes, pixel_boxes))

    # if NMS_choice=='lanms': #使用lanms模块的nms
    #     total_boxes = lanms.merge_quadrangle_n9(total_boxes.astype('float32'), NMS_thresh)
    #     final_boxes = adjust_ratio(total_boxes, ratio_w, ratio_h)
    #
    #     final_boxes[:,:8]/=img_size
    #     final_boxes[:,[0,2,4,6]]*=width
    #     final_boxes[:,[1,3,5,7]]*=heigth
    #     # print('-----------------------经过NMS的final_boxes:',final_boxes)
    #     # plot_img = plot_boxes(orignal_img, final_boxes)
    #     # plot_img.save(img_save_pths)
    #     return final_boxes

    if NMS_choice == 'ssd':  # 使用SSD模块的nms
        score = total_boxes[:, 8]
        total_boxes = total_boxes[:, :8]
        total_boxes = total_boxes.reshape(-1, 4, 2)
        keep = non_max_suppression_poly(total_boxes, score, NMS_thresh)  # anchor部分NMS
        total_boxes = total_boxes[keep]
        total_boxes = total_boxes.reshape(-1, 8)
        total_boxes /= img_size
        total_boxes[:, [0, 2, 4, 6]] *= width
        total_boxes[:, [1, 3, 5, 7]] *= heigth

        final_boxes = adjust_ratio(total_boxes, ratio_w, ratio_h)

        print('-----------------------经过NMS的final_boxes:', final_boxes)
        plot_img = plot_boxes(orignal_img, final_boxes)
        plot_img.save(img_save_pths)
        return final_boxes


if __name__ == '__main__':
    # ----set_device----
    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # ----get_model----
    net = PixelAnchornet(pretrained=False)  # .to(device)
    model_checkpoint = torch.load(pre_model_weight, map_location=torch.device('cpu'))
    net.load_state_dict(model_checkpoint)
    net.eval()
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
            print('-----------------传入图片的img.size:', img.size)
            pixle_anchor_detect(img, net,
                                NMS_choice='ssd',
                                img_save_pths=os.path.join(image_save_path,
                                                           f'{im_name}_{i}.jpg'),
                                img_size=2048
                                )
