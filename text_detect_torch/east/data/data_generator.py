import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from text_detect_torch.east import cfg


class getDataset(Dataset):
    def __init__(self, is_val=False):
        train_image_dir = os.path.join(cfg.data_dir, cfg.train_image_dir_name)
        train_label_dir = os.path.join(cfg.data_dir, cfg.train_label_dir_name)
        self.img_h, self.img_w = cfg.max_train_img_height, cfg.max_train_img_width
        if is_val:
            with open(os.path.join(cfg.data_dir, cfg.val_fname), 'r') as f_val:
                f_list = f_val.readlines()
        else:
            with open(os.path.join(cfg.data_dir, cfg.train_fname), 'r') as f_tr:
                f_list = f_tr.readlines()
        self.image_path_list = []
        self.labels_path_dic = {}
        for line in f_list:
            img_name = str(line).strip().split(',')[0]
            img_path = os.path.join(train_image_dir, img_name)
            self.image_path_list.append(img_path)
            gt_file = os.path.join(train_label_dir, img_name[:-4] + '_gt.npy')
            self.labels_path_dic[img_path] = gt_file
        self.n_sample = len(self.image_path_list)

    def __getitem__(self, index):
        img_path = self.image_path_list[index]
        label = np.load(self.labels_path_dic[img_path])
        try:
            img = Image.open(img_path).convert('RGB')
        except IOError:
            print(f'Corrupted image for {index}')
            img = Image.new('RGB', (self.img_w, self.img_h))
        img_tensor = transforms.ToTensor()(img)
        label = np.transpose(label, (2, 0, 1))
        return img_tensor, label

    def __len__(self):
        return self.n_sample
