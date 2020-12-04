import os
import random


def data_split(full_list, ratio, shuffle=False):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


if __name__ == '__main__':
    path = '/home/ai/lvsongnan/data/crnn_data_jpn_random/jpn/tmp_labels.txt'
    save_dir = '/home/ai/lvsongnan/data/crnn_data_jpn_random/jpn'
    with open(path, 'r', encoding='utf-8') as f:
        samples = f.readlines()
    train_list, test_list = data_split(samples, 0.8, shuffle=True)
    with open(os.path.join(save_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.writelines(train_list)
    with open(os.path.join(save_dir, 'test.txt'), 'w', encoding='utf-8') as f:
        f.writelines(test_list)
