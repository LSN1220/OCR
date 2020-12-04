import os

root_dir = os.path.join(os.path.abspath(
    os.path.dirname(__file__)
).split('east')[0], 'east')

# data_config
data_dir = 'D:/GIT/github/data/icpr_text/train_1000'
train_task_id = '3T512'
train_image_dir_name = 'images_%s/' % train_task_id
train_label_dir_name = 'labels_%s/' % train_task_id
train_fname = 'train_%s.txt' % train_task_id
val_fname = 'val_%s.txt' % train_task_id
max_train_img_height = 256  # 1088  # 17 * 64
max_train_img_width = 256  # 768
pixel_size = 4

# training_config
batch_size = 1
pretrain = False
pretrain_weight = ''
pths_path = os.path.join(root_dir, '/new_pths')
lr = 1e-3
decay = 5e-4
epochs = 25
interval = 5
log_file = os.path.join(root_dir, '/new_log_file.txt')

# losses_config
epsilon = 1e-4
lambda_inside_score_loss = 4.0
lambda_side_v_code_loss = 1.0
lambda_side_v_coord_loss = 1.0

# eval_config
pixel_threshlod = 0.9
quiet = True