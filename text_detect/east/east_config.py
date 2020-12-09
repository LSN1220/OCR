import os

job_dir = os.path.join(os.path.abspath(
    os.path.dirname(__file__)
).split('east')[0], 'east')

# data generator set
valid_split_ratio = 0.1

# data dir
data_id = '3T256'
data_dir = 'D:/GIT/github/data/icpr_text/train_1000/'
origin_image_dir_name = 'image_1000/'
origin_txt_dir_name = 'txt_1000/'
train_image_dir_name = 'images_%s/' % data_id
train_label_dir_name = 'labels_%s/' % data_id
train_fname = 'train_%s.txt' % data_id
val_fname = 'val_%s.txt' % data_id
"""
data_id = 'contract'
data_dir = 'D:/GIT/github/data/Contract_data'
origin_image_dir_name = 'images'
origin_txt_dir_name = 'images'
train_image_dir_name = 'images_%s/' % data_id
train_label_dir_name = 'labels_%s/' % data_id
train_fname = 'train_%s.txt' % data_id
val_fname = 'val_%s.txt' % data_id
"""

# training config
job_name = 'job_east_1208'
start_epoch = 0
stop_epoch = 25
max_train_img_size = 256  # (1088, 768)
lr = 1e-3
batch_size = 8
load_weights = os.path.join(job_dir, 'pre_model/east_model_weights_1.h5')  # None  # None or weight path
downsample_factor = 4

# predict config
max_pred_img_size = (1088, 768)
trunc_threshold = 0.1
side_vertex_pixel_threshold = 0.8

# east network set
locked_layers = False
feature_layers_range = range(5, 1, -1)
feature_layers_num = len(feature_layers_range)
lambda_inside_score_loss = 4.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0

# workspace
job_output_path = os.path.join(job_dir, job_name)
models_dir = os.path.join(job_output_path, 'checkpoints/')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


