import os
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import east_config as cfg
from net.east_network import east_network
from data.data_generator import gen


def train(start_epoch, stop_epoch, max_train_img_size, downsample_factor, lr, batch_size, load_weight):
    max_train_img_h = 0
    max_train_img_w = 0
    if isinstance(max_train_img_size, int):
        max_train_img_h, max_train_img_w = max_train_img_size, max_train_img_size
    elif isinstance(max_train_img_size, tuple):
        max_train_img_h, max_train_img_w = max_train_img_size

    train_gen = gen(max_train_img_h, max_train_img_w, downsample_factor, batch_size, is_val=False)
    valid_gen = gen(max_train_img_h, max_train_img_w, downsample_factor, 1, is_val=True)
    with open(os.path.join(cfg.data_dir, cfg.train_fname), 'r') as f_train:
        f_list = f_train.readlines()
        train_img_num = len(f_list)
    with open(os.path.join(cfg.data_dir, cfg.val_fname), 'r') as f_val:
        f_list = f_val.readlines()
        valid_img_num = len(f_list)

    model, input, y_pred = east_network()

    test_func = K.function([input], [y_pred])

    # viz_cb = VizCallback()

    tensorboard = TensorBoard(log_dir=os.path.join(cfg.job_output_path, 'logs'),
                              histogram_freq=0, write_graph=True, embeddings_freq=0)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0,
                               patience=100, verbose=0, mode='auto')
    callbacks_list = [tensorboard,
                      # viz_cb,
                      early_stop]

    if start_epoch > 0:
        weight_file = os.path.join(cfg.models_dir, 'east_detect-%04d.hdf5' % (start_epoch))
        model.load_weights(weight_file)
    elif load_weight is not None:
        weight_file = load_weight
        model.load_weights(weight_file)

    model.fit_generator(generator=train_gen,
                        steps_per_epoch=train_img_num // batch_size,
                        epochs=stop_epoch,
                        validation_data=valid_gen,
                        validation_steps=valid_img_num // batch_size,
                        initial_epoch=start_epoch,
                        callbacks=callbacks_list)


if __name__ == "__main__":
    train(cfg.start_epoch, cfg.stop_epoch, cfg.max_train_img_size, cfg.downsample_factor, cfg.lr, cfg.batch_size, cfg.load_weights)
    K.clear_session()
