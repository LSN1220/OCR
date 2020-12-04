import os
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from text_detect_torch.east import cfg
from text_detect_torch.east.data.data_generator import getDataset
from text_detect_torch.east.net.network import east
from text_detect_torch.east.net.losses import quad_loss
# from text_detect_torch.east.net.utils import eval_pre_rec_f1

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


def train(batch_size, pretrain=False, epochs=25, interval=5):
    train_data = getDataset()
    val_data = getDataset(is_val=True)
    train_file_num = len(train_data)
    val_file_num = len(val_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True,
                            num_workers=0, pin_memory=True)
    model = east()

    if pretrain and os.path.exists(cfg.pretrain_weight):
        model.load_state_dict(torch.load(cfg.pretrain_weight, map_location=device))

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.lr,
                                 weight_decay=cfg.decay)
    loss_func = quad_loss

    # eval_p_r_f = eval_pre_rec_f1()

    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0
        epoch_val_loss = 0
        epoch_time = time.time()
        for i, (image_tensors, labels) in enumerate(train_loader):
            start_time = time.time()
            batch_x = image_tensors.to(device).float()
            batch_y = labels.to(device).float()

            pred = model(batch_x)
            loss = loss_func(batch_y, pred)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(
                epoch + 1, epochs, i + 1, int(train_file_num / batch_size), time.time() - start_time, loss.item()))

        model.eval()
        for i, (image_tensors, labels) in enumerate(val_loader):
            batch_x = image_tensors.to(device).float()
            batch_y = labels.to(device).float()

            pred = model(batch_x)
            loss = loss_func(batch_y, pred)
            epoch_val_loss += loss

        print('epoch_loss is {:.8f}, epoch_val_loss is {:.8f}, epoch_time is {:.8f}'.format(
            epoch_loss / int(train_file_num / batch_size),
            epoch_val_loss / int(val_file_num / 1),
            time.time() - epoch_time
        ))

        with open(cfg.log_file, 'a') as f:
            f.write('epoch is {},epoch_loss is {}, epoch_val_loss is {}\n'.format(
                epoch,
                epoch_loss / int(train_file_num / batch_size),
                epoch_val_loss / int(val_file_num / 1)
            ))

        if (epoch + 1) % interval == 0:
            state_dict = model.state_dict()
            if not os.path.exists(cfg.pths_path):
                os.mkdir(cfg.pths_path)
            torch.save(state_dict, os.path.join(cfg.pths_path, 'model_epoch_{}.pth'.format(epoch + 1)))


if __name__ == "__main__":
    train(cfg.batch_size, cfg.pretrain, cfg.epochs, cfg.interval)
