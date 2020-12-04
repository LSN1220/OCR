import torch
import torch.nn as nn
from torchvision import models

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


class east(nn.Module):
    def __init__(self):
        super(east, self).__init__()
        feats = list(models.vgg16(pretrained=True).features.children())
        self.lock_layers = nn.Sequential(*feats[0:4])
        self.feat1 = nn.Sequential(*feats[4:5])  # 64
        self.feat2 = nn.Sequential(*feats[5:10])  # 128
        self.feat3 = nn.Sequential(*feats[10:17])  # 256
        self.feat4 = nn.Sequential(*feats[17:24])  # 512
        self.feat5 = nn.Sequential(*feats[24:])  # 512
        self.conv1 = nn.Conv2d(1024, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(384, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(192, 32, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()

        self.inside_score = nn.Conv2d(32, 1, 1)
        # self.sigmoid1 = nn.Sigmoid()
        self.side_v_code = nn.Conv2d(32, 2, 1)
        # self.sigmoid2 = nn.Sigmoid()
        self.side_v_coord = nn.Conv2d(32, 4, 1)
        # self.sigmoid3 = nn.Sigmoid()

        self.upsamlpe1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamlpe2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamlpe3 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, images):
        # images = mean_image_subtraction(images)
        with torch.no_grad():
            lock_layers = self.lock_layers(images)
        f1 = self.feat1(lock_layers)
        f2 = self.feat2(f1)
        f3 = self.feat3(f2)
        f4 = self.feat4(f3)
        f5 = self.feat5(f4)

        g = self.upsamlpe1(f5)
        c = self.conv1(torch.cat((g, f4), 1))
        c = self.bn1(c)
        c = self.relu1(c)

        h = self.conv2(c)
        h = self.bn2(h)
        h = self.relu2(h)
        g = self.upsamlpe2(h)
        c = self.conv3(torch.cat((g, f3), 1))
        c = self.bn3(c)
        c = self.relu3(c)

        h = self.conv4(c)
        h = self.bn4(h)
        h = self.relu4(h)
        g = self.upsamlpe3(h)
        c = self.conv5(torch.cat((g, f2), 1))
        c = self.bn5(c)
        c = self.relu5(c)

        h = self.conv6(c)
        h = self.bn6(h)
        h = self.relu6(h)
        g = self.conv7(h)
        g = self.bn7(g)
        g = self.relu7(g)

        inside_score = self.inside_score(g)
        # inside_score = self.sigmoid1(inside_score)
        side_v_code = self.side_v_code(g)
        # side_v_code = self.sigmoid2(side_v_code)
        side_v_coord = self.side_v_coord(g)
        # side_v_coord = self.sigmoid3(side_v_coord)

        east_detect = torch.cat((inside_score, side_v_code, side_v_coord), 1)
        return east_detect


# vgg_pretrained_features = list(models.vgg16(pretrained=True).features.children())
# print(nn.Sequential(*vgg_pretrained_features[4:5]))
# model = east()
# print(model)
