import torch
import torch.nn as nn
from text_detect.pixel_anchor.model.resnet50 import Resnet
from text_detect.pixel_anchor.model.anchornet import Anchornet
from text_detect.pixel_anchor.model.pixelnet import Pixelnet
import torchvision.models as models

resnet50 = models.resnet50(pretrained=True)


class PixelAnchornet(nn.Module):
    def __init__(self, pretrained):
        print("----进入PixelAnchornet----")
        super(PixelAnchornet, self).__init__()
        self.resnet = Resnet([3, 4, 6, 3], 1000)
        # -----------------------------在主干网络中加载与resnet50训练模型----------------
        if pretrained:
            pretrained_dict = resnet50.state_dict()
            resnet_dict = self.resnet.state_dict()

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in resnet_dict}
            resnet_dict.update(pretrained_dict)

            self.resnet.load_state_dict(resnet_dict)
        # -----------------------------在主干网络中加载resnet50训练模型
        self.anchornet = Anchornet(256)
        self.pixelnet = Pixelnet()  # 已加激活函数和bn

    def forward(self, x):
        # print('------------------传入PixelAnchornet中的输入x:',x)
        resnet_list = self.resnet(x)
        one_div4 = resnet_list[0]  # torch.Size([5, 256, 160, 160])
        # print('-----------------传入PixelAnchornet中1/4features.sum():',one_div4.sum())
        # print('输入图片1/4的size：', one_div4.size())
        one_div8 = resnet_list[1]  # torch.Size([5, 512, 80, 80])
        # print('-----------------传入PixelAnchornet中1/8features.sum():',one_div8.sum())
        # print('one_div8.size():', one_div8.size())
        one_div16 = resnet_list[2]  # torch.Size([5, 1024, 40, 40])
        # print('-----------------传入PixelAnchornet中1/16features.sum():', one_div16.sum())
        # print('one_div16.size():', one_div16.size())

        score, geo, attention_map = self.pixelnet(resnet_list)
        # print('score.size():',score.size())
        # print('geo.size():',geo.size())
        # print('attention_amp:',attention_map.size())

        pre_location, pre_class = self.anchornet(one_div4, one_div16, attention_map)
        # print('--------------pre_location.size()----------:',pre_location.size())
        # print('--------------pre_Class.size()--------------:',pre_class.size())
        return score, geo, attention_map, pre_location, pre_class


if __name__ == '__main__':
    for i in range(100):
        a = torch.randn(1, 3, 512, 512)
        mybackbone = PixelAnchornet(pretrained=False)
        score, geo, pre_location, pre_class = mybackbone(a)
        print('score:', score.size())
        print('geo:', geo.size())
        print('pre_location.size()', pre_location.size())
        print('pre_class.size()', pre_class.size())
