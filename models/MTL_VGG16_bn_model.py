import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms

from torch.autograd import Variable

# from config import config, parser
from copy import deepcopy
import numpy as np

# __all__ = [
#     'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
#     'vgg19_bn', 'vgg19',
# ]


# model_urls = {
#     'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
#     'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
#     'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
#     'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
#     'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
#     'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
#     'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
#     'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
# }


# class VGG(nn.Module):

#     def __init__(self, features, num_classes=1000, init_weights=True):
#         super(VGG, self).__init__()
#         self.features = features
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes),
#         )
#         if init_weights:
#             self._initialize_weights()

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)


# def make_layers(cfg, batch_norm=False):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)


# cfg = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

# def vgg16(pretrained=False, **kwargs):
#     """VGG 16-layer model (configuration "D")
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG(make_layers(cfg['D']), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
#     return model


# def vgg16_bn(pretrained=False, **kwargs):
#     """VGG 16-layer model (configuration "D") with batch normalization
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
#     return model



class MTL_VGG16_bn_model(torch.nn.Module):
  def __init__(self, gen_classes= 2, smile_classes = 2, emo_classes = 7, age_classes = 100):
    super(MTL_VGG16_bn_model, self).__init__()

    self.MTL_vgg16_bn_features = models.vgg16_bn(pretrained=True).features

    self.features_length = 25088

    self.use_gpu = torch.cuda.is_available()

    # gender branch
    self.gender_clf = nn.Sequential(
        nn.Linear(self.features_length, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(256, gen_classes)
    )

    # smile branch
    self.smile_clf = nn.Sequential(
        nn.Linear(self.features_length, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(256, smile_classes)
    )
    
    # emotion branch
    self.emotion_clf = nn.Sequential(
        nn.Linear(self.features_length, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(256, emo_classes)
    )

    self.age_clf = nn.Sequential(
        nn.Linear(self.features_length, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(256, age_classes)
    )




#   def get_face_attribute(self, last_conv_out):
#     # last_conv_out = self.vgg11_bn.avgpool(last_conv_out)
#     last_conv_out = last_conv_out.view(last_conv_out.size(0), -1)

#     gen_pred = self.dropout(F.relu(self.fc1(last_conv_out)))
#     gen_pred = self.gen_cls_pred(gen_pred)

#     smile_pred   = self.dropout(F.relu(self.fc2(last_conv_out)))
#     smile_pred   = self.smile_cls_pred(smile_pred)

#     emo_pred = self.dropout(F.relu(self.fc3(last_conv_out)))
#     emo_pred = self.emo_cls_pred(emo_pred)

#     age_pred = self.dropout(F.relu(self.fc4(last_conv_out)))
#     age_pred = F.softmax(self.age_cls_pred(age_pred), 1)

#     return gen_pred, smile_pred, emo_pred, age_pred


  def forward(self, x):
    # last1 = self.get_convs_out(x)
    x = self.MTL_vgg16_bn_features(x)
    x = x.view(x.size(0), -1)

    gen_pred  = self.gender_clf(x)
    smile_pred  = self.smile_clf(x)
    emo_pred  = self.emotion_clf(x)
    age_pred  = self.age_clf(x)

    return gen_pred, smile_pred, emo_pred, age_pred 


if __name__ == "__main__":
    model = MTL_VGG16_bn_model()
    print("MTL_VGG16_bn_model")
    print(model)