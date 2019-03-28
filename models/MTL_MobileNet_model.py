import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

import torch.nn.functional as F

from torchvision import models


class MTL_MobileNet_model(torch.nn.Module):
    def __init__(self):
        super(MTL_MobileNet_model, self).__init__()
        self.mobilenet = models.mobilenet(pretrained=True)

        # age branch
        self.fc1          = nn.Linear(50176, 512)
        self.age_cls_pred = nn.Linear(512, 100)

        # gender branch
        self.fc2          = nn.Linear(50176, 512)
        self.gen_cls_pred = nn.Linear(512, 2)

        # emotion branch
        self.dropout      = nn.Dropout(p=0.5, inplace=False)
        self.fc3          = nn.Linear(50176, 512)
        self.emo_cls_pred = nn.Linear(512, 7)

        # smile branch
        self.fc4          = nn.Linear(50176, 512)
        self.smile_cls_pred = nn.Linear(512, 2)

    def get_densenet_convs_out(self, x):
        """
        get outputs from convolutional layers of ResNet
        :param x: image input
        :return: middle ouput from layer2, and final ouput from layer4
        """
        x = self.mobilenet.features(x)    # out = [N, 64, 112, 112]

        return x

    def get_age_gender_emotion(self, last_conv_out):
        # last_conv_out = self.denseNet.avgpool(last_conv_out)
        last_conv_out = last_conv_out.view(last_conv_out.size(0), -1)

        # print("last_conv_out: ", last_conv_out.size())

        age_pred = F.relu(self.dropout(self.fc1(last_conv_out)))
        age_pred = F.softmax(self.age_cls_pred(age_pred), 1)

        gen_pred = F.relu(self.dropout(self.fc2(last_conv_out)))
        gen_pred = self.gen_cls_pred(gen_pred)

        emo_pred = F.relu(self.dropout(self.fc3(last_conv_out)))
        emo_pred = self.emo_cls_pred(emo_pred)

        smile_pred   = F.relu(self.dropout(self.fc4(last_conv_out)))
        smile_pred   = self.smile_cls_pred(smile_pred)

        return gen_pred, age_pred, emo_pred, smile_pred
    
    def forward(self, x, return_atten_info = False):
        last1 = self.get_densenet_convs_out(x)
        gen_pred, age_pred, emo_pred, smile_pred = self.get_age_gender_emotion(last1)
        return gen_pred, age_pred, emo_pred, smile_pred


if __name__ == "__main__":
    mobilenet_model = MTL_MobileNet_model()
    print("MobileNet done")