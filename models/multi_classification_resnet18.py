import torch
import torch.nn as nn
from torchvision import models


class MultiClassificationResNet18Model(torch.nn.Module):
    def __init__(self, args):
        super(MultiClassificationResNet18Model, self).__init__()
        self.resNet = models.resnet18(pretrained=True)
        self.use_gpu = torch.cuda.is_available()
        self.args = args
        self.features_length = 512
        self.age_divide_100_classes, self.age_divide_20_classes, self.age_divide_10_classes, self.age_divide_5_classes = self.get_age_cls_class()
        self.age_clf_100_classes = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 100)
        )
        self.age_clf_20_classes = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 20)
        )
        self.age_clf_10_classes = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 10)
        )
        self.age_clf_5_classes = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 5)
        )

    def get_resnet_convs_out(self, x):
        """
        get outputs from convolutional layers of ResNet
        :param x: image input
        :return: middle ouput from layer2, and final ouput from layer4
        """
        x = self.resNet.conv1(x)  # out = [N, 64, 112, 112]
        x = self.resNet.bn1(x)
        x = self.resNet.relu(x)
        x = self.resNet.maxpool(x)  # out = [N, 64, 56, 56]
        x = self.resNet.layer1(x)  # out = [N, 64, 56, 56]
        x = self.resNet.layer2(x)  # out = [N, 128, 28, 28]
        x = self.resNet.layer3(x)  # out = [N, 256, 14, 14]
        x = self.resNet.layer4(x)  # out = [N, 512, 7, 7]
        return x

    def get_age_cls_class(self):
        age_divide_100_classes = False
        age_divide_20_classes = False
        age_divide_10_classes = False
        age_divide_5_classes = False
        if self.args.age_classification_combination == [1, 0, 0, 0]:
            age_divide_100_classes = True
            age_divide_20_classes = False
            age_divide_10_classes = False
            age_divide_5_classes = False
        elif self.args.age_classification_combination == [1, 1, 0, 0]:
            age_divide_100_classes = True
            age_divide_20_classes = True
            age_divide_10_classes = False
            age_divide_5_classes = False
        elif self.args.age_classification_combination == [1, 1, 1, 0]:
            age_divide_100_classes = True
            age_divide_20_classes = True
            age_divide_10_classes = True
            age_divide_5_classes = False
        elif self.args.age_classification_combination == [1, 1, 1, 1]:
            age_divide_100_classes = True
            age_divide_20_classes = True
            age_divide_10_classes = True
            age_divide_5_classes = True
        else:
            print("age_divide_100_classes, age_divide_20_classes, age_divide_10_classes, age_divide_5_classes")
            ValueError
        return age_divide_100_classes, age_divide_20_classes, age_divide_10_classes, age_divide_5_classes

    def forward(self, x, return_atten_info=False):
        x = self.get_resnet_convs_out(x)
        x = x.view(x.size(0), -1)
        age_pred_100_classes, age_pred_20_classes, age_pred_10_classes, age_pred_5_classes = None, None, None, None
        if self.age_divide_100_classes == True:
            age_pred_100_classes = self.age_clf_100_classes(x)
        if self.age_divide_20_classes == True:
            age_pred_20_classes = self.age_clf_20_classes(x)
        if self.age_divide_10_classes == True:
            age_pred_10_classes = self.age_clf_10_classes(x)
        if self.age_divide_5_classes == True:
            age_pred_5_classes = self.age_clf_5_classes(x)
        return age_pred_100_classes, age_pred_20_classes, age_pred_10_classes, age_pred_5_classes