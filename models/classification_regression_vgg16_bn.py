import torch
import torch.nn as nn
from torchvision import models

class RegressionAndClassificationVGG16bn(torch.nn.Module):
    def __init__(self, args):
        super(RegressionAndClassificationVGG16bn, self).__init__()
        self.vgg16_bn_features = models.vgg16_bn(pretrained=True).features
        self.features_length = 25088
        self.use_gpu = torch.cuda.is_available()
        self.args = args
        self.age_divide_100_classes, self.age_divide_20_classes, \
        self.age_divide_10_classes, self.age_divide_5_classes = self.get_age_cls_class()
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
        self.age_regression = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 1)   # output layer, output one neurons, to do regression.
        )

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
            raise ValueError

        return age_divide_100_classes, age_divide_20_classes, age_divide_10_classes, age_divide_5_classes

    def forward(self, x):
        x = self.vgg16_bn_features(x)
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
        if self.args.mse_regression_loss == True:
            age_regression = self.age_regression(x)

        return [age_pred_100_classes, age_pred_20_classes, age_pred_10_classes, age_pred_5_classes], age_regression
