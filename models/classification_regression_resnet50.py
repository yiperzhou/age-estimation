import torch
import torch.nn as nn
from torchvision import models


class RegressionAndClassificationResNet50(torch.nn.Module):
    def __init__(self, args):
        super(RegressionAndClassificationResNet50, self).__init__()
        resnet50_model = models.resnet50(pretrained=True)
        self.resnet50_features = nn.Sequential(
            resnet50_model.conv1,
            resnet50_model.bn1,
            resnet50_model.relu,
            resnet50_model.maxpool,
            resnet50_model.layer1,
            resnet50_model.layer2,
            resnet50_model.layer3,
            resnet50_model.layer4,
            resnet50_model.avgpool
        )
        self.features_length = 2048
        self.use_gpu = torch.cuda.is_available()
        self.args = args

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

        self.age_regression_100_classes = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 1)  # output layer, output one neurons, to do regression.
        )

        self.age_regression_20_classes = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 1)  # output layer
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
            ValueError
        return age_divide_100_classes, age_divide_20_classes, age_divide_10_classes, age_divide_5_classes

    def forward(self, x):
        x = self.resnet50_features(x)
        x = x.view(x.size(0), -1)
        age_pred_100_classes, age_pred_20_classes, age_pred_10_classes, age_pred_5_classes = None, None, None, None
        if self.args.age_divide_100_classes == True:
            age_pred_100_classes = self.age_clf_100_classes(x)
        if self.args.age_divide_20_classes == True:
            age_pred_20_classes = self.age_clf_20_classes(x)
        if self.args.age_divide_10_classes == True:
            age_pred_10_classes = self.age_clf_10_classes(x)
        if self.args.age_divide_5_classes == True:
            age_pred_5_classes = self.age_clf_5_classes(x)
        if self.args.mse_regression_loss == True:
            # age_regression = self.age_regression(x)
            [age_regression_100_classes, age_regression_20_classes] = self.age_regression_100_classes(x), self.age_regression_20_classes(x)

        return [age_pred_100_classes, age_pred_20_classes, age_pred_10_classes, age_pred_5_classes], [age_regression_100_classes, age_regression_20_classes]
