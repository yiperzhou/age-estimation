import torch
from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                      padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim,
                       stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(
            input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def mobilenet_v2(pretrained=False, **kwargs):
    return MobileNetV2(**kwargs)


class RegressionAndClassificationMobileNetv1(torch.nn.Module):
    def __init__(self, args):
        super(RegressionAndClassificationMobileNetv1, self).__init__()
        backbone_model = mobilenet_v2(pretrained=True)
        self.MobileNet_V1_model_feature = nn.Sequential(
            backbone_model.features,
            torch.nn.AvgPool2d(kernel_size=(7, 7))
        )
        self.features_length = 1280
        self.args = args
        self.use_gpu = torch.cuda.is_available()
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
            print(
                "age_divide_100_classes, age_divide_20_classes, age_divide_10_classes, age_divide_5_classes")
            raise ValueError

        return age_divide_100_classes, age_divide_20_classes, age_divide_10_classes, age_divide_5_classes

    def forward(self, x):
        x = self.MobileNet_V1_model_feature(x)
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
