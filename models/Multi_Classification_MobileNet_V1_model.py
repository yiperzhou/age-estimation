from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch
import os

from models.Elastic_MobileNet_V1 import Elastic_MobileNet

# not official model weights
model_urls = {
    'mobilenetV1': '/home/yi/Narvi_yi_home/projects/MultitaskLearningFace/resources/mobilenet_sgd_68.848.pth.tar'
}



class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()


        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3,  32, 2),
            conv_dw(32,  64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )

        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):

        x = self.model(x)

        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class Multi_Classification_MobileNet_V1_model(torch.nn.Module):
    def __init__(self, args, age_classes=100):
        super(Multi_Classification_MobileNet_V1_model, self).__init__()
    
        mobilenet_v1_model = Elastic_MobileNet()
        
        self.Multi_Classification_MobileNet_V1_model_feature = mobilenet_v1_model.model

        self.features_length = 1024

        self.use_gpu = torch.cuda.is_available()

        self.args = args

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
 

    def load_MobileNet_V1_ImageNet_pretrained_weight(self, mobilenet_v1_model):
        tar = torch.load(model_urls['mobilenetV1'])
        state_dict = tar['state_dict']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        mobilenet_v1_model = mobilenet_v1_model.load_state_dict(new_state_dict)

        # model.load_state_dict(model_zoo.load_url(model_urls['mobilenetV1']))
        # LOG("loaded ImageNet pretrained weights", logfile)
        print("[MobileNet_v1] loaded ImageNet pretrained weights")

        return mobilenet_v1_model

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
        x = self.Multi_Classification_MobileNet_V1_model_feature(x)

        # print(x.size())  # torch.Size([8, 1280, 7, 7])
        x = x.view(x.size(0), -1)
        # print(x.size())   # torch.Size([8, 62720])

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
