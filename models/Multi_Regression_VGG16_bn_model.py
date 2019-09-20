import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms

from torch.autograd import Variable

# from config import config, parser
from copy import deepcopy
import numpy as np


class Multi_Regression_VGG16_bn_model(torch.nn.Module):
    def __init__(self, args, age_classes = 100):
        super(Multi_Regression_VGG16_bn_model, self).__init__()

        self.MTL_vgg16_bn_features = models.vgg16_bn(pretrained=True).features

        self.features_length = 25088

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



    # # gender branch
    # self.gender_clf = nn.Sequential(
    #     nn.Linear(self.features_length, 256),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(p=0.5, inplace=False),
    #     nn.Linear(256, gen_classes)
    # )

    # # smile branch
    # self.smile_clf = nn.Sequential(
    #     nn.Linear(self.features_length, 256),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(p=0.5, inplace=False),
    #     nn.Linear(256, smile_classes)
    # )
    
    # # emotion branch
    # self.emotion_clf = nn.Sequential(
    #     nn.Linear(self.features_length, 256),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(p=0.5, inplace=False),
    #     nn.Linear(256, emo_classes)
    # )

    # self.age_clf = nn.Sequential(
    #     nn.Linear(self.features_length, 256),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(p=0.5, inplace=False),
    #     nn.Linear(256, age_classes)
    # )

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

        # gen_pred  = self.gender_clf(x)
        # smile_pred  = self.smile_clf(x)
        # emo_pred  = self.emotion_clf(x)
        # age_pred  = self.age_clf(x)

        # return gen_pred, smile_pred, emo_pred, age_pred 

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



if __name__ == "__main__":
    model = MTL_VGG16_bn_model()
    print("MTL_VGG16_bn_model")
    print(model)