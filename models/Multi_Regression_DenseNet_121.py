import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

import torch.nn.functional as F

from torchvision import models

    
class Multi_Regression_DenseNet_121_model(torch.nn.Module):
    def __init__(self, args, age_classes=100):
        super(Multi_Regression_DenseNet_121_model, self).__init__()

        self.MTL_DenseNet_features = models.densenet121(pretrained=True).features
        
        self.features_length = 50176

        self.args = args
        # self.age_divide = self.get_age_rgs_number_class()  

        self.use_gpu = torch.cuda.is_available()

        # self.args = args

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
        #     nn.Linear(self.features_length, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(512, gen_classes)
        # )

        # # smile branch
        # self.smile_clf = nn.Sequential(
        #     nn.Linear(self.features_length, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(512, smile_classes)
        # )

        # # emotion branch
        # self.emotion_clf = nn.Sequential(
        #     nn.Linear(self.features_length, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(512, emo_classes)
        # )

        # # age branch
        # self.age_clf = nn.Sequential(
        #     nn.Linear(self.features_length, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(512, age_classes)
        # )

      
        # self.age_rgs_clf = nn.Sequential(
        #     nn.Linear(self.features_length, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(256, self.age_divide)
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


    
    def forward(self, x):
        x = self.MTL_DenseNet_features(x)
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

