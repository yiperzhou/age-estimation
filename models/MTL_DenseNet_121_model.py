import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

import torch.nn.functional as F

from torchvision import models

    
class MTL_DenseNet_121_model(torch.nn.Module):
    def __init__(self, args, gen_classes=2, smile_classes=2, emo_classes=7, age_classes=100):
        super(MTL_DenseNet_121_model, self).__init__()

        self.MTL_DenseNet_features = models.densenet121(pretrained=True).features
        
        self.features_length = 50176

        self.args = args
        self.age_divide = self.get_age_rgs_number_class()  

        # gender branch
        self.gender_clf = nn.Sequential(
            nn.Linear(self.features_length, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512, gen_classes)
        )

        # smile branch
        self.smile_clf = nn.Sequential(
            nn.Linear(self.features_length, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512, smile_classes)
        )

        # emotion branch
        self.emotion_clf = nn.Sequential(
            nn.Linear(self.features_length, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512, emo_classes)
        )

        # age branch
        self.age_clf = nn.Sequential(
            nn.Linear(self.features_length, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512, age_classes)
        )

      
        self.age_rgs_clf = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, self.age_divide)
        )     

    
    def get_age_rgs_number_class(self):

        if self.args.age_loss_type == "10_age_cls_loss":
            age_divide = 10
            # print("10_age_cls_loss")
        elif self.args.age_loss_type == "5_age_cls_loss":
            age_divide = 20
            # print("5_age_cls_loss")
        elif self.args.age_loss_type == "20_age_cls_loss":
            age_divide = 5
            # print("20_age_cls_loss")
        else:
            print("10_age_cls_loss, 5_age_cls_loss, 20_age_cls_loss")
            ValueError

        return age_divide

    def get_age_gender_emotion(self, last_conv_out):
        last_conv_out = last_conv_out.view(last_conv_out.size(0), -1)

        print("MTL_DenseNet_121, last_conv_out: ", last_conv_out.size())

        gen_pred = F.relu(self.dropout(self.fc2(last_conv_out)))
        gen_pred = F.softmax(self.gen_cls_pred(gen_pred))

        smile_pred   = F.relu(self.dropout(self.fc4(last_conv_out)))
        smile_pred   = F.softmax(self.smile_cls_pred(smile_pred))

        emo_pred = F.relu(self.dropout(self.fc3(last_conv_out)))
        emo_pred = F.softmax(self.emo_cls_pred(emo_pred))

        age_pred = F.relu(self.dropout(self.fc1(last_conv_out)))
        age_pred = F.softmax(self.age_cls_pred(age_pred), 1)


        return gen_pred, smile_pred, emo_pred, age_pred

    
    def forward(self, x):
        x = self.MTL_DenseNet_features(x)
        x = x.view(x.size(0), -1)

        gen_pred = self.gender_clf(x)
        smile_pred = self.smile_clf(x)
        emo_pred = self.emotion_clf(x)
        age_pred = self.age_clf(x)

        age_pred_rgs = self.age_rgs_clf(x)

        return gen_pred, smile_pred, emo_pred, age_pred, age_pred_rgs

