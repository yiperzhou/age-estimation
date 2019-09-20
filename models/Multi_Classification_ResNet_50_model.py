import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms

from torch.autograd import Variable

# from config import config, parser
from copy import deepcopy
import numpy as np


class Multi_Classification_ResNet_50_model(torch.nn.Module):
  def __init__(self, args, age_classes = 100):
    super(Multi_Classification_ResNet_50_model, self).__init__()

    resnet50_model = models.resnet50(pretrained=True)

    self.Multi_Classification_ResNet_50_features = nn.Sequential(
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
    x = self.Multi_Classification_ResNet_50_features(x)
    
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
