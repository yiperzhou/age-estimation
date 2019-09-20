import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms

from torch.autograd import Variable

# from config import config, parser
from copy import deepcopy
import numpy as np


class Multi_Regression_ResNet_50_model(torch.nn.Module):
  def __init__(self, args, age_classes = 100):
    super(Multi_Regression_ResNet_50_model, self).__init__()

    resnet50_model = models.resnet50(pretrained=True)

    self.MTL_ResNet_50_features = nn.Sequential(
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


  def forward(self, x):
    x = self.MTL_ResNet_50_features(x)
    
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

    # gen_pred  = self.gender_clf(x)
    # smile_pred  = self.smile_clf(x)
    # emo_pred  = self.emotion_clf(x)
    # age_pred  = self.age_clf(x)

    # return gen_pred, smile_pred, emo_pred, age_pred 



#   def evaluate(self, faces):
#     preds = []
#     weigh = np.linspace(1, self.age_cls_unit, self.age_cls_unit)

#     for face in faces:
#       face = Variable(torch.unsqueeze(face, 0)).cuda()
#       # face = torch.autograd.Variable(torch.unsqueeze(face, 0))
#       # age_out shape, [1, 60]; gen_out shape, [1,3]
#       # gen_out, age_out = self.forward(face)
#       gen_out, age_out, emo_out = self.forward(face)

#       gen_out = F.softmax(gen_out, 1)
#       gen_prob, gen_pred = torch.max(gen_out, 1)

#       gen_pred = gen_pred.cpu().data.numpy()[0]
#       gen_prob = gen_prob.cpu().data.numpy()[0]

#       age_probs = age_out.cpu().data.numpy()
#       age_probs.resize((self.age_cls_unit,))

#       # expectation and variance
#       age_pred = sum(age_probs * weigh)
#       age_var = np.square(np.mean(age_probs * np.square(weigh - age_pred)))

#       emo_out = F.softmax(emo_out, 1)
#       emo_prob, emo_pred = torch.max(emo_out, 1)
#       emo_pred = emo_pred.cpu().data.numpy()[0]
#       emo_prob = emo_prob.cpu().data.numpy()[0]
      

#       preds.append([gen_pred, gen_prob, age_pred, age_var, emo_pred, emo_prob])
#     return preds
  