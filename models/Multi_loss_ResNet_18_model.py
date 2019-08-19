import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms

from torch.autograd import Variable

# from config import config, parser
from copy import deepcopy
import numpy as np


class Multi_loss_ResNet_18_model(torch.nn.Module):
  def __init__(self, args):
    super(Multi_loss_ResNet_18_model, self).__init__()
    self.resNet = models.resnet18(pretrained=True)

    self.use_gpu = torch.cuda.is_available()
    # self.age_divide = float(parser['DATA']['age_divide'])
    # self.age_cls_unit = int(parser['RacNet']['age_cls_unit'])

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
    x = self.resNet.conv1(x)    # out = [N, 64, 112, 112]
    x = self.resNet.bn1(x)
    x = self.resNet.relu(x)
    x = self.resNet.maxpool(x)  # out = [N, 64, 56, 56]

    x = self.resNet.layer1(x)   # out = [N, 64, 56, 56]
    x = self.resNet.layer2(x)   # out = [N, 128, 28, 28]
    x = self.resNet.layer3(x)   # out = [N, 256, 14, 14]
    x = self.resNet.layer4(x)   # out = [N, 512, 7, 7]

    return x

  # def get_age_rgs_number_class(self):

  #     if self.args.age_loss_type == "10_age_cls_loss":
  #         age_divide = 10
  #         # print("10_age_cls_loss")
  #     elif self.args.age_loss_type == "5_age_cls_loss":
  #         age_divide = 20
  #         # print("5_age_cls_loss")
  #     elif self.args.age_loss_type == "20_age_cls_loss":
  #         age_divide = 5
  #         # print("20_age_cls_loss")
  #     else:
  #         print("10_age_cls_loss, 5_age_cls_loss, 20_age_cls_loss")
  #         ValueError

  #     return age_divide


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


    # return gen_pred, smile_pred, emo_pred, age_pred, age_pred_rgs
  
  def forward(self, x, return_atten_info = False):
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


    # gen_pred, smile_pred, emo_pred, age_pred, age_pred_rgs = self.get_age_gender_emotion(last1)
    # return gen_pred, smile_pred, emo_pred, age_pred, age_pred_rgs



  # def evaluate(self, faces):
  #   preds = []
  #   weigh = np.linspace(1, self.age_cls_unit, self.age_cls_unit)

  #   for face in faces:
  #     face = Variable(torch.unsqueeze(face, 0)).cuda()
  #     # face = torch.autograd.Variable(torch.unsqueeze(face, 0))
  #     # age_out shape, [1, 60]; gen_out shape, [1,3]
  #     # gen_out, age_out = self.forward(face)
  #     gen_out, age_out, emo_out = self.forward(face)

  #     gen_out = F.softmax(gen_out, 1)
  #     gen_prob, gen_pred = torch.max(gen_out, 1)

  #     gen_pred = gen_pred.cpu().data.numpy()[0]
  #     gen_prob = gen_prob.cpu().data.numpy()[0]

  #     age_probs = age_out.cpu().data.numpy()
  #     age_probs.resize((self.age_cls_unit,))

  #     # expectation and variance
  #     age_pred = sum(age_probs * weigh)
  #     age_var = np.square(np.mean(age_probs * np.square(weigh - age_pred)))

  #     emo_out = F.softmax(emo_out, 1)
  #     emo_prob, emo_pred = torch.max(emo_out, 1)
  #     emo_pred = emo_pred.cpu().data.numpy()[0]
  #     emo_prob = emo_prob.cpu().data.numpy()[0]
      

  #     preds.append([gen_pred, gen_prob, age_pred, age_var, emo_pred, emo_prob])
  #   return preds
