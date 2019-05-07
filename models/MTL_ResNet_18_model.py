import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms

from torch.autograd import Variable

# from config import config, parser
from copy import deepcopy
import numpy as np


class MTL_ResNet_18_model(torch.nn.Module):
  def __init__(self):
    super(MTL_ResNet_18_model, self).__init__()
    self.resNet = models.resnet18(pretrained=True)

    self.use_gpu = torch.cuda.is_available()
    # self.age_divide = float(parser['DATA']['age_divide'])
    # self.age_cls_unit = int(parser['RacNet']['age_cls_unit'])

    # gender branch
    self.fc1          = nn.Linear(512, 512)
    self.gen_cls_pred = nn.Linear(512, 2)

    # smile branch
    self.fc2          = nn.Linear(512, 512)
    self.smile_cls_pred = nn.Linear(512, 2)
    
    # emotion branch
    self.dropout      = nn.Dropout(p=0.5, inplace=False)
    
    self.fc3          = nn.Linear(512, 512)
    self.emo_cls_pred = nn.Linear(512, 7)

    # age branch
    self.fc4          = nn.Linear(512, 256)
    # self.age_cls_pred = nn.Linear(512, 100)
    self.age_cls_pred = nn.Linear(256, 88)

    self.fc5          = nn.Linear(512, 256)
    self.age_rgs_pred = nn.Linear(256, 10)

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


  def get_age_gender_emotion(self, last_conv_out):
    last_conv_out = self.resNet.avgpool(last_conv_out)
    last_conv_out = last_conv_out.view(last_conv_out.size(0), -1)

    # print("last_conv_out: ", last_conv_out.size())

    gen_pred = F.relu(self.dropout(self.fc1(last_conv_out)))
    gen_pred = self.gen_cls_pred(gen_pred)

    smile_pred   = F.relu(self.dropout(self.fc2(last_conv_out)))
    smile_pred   = self.smile_cls_pred(smile_pred)

    emo_pred = F.relu(self.dropout(self.fc3(last_conv_out)))
    emo_pred = self.emo_cls_pred(emo_pred)

    age_pred = F.relu(self.dropout(self.fc4(last_conv_out)))
    age_pred = F.softmax(self.age_cls_pred(age_pred), 1)

    age_pred_rgs = F.relu(self.dropout(self.fc5(last_conv_out)))
    age_pred_rgs = F.softmax(self.age_rgs_pred(age_pred_rgs), 1)

    return gen_pred, smile_pred, emo_pred, age_pred, age_pred_rgs
  
  def forward(self, x, return_atten_info = False):
    last1 = self.get_resnet_convs_out(x)
    gen_pred, smile_pred, emo_pred, age_pred, age_pred_rgs = self.get_age_gender_emotion(last1)
    return gen_pred, smile_pred, emo_pred, age_pred, age_pred_rgs



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
