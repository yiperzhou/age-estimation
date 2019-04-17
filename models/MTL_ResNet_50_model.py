import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms

from torch.autograd import Variable

# from config import config, parser
from copy import deepcopy
import numpy as np


class MTL_ResNet_50_model(torch.nn.Module):
  def __init__(self, gen_classes= 2, smile_classes = 2, emo_classes = 7, age_classes = 100):
    super(MTL_ResNet_50_model, self).__init__()

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


    # gender branch
    self.gender_clf = nn.Sequential(
        nn.Linear(self.features_length, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(256, gen_classes)
    )

    # smile branch
    self.smile_clf = nn.Sequential(
        nn.Linear(self.features_length, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(256, smile_classes)
    )
    
    # emotion branch
    self.emotion_clf = nn.Sequential(
        nn.Linear(self.features_length, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(256, emo_classes)
    )

    self.age_clf = nn.Sequential(
        nn.Linear(self.features_length, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(256, age_classes)
    )

  def forward(self, x):
    x = self.MTL_ResNet_50_features(x)
    
    x = x.view(x.size(0), -1)

    gen_pred  = self.gender_clf(x)
    smile_pred  = self.smile_clf(x)
    emo_pred  = self.emotion_clf(x)
    age_pred  = self.age_clf(x)

    return gen_pred, smile_pred, emo_pred, age_pred 



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
  