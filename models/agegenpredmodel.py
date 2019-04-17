import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms

from torch.autograd import Variable

# from config import config, parser
# from resnet import resnet50
from copy import deepcopy
import numpy as np


def image_transformer():
  """
  :return:  A transformer to convert a PIL image to a tensor image
            ready to feed into a neural network
  """
  return {
      'train': transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
    }

# from layer_utils.roi_align.roi_align import CropAndResizeFunction

"""
0 ResNet(
  0 (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  1(bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
  2 (relu): ReLU(inplace)
  3 (maxpool): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), ceil_mode=False)
  4 (stack1): Sequential 
  5 (stack2): Sequential 
  6 (stack3): Sequential 
  7 (stack4): Sequential 
  8 (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
  9 (fc): Linear(in_features=2048, out_features=2000, bias=True)
)
1 Linear(in_features=2048, out_features=512, bias=True)
2 Linear(in_features=512, out_features=99, bias=True)
3 Linear(in_features=2048, out_features=512, bias=True)
4 Linear(in_features=512, out_features=2, bias=True)
"""

class AgeGenPredModel(torch.nn.Module, ):
  def __init__(self):
    super(AgeGenPredModel, self).__init__()
    self.resNet = models.resnet18(pretrained=True)

    self.use_gpu = torch.cuda.is_available()
    self.age_divide = float(10)
    self.age_cls_unit = int(88)
    self.features_length = 512

    self.fc1          = nn.Linear(512, 512)
    self.age_cls_pred = nn.Linear(512, self.age_cls_unit)

    self.fc2          = nn.Linear(512, 512)
    self.gen_cls_pred = nn.Linear(512, 2)

    # smile branch
    self.smile_clf = nn.Sequential(
        nn.Linear(self.features_length, 256),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(256, 2)
    )
    
    # emotion branch
    self.emotion_clf = nn.Sequential(
        nn.Linear(self.features_length, 256),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(256, 7)
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

    return x # out = [N, 512, 1 ,1]


  # def crop_and_resize(self, last, org_img):
  #   """
  #   crop the image with the highest attention point as center
  #   :param atten_mask: 7x7 attention mask
  #   :param org_img: original image
  #   :return: new croped and resized image
  #   """
  #   # find max response point
  #   atten_mask = self.mapreduce(last)  # [N, 1, 7, 7]
  #   atten_mask = atten_mask.view(atten_mask.size(0), -1) # [N, 49]
  #   _, max_idx = torch.max(atten_mask, 1)  # idx = [0 ~ 49]
  #
  #   # get cords with max respons as center
  #   x1 = 32 * torch.remainder(max_idx, 7)
  #   y1 = 32 * (max_idx - x1) / 7
  #   x2 = x1 + 128
  #   y2 = y1 + 128
  #
  #   x1 = x1.view(-1, 1)
  #   x2 = x2.view(-1, 1)
  #   y1 = y1.view(-1, 1)
  #   y2 = y2.view(-1, 1)
  #
  #   # crop the image
  #   boxes = torch.cat((y1, x1, y2, x2), 1).type(torch.FloatTensor)
  #   box_ind = Variable(torch.IntTensor(range(boxes.size(0))))
  #   if self.use_gpu:
  #     boxes = boxes.cuda()
  #     box_ind = box_ind.cuda()
  #
  #   croped_img = CropAndResizeFunction(224, 224)(org_img, boxes, box_ind)
  #
  #   return croped_img, max_idx, atten_mask


  def get_age_gender(self, last_conv_out):
    last_conv_out = self.resNet.avgpool(last_conv_out)
    last_conv_out = last_conv_out.view(last_conv_out.size(0), -1)

    age_pred = F.relu(self.fc1(last_conv_out))
    age_pred = F.softmax(self.age_cls_pred(age_pred), 1)

    gen_pred = F.relu(self.fc2(last_conv_out))
    gen_pred = self.gen_cls_pred(gen_pred)


    smile_pred = self.smile_clf(last_conv_out)
    emo_pred = self.emotion_clf(last_conv_out)    

    return gen_pred, smile_pred, emo_pred, age_pred


  def forward(self, x, return_atten_info = False):
    last1 = self.get_resnet_convs_out(x)
    gen_pred, smile_pred, emo_pred, age_pred = self.get_age_gender(last1)
    return gen_pred, smile_pred, emo_pred, age_pred


  def evaluate(self, faces):
    preds = []
    weigh = np.linspace(1, self.age_cls_unit, self.age_cls_unit)

    for face in faces:
      face = Variable(torch.unsqueeze(face, 0))
      gen_out, age_out = self.forward(face)

      gen_out = F.softmax(gen_out, 1)
      gen_prob, gen_pred = torch.max(gen_out, 1)

      gen_pred = gen_pred.cpu().data.numpy()[0]
      gen_prob = gen_prob.cpu().data.numpy()[0]

      age_probs = age_out.cpu().data.numpy()
      age_probs.resize((self.age_cls_unit,))

      # expectation and variance
      age_pred = sum(age_probs * weigh)
      age_var = np.square(np.mean(age_probs * np.square(weigh - age_pred)))

      preds.append([gen_pred, gen_prob, age_pred, age_var])
    return preds


if __name__ == "__main__":
  a = AgeGenPredModel()
  print("All Good")
  pass
