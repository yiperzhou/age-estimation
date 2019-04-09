import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.nn.functional as F

class All_In_One_model(nn.Module):
    '''
    reimplement the paper, "An All-In-One Convolutional Neural Network for Face Analysis"
    '''

    def __init__(self):
        super(All_In_One_model, self).__init__()

        self.conv1 = nn.Conv2d(11, 94, kernel_size =11)
        self.conv1 = nn.Conv2d(11, 94, kernel_size =5)
        self.conv1 = nn.Conv2d(11, 94, kernel_size =3)
        self.conv1 = nn.Conv2d(11, 94, kernel_size =3)
        self.conv1 = nn.Conv2d(11, 94, kernel_size =3)
        self.conv1 = nn.Conv2d(11, 94, kernel_size =3)
        self.conv1 = nn.Conv2d(11, 94, kernel_size =3)
        



    def forward(self, x):

        return x