import os
import re
import time
import copy
import math
import glob
import datetime
import numpy as np
import pandas as pd

from multiprocessing import cpu_count
from collections import OrderedDict
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from models import *
from data_load import *
from utils import *

from utils.helper_4 import convert_to_onehot_tensor

def age_l1_criterion_encapsulation(age_criterion, age_pred, age_label):
    
    age_pred = age_pred.type(torch.cuda.FloatTensor)
    age_label = convert_to_onehot_tensor(age_label, 100)

    age_label = age_label.type(torch.cuda.FloatTensor)


    age_loss_l1 = age_criterion(age_pred, age_label)

    return age_loss_l1


def age_mapping_function(origin_value, age_divide):
    # print("origin_value: ", origin_value)
    origin_value = origin_value.cpu()

    y_true_rgs = torch.div(origin_value, age_divide)

    # print("y_true_rgs: ", y_true_rgs)

    # y_true_rgs = torch.ceil(y_true_rgs)
    
    return  y_true_rgs


def age_rgs_criterion_encapsulation(age_criterion, age_out_rgs, age_label, classification_type):
    
    if classification_type == "100_classes":
        age_divide = 1
    elif classification_type == "20_classes":
        age_divide = 5
    elif classification_type == "10_classes":
        age_divide = 10
    elif classification_type == "5_classes":
        age_divide = 20
    else:
        print("classification type value is wrong")
        raise ValueError

    mapped_age_label = age_mapping_function(age_label, age_divide)

    mapped_age_label = mapped_age_label.type(torch.cuda.FloatTensor)

    mapped_age_label = mapped_age_label.type(torch.cuda.LongTensor)

    age_rgs_loss = age_criterion(age_out_rgs, mapped_age_label)

    return age_rgs_loss


def train_valid(model, loader, criterion, optimizer, epoch, logFile, args, pharse):

    LOG("[" + pharse + "]: Starting, Epoch: " + str(epoch), logFile)

    best_age_mae = 99.

    age_l1_rgs_epoch_loss = AverageMeter()                   

    age_epoch_l1 = AverageMeter()

    age_epoch_mae_own_list = []

    if pharse == "train":
        model.train()
    elif pharse == "valid":
        model.eval()
    else:
        NotImplementedError
    torch.cuda.empty_cache()

    epoch_start_time = time.time()

    age_l1_mae_criterion = criterion[0]

    age_loader = loader[0]

    for j_batch_idx, (age_img, age_label) in enumerate(age_loader):

        if pharse == "train":
            optimizer.zero_grad()
        # age
        age_img = age_img.cuda()
        age_label = age_label.cuda()

        age_pred = model(age_img)

        # age l1 regrssion loss
        age_loss_rgs_l1 = age_l1_criterion_encapsulation(age_l1_mae_criterion, age_pred, age_label)

        age_epoch_l1.update(age_loss_rgs_l1.item(), 1)

        age_l1_rgs_epoch_loss.update(age_loss_rgs_l1.item(), 1)

        if pharse == "train":
            age_loss_rgs_l1.backward()
            optimizer.step()
        elif pharse == "valid":
            # print("valid pharse")
            continue
        else:
            print("pharse should be in [train, valid]")
            NotImplementedError

    losses = [age_l1_rgs_epoch_loss.avg]

    LOG("[" + pharse +"] [Loss  ], [l1 ]: " + str(losses), logFile)
    LOG("[" + pharse +"] , l1                  : " + str(age_epoch_l1.avg), logFile)
    try:
        lr = float(str(optimizer).split("\n")[3].split(" ")[-1])
    except:
        lr = 100
    LOG("lr: " + str(lr), logFile)
    
    return losses, lr, model