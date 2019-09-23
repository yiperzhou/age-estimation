import os
import re
import cv2
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

def age_mae_criterion_encapsulation(age_criterion, age_out_cls, age_label):
    # print("age_out_cls: ", age_out_cls)

    # print("age_out_cls: ", age_out_cls)
    # print("age_label: ", age_label)

    _, pred_ages = torch.max(age_out_cls, 1)
    # pred_age = pred_age.view(-1).cpu().numpy()
    # pred_ages = []
    # for p in pred_age:
    #     pred_ages.append([p])

    # # print("pred_ages: ", pred_ages)

    # pred_ages = torch.FloatTensor(pred_ages)
    # pred_ages = pred_ages.cuda()
    
    # age_label = age_label.unsqueeze(0)
    # age_label = age_label.type(torch.cuda.LongTensor)

    # age_label = age_label.reshape([age_label.size()[1], 1])
    # age_label = age_label.squeeze(1)
    # # print("age_out_cls: ", age_out_cls.size())
    # # print("age_label: ", age_label)
    # # print(age_label)

    # # print("[age_label] after: ", age_label)
    pred_ages = pred_ages.type(torch.cuda.FloatTensor)
    age_label = age_label.type(torch.cuda.FloatTensor)
    # print("pred_ages: ", pred_ages)
    # print("age_label: ", age_label)

    age_loss_mae = age_criterion(pred_ages, age_label)
    # age_loss_mae = Variable(age_loss_mae, requires_grad = True) 
    
    # print("age_loss_mae: ", age_loss_mae)

    return age_loss_mae


def age_mapping_function(origin_value, age_divide):
    # print("origin_value: ", origin_value)
    origin_value = origin_value.cpu()

    y_true_rgs = torch.div(origin_value, age_divide)

    # print("y_true_rgs: ", y_true_rgs)

    # y_true_rgs = torch.ceil(y_true_rgs)
    
    return  y_true_rgs


def age_cls_criterion_encapsulation(age_criterion, age_out_cls, age_label, classification_type):
    
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

    age_cls_loss = age_criterion(age_out_cls, mapped_age_label)

    return age_cls_loss


def train_valid(model, loader, criterion, optimizer, epoch, logFile, args, pharse):

    LOG("[" + pharse + "]: Starting, Epoch: " + str(epoch), logFile)

    best_age_mae = 99.

    loss = 0

    age_cls_epoch_loss = AverageMeter()
    age_l1_rgs_epoch_loss = AverageMeter()                   
    
    age_epoch_loss = AverageMeter()
    age_epoch_acc = AverageMeter()

    age_epoch_mae = AverageMeter()
    total_loss = AverageMeter()

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

        #
        age_pred = model(age_img)

            age_loss_cls_100_classes = age_cls_criterion_encapsulation(age_l1_mae_criterion, age_pred, age_label, args.l1_regression_loss)


        else:
            print("age_divide_100_classes, age_divide_20_classes, age_divide_10_classes, age_divide_5_classes")
            ValueError

        # age l1 regrssion loss
        age_loss_rgs_l1 = age_mae_criterion_encapsulation(age_l1_mae_criterion, age_pred_100_classes, age_label)

        age_prec1 = accuracy(age_pred_100_classes.data, age_label)
        age_epoch_acc.update(age_prec1[0].item(), age_label.size(0))

        age_epoch_mae.update(age_loss_rgs_l1.item(), 1)

        # age_loss_cls = age_loss_cls_100_classes + age_loss_cls_20_classes \
        #                + age_loss_cls_10_classes + age_loss_cls_5_classes
        
        if args.age_classification_combination == [1, 0, 0, 0]:
            age_loss_cls = age_loss_cls_100_classes

        elif args.age_classification_combination == [1, 1, 0, 0]:
            age_loss_cls = age_loss_cls_100_classes + age_loss_cls_20_classes   # + age_loss_cls_10_classes + age_loss_cls_5_classes
        
        elif  args.age_classification_combination == [1, 1, 1, 0]:
            age_loss_cls = age_loss_cls_100_classes + age_loss_cls_20_classes + age_loss_cls_10_classes  
        
        elif  args.age_classification_combination == [1, 1, 1, 1]:
            age_loss_cls = age_loss_cls_100_classes + age_loss_cls_20_classes + age_loss_cls_10_classes + age_loss_cls_5_classes

        else:
            raise ValueError
            
        age_cls_epoch_loss.update(age_loss_cls.item(), 1)
        age_l1_rgs_epoch_loss.update(age_loss_rgs_l1.item(), 1)

        loss = age_loss_cls

        total_loss.update(loss.item(), 1)

        if pharse == "train":
            loss.backward()
            optimizer.step()
        elif pharse == "valid":
            # print("valid pharse")
            continue
        else:
            print("pharse should be in [train, valid]")
            NotImplementedError

    accs = [age_epoch_acc.avg]
    losses = [total_loss.avg, age_cls_epoch_loss.avg, age_l1_rgs_epoch_loss.avg]

    LOG("[" + pharse +"] [ACC(%)], [age        ]: " + str(accs), logFile)
    LOG("[" + pharse +"] [Loss  ], [total, cls, l1 ]: " + str(losses), logFile)
    LOG("[" + pharse +"] , MAE                  : " + str(age_epoch_mae.avg), logFile)
    try:
        lr = float(str(optimizer).split("\n")[3].split(" ")[-1])
    except:
        lr = 100
    LOG("lr: " + str(lr), logFile)
    
    return accs, losses, lr, model