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
# from train_valid.age_losses_methods import Age_rgs_loss

from train_valid.age_losses_methods import apply_label_smoothing


def age_mae_criterion_Encapsulation(age_criterion, age_out_cls, age_label):
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



def age_rgs_criterion_Encapsulation(age_rgs_criterion, y_pred, y_true, args):
    '''
    age regression loss, reference, the github repository, Age-Gender-Pred, repository
    '''
    
    if args.age_loss_type == "10_age_cls_loss":
        age_divide = torch.tensor(10)
        # print("10_age_cls_loss")
    elif args.age_loss_type == "5_age_cls_loss":
        age_divide = torch.tensor(5)
        # print("5_age_cls_loss")
    elif args.age_loss_type == "20_age_cls_loss":
        age_divide = torch.tensor(20)
        # print("20_age_cls_loss")
    else:
        print("10_age_cls_loss, 5_age_cls_loss, 20_age_cls_loss")
        

    y_true_rgs = torch.div(y_true, age_divide)
    
    y_true_rgs = y_true_rgs.type(torch.FloatTensor)

    y_true_rgs = torch.ceil(y_true_rgs)
    
    y_true_rgs = y_true_rgs.type(torch.cuda.LongTensor)

    age_loss_rgs = age_rgs_criterion(y_pred, y_true_rgs)

    return age_loss_rgs


def Train_Valid(model, loader, criterion, optimizer, epoch, logFile, args, pharse, debug):

    if debug == True:
        LOG("enter [DEBUG] Tran_Valid", logFile)

    LOG("[" + pharse + "]: Starting, Epoch: " + str(epoch), logFile)

    best_age_mae = 99.

    loss = 0

    age_cls_epoch_loss = AverageMeter()
    age_l1_rgs_epoch_loss = AverageMeter()                   
    age_euclidean_epoch_loss = AverageMeter()
    age_gaussian_epoch_loss = AverageMeter()
    
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

    age_cls_criterion, age_l1_mae_criterion, age_euclidean_loss_criterion, age_gaussian_loss_criterion = criterion[0], criterion[1], criterion[2], criterion[3]

    age_loader = loader[0]

    for j_batch_idx, (age_img, age_label) in enumerate(age_loader):

        if pharse == "train":
            optimizer.zero_grad()
        
        # age
        age_img = age_img.cuda()
        age_label = age_label.cuda()
        # gender_out_1, smile_out_1, emo_out_1, 

        age_out_cls, age_out_rgs  = model(age_img)
        # age_out_cls is for classification, age_out_rgs is for regression including l1 regression, euclidean distance, gaussian loss.

        # age classification crossentropy loss

        # print("age_out_cls: ", age_out_cls)
        # print("age_label: ", age_label)
        # age_label_one_hot = convert_to_onehot_tensor(age_label, 88)
        # print("age_label_one_hot: ", age_label_one_hot)
        # age_loss = age_cls_criterion(age_out_cls, age_label_one_hot)


        # add label smoothing technique to test the effect.
        age_label_one_hot = apply_label_smoothing(age_label)

        age_label_one_hot = age_label_one_hot.type(torch.cuda.FloatTensor)
        
        age_loss_cls = age_cls_criterion(age_out_cls, age_label_one_hot)

        # print("age_cls_label: ", age_label)
        # print("age_out_cls      : ", age_out_cls)

        # # age l1 regrssion loss
        age_loss_rgs_l1 = age_mae_criterion_Encapsulation(age_l1_mae_criterion, age_out_rgs, age_label)

        # age_loss_rgs_l1 = 0

        # print("age_loss_mae.size(0): ", age_loss_mae.size(0))

        # age_epoch_loss.update(age_loss.item(), age_label.size(0))

        # age_prec1 = accuracy(age_out_cls.data, age_label)
        # age_epoch_acc.update(age_prec1[0].item(), age_label.size(0))

        age_epoch_mae.update(age_loss_rgs_l1.item(), 1)

        age_epoch_mae_own_list.append(age_loss_rgs_l1.item())
        
        # print("age_loss   : ", age_loss)
        # print("age_cls_acc: ", age_prec1[0].item())

        # # age euclidean loss
        # age_loss_rgs_euclidean = age_euclidean_loss_criterion(age_out_rgs, age_label)
        #
        # # age gaussian loss
        # age_loss_gaussian = age_gaussian_loss_criterion(age_out_rgs, age_label)

        # age_loss_rgs_l1 = age_rgs_criterion_Encapsulation(age_rgs_criterion, age_out_rgs, age_label, args)
        # age_l1_rgs_epoch_loss.update(age_loss_rgs_l1, age_label.size())

        reduce_age_cls_loss, reduce_age_l1_rgs_loss, reduce_age_euclidean_loss, reduce_age_gaussian_loss  = args.loss_weights[0], args.loss_weights[1], args.loss_weights[2], args.loss_weights[3]

        age_loss_cls = age_loss_cls * reduce_age_cls_loss 
        # age_loss_rgs_l1 = age_loss_rgs_l1 * reduce_age_l1_rgs_loss
        # age_loss_rgs_euclidean = age_loss_rgs_euclidean * reduce_age_euclidean_loss
        # age_loss_gaussian = age_loss_gaussian * reduce_age_gaussian_loss


        age_cls_epoch_loss.update(age_loss_cls.item(), 1)
        # age_l1_rgs_epoch_loss.update(age_loss_rgs_l1.item(), 1)
        # age_euclidean_epoch_loss.update(age_loss_rgs_euclidean.item(), 1)
        # age_gaussian_epoch_loss.update(age_loss_gaussian.item(), 1)


        # print("[age_loss_cls, age_loss_rgs_l1, age_loss_rgs_euclidean, age_gaussian_loss]: ", [age_loss_cls, age_loss_rgs_l1, age_loss_rgs_euclidean, age_loss_gaussian])

        loss = age_loss_cls
        # loss = age_loss_cls + age_loss_rgs_l1 + age_loss_rgs_euclidean + age_loss_gaussian

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
    losses = [total_loss.avg, age_cls_epoch_loss.avg, age_l1_rgs_epoch_loss.avg, age_euclidean_epoch_loss.avg, age_gaussian_epoch_loss.avg]

    LOG("[" + pharse +"] [ACC(%)], [age        ]: " + str(accs), logFile)
    LOG("[" + pharse +"] [Loss  ], [total, cls, l1, euclidean, gaussian ]: " + str(losses), logFile)
    LOG("[" + pharse +"] , MAE                  : " + str(age_epoch_mae.avg), logFile)
    try:
        lr = float(str(optimizer).split("\n")[3].split(" ")[-1])
    except:
        lr = 100
    LOG("lr: " + str(lr), logFile)
    
    return accs, losses, lr, model
