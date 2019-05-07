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

def age_mae_criterion_Encapsulation(age_criterion, age_out_1, age_label):
    # print("age_out_1: ", age_out_1)

    
    # print("age_out_1: ", age_out_1)
    # print("age_label: ", age_label)

    _, pred_ages = torch.max(age_out_1, 1)
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
    # # print("age_out_1: ", age_out_1.size())
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



def age_rgs_criterion_Encapsulation(age_rgs_criterion, y_pred, y_true):
    '''
    age regression loss, reference, the github repository, Age-Gender-Pred, repository
    '''
    
    age_divde = torch.tensor(10)
        
        
    # def forward(self, y_pred, y_true, age_rgs_criterion):

    y_true_rgs = torch.div(y_true, age_divde)
    
    y_true_rgs = y_true_rgs.type(torch.FloatTensor)

    y_true_rgs = torch.ceil(y_true_rgs)
    
    y_true_rgs = y_true_rgs.type(torch.cuda.LongTensor)

    age_loss_rgs = age_rgs_criterion(y_pred, y_true_rgs)

    return age_loss_rgs


def Train_Valid_debug(model, loader, criterion, optimizer, epoch, logFile, args, pharse):

    LOG("[" + pharse + "]: Starting, Epoch: " + str(epoch), logFile)

    best_gen_acc = 0.
    best_age_mae = 99.
    best_emotion_acc = 0.
    
    not_reduce_rounds = 0

    loss = 0

    gender_epoch_loss = AverageMeter()                      
    smile_epoch_loss = AverageMeter()
    emo_epoch_loss = AverageMeter()
    age_epoch_loss = AverageMeter()

    gender_epoch_acc = AverageMeter()
    smile_epoch_acc = AverageMeter()
    emo_epoch_acc = AverageMeter()
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

    print("four tasks weights: ", args.loss_weights)

    gender_criterion, age_mae_criterion, smile_criterion, age_cls_criterion, emotion_criterion = criterion[0], criterion[1], criterion[2], criterion[3], criterion[4]

    age_gaussian_loss_criterion, age_euclidean_loss_criterion, age_rgs_criterion  = criterion[5], criterion[6], criterion[7]

    gender_loader, age_loader, smile_loader, emotion_loader = loader[0], loader[1], loader[2], loader[3]


    del gender_loader, smile_loader, emotion_loader
    del gender_criterion, smile_criterion, emotion_criterion

    # for j, (emo_img, emo_label) in enumerate(emotion_loader):
    for j_batch_idx, (age_img, age_label) in enumerate(age_loader):

        if pharse == "train":
            optimizer.zero_grad()

        # print("train loader")
        # # age_img, gender_img, emotion_img = data
        # # age_label, gender_label, emotion_label = labels
        # print("age label: ", age_label)
        # print("age_img: ", age_img.size())

        # print("gender_label: ", gender_label)
        # print("gender_img: ", gender_img.size())

        # print("smile_label: ", smile_label)
        # print("smile_img: ", smile_img.size())

        # print("gender_label: ", gender_label)
        # print("emotion_label: ", emotion_label)
        
        # age
        age_img = age_img.cuda()
        age_label = age_label.cuda()
        gender_out_1, smile_out_1, emo_out_1, age_out_1, age_out_rgs_1  = model(age_img)
        # print("age_out_1: ", age_out_1)
        # print("age_label: ", age_label)
        # age_label_one_hot = convert_to_onehot_tensor(age_label, 88)
        # print("age_label_one_hot: ", age_label_one_hot)
        # age_loss = age_cls_criterion(age_out_1, age_label_one_hot)
        age_loss = age_cls_criterion(age_out_1, age_label)

        # print("age_cls_label: ", age_label)
        # print("age_out      : ", age_out_1)

        age_loss_mae = age_mae_criterion_Encapsulation(age_mae_criterion, age_out_1, age_label)
        # print("age_loss_mae.size(0): ", age_loss_mae.size(0))

        age_epoch_loss.update(age_loss.item(), age_label.size(0))
        age_prec1 = accuracy(age_out_1.data, age_label)
        age_epoch_acc.update(age_prec1[0].item(), age_label.size(0))
        age_epoch_mae.update(age_loss_mae.item(), 1)

        age_epoch_mae_own_list.append(age_loss_mae.item())
        
        # print("age_loss   : ", age_loss)
        # print("age_cls_acc: ", age_prec1[0].item())


        age_loss_gaussian = age_gaussian_loss_criterion(age_out_1, age_label)

        


        age_loss_rgs = age_rgs_criterion_Encapsulation(age_rgs_criterion, age_out_rgs_1, age_label)
        # print("age_loss_rgs: ", age_loss_rgs)


        # age_loss_euclidean = age_euclidean_loss_criterion(age_out_1, age_label)

        # print("age_loss_euclidean: ", age_loss_euclidean)



        
        # # gender
        # gender_img = gender_img.cuda()
        # gender_label = gender_label.cuda()
        # gender_out_2, smile_out_2, emo_out_2, age_out_2 = model(gender_img)
        # # print("gender_out_2: ", gender_out_2)
        # # gender_target = gender_label.view(-1)
        # gender_loss = gender_criterion(gender_out_2, gender_label)
        # # gender_loss = Variable(gender_loss, requires_grad = True) 
        # gender_epoch_loss.update(gender_loss.item(), gender_label.size(0))
        # # print("gender loss: ", gender_loss.item())
        # gender_prec1 = accuracy(gender_out_2.data, gender_label)
        # gender_epoch_acc.update(gender_prec1[0].item(), gender_label.size(0))


        # # emotion 
        # emo_img = emo_img.cuda()
        # emo_label = emo_label.cuda()
        # gender_out_3, smile_out_3, emo_out_3, age_out_3 = model(emo_img)
        # emo_loss = emotion_criterion(emo_out_3, emo_label)
        # emo_epoch_loss.update(emo_loss.item(), emo_label.size(0))
        # emo_prec1 = accuracy(emo_out_3.data, emo_label)
        # emo_epoch_acc.update(emo_prec1[0].item(), emo_label.size(0))


        # # smile            
        # smile_img = smile_img.cuda()
        # smile_label = smile_label.cuda()
        # gender_out_4, smile_out_4, emo_out_4, age_out_4 = model(smile_img)
        # smile_loss = smile_criterion(smile_out_4, smile_label)
        # smile_epoch_loss.update(smile_loss.item(), smile_label.size(0))
        # smile_prec1 = accuracy(smile_out_4.data, smile_label)
        # smile_epoch_acc.update(smile_prec1[0].item(), smile_label.size(0))


        reduce_gen_loss, reduce_smile_loss, reduce_emo_loss, reduce_age_cls_loss  = args.loss_weights[0], args.loss_weights[1], args.loss_weights[2], args.loss_weights[3]
        
        loss = age_loss * reduce_age_cls_loss
        
        loss = loss * 1 + age_loss_rgs * 1

        loss = loss * 1 + age_loss_gaussian * 0


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

        # print("[", pharse," LOSS]", "total: ", loss.item())
        # print("              Gender: ", gender_loss.item())
        # print("              Smile: ", smile_loss.item())
        # print("              Emotion: ", emo_loss.item()) 
        # print("              Age: ", age_loss.item())

        # # age_prec1 = accuracy(age_out.data, age_target.view(-1))

        # # age_epoch_loss.update(age_loss.item(), age_label.size(0))
        # # print("age loss: ", age_loss.item())
        # # age_epoch_acc.update(age_prec1[0].item(), inputs.size(0))
        # # print("acc: ", age_prec1[0].item())

        # _, pred_age = torch.max(pred_ages, 1)
        # pred_age = pred_age.view(-1).cpu().numpy()
        # # age_true    = age_target.view(-1).cpu().numpy()
        # age_rgs_loss = sum(np.abs(pred_age - age_label.view(-1)))/age_label.size(0)
        # # print("age prediction MAE in batch size:", age_rgs_loss)
        # age_epoch_rgs_loss.update(age_rgs_loss, age_label.size(0))



        # emotion_prec1 = accuracy(emo_out_3.data, emotion_label)
        # emotion_epoch_loss.update(emotion_prec1[0].item(), emotion_label.size(0))

        # # (age_rgs_loss*0.1).backward()
        # # optimizer.step()

    accs = [gender_epoch_acc.avg, smile_epoch_acc.avg, emo_epoch_acc.avg, age_epoch_acc.avg]
    losses = [gender_epoch_loss.avg, smile_epoch_loss.avg, emo_epoch_loss.avg, age_epoch_loss.avg, age_epoch_mae.avg, total_loss.avg]
    
    LOG("[" + pharse +"] [ACC(%)], [gender, smile, emotion, age        ]: " + str(accs), logFile)
    LOG("[" + pharse +"] [Loss  ], [gender, smile, emotion, age, age_mae, total]: " + str(losses), logFile)
    LOG("[" + pharse +"] age_epoch_mae_own_list, mean age                       : " + str(np.mean(age_epoch_mae_own_list)), logFile)
    try:
        lr = float(str(optimizer).split("\n")[3].split(" ")[-1])
    except:
        lr = 100
    LOG("lr: " + str(lr), logFile)
    
    return accs, losses, lr, model
