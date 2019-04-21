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


def Train_Valid(model, trainloader, criterion, optimizer, epoch, logFile, args, pharse):

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

    print("four tasks weights: ", args.loss_weights)

    gender_criterion, age_criterion, smile_criterion, age_cls_criterion, emotion_criterion = criterion[0], criterion[1], criterion[2], criterion[3], criterion[4]
    gender_train_loader, age_train_loader, smile_train_loader, emotion_train_loader = trainloader[0], trainloader[1], trainloader[2], trainloader[3]

    epoch_start_time = time.time()
    for (i_batch_idx, (gender_img, gender_label)), (j_batch_idx, (age_img, age_label)), (k_batch_idx, (emo_img, emo_label)), (m, (smile_img, smile_label)) \
        in zip(enumerate(gender_train_loader), enumerate(age_train_loader), enumerate(emotion_train_loader), enumerate(smile_train_loader)):

        if pharse == "train":
            optimizer.zero_grad()

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
        gender_out_1, smile_out_1, emo_out_1, age_out_1  = model(age_img)
        age_loss = age_cls_criterion(age_out_1, age_label)

        # print("age_cls_label: ", age_label)
        # print("age_out      : ", age_out_1)

        age_loss_mae = age_mae_criterion_Encapsulation(age_criterion, age_out_1, age_label)
        
        age_prec1 = accuracy(age_out_1.data, age_label)
        age_epoch_acc.update(age_prec1[0].item(), age_label.size(0))
        
        age_epoch_mae.update(age_loss_mae.item(), 1)
        age_epoch_mae_own_list.append(age_loss_mae.item())
        
        
        # gender
        gender_img = gender_img.cuda()
        gender_label = gender_label.cuda()
        gender_out_2, smile_out_2, emo_out_2, age_out_2 = model(gender_img)
        # print("gender_out_2: ", gender_out_2)
        # gender_target = gender_label.view(-1)
        gender_loss = gender_criterion(gender_out_2, gender_label)
        # gender_loss = Variable(gender_loss, requires_grad = True) 
        
        # print("gender loss: ", gender_loss.item())
        gender_prec1 = accuracy(gender_out_2.data, gender_label)
        gender_epoch_acc.update(gender_prec1[0].item(), gender_label.size(0))


        # emotion 
        emo_img = emo_img.cuda()
        emo_label = emo_label.cuda()
        gender_out_3, smile_out_3, emo_out_3, age_out_3 = model(emo_img)
        emo_loss = emotion_criterion(emo_out_3, emo_label)
        
        emo_prec1 = accuracy(emo_out_3.data, emo_label)
        emo_epoch_acc.update(emo_prec1[0].item(), emo_label.size(0))


        # smile            
        smile_img = smile_img.cuda()
        smile_label = smile_label.cuda()
        gender_out_4, smile_out_4, emo_out_4, age_out_4 = model(smile_img)
        smile_loss = smile_criterion(smile_out_4, smile_label)
        
        smile_prec1 = accuracy(smile_out_4.data, smile_label)
        smile_epoch_acc.update(smile_prec1[0].item(), smile_label.size(0))


        reduce_gen_loss, reduce_smile_loss, reduce_emo_loss,reduce_age_cls_loss  = args.loss_weights[0], args.loss_weights[1], args.loss_weights[2], args.loss_weights[3]
        gender_loss = gender_loss * reduce_gen_loss
        smile_loss = smile_loss * reduce_smile_loss
        emo_loss = emo_loss * reduce_emo_loss
        age_loss = age_loss * reduce_age_cls_loss

        loss = gender_loss +  smile_loss + emo_loss + age_loss

        

        gender_epoch_loss.update(gender_loss.item(), gender_label.size(0))
        age_epoch_loss.update(age_loss.item(), age_label.size(0))
        emo_epoch_loss.update(emo_loss.item(), emo_label.size(0))
        smile_epoch_loss.update(smile_loss.item(), smile_label.size(0))
        
        # total_loss.update(loss[0].item(), 1)
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


        # print("train loader")
        if j_batch_idx % 100 == 0:
            print(
                'Epoch: {} [{}/{}]'.
                format(epoch, j_batch_idx * len(age_img), len(age_train_loader.dataset)))

            print("[" + pharse +"] [ACC(%)], [gender, smile, emotion, age        ]: " + str([gender_epoch_acc.avg, smile_epoch_acc.avg, emo_epoch_acc.avg, age_epoch_acc.avg]))
            print("[" + pharse +"] [Loss  ], [gender, smile, emotion, age, age_mae, total]: " + str([gender_epoch_loss.avg, smile_epoch_loss.avg, emo_epoch_loss.avg, age_epoch_loss.avg, age_epoch_mae.avg, total_loss.avg]))            


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
        # else:
        #     break


    LOG("One [" + pharse + "] epoch took {} minutes".format((time.time() - epoch_start_time)/60), logFile)

    accs = [gender_epoch_acc.avg, smile_epoch_acc.avg, emo_epoch_acc.avg, age_epoch_acc.avg]
    losses = [gender_epoch_loss.avg, smile_epoch_loss.avg, emo_epoch_loss.avg, age_epoch_loss.avg, age_epoch_mae.avg, total_loss.avg]
    
    losses_perc = [
        gender_epoch_loss.avg/total_loss.avg,
        smile_epoch_loss.avg/total_loss.avg,
        emo_epoch_loss.avg/total_loss.avg,
        age_epoch_loss.avg/total_loss.avg,
        (gender_epoch_loss.avg + smile_epoch_loss.avg + emo_epoch_loss.avg + age_epoch_loss.avg)/total_loss.avg
    ]
    
    LOG("[" + pharse +"] [ACC(%)], [gender, smile, emotion, age        ]: " + str(accs), logFile)
    LOG("[" + pharse +"] [Loss  ], [gender, smile, emotion, age, age_mae, total]: " + str(losses), logFile)
    LOG("[" + pharse +"] [losses Perc], [gender, smile, emotion, age, sum/total]: " + str(losses_perc), logFile)
    LOG("[" + pharse +"] age_epoch_mae_own_list, mean age                       : " + str(np.mean(age_epoch_mae_own_list)), logFile)

    try:
        lr = float(str(optimizer).split("\n")[3].split(" ")[-1])
    except:
        lr = 100
    LOG("lr: " + str(lr), logFile)
    
    return accs, losses, lr, model
