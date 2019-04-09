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

def Train_Valid(model, trainloader, criterion, optimizer, epoch, logFile, args, pharse):

    LOG("[" + pharse + "]: Starting, Epoch: " + str(epoch), logFile)


    best_gen_acc = 0.
    best_age_mae = 99.
    best_emotion_acc = 0.
    
    not_reduce_rounds = 0

    loss = 0

    age_epoch_loss = AverageMeter()
    gender_epoch_loss = AverageMeter()                      
    smile_epoch_loss = AverageMeter()
    emo_epoch_loss = AverageMeter()

    age_epoch_mae = AverageMeter()
    gender_epoch_acc = AverageMeter()
    smile_epoch_acc = AverageMeter()
    emo_epoch_acc = AverageMeter()

    if pharse == "train":
        model.train()

    elif pharse == "valid":
        model.eval()

    else:
        NotImplementedError


    torch.cuda.empty_cache()

    epoch_age_tp = 0.
    epoch_age_mae = 0.
    epoch_gender_tp = 0.
    epoch_emotion_tp = 0

    processed_data = 0
    epoch_start_time = time.time()

    print("four tasks weights: ", args.loss_weights)

    gender_criterion, age_criterion, smile_criterion, age_cls_criterion, emotion_criterion = criterion[0], criterion[1], criterion[2], criterion[3], criterion[4]
    gender_train_loader, age_train_loader, smile_train_loader, emotion_train_loader = trainloader[0], trainloader[1], trainloader[2], trainloader[3]


    for (i, (gender_img, gender_label)), (j, (age_img, age_label)), (k, (emo_img, emo_label)), (m, (smile_img, smile_label)) in zip(enumerate(gender_train_loader), enumerate(age_train_loader), enumerate(emotion_train_loader), enumerate(smile_train_loader)):
        # print("i: ", i, ", inside the for loop")

        # print("train loader")
        if i < 10000000:
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

            age_img = age_img.cuda()
            gender_img = gender_img.cuda()
            # smile_img = smile_img.cuda()

            age_label = age_label.cuda()
            gender_label = gender_label.cuda()
            
            smile_img = smile_img.cuda()
            smile_label = smile_label.cuda()

            if pharse == "train":
                optimizer.zero_grad()

            # gender_out, age_out, emo_out = model(inputs)

            # 难道这里也是三个作为一组然后输入进去吗？
            # 12.03.2019, 再次看这个源码，发现我还是没有很清楚的答案去回答上面的问题，
                        
            # age
            gender_out_1, smile_out_1, emo_out_1, age_out_1  = model(age_img)

            _, pred_age = torch.max(age_out_1, 1)
            pred_age = pred_age.view(-1).cpu().numpy()
            pred_ages = []
            for p in pred_age:
                pred_ages.append([p])
            pred_ages = torch.FloatTensor(pred_ages)
            pred_ages = pred_ages.cuda()
            
            age_label = age_label.unsqueeze(0)
            age_label = age_label.type(torch.cuda.LongTensor)

            age_label = age_label.reshape([age_label.size()[1], 1])
            age_label = age_label.squeeze(1)
            # print("age_out_1: ", age_out_1.size())
            # print("age_label: ", age_label.size())
            # print(age_label)
            age_loss = age_cls_criterion(age_out_1, age_label)

            age_label = age_label.type(torch.cuda.FloatTensor)
            
            age_loss_mae = age_criterion(pred_ages, age_label)
            # # age_loss = Variable(age_loss, requires_grad = True)

            age_epoch_loss.update(age_loss.item(), age_label.size(0))

            age_epoch_mae.update(age_loss_mae.item(), age_label.size(0))

            # gender
            gender_out_2, smile_out_2, emo_out_2, age_out_2 = model(gender_img)
            # print("gender_out_2: ", gender_out_2)
            # gender_target = gender_label.view(-1)
            gender_loss = gender_criterion(gender_out_2, gender_label)
            # gender_loss = Variable(gender_loss, requires_grad = True) 
            gender_epoch_loss.update(gender_loss.item(), gender_label.size(0))
            # print("gender loss: ", gender_loss.item())


            gender_prec1 = accuracy(gender_out_2.data, gender_label)


            gender_epoch_acc.update(gender_prec1[0].item(), gender_label.size(0))


            emo_img = emo_img.cuda()
            emo_label = emo_label.cuda()
            
            gender_out_3, smile_out_3, emo_out_3, age_out_3 = model(emo_img)
            
            # print
            
            emo_loss = emotion_criterion(emo_out_3, emo_label)
            

            emo_epoch_loss.update(emo_loss.item(), emo_label.size(0))
            emo_prec1 = accuracy(emo_out_3.data, emo_label)
            emo_epoch_acc.update(emo_prec1[0].item(), emo_label.size(0))

            
        
            # # emotion 
            # gender_out_3, age_out_3, smile_out_3 = model(smile_img)
            # print("smile_out_3: ", smile_out_3)

            # _, pred_emo = torch.max(smile_out_3, 1)
            # pred_emo = pred_emo.view(-1).cpu().numpy()
            # pred_emos = []
            # for p in pred_emo:
            #     pred_emos.append([p])
            # pred_emos = torch.FloatTensor(pred_emos)

            # emotion_label = smile_label.type(torch.cuda.LongTensor)
            # pred_emos = pred_emos.cuda()
            # emo_loss = smile_criterion(pred_emos, emotion_label)
            # # emo_loss = Variable(emo_loss, requires_grad = True)
            smile_loss = 0 
            
            # print("gender loss: ", gender_loss)
            # print("age loss: ", age_loss.item())
            
            # # print("emotion loss: ", emo_loss)
            # reduce_gen_loss     = 1
            # reduce_age_mae      = 0.1
            # reduce_emo_loss     = 1


            # smile
            gender_out_4, smile_out_4, emo_out_4, age_out_4 = model(smile_img)

            smile_loss = smile_criterion(smile_out_4, smile_label)
            # gender_loss = Variable(gender_loss, requires_grad = True) 

            smile_epoch_loss.update(smile_loss.item(), smile_label.size(0))

            # print("gender loss: ", gender_loss.item())

            smile_prec1 = accuracy(smile_out_4.data, smile_label)

            smile_epoch_acc.update(smile_prec1[0].item(), smile_label.size(0))


            reduce_gen_loss, reduce_smile_loss, reduce_emo_loss,reduce_age_cls_loss  = args.loss_weights[0], args.loss_weights[1], args.loss_weights[2], args.loss_weights[3]

            loss = gender_loss * reduce_gen_loss +  smile_loss * reduce_smile_loss + emo_loss * reduce_emo_loss + age_loss * reduce_age_cls_loss

            if pharse == "train":
                loss.backward()
                optimizer.step()
            elif pharse == "valid":
                print("valid pharse")
            else:
                print("pharse should be in [train, valid]")
                NotImplementedError

            print("[", pharse," LOSS]", "total: ", loss.item())
            print("              Gender: ", gender_loss.item())
            print("              Smile: ", smile_loss.item())
            print("              Emotion: ", emo_loss.item()) 
            print("              Age: ", age_loss.item())

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
        else:
            break

    # # train emotion task
    # for m, (emo_img, emo_label) in enumerate(emotion_train_loader):
    #     # emotion
    #     optimizer.zero_grad()
        
    #     emo_img = emo_img.cuda()
    #     emo_label = emo_label.cuda()
        
    #     gender_out_4, age_out_4, emo_out_4 = model(emo_img)
    #     emo_loss = emotion_criterion(emo_out_4, emo_label)
        
    #     emo_epoch_loss.update(emo_loss.item(), emo_label.size(0))
    #     emo_prec1 = accuracy(emo_out_4.data, emo_label)
    #     emo_epoch_acc.update(emo_prec1[0].item(), emo_label.size(0))

    #     emo_loss.backward()

    #     optimizer.step()

    # LOG(" args.loss_weights[0], args.loss_weights[1], args.loss_weights[2], args.loss_weights[3]: " + str(args.loss_weights[0]) + ", " + str(args.loss_weights[1]) + ", " + str(args.loss_weights[2])+ ", " + str(args.loss_weights[3]), logFile)
    accs = [gender_epoch_acc.avg, smile_epoch_acc.avg]
    losses = [gender_epoch_loss.avg, age_epoch_loss.avg, smile_epoch_loss.avg]
    LOG("[" + pharse + "] accs: " + str(accs), logFile)
    LOG("[" + pharse +"] Smile acc: " + str(smile_epoch_acc.avg), logFile)
    LOG("[" + pharse +"] emotion acc: " + str(emo_epoch_acc.avg), logFile)

    LOG("[" + pharse + "] losses: " + str(losses), logFile)
    LOG("[" + pharse +"] age mae: " + str(age_epoch_mae.avg), logFile)
    LOG("[" + pharse +"] Smile loss: " + str(smile_epoch_loss.avg), logFile)
    LOG("[" + pharse +"] emotion loss: " + str(emo_epoch_loss.avg), logFile )
    

    # print("lr: ", str(optimizer))

    try:
        lr = float(str(optimizer).split("\n")[3].split(" ")[-1])
    except:
        lr = 100
    LOG("lr : " + str(lr), logFile)

    return accs, losses, lr, model
