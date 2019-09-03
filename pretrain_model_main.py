# pretrain model with IMDB-WIKI dataset
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
from tqdm import tqdm

from models import *
from data_load import *
from utils import *

from pretrain_opts import args


def age_mae_criterion_Encapsulation_IMDB_WIKI(age_criterion, age_out_1, age_label):
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
    # # print("pred_ages: ", pred_ages)
    # # print("age_label: ", age_label)
    # print("pred_ages: ", pred_ages)
    # print("age_label: ", age_label)

    try:
        age_loss_mae = age_criterion(pred_ages, age_label)

    except ValueError as identifier:
        LOG("pred_ages: " + str(pred_ages), logFile)
        LOG("age_label: " + str(age_label), logFile)

        pass
    
    # age_loss_mae = Variable(age_loss_mae, requires_grad = True) 
    
    # print("age_loss_mae: ", age_loss_mae)

    return age_loss_mae


def convert_one_hot_to_index_format(age_cls_label):

    age_cls_label = age_cls_label.reshape([len(age_cls_label), 101])
    _, age_cls_label = age_cls_label.max(1)

    age_cls_label = age_cls_label.type(torch.cuda.LongTensor)

    return age_cls_label

def Train_Valid(model, loader, criterion, optimizer, epoch, logFile, args, pharse):

    LOG("[" + pharse + "]: Starting, Epoch: " + str(epoch), logFile)

    best_gen_acc = 0.
    best_age_mae = 99.
    best_emotion_acc = 0.
    
    not_reduce_rounds = 0

    loss = 0

    age_epoch_loss = AverageMeter()
    gender_epoch_loss = AverageMeter()
    total_epoch_loss = AverageMeter()

    age_epoch_mae = AverageMeter()
    gender_epoch_acc = AverageMeter()
    age_epoch_acc = AverageMeter()

    age_epoch_mae_own_list = []

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

    print("[Age, Gender] tasks weights: ", args.loss_weights)

    gender_criterion, age_criterion, age_cls_criterion = criterion[0], criterion[1], criterion[2]
    pbar = tqdm(total=len(loader.dataset))

    for batch_idx, input_data in tqdm(enumerate(loader)):
        
        image, gender_label, age_rgs_label, age_cls_label = input_data[0], input_data[1], input_data[2], input_data[3]
        input_img = image.cuda()

        gender_label = gender_label.squeeze(-1)
        gender_label = gender_label.cuda()

        age_cls_label = age_cls_label.type(torch.cuda.LongTensor)
        # age_cls_label = age_cls_label.cuda()

        if args.multitask_training_type == 3:
            # optimizer.zero_grad()

            gender_out, age_out, emo_out, smile_out = model(input_img)

            _, age_out = age_out.max(1)
            age_out = age_out.type(torch.cuda.FloatTensor)

            age_loss = age_criterion(age_out, age_cls_label)
            age_loss = torch.autograd.Variable(age_loss, requires_grad = True)

            print("age loss: ", age_loss)
            age_loss.backward()
            optimizer.step()

            # gender
            optimizer.zero_grad()

            gender_label = gender_label.squeeze(-1)

            gender_loss = gender_criterion(gender_out, gender_label)
            gender_loss.backward()
            optimizer.step()

        elif args.multitask_training_type == 2:

            if pharse == "train":
                optimizer.zero_grad()

            gender_out, smile_out, emo_out, age_out = model(input_img)

            age_cls_loss = age_cls_criterion(age_out, age_cls_label)

            age_loss_mae = age_mae_criterion_Encapsulation_IMDB_WIKI(age_criterion, age_out, age_cls_label)

            # print("gender_label: ", gender_label)
            # print("gender_out  : ", gender_out)

            gender_loss = gender_criterion(gender_out, gender_label)

            # total loss
            reduce_gen_loss, reduce_age_cls_loss  = args.loss_weights[0], args.loss_weights[1]
            
            gender_loss = gender_loss * reduce_gen_loss
            age_loss = age_cls_loss * reduce_age_cls_loss
            loss = gender_loss + age_loss

            age_epoch_loss.update(age_cls_loss.item(), 1)
            age_epoch_mae.update(age_loss_mae.item(), 1)
            gender_epoch_loss.update(gender_loss.item(), 1)
            
            age_epoch_mae_own_list.append(age_loss_mae.item())

            # print("gender_out.data:  ", gender_out.data)
            # print("gender_label   :  ", gender_label)
            gender_prec1 = accuracy(gender_out.data, gender_label)
            gender_epoch_acc.update(gender_prec1[0].item(), gender_label.size(0))

            # print("age_out.data    :  ", age_out.data)
            # print("age_cls_label   :  ", age_cls_label)
            age_prec1 = accuracy(age_out.data, age_cls_label)
            age_epoch_acc.update(age_prec1[0].item(), age_cls_label.size(0))

            if pharse == "train":
                loss.backward()
                optimizer.step()
            elif pharse == "valid":
                # print("valid pharse")
                continue
            else:
                print("pharse should be in [train, valid]")
                NotImplementedError
            total_epoch_loss.update(loss.item(), 1)     
            # print("[", pharse," LOSS]", "total: ", loss.item())
            # print("                 Age: ", age_cls_loss.item())
            # print("             MAE Age: ", age_loss_mae.item())
            # print("              Gender: ", gender_loss.item())
        else:
            pass

        # print("train loader")
        if batch_idx % 1000 == 0:
            print(
                "[{}] Epoch: {} [{}/{}]".
                format(pharse, epoch, batch_idx * len(input_img), len(loader.dataset)))
            # pbar.update(batch_idx * len(input_img))

            print("[" + pharse +"] [ACC(%)], [gender, age        ]: " + str([gender_epoch_acc.avg, age_epoch_acc.avg]))
            print("[" + pharse +"] [Loss  ], [gender, age, age_mae, total]: " + str([gender_epoch_loss.avg, age_epoch_mae.avg, age_epoch_loss.avg, total_epoch_loss.avg]))
    
    LOG("One [" + pharse + "] epoch took {} minutes".format((time.time() - epoch_start_time)/60), logFile)

    # one epoch end
    accs = [gender_epoch_acc.avg, age_epoch_acc.avg]
    losses = [gender_epoch_loss.avg, age_epoch_mae.avg, age_epoch_loss.avg, total_epoch_loss.avg]

    LOG("[" + pharse + "] [ACC(%)], [gender, age                ]: " + str(accs), logFile)
    LOG("[" + pharse + "] losses:   [gender, age_mae, age, total]: " + str(losses), logFile)
    LOG("[" + pharse +"] age_epoch_mae_own_list, mean age                       : " + str(np.mean(age_epoch_mae_own_list)), logFile)
    # pbar.close()

    try:
        lr = float(str(optimizer).split("\n")[3].split(" ")[-1])
    except:
        lr = 100
    LOG("lr : " + str(lr), logFile)

    return accs, losses, lr, model 


def parse_loss_weight(args):

    folder_sub_name = "_" + args.subtasks[0]+ "_" + str(args.loss_weights[0]) + "_" + args.subtasks[1] +"_" + str(args.loss_weights[1])

    return folder_sub_name


def main(**kwargs):
    global args
    for arg, v in kwargs.items():
        args.__setattr__(arg, v)

    # parse loss weight to sub folder name
    args.folder_sub_name = parse_loss_weight(args)

    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')

    path = "./results" + os.sep + "pretrained_" + args.model + os.sep \
           + args.folder_sub_name + "_" + args.dataset + os.sep + ts_str
    tensorboard_folder = path + os.sep + "Graph"
    # csv_path = path + os.sep + "log.csv"
    
    os.makedirs(path)

    global logFile

    logFile = path + os.sep + "log.txt"

    LOG(args, logFile)

    global writer
    writer = SummaryWriter(tensorboard_folder)

    IMDB_WIKI_train_loader, IMDB_WIKI_val_loader = load_IMDB_WIKI_dataset(args)

    model = get_model(args, logFile)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_rate, weight_decay=args.weight_decay)

    age_cls_criterion = nn.CrossEntropyLoss()
    age_criterion = nn.L1Loss()
    gender_criterion = nn.CrossEntropyLoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=1e-5, patience=10)
    # log model to logfile, so that I can check the logfile later to know the model detail,
    # for example, the number of class for the age estimation
    LOG(model, logFile)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
        age_cls_criterion = age_cls_criterion.cuda()

        gender_criterion = gender_criterion.cuda()
        age_criterion = age_criterion.cuda()
    
    epochs_train_total_loss, epochs_valid_total_loss = [], []

    epochs_train_age_mae, epochs_train_age_losses = [], []

    epochs_valid_age_mae, epochs_valid_age_losses = [], []

    epochs_train_gender_accs, epochs_train_gender_losses = [], []

    epochs_valid_gender_accs, epochs_valid_gender_losses = [], []

    # epochs_train_emotion_accs, epochs_train_emotion_losses = [], []
    
    # epochs_valid_emotion_accs, epochs_valid_emotion_losses = [], []
    epochs_train_age_accs, epochs_valid_age_accs = [], []

    epochs_train_lr = []

    lowest_loss = 100000

    columns = ['Timstamp', 'Epoch', 'lr', 
                'train_gender_loss', 'train_age_loss', 'train_age_mae', 'train_total_loss'
                'train_gender_acc', 'train_age_acc',
                'val_gender_loss', 'val_age_loss', 'val_age_mae', 'val_total_loss',
                'val_gender_acc', 'val_age_loss']

    csv_checkpoint = pd.DataFrame(data=[], columns=columns)


    for epoch in range(0, args.epochs):

        message = '\n\nEpoch: {}/{}: '.format(epoch + 1, args.epochs)
        LOG(message, logFile)

        train_accs, train_losses, lr, model = Train_Valid(model, IMDB_WIKI_train_loader, 
                                        [gender_criterion, age_criterion, age_cls_criterion], optimizer, epoch, logFile, args, "train")

        LOG_variables_to_board([epochs_train_gender_losses, epochs_train_age_losses, epochs_train_age_mae, epochs_train_total_loss],
                                train_losses,
                                ['train_gender_loss', 'train_age_loss', 'train_age_mae', 'train_total_loss'],
                                [epochs_train_gender_accs, epochs_train_age_accs],
                                train_accs,
                                ['train_gender_acc', 'train_age_acc'],
                                "Train", tensorboard_folder, epoch, logFile, writer)


        val_accs, val_losses, lr, model = Train_Valid(model, IMDB_WIKI_val_loader,
                                                [gender_criterion, age_criterion, age_cls_criterion], optimizer, epoch, logFile, args, "valid")

        LOG_variables_to_board([epochs_valid_gender_losses, epochs_valid_age_losses, epochs_valid_age_mae, epochs_valid_total_loss],
                                val_losses,
                                ['val_gender_loss', 'val_age_loss', 'val_age_mae', 'val_total_loss'],
                                [epochs_valid_gender_accs, epochs_valid_age_accs],
                                val_accs,
                                ['val_gender_acc', 'val_age_acc'],
                                "Valid", tensorboard_folder, epoch, logFile, writer)

        LOG("\n", logFile)

        epochs_train_lr.append(lr)
        writer.add_scalar(tensorboard_folder + os.sep + 'lr', lr, epoch)

        message = '\n\nEpoch: {}/{}: '.format(epoch + 1, args.epochs)
        LOG(message, logFile)
        LOG(args, logFile)

        scheduler.step(val_losses[-1])
        if lowest_loss > val_losses[-1]:
            lowest_loss = val_losses[-1]

            save_checkpoint({
                'epoch': epoch,
                'model': args.model,
                'state_dict': model.state_dict(),
                'lowest_loss': lowest_loss,
                'optimizer': optimizer.state_dict(),
            }, path)

    writer.close()
    LOG("done", logFile)
    LOG(args, logFile)



if __name__ == "__main__":
    main()
