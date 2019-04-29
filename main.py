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
from train_valid import *

from opts import args
from train_valid.train_valid_variant_1_debug_version import Train_Valid_debug

from utils.config import process_config

def parse_loss_weight(args):

    folder_sub_name = "_" + args.subtasks[0]+ "_" + str(args.loss_weights[0]) + "_" + args.subtasks[1] +"_" + str(args.loss_weights[1]) + \
                        "_" + args.subtasks[2] + "_" +  str(args.loss_weights[2]) + "_" + args.subtasks[3] + "_" + str(args.loss_weights[3])

    return folder_sub_name




def main(**kwargs):
    global args
    for arg, v in kwargs.items():
        args.__setattr__(arg, v)

    # # parse config 
    # config = process_config(args.config)

    # parse loss weight to sub folder name
    args.folder_sub_name = parse_loss_weight(args)

    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')

    if args.debug:
        print("[Debug mode]")
        path = "./results" + os.sep + "Debug-" + args.model + os.sep + args.folder_sub_name + "_" + args.dataset + os.sep + ts_str
    else:
        path = "./results" + os.sep + args.model + os.sep + args.folder_sub_name + "_" + args.dataset + os.sep + ts_str

    if args.load_IMDB_WIKI_pretrained_model:
        print("load IMDB WIKI pretrained model")
        path = "./results" + os.sep + "loaded_pretrained-" + args.model + os.sep + args.folder_sub_name + "_" + args.dataset + os.sep + ts_str
    else:
        path = "./results" + os.sep + args.model + os.sep + args.folder_sub_name + "_" + args.dataset + os.sep + ts_str


    tensorboard_folder = path + os.sep + "Graph"
    # csv_path = path + os.sep + "log.csv"
    
    os.makedirs(path)

    global logFile
    logFile = path + os.sep + "log.txt"

    LOG(args, logFile)

    global writer
    writer = SummaryWriter(tensorboard_folder)

    age_train_loader, age_test_loader, gender_train_loader, gender_test_loader, smile_train_loader, smile_test_loader = get_CVPR_Age_Gender_Smile_data(args)

    emotion_train_loader, emotion_test_loader = get_FER2013_Emotion_data(args)

    pretrained_model_weight_path = get_pretrained_model_weights_path(args)

    model = get_model(args, logFile)


    #load pretrained model 
    if args.load_IMDB_WIKI_pretrained_model:
        model = load_model_weights(model, pretrained_model_weight_path)
        LOG("load pretrained weight, DONE", logFile)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), momentum=0.9, lr=args.lr_rate, weight_decay=args.weight_decay)

    age_cls_criterion = nn.CrossEntropyLoss()
    # age_cls_criterion = nn.BCELoss()
    age_mae_criterion = nn.L1Loss()
    gender_criterion = nn.CrossEntropyLoss()
    smile_criterion = nn.CrossEntropyLoss()
    emotion_criterion = nn.CrossEntropyLoss()

    
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=1e-5, patience=10)
    
    # log model to logfile, so that I can check the logfile later to know the model detail, for example, the number of class for the age estimation
    LOG(model, logFile)


    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
        age_cls_criterion = age_cls_criterion.cuda()

        gender_criterion = gender_criterion.cuda()
        age_mae_criterion = age_mae_criterion.cuda()
        smile_criterion = smile_criterion.cuda()
        emotion_criterion = emotion_criterion.cuda()

    epochs_train_total_loss, epochs_valid_total_loss = [], []

    epochs_train_smile_accs, epochs_train_smile_losses = [], []
    epochs_valid_smile_accs, epochs_valid_smile_losses = [], []

    epochs_train_gender_accs, epochs_train_gender_losses = [], []
    epochs_valid_gender_accs, epochs_valid_gender_losses = [], []

    epochs_train_emotion_accs, epochs_train_emotion_losses = [], []
    epochs_valid_emotion_accs, epochs_valid_emotion_losses = [], []

    epochs_train_age_accs, epochs_train_age_losses = [], []
    epochs_valid_age_accs, epochs_valid_age_losses = [], []

    epochs_train_age_mae_losses = []
    epochs_valid_age_mae_losses = []

    epochs_train_lr = []

    lowest_loss = 100000

    columns = ['Timstamp', 'Epoch', 'lr', 
                'train_gender_loss', 'train_smile_loss', 'train_emotion_loss', 'train_age_loss', 'train_age_mae', 'train_total_loss'
                'train_gender_acc', 'train_smile_acc', 'train_emotion_acc', 'train_age_acc',
                'val_gender_loss', 'val_smile_loss',  'val_emotion_loss', 'val_age_loss', 'val_age_mae', 'val_total_loss',
                'val_gender_acc', 'val_smile_acc', 'val_emotion_acc', 'val_age_loss']

    csv_checkpoint = pd.DataFrame(data=[], columns=columns)
    


    for epoch in range(0, args.epoch):


        if args.multitask_training_type == "Train_Valid":
            if args.debug == True:
                LOG("enter [DEBUG] Tran_Valid", logFile)
                train_accs, train_losses, lr, model = Train_Valid_debug(model, [gender_train_loader, age_train_loader, smile_train_loader, emotion_train_loader], 
                                                [gender_criterion, age_mae_criterion, smile_criterion, age_cls_criterion, emotion_criterion], optimizer, epoch, logFile, args, "train")

            else:
                LOG("enter [Normal] Tran_Valid", logFile)
                train_accs, train_losses, lr, model = Train_Valid(model, [gender_train_loader, age_train_loader, smile_train_loader, emotion_train_loader], 
                                                [gender_criterion, age_mae_criterion, smile_criterion, age_cls_criterion, emotion_criterion], optimizer, epoch, logFile, args, "train")                

        elif args.multitask_training_type == "Train_Valid_2":
            train_accs, train_losses, lr, model = Train_Valid_2(model, [gender_train_loader, age_train_loader, smile_train_loader, emotion_train_loader], 
                                            [gender_criterion, age_mae_criterion, smile_criterion, age_cls_criterion, emotion_criterion], optimizer, epoch, logFile, args, "train")
        else:
            LOG("[Train_Valid, Train_Valid_2]", logFile)


        LOG_variables_to_board([epochs_train_gender_losses, epochs_train_smile_losses, epochs_train_emotion_losses, epochs_train_age_losses, epochs_train_age_mae_losses],
                                train_losses, 
                                ['train_gender_loss', 'train_smile_loss', 'train_emotion_loss', 'train_age_loss', 'train_age_mae', 'train_total_loss'],
                                [epochs_train_gender_accs, epochs_train_smile_accs ,epochs_train_emotion_accs, epochs_train_age_accs, epochs_train_total_loss],
                                train_accs,
                                ['train_gender_acc', 'train_smile_acc', 'train_emotion_acc', 'train_age_acc'],
                                "Train", tensorboard_folder, epoch, logFile, writer)


        if args.multitask_training_type == "Train_Valid":
            if args.debug == True:
                LOG("enter [DEBUG] Tran_Valid", logFile)
                val_accs, val_losses, lr, model = Train_Valid_debug(model,[gender_test_loader, age_test_loader, smile_test_loader, emotion_test_loader],
                                                        [gender_criterion, age_mae_criterion, smile_criterion, age_cls_criterion, emotion_criterion], optimizer, epoch, logFile, args, "valid")
            else:
                LOG("enter [Normal] Tran_Valid", logFile)
                val_accs, val_losses, lr, model = Train_Valid(model,[gender_test_loader, age_test_loader, smile_test_loader, emotion_test_loader],
                                                        [gender_criterion, age_mae_criterion, smile_criterion, age_cls_criterion, emotion_criterion], optimizer, epoch, logFile, args, "valid")

        elif args.multitask_training_type == "Train_Valid_2":
            val_accs, val_losses, lr, model = Train_Valid_2(model,[gender_test_loader, age_test_loader, smile_test_loader, emotion_test_loader],
                                                    [gender_criterion, age_mae_criterion, smile_criterion, age_cls_criterion, emotion_criterion], optimizer, epoch, logFile, args, "valid")
        else:
            LOG("[Train_Valid, Train_Valid_2]", logFile)


        LOG_variables_to_board([epochs_valid_gender_losses, epochs_valid_smile_losses, epochs_valid_emotion_losses, epochs_valid_age_losses, epochs_valid_age_mae_losses],
                                val_losses,
                                ['val_gender_loss', 'val_smile_loss',  'val_emotion_loss', 'val_age_loss', 'val_age_mae', 'val_total_loss'],
                                [epochs_valid_gender_accs, epochs_valid_smile_accs, epochs_valid_emotion_accs, epochs_valid_age_accs, epochs_valid_total_loss],
                                val_accs,
                                ['val_gender_acc', 'val_smile_acc', 'val_emotion_acc', 'val_age_acc'],
                                "Valid", tensorboard_folder, epoch, logFile, writer)

        LOG("\n", logFile)

        epochs_train_lr.append(lr)
        writer.add_scalar(tensorboard_folder + os.sep + 'lr', lr, epoch)


        message = '\n\nEpoch: {}/{}: '.format(epoch + 1, args.epoch)
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
    
        # # save csv logging file
        # save_csv_logging(csv_checkpoint, epoch, lr, train_losses, train_accs, val_losses, val_accs, csv_path, logFile)


    writer.close()
    LOG("done", logFile)
    LOG(args, logFile)





if __name__ == "__main__":
    main()
