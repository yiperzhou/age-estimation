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

def parse_loss_weight(args):

    folder_sub_name = "_" + args.subtasks[0]+ "_" + str(args.loss_weights[0]) + "_" + args.subtasks[1] +"_" + str(args.loss_weights[1]) + \
                        "_" + args.subtasks[2] + "_" +  str(args.loss_weights[2]) + "_" + args.subtasks[3] + "_" + str(args.loss_weights[3])

    return folder_sub_name




def main(**kwargs):
    global args
    for arg, v in kwargs.items():
        args.__setattr__(arg, v)

    # parse loss weight to sub folder name
    args.folder_sub_name = parse_loss_weight(args)

    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')

    path = "./results" + os.sep + args.model + os.sep + args.folder_sub_name + "_" + args.dataset + os.sep + ts_str
    tensorboard_folder = path + os.sep + "Graph"
    csv_path = path + os.sep + "log.csv"
    
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
    if args.load_pretrained_model:
        model = load_model_weights(model, pretrained_model_weight_path)
        LOG("load pretrained weight, DONE", logFile)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), momentum=0.9, lr=args.lr_rate, weight_decay=args.weight_decay)

    age_cls_criterion = nn.CrossEntropyLoss()
    age_criterion = nn.L1Loss()
    gender_criterion = nn.CrossEntropyLoss()
    smile_criterion = nn.CrossEntropyLoss()
    emotion_criterion = nn.CrossEntropyLoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=1e-5, patience=10)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
        age_cls_criterion = age_cls_criterion.cuda()

        gender_criterion = gender_criterion.cuda()
        age_criterion = age_criterion.cuda()
        smile_criterion = smile_criterion.cuda()
        emotion_criterion = emotion_criterion.cuda()

    epochs_train_loss, epochs_valid_loss = [], []

    epochs_train_age_mae, epochs_train_age_losses = [], []

    epochs_valid_age_mae, epochs_valid_age_losses = [], []

    epochs_train_gender_accs, epochs_train_gender_losses = [], []

    epochs_valid_gender_accs, epochs_valid_gender_losses = [], []

    epochs_train_emotion_accs, epochs_train_emotion_losses = [], []
    
    epochs_valid_emotion_accs, epochs_valid_emotion_losses = [], []

    epochs_train_lr = []

    lowest_loss = 100000

    columns = ['Timstamp', 'Epoch', 'lr', 
                'train_gender_acc', 'train_emotion_acc', 'train_gender_loss', 'train_age_mae', 'train_emotion_loss',
                'valid_gender_acc', 'valid_emotion_acc', 'valid_gender_loss', 'valid_age_mae', 'valid_emotion_loss', 
                'train_total_loss', 'val_total_loss']

    csv_checkpoint = pd.DataFrame(data=[], columns=columns)
    


    for epoch in range(0, args.epochs):

        accs, losses, lr, model = Train_Valid(model, [gender_train_loader, age_train_loader, smile_train_loader, emotion_train_loader], 
                                        [gender_criterion, age_criterion, smile_criterion, age_cls_criterion, emotion_criterion], optimizer, epoch, logFile, args, "train")

        LOG_variables_to_board([epochs_train_gender_losses, epochs_train_age_losses, epochs_train_emotion_losses], losses, ['train_gender_loss', 'train_age_loss', 'train_emotion_loss'],
                                [epochs_train_gender_accs, epochs_train_emotion_accs], accs, ['train_gender_acc', 'train_emotion_acc'],
                                "Train", tensorboard_folder, epoch, logFile, writer)


        val_accs, val_losses, lr, model = Train_Valid(model,[gender_test_loader, age_test_loader, smile_test_loader, emotion_test_loader],
                                                [gender_criterion, age_criterion, smile_criterion, age_cls_criterion, emotion_criterion], optimizer, epoch, logFile, args, "valid")

        LOG_variables_to_board([epochs_valid_gender_losses, epochs_valid_age_losses, epochs_valid_emotion_losses], val_losses, ['val_gender_loss', 'val_age_loss', 'val_emotion_loss'],
                                [epochs_valid_gender_accs, epochs_valid_emotion_accs], val_accs, ['val_gender_acc', 'val_emotion_acc'],
                                "Valid", tensorboard_folder, epoch, logFile, writer)

        LOG("\n", logFile)


        total_train_loss = losses[0]+losses[1]+losses[2]
        epochs_train_loss.append(total_train_loss)
        writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'total_loss', total_train_loss, epoch)

        LOG("total_loss: " + str(total_train_loss), logFile)

        LOG("in epoch loop: " + str(lr), logFile)

        epochs_train_lr.append(lr)
        writer.add_scalar(tensorboard_folder + os.sep + 'lr', lr, epoch)
        LOG("lr: " + str(lr), logFile)


        total_val_loss = val_losses[0] + val_losses[1] + val_losses[2]
        epochs_valid_loss.append(total_val_loss)
        writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'val_total_loss', total_val_loss, epoch)
        LOG("val_total_loss: " + str(total_val_loss), logFile)

        message = '\n\nEpoch: {}/{}: '.format(epoch + 1, args.epochs)
        LOG(message + args.model, logFile)


        scheduler.step(total_val_loss)
        if lowest_loss > total_val_loss:
            lowest_loss = total_val_loss

            save_checkpoint({
                'epoch': epoch,
                'model': "MTLModel",
                'state_dict': model.state_dict(),
                'lowest_loss': lowest_loss,
                'optimizer': optimizer.state_dict(),
            }, path)
    
        # save csv logging file
        save_csv_logging(csv_checkpoint, epoch, lr, losses, val_losses, accs, val_accs, total_train_loss, total_val_loss, csv_path, logFile)

    writer.close()
    LOG("done", logFile)
    LOG(args, logFile)





if __name__ == "__main__":
    main()
