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



def parse_args(args):


    if args.multitask_weight_type == 1:
        args.weights = [1, 0, 0]                 # train only for age
        args.folder_sub_name = "_only_age"

    elif args.multitask_weight_type == 2:
        args.weights = [0, 1, 0]                 # train only for gender
        args.folder_sub_name = "_only_gender"

    elif args.multitask_weight_type == 3:
        args.weights = [0, 0, 1]                 # train only for emotion
        args.folder_sub_name = "_only_emotion"

    elif args.multitask_weight_type == 4:
        args.weights = [1, 1, 1]                 # train age, gender, emotion together
        args.folder_sub_name = "_age_gender_emotion_1_1_1"
    else:
        LOG("weight type should be in [0,1,2,3]", logFile)
        exit()
    args.weights = [1, 1, 1]
    args.folder_sub_name = "_age_gender_emotion_smile_0_0_0_1"

    args.train_type = args.multitask_weight_type

    return args



def main(**kwargs):
    global args
    for arg, v in kwargs.items():
        args.__setattr__(arg, v)

    # parse args
    args = parse_args(args)

    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')

    path = "./results" + os.sep + args.model + "_" + args.folder_sub_name + "_" + args.dataset + os.sep + ts_str
    tensorboard_folder = path + os.sep + "Graph"
    csv_path = path + os.sep + "log.csv"
    
    os.makedirs(path)

    global logFile

    logFile = path + os.sep + "log.txt"

    LOG("[Age, Gender, Emotion] load: start ...", logFile)

    LOG(args, logFile)

    global writer
    writer = SummaryWriter(tensorboard_folder)

    age_train_loader, age_test_loader, gender_train_loader, gender_test_loader, smile_train_loader, smile_test_loader = get_CVPR_Age_Gender_Smile_data(args)

    emotion_train_loader, emotion_test_loader = get_FER2013_Emotion_data(args)

    # model = "MTL_ResNet_18"
    # model_type = "res18_cls70"

    if args.model == "MTL_ResNet_18":
        # load model
        model = MTL_ResNet_18_model()

        #load pretrained model 
        if args.load_pretrained_model:
            model = load_pretrained_model(model, "/media/yi/harddrive/codes/MultitaskLearningFace/results/2019-03-24-12-56-49--Age_Gender--IMDB_WIKI/save_models/model_best.pth.tar")

    elif args.model == "res18_cls70":

        model = AgeGenPredModel()
        model = load_pretrained_model(model, "/media/yi/harddrive/codes/Age-Gender-Pred/models/30_epoch_trained_res18_cls70/res18_cls70_best.nn")

        LOG("load ResNet18-Cls70 pretrained weight finished", logFile)
        
        # replace the last fully connected layer.
        model.age_cls_pred = nn.Linear(in_features = 512, out_features=100, bias=True)

        LOG("modify the age prediction range from [0, 60] -> [0, 99]", logFile)


    elif args.model == "MTL_ResNet_50":
        model = MTL_ResNet_50_model()

    elif args.model == "MTL_DenseNet_121_model":
        model = MTL_DenseNet_121_model()

    elif args.model == "Elastic_MTL_DenseNet_121_model":
        model = Elastic_MTL_DenseNet_121_model(args, logFile)

    else:
        NotImplementedError


    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_rate, weight_decay=args.weight_decay)


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

        message = '\n\nEpoch: {}/{}: '.format(epoch + 1, args.epochs)
        LOG(message, logFile)


        accs, losses, lr, model = Train_Valid_Cvpr_age_gender_emotion(model, [gender_train_loader, age_train_loader, smile_train_loader, emotion_train_loader], 
                                        [gender_criterion, age_criterion, smile_criterion, age_cls_criterion, emotion_criterion], optimizer, epoch, logFile, args, "train")

        # accs, losses, lr = train(model, [gender_train_loader, age_train_loader, smile_train_loader], 
        #                                 [gender_criterion, age_criterion, emotion_criterion], optimizer, epoch, args.train_type)

        LOG_variables_to_board([epochs_train_gender_losses, epochs_train_age_losses, epochs_train_emotion_losses], losses, ['gender_loss', 'age_loss', 'emotion_loss'],
                                [epochs_train_gender_accs, epochs_train_emotion_accs], accs, ['gender_acc', 'emotion_acc'],
                                "Train", tensorboard_folder, epoch, logFile, writer)


        val_accs, val_losses, lr, model = Train_Valid_Cvpr_age_gender_emotion(model,[gender_test_loader, age_test_loader, smile_test_loader, emotion_test_loader],
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





if __name__ == "__main__":
    main()
