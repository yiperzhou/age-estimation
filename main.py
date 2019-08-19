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

from data_load.CVPR_16_ChaLearn_data_loader import get_CVPR_Age_data


from utils.utils_1 import get_pretrained_model_weights_path, get_model
from utils.helper_2 import LOG_variables_to_board, LOG
from utils.helper import save_checkpoint, load_model_weights
from utils.config import process_config

from train_valid.train_valid_variant_1_debug_version import Train_Valid
from train_valid.age_losses_methods import Gaussian_age_loss, Euclidean_age_loss, Age_rgs_loss


from opts import args


def parse_loss_weight(args):

    folder_sub_name = "_" + args.all_losses[0]+ "_" + str(args.loss_weights[0]) + "_" + args.all_losses[1] +"_" + str(args.loss_weights[1]) + \
                        "_" + args.all_losses[2] + "_" +  str(args.loss_weights[2]) + args.all_losses[3] + "_" +  str(args.loss_weights[3])

    return folder_sub_name




def main(**kwargs):
    global args
    for arg, v in kwargs.items():
        args.__setattr__(arg, v)

    # parse loss weight to sub folder name
    args.folder_sub_name = parse_loss_weight(args)

    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')

    if args.debug:
        print("[Debug mode]")
        path = os.path.join("./results", "Debug-"+ args.model, "_" + args.dataset, args.folder_sub_name, ts_str)
    else:
        path = "./results" + os.sep + args.model + "_" + args.dataset + os.sep + args.folder_sub_name + os.sep + ts_str

    if args.load_IMDB_WIKI_pretrained_model:
        print("load IMDB WIKI pretrained model")
        path = "./results" + os.sep + "loaded_pretrained-" + args.model + os.sep + "_" + args.dataset + os.sep + args.folder_sub_name + os.sep + ts_str
    else:
        path = "./results" + os.sep + args.model + os.sep + "_" + args.dataset + os.sep + args.folder_sub_name + os.sep + ts_str


    tensorboard_folder = os.path.join(path, "Graph")
    
    os.makedirs(path)

    global logFile
    logFile = path + os.sep + "log.txt"

    LOG(args, logFile)

    global writer
    writer = SummaryWriter(tensorboard_folder)

    age_train_loader, age_test_loader = get_CVPR_Age_data(args)

    pretrained_model_weight_path = get_pretrained_model_weights_path(args)

    model = get_model(args, logFile)


    #load pretrained model 
    if args.load_IMDB_WIKI_pretrained_model:
        model = load_model_weights(model, pretrained_model_weight_path)
        LOG("load pretrained weight, DONE", logFile)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), momentum=0.9, lr=args.lr_rate, weight_decay=args.weight_decay)

    age_cls_criterion = nn.CrossEntropyLoss()
    age_mae_criterion = nn.L1Loss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=1e-5, patience=10)
    
    # log model to logfile, so that I can check the logfile later to know the model detail, for example, the number of class for the age estimation
    LOG(model, logFile)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
        age_cls_criterion = age_cls_criterion.cuda()

    epochs_train_total_loss, epochs_valid_total_loss = [], []
    epochs_train_age_cls_loss, epochs_valid_age_cls_loss = [], []
    epochs_train_age_rgs_mae_loss, epochs_valid_age_rgs_mae_loss = [], []

    epochs_train_age_accs, epochs_valid_age_accs = [], []

    epochs_train_lr = []

    lowest_loss = 100000

    columns = ['Timstamp', 'Epoch', 'lr', 
                'train_total_loss', 'train_age_cls_loss', 'train_age_l1_mae_loss',
                'train_age_acc',
                'val_total_loss', 'val_age_cls_loss', 'val_age_l1_mae_loss',
                'val_age_acc']

    csv_checkpoint = pd.DataFrame(data=[], columns=columns)
    


    for epoch in range(0, args.epoch):

        train_accs, train_losses, lr, model = Train_Valid(model, [age_train_loader], 
                                                                [age_cls_criterion, age_mae_criterion],
                                                                optimizer, epoch, logFile, args, "train", args.debug)

        LOG_variables_to_board([epochs_train_total_loss, epochs_train_age_cls_loss, epochs_train_age_rgs_mae_loss],
                                train_losses, 
                                ['train_total_loss', 'train_age_cls_loss', 'train_age_mae_loss'],
                                [epochs_train_age_accs],
                                train_accs,
                                ['train_age_acc'],
                                "Train", tensorboard_folder, epoch, logFile, writer)

        val_accs, val_losses, lr, model = Train_Valid(model,[age_test_loader],
                                                            [age_cls_criterion, age_mae_criterion],
                                                            optimizer, epoch, logFile, args, "valid", args.debug)


        LOG_variables_to_board([epochs_valid_total_loss, epochs_valid_age_cls_loss, epochs_valid_age_rgs_mae_loss],
                                val_losses,
                                ['val_total_loss', 'val_age_cls_loss', 'val_age_l1_mae_loss'],
                                [epochs_valid_age_accs],
                                val_accs,
                                ['val_age_acc'],
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



    writer.close()
    LOG("done", logFile)
    LOG(args, logFile)
    LOG(path, logFile)


if __name__ == "__main__":
    main()