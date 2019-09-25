import os
import re
import time
import datetime
import numpy as np
import pandas as pd

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

from data_load.CVPR_16_ChaLearn_data_loader import get_CVPR_age_data


from utils.utils_1 import get_model
from utils.helper_2 import log_variables_to_board, LOG
from utils.helper import save_checkpoint, load_model_weights
from train_valid.train_valid import train_valid
from train_valid.age_losses_methods import Age_rgs_loss


from opts import args


def parse_sub_folder(args):

    sub_folder_name = "_" + args.model + "_" + args.l1_regression_loss

    return sub_folder_name


def main(**kwargs):
    global args
    for arg, v in kwargs.items():
        args.__setattr__(arg, v)

    # parse loss weight to sub folder name
    args.sub_folder_name = parse_sub_folder(args)

    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')


    path = "./results" + os.sep + args.model + "_" + args.dataset + os.sep + args.sub_folder_name + os.sep + ts_str
    
    tensorboard_folder = os.path.join(path, "Graph")
    
    os.makedirs(path)

    global logFile
    logFile = path + os.sep + "log.txt"

    LOG(args, logFile)

    global writer
    writer = SummaryWriter(tensorboard_folder)

    age_train_loader, age_test_loader = get_CVPR_age_data(args)

    model = get_model(args, logFile)


    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), momentum=0.9,
                          lr=args.lr_rate, weight_decay=args.weight_decay)

    age_l1_criterion = nn.L1Loss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=1e-5, patience=10)
    
    LOG(model, logFile)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.cuda.empty_cache()
        model = model.cuda()

    epochs_train_age_rgs_l1_loss, epochs_valid_age_rgs_l1_loss = [], []

    epochs_train_lr = []

    lowest_loss = 100000

    columns = ['Timstamp', 'Epoch', 'lr',
               'train_age_l1_loss',
               'val_age_l1_loss']

    csv_checkpoint = pd.DataFrame(data=[], columns=columns)

    for epoch in range(0, args.epoch):

        train_losses, lr, model = train_valid(model, [age_train_loader], 
                                                                [age_l1_criterion],
                                                                optimizer, epoch, logFile, args, "train")

        log_variables_to_board([epochs_train_age_rgs_l1_loss],
                                train_losses, 
                                ['train_age_l1_loss'],
                                "Train", tensorboard_folder, epoch, logFile, writer)

        val_losses, lr, model = train_valid(model,[age_test_loader],
                                                            [age_l1_criterion],
                                                            optimizer, epoch, logFile, args, "valid")

        log_variables_to_board([epochs_valid_age_rgs_l1_loss],
                                val_losses,
                                ['val_age_l1_loss'],
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
