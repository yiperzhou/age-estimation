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



# from emotion_preprocess import FER2013
from dataload import Mix_Age_Gender_Emotion_Dataset, CVPE_AGE_load_dataset, get_model_data
from model import MTLModel
from opts import args
from helper import emotion_transform_train, emotion_transform_valid, emotion_transform_test, LOG, getstd, load_chalearn_dataset, imdb_image_transformer, AverageMeter, accuracy, save_checkpoint, indexes_to_one_hot, calculate_age_loss


def LOG_variables_to_board(epoch_losses, losses, losses_name, epoch_accs, accs, accs_name, phrase, folder, epoch, logFile):

    global writer
    LOG(phrase + " epoch    " + str(epoch+1) + ":", logFile)
    for e_loss, l, l_n in zip(epoch_losses, losses, losses_name):
        e_loss.append(l)
        writer.add_scalar(folder + os.sep + "data" + os.sep + l_n, l, epoch)
        LOG("          " + l_n + ": "+ str(l), logFile)

    LOG("\n", logFile)

    for e_acc, ac, ac_n in zip(epoch_accs, accs, accs_name):
        e_acc.append(ac)
        writer.add_scalar(folder + os.sep + "data" + os.sep + ac_n, ac, epoch)
        LOG("          " + ac_n + ": "+ str(ac), logFile)
    LOG("---------------", logFile)
    

def save_csv_logging(csv_checkpoint, epoch, lr, losses, val_losses, accs, val_accs, total_loss, total_val_loss, csv_path):
    try:



        csv_checkpoint.loc[len(csv_checkpoint)] = [str(datetime.datetime.now()), epoch, lr, 
                                                    accs[0], accs[1], losses[0], losses[1], losses[2],
                                                    val_accs[0], val_accs[1], val_losses[0], val_losses[1], val_losses[2], 
                                                    total_loss, total_val_loss]
        csv_checkpoint.to_csv(csv_path, index=False)

    except:
        LOG(
            "Error when saving csv files! [tip]: Please check csv column names.", logFile)

        # print(csv_checkpoint.columns)
    return csv_checkpoint


def validate(model, validloader, criterion, optimizer, epoch, train_type):

    global logFile

    LOG("[Main3, Model] Validate: Starting, Epoch: " + str(epoch), logFile)

    best_gen_acc, best_age_mae, best_emotion_acc = 0., 99., 0.
    not_reduce_rounds = 0

    # checkpoint_path = self.checkpoint_best if self.load_best else self.checkpoint_last

    loss = 0

    age_epoch_loss, gender_epoch_loss, emotion_epoch_loss  = AverageMeter(), AverageMeter(), AverageMeter()

    age_epoch_mae, gender_epoch_acc, emotion_epoch_acc = AverageMeter(), AverageMeter(), AverageMeter()

    model.eval()
    torch.cuda.empty_cache()

    epoch_age_tp = 0.
    epoch_age_mae = 0.
    epoch_gender_tp = 0.
    epoch_emotion_tp = 0

    processed_data = 0
    epoch_start_time = time.time()

    gender_criterion, age_criterion, emotion_criterion = criterion[0], criterion[1], criterion[2]
    gender_valid_loader, age_valid_loader, emotion_valid_loader = validloader[0], validloader[1], validloader[2]

    if args.multitask_training_type == 3:

        # valid age task
        for i, (input_img, target) in enumerate(age_valid_loader):
            input_img = input_img.cuda()


            gender_out, age_out, emo_out = model(input_img)
            
            age_loss = calculate_age_loss(age_criterion, age_out, target)
            age_loss = Variable(age_loss, requires_grad = True)

            # print("age loss: ", age_loss)

            age_epoch_mae.update(age_loss[0].item(), 1)
            
        print("valid age task")

        # train gender task
        for i, (input_img, target) in enumerate(gender_valid_loader):
            input_img = input_img.cuda()
            gender_label = target.cuda()

            gender_out, age_out, emo_out = model(input_img)
            gender_loss = gender_criterion(gender_out, gender_label)

            gender_epoch_loss.update(gender_loss.item(), gender_label.size(0))

            gender_prec1 = accuracy(gender_out.data, gender_label)
            gender_epoch_acc.update(gender_prec1[0].item(), gender_label.size(0))

        print("valid Gender task")
        
        # train smile task
        num_smile_img = 0
        for i, (input_img, target) in enumerate(emotion_valid_loader):
            # print("smile task 1")
            try:
                try:
                    input_img = input_img.cuda()
                    #.cuda()
                except:
                    input_img = input_img
                    
                smile_label = target.cuda()
                #.cuda()

                gender_out, age_out, emo_out = model(input_img)
                emo_loss = emotion_criterion(emo_out, smile_label)

                emotion_epoch_loss.update(emo_loss.item(), smile_label.size(0))
                emotion_prec1 = accuracy(emo_out.data, smile_label)
                emotion_epoch_acc.update(emotion_prec1[0].item(), smile_label.size(0))
                num_smile_img += 1

            except:
                continue

        print("num_smile_img: ", num_smile_img)


    elif args.multitask_training_type == 2:
        for i, (data, labels) in enumerate(age_train_loader):
            print("i: ", i, ", inside the for loop")

            # print("train loader")
            if i < 10000000:
                age_img, gender_img, emotion_img = data
                age_label, gender_label, emotion_label = labels
                # print("age label: ", age_label)
                # print("gender_label: ", gender_label)
                # print("emotion_label: ", emotion_label)

                age_img = age_img.cuda()
                gender_img = gender_img.cuda()
                emotion_img = emotion_img.cuda()

                age_label = age_label.cuda()
                gender_label = gender_label.cuda()
                emotion_label = emotion_label.cuda()

                optimizer.zero_grad()
                # gender_out, age_out, emo_out = model(inputs)

                # 难道这里也是三个作为一组然后输入进去吗？
                # 12.03.2019, 再次看这个源码，发现我还是没有很清楚的答案去回答上面的问题，
                gender_out_1, age_out_1, emo_out_1 = model(age_img)
                

                # age
                # age_target = age_target.view(-1, 60)
                _, pred_age = torch.max(age_out_1, 1)
                pred_age = pred_age.view(-1).cpu().numpy()
                pred_ages = []
                for p in pred_age:
                    pred_ages.append([p])
                pred_ages = torch.FloatTensor(pred_ages)
                pred_ages = pred_ages.cuda()
                
                age_label = age_label.unsqueeze(0)
                age_label = age_label.type(torch.cuda.FloatTensor)
                age_loss = age_criterion(pred_ages, age_label)
                age_loss = Variable(age_loss, requires_grad = True)


                age_loss.backward()
                optimizer.step()

                optimizer.zero_grad()

                # gender
                gender_out_2, age_out_2, emo_out_2 = model(gender_img)
                gender_target = gender_label.view(-1)
                gender_loss = gender_criterion(gender_out_2, gender_target)
                gender_loss = Variable(gender_loss, requires_grad = True) 

                gender_loss.backward()
                optimizer.step()

                optimizer.zero_grad()
                
            
                # emotion 
                gender_out_3, age_out_3, emo_out_3 = model(emotion_img)
                _, pred_emo = torch.max(emo_out_3, 1)
                pred_emo = pred_emo.view(-1).cpu().numpy()
                pred_emos = []
                for p in pred_emo:
                    pred_emos.append([p])
                pred_emos = torch.FloatTensor(pred_emos)

                emotion_label = emotion_label.type(torch.cuda.LongTensor)
                pred_emos = pred_emos.cuda()
                emo_loss = emotion_criterion(pred_emos, emotion_label)
                emo_loss = Variable(emo_loss, requires_grad = True) 

                # print("age loss: ", age_loss)
                # print("gender loss: ", gender_loss)
                # print("emotion loss: ", emo_loss)

                # loss = age_loss * weights[0] + gender_loss * weights[1] + emo_loss * weights[2]

                emo_loss.backward()

                optimizer.step()

                # age_prec1 = accuracy(age_out.data, age_target.view(-1))

                # age_epoch_loss.update(age_loss.item(), age_label.size(0))
                # print("age loss: ", age_loss.item())
                # age_epoch_acc.update(age_prec1[0].item(), inputs.size(0))
                # print("acc: ", age_prec1[0].item())

                _, pred_age = torch.max(pred_ages, 1)
                pred_age = pred_age.view(-1).cpu().numpy()
                # age_true    = age_target.view(-1).cpu().numpy()
                age_rgs_loss = sum(np.abs(pred_age - age_label.view(-1)))/age_label.size(0)
                # print("age prediction MAE in batch size:", age_rgs_loss)
                age_epoch_rgs_loss.update(age_rgs_loss, age_label.size(0))

                gender_prec1 = accuracy(gender_out_2.data, gender_target)

                gender_epoch_loss.update(gender_loss.item(), gender_label.size(0))
                # print("gender loss: ", gender_loss.item())
                gender_epoch_acc.update(gender_prec1[0].item(), gender_label.size(0))

                emotion_prec1 = accuracy(emo_out_3.data, emotion_label)
                emotion_epoch_loss.update(emotion_prec1[0].item(), emotion_label.size(0))

                # (age_rgs_loss*0.1).backward()
                # optimizer.step()
            else:
                break


    else:
        NotImplementedError


    # LOG("valid epoch     :" + str(epoch), logFile)
    # LOG("              gender loss : " + str(gender_epoch_loss.avg), logFile)
    # LOG("              age mae : " + str(age_epoch_mae.avg), logFile)
    # LOG("              emotion loss : " + str(emotion_epoch_loss.avg), logFile)

    # LOG("              gender acc : " + str(gender_epoch_acc.avg), logFile)
    # LOG("              emotion acc : " + str(emotion_epoch_acc.avg), logFile)

    # LOG("valid age rgs epoch loss: " + str(age_epoch_loss.avg), logFile)

    # accs = [gender_epoch_acc.avg. emotion_epoch_acc.avg]
    accs = [gender_epoch_acc.avg, emotion_epoch_acc.avg]
    losses = [gender_epoch_loss.avg, age_epoch_mae.avg, emotion_epoch_loss.avg]

    # LOG("---------------- Done" + str(epoch), logFile)
    return accs, losses


def train(model, trainloader, criterion, optimizer, epoch, train_type):
    
    global args

    LOG("[Main3, Model] train: Starting, Epoch: " + str(epoch), logFile)


    best_gen_acc = 0.
    best_age_mae = 99.
    best_emotion_acc = 0.
    
    not_reduce_rounds = 0

    # checkpoint_path = self.checkpoint_best if self.load_best else self.checkpoint_last

    loss = 0

    age_epoch_loss = AverageMeter()
    gender_epoch_loss = AverageMeter()                      
    emotion_epoch_loss = AverageMeter()

    age_epoch_mae = AverageMeter()
    gender_epoch_acc = AverageMeter()
    emotion_epoch_acc = AverageMeter()

    model.train()
    torch.cuda.empty_cache()

    epoch_age_tp = 0.
    epoch_age_mae = 0.
    epoch_gender_tp = 0.
    epoch_emotion_tp = 0

    processed_data = 0
    epoch_start_time = time.time()

    print("three tasks weights: ", args.weights)

    gender_criterion, age_criterion, emotion_criterion = criterion[0], criterion[1], criterion[2]
    gender_train_loader, age_train_loader, emotion_train_loader = trainloader[0], trainloader[1], trainloader[2]

    if args.multitask_training_type == 3:

        # train age task
        for i, (input_img, target) in enumerate(age_train_loader):
            input_img = input_img.cuda()

            optimizer.zero_grad()

            gender_out, age_out, emo_out = model(input_img)
            
            age_loss = calculate_age_loss(age_criterion, age_out, target)
            age_loss = Variable(age_loss, requires_grad = True)

            # print("age loss: ", age_loss)
            age_loss.backward()
            optimizer.step()

            age_epoch_mae.update(age_loss[0].item(), 1)
            
        print("age task")

        # # train gender task
        # for i, (input_img, target) in enumerate(gender_train_loader):
        #     input_img = input_img.cuda()
        #     gender_label = target.cuda()

        #     optimizer.zero_grad()

        #     gender_out, age_out, emo_out = model(input_img)
        #     gender_loss = gender_criterion(gender_out, gender_label)
        #     gender_loss.backward()
        #     optimizer.step()

        #     gender_epoch_loss.update(gender_loss.item(), 1)
        #     gender_prec1 = accuracy(gender_out.data, gender_label)
        #     gender_epoch_acc.update(gender_prec1[0].item(), gender_label.size(0))

        # print("Gender task")
        
        # # train smile task
        # num_smile_img = 0
        # for i, (input_img, target) in enumerate(emotion_train_loader):
        #     # print("smile task 1")
        #     try:
        #         # print("input img: ", input_img)
        #         # print("image size: ", input_img.size())
        #         try:
        #             input_img = input_img.cuda()
        #             #.cuda()
        #         except:
        #             input_img = input_img
                    
        #         smile_label = target.cuda()
        #         #.cuda()

        #         optimizer.zero_grad()

        #         gender_out, age_out, emo_out = model(input_img)
        #         emo_loss = emotion_criterion(emo_out, smile_label)
        #         emo_loss.backward()
        #         optimizer.step()

        #         emotion_epoch_loss.update(emo_loss.item(), 1)
        #         emotion_prec1 = accuracy(emo_out.data, smile_label)
        #         emotion_epoch_acc.update(emotion_prec1[0].item(), smile_label.size(0))

        #         # print("num_smile_img: ", num_smile_img)
        #         num_smile_img += 1

        #     except:
        #         continue
        # print("Smile task")
            

    elif args.multitask_training_type == 2:
        for i, (data, labels) in enumerate(gender_train_loader):
            print("i: ", i, ", inside the for loop")

            # print("train loader")
            if i < 10000000:
                age_img, gender_img, emotion_img = data
                age_label, gender_label, emotion_label = labels
                # print("age label: ", age_label)
                # print("gender_label: ", gender_label)
                # print("emotion_label: ", emotion_label)

                age_img = age_img.cuda()
                gender_img = gender_img.cuda()
                emotion_img = emotion_img.cuda()

                age_label = age_label.cuda()
                gender_label = gender_label.cuda()
                emotion_label = emotion_label.cuda()

                optimizer.zero_grad()
                # gender_out, age_out, emo_out = model(inputs)

                # 难道这里也是三个作为一组然后输入进去吗？
                # 12.03.2019, 再次看这个源码，发现我还是没有很清楚的答案去回答上面的问题，
                gender_out_1, age_out_1, emo_out_1 = model(age_img)
                

                # age
                # age_target = age_target.view(-1, 60)
                _, pred_age = torch.max(age_out_1, 1)
                pred_age = pred_age.view(-1).cpu().numpy()
                pred_ages = []
                for p in pred_age:
                    pred_ages.append([p])
                pred_ages = torch.FloatTensor(pred_ages)
                pred_ages = pred_ages.cuda()
                
                age_label = age_label.unsqueeze(0)
                age_label = age_label.type(torch.cuda.FloatTensor)
                age_loss = age_criterion(pred_ages, age_label)
                age_loss = Variable(age_loss, requires_grad = True)


                age_loss.backward()
                optimizer.step()

                optimizer.zero_grad()

                # gender
                gender_out_2, age_out_2, emo_out_2 = model(gender_img)
                gender_target = gender_label.view(-1)
                gender_loss = gender_criterion(gender_out_2, gender_target)
                gender_loss = Variable(gender_loss, requires_grad = True) 

                gender_loss.backward()
                optimizer.step()

                optimizer.zero_grad()
                
            
                # emotion 
                gender_out_3, age_out_3, emo_out_3 = model(emotion_img)
                _, pred_emo = torch.max(emo_out_3, 1)
                pred_emo = pred_emo.view(-1).cpu().numpy()
                pred_emos = []
                for p in pred_emo:
                    pred_emos.append([p])
                pred_emos = torch.FloatTensor(pred_emos)

                emotion_label = emotion_label.type(torch.cuda.LongTensor)
                pred_emos = pred_emos.cuda()
                emo_loss = emotion_criterion(pred_emos, emotion_label)
                emo_loss = Variable(emo_loss, requires_grad = True) 

                # print("age loss: ", age_loss)
                # print("gender loss: ", gender_loss)
                # print("emotion loss: ", emo_loss)

                # loss = age_loss * weights[0] + gender_loss * weights[1] + emo_loss * weights[2]

                emo_loss.backward()

                optimizer.step()

                # age_prec1 = accuracy(age_out.data, age_target.view(-1))

                # age_epoch_loss.update(age_loss.item(), age_label.size(0))
                # print("age loss: ", age_loss.item())
                # age_epoch_acc.update(age_prec1[0].item(), inputs.size(0))
                # print("acc: ", age_prec1[0].item())

                _, pred_age = torch.max(pred_ages, 1)
                pred_age = pred_age.view(-1).cpu().numpy()
                # age_true    = age_target.view(-1).cpu().numpy()
                age_rgs_loss = sum(np.abs(pred_age - age_label.view(-1)))/age_label.size(0)
                # print("age prediction MAE in batch size:", age_rgs_loss)
                age_epoch_rgs_loss.update(age_rgs_loss, age_label.size(0))

                gender_prec1 = accuracy(gender_out_2.data, gender_target)

                gender_epoch_loss.update(gender_loss.item(), gender_label.size(0))
                # print("gender loss: ", gender_loss.item())
                gender_epoch_acc.update(gender_prec1[0].item(), gender_label.size(0))

                emotion_prec1 = accuracy(emo_out_3.data, emotion_label)
                emotion_epoch_loss.update(emotion_prec1[0].item(), emotion_label.size(0))

                # (age_rgs_loss*0.1).backward()
                # optimizer.step()
            else:
                break


    else:
        NotImplementedError



    accs = [gender_epoch_acc.avg, emotion_epoch_acc.avg]
    losses = [gender_epoch_loss.avg, age_epoch_mae.avg, emotion_epoch_loss.avg]
    # print("[Train] losses: ", losses)
    # print("[Train] accs: ", accs)

    try:
        lr = float(str(optimizer).split("\n")[5].split(" ")[-1])
    except:
        lr = 100
    LOG("lr : " + str(lr), logFile)

    return accs, losses, lr


def parse_args(args):


    train_type = args.multitask_weight_type

    if train_type == 0:
        args.folder_sub_name = "_only_age"
    elif train_type == 1:
        args.folder_sub_name = "_only_gender"
    elif train_type == 2:
        args.folder_sub_name = "_only_emotion"
    elif train_type == 3:
        args.folder_sub_name = "_age_gender_emotion"
    else:
        args.folder_sub_name = "_wrong"
        exit()


    if args.multitask_weight_type == 0:
        args.weights = [1, 0, 0]                 # train only for age
    elif args.multitask_weight_type == 1:
        args.weights = [0, 1, 0]                 # train only for gender
    elif args.multitask_weight_type == 2:
        args.weights = [0, 0, 1]                 # train only for emotion
    elif args.multitask_weight_type == 3:
        args.weights = [1, 1, 1]                 # train age, gender, emotion together
    else:
        LOG("weight type should be in [0,1,2,3]", logFile)
        exit()

    args.train_type = train_type
    args.num_epochs = 50

    return args


def main(**kwargs):
    global args
    for arg, v in kwargs.items():
        args.__setattr__(arg, v)

    # parse args
    args = parse_args(args)

    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')

    path = "./models" + os.sep + ts_str+args.folder_sub_name
    csv_path = "./models" + os.sep + ts_str + args.folder_sub_name + os.sep + "log.csv"

    tensorboard_folder = path + os.sep + "Graph"
    os.makedirs(path)

    global logFile

    logFile = "./models" + os.sep + ts_str+args.folder_sub_name + os.sep + "log.txt"

    LOG("[Age, Gender, Emotion] load: start ...", logFile)

    LOG(args, logFile)

    global writer
    writer = SummaryWriter(tensorboard_folder)

    age_train_loader, age_test_loader, gender_train_loader, gender_test_loader, smile_train_loader, smile_test_loader = get_model_data(args)


    # load model
    model = MTLModel()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_rate, weight_decay=args.weight_decay)

    # age_cls_criterion = nn.CrossEntropyLoss()
    age_criterion = nn.L1Loss()
    gender_criterion = nn.CrossEntropyLoss()
    emotion_criterion = nn.CrossEntropyLoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=1e-5, patience=10)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
        # age_cls_criterion = age_cls_criterion.cuda()

        gender_criterion = gender_criterion.cuda()
        age_criterion = age_criterion.cuda()
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


    for epoch in range(0, args.num_epochs):

        message = '\n\nEpoch: {}/{}: '.format(epoch + 1, args.num_epochs)
        LOG(message, logFile)

        accs, losses, lr = train(model, [gender_train_loader, age_train_loader, smile_train_loader], 
                                        [gender_criterion, age_criterion, emotion_criterion], optimizer, epoch, args.train_type)

        LOG_variables_to_board([epochs_train_gender_losses, epochs_train_age_losses, epochs_train_emotion_losses], losses, ['gender_loss', 'age_loss', 'emotion_loss'],
                                [epochs_train_gender_accs, epochs_train_emotion_accs], accs, ['gender_acc', 'emotion_acc'],
                                "Train", tensorboard_folder, epoch, logFile)


        val_accs, val_losses = validate(model,[gender_test_loader, age_test_loader, smile_test_loader],
                                                [gender_criterion, age_criterion, emotion_criterion], optimizer, epoch, args.train_type)

        LOG_variables_to_board([epochs_valid_gender_losses, epochs_valid_age_losses, epochs_valid_emotion_losses], val_losses, ['val_gender_loss', 'val_age_loss', 'val_emotion_loss'],
                                [epochs_valid_gender_accs, epochs_valid_emotion_accs], val_accs, ['val_gender_acc', 'val_emotion_acc'],
                                "Valid", tensorboard_folder, epoch, logFile)

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
        save_csv_logging(csv_checkpoint, epoch, lr, losses, val_losses, accs, val_accs, total_train_loss, total_val_loss, csv_path)

    writer.close()
    LOG("done", logFile)





if __name__ == "__main__":
    main()
