import time
import torch
import torch.nn as nn
from utils import *
from utils.helper_4 import convert_to_onehot_tensor

def age_mae_criterion_encapsulation(age_criterion, age_out_cls, age_label):
    _, pred_ages = torch.max(age_out_cls, 1)
    pred_ages = pred_ages.type(torch.cuda.FloatTensor)
    age_label = age_label.type(torch.cuda.FloatTensor)
    age_loss_mae = age_criterion(pred_ages, age_label)
    return age_loss_mae

def age_mapping_function(origin_value, age_divide):
    origin_value = origin_value.cpu()
    y_true_rgs = torch.div(origin_value, age_divide)
    return  y_true_rgs

def age_cls_criterion_encapsulation(age_criterion, age_out_cls, age_label, classification_type):
    if classification_type == "100_classes":
        age_divide = 1
    elif classification_type == "20_classes":
        age_divide = 5
    elif classification_type == "10_classes":
        age_divide = 10
    elif classification_type == "5_classes":
        age_divide = 20
    else:
        print("classification type value is wrong")
        raise ValueError
    mapped_age_label = age_mapping_function(age_label, age_divide)
    mapped_age_label = mapped_age_label.type(torch.cuda.FloatTensor)
    mapped_age_label = mapped_age_label.type(torch.cuda.LongTensor)
    age_cls_loss = age_criterion(age_out_cls, mapped_age_label)
    return age_cls_loss

def train_valid(model, loader, criterion, optimizer, epoch, logFile, args, pharse):
    LOG("[" + pharse + "]: Starting, Epoch: " + str(epoch), logFile)
    best_age_mae = 99.
    loss = 0
    age_cls_epoch_loss = AverageMeter()
    age_l1_rgs_epoch_loss = AverageMeter()
    age_epoch_loss = AverageMeter()
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
    age_cls_criterion = criterion[0]
    age_loader = loader[0]
    for j_batch_idx, (age_img, age_label) in enumerate(age_loader):
        if pharse == "train":
            optimizer.zero_grad()
        # age
        age_img = age_img.cuda()
        age_label = age_label.cuda()
        age_pred_100_classes, age_pred_20_classes, age_pred_10_classes, age_pred_5_classes = model(age_img)

        if args.age_classification_combination == [1, 0, 0, 0]:
            age_loss_cls_100_classes = age_cls_criterion_encapsulation(age_cls_criterion, age_pred_100_classes, age_label, "100_classes")
        elif args.age_classification_combination == [1, 1, 0, 0]:
            # print("age classification combination list [1, 1, 0, 0]")
            age_loss_cls_100_classes = (age_cls_criterion, age_pred_100_classes, age_label, "100_classes")
            age_loss_cls_20_classes = age_cls_criterion_encapsulation(age_cls_criterion, age_pred_20_classes, age_label, "20_classes")
        elif args.age_classification_combination == [1, 1, 1, 0]:
            age_loss_cls_100_classes = age_cls_criterion_encapsulation(age_cls_criterion, age_pred_100_classes, age_label, "100_classes")
            age_loss_cls_20_classes = age_cls_criterion_encapsulation(age_cls_criterion, age_pred_20_classes, age_label, "20_classes")
            age_loss_cls_10_classes = age_cls_criterion_encapsulation(age_cls_criterion, age_pred_10_classes, age_label, "10_classes")
        elif args.age_classification_combination == [1, 1, 1, 1]:
            age_loss_cls_100_classes = age_cls_criterion_encapsulation(age_cls_criterion, age_pred_100_classes, age_label, "100_classes")
            age_loss_cls_20_classes = age_cls_criterion_encapsulation(age_cls_criterion, age_pred_20_classes, age_label, "20_classes")
            age_loss_cls_10_classes = age_cls_criterion_encapsulation(age_cls_criterion, age_pred_10_classes, age_label, "10_classes")
            age_loss_cls_5_classes = age_cls_criterion_encapsulation(age_cls_criterion, age_pred_5_classes, age_label, "5_classes")
        else:
            print("age_divide_100_classes, age_divide_20_classes, age_divide_10_classes, age_divide_5_classes")
            ValueError
        # age l1 regrssion to calculate the age MAE value
        age_l1_calculation = nn.L1Loss()
        age_loss_rgs_l1 = age_mae_criterion_encapsulation(age_l1_calculation, age_pred_100_classes, age_label)
        age_epoch_mae.update(age_loss_rgs_l1.item(), 1)
        if args.age_classification_combination == [1, 0, 0, 0]:
            age_loss_cls = age_loss_cls_100_classes
        elif args.age_classification_combination == [1, 1, 0, 0]:
            age_loss_cls = age_loss_cls_100_classes + age_loss_cls_20_classes
        
        elif  args.age_classification_combination == [1, 1, 1, 0]:
            age_loss_cls = age_loss_cls_100_classes + age_loss_cls_20_classes + age_loss_cls_10_classes  
        
        elif  args.age_classification_combination == [1, 1, 1, 1]:
            age_loss_cls = age_loss_cls_100_classes + age_loss_cls_20_classes + age_loss_cls_10_classes + age_loss_cls_5_classes
        else:
            raise ValueError
        age_cls_epoch_loss.update(age_loss_cls.item(), 1)
        age_l1_rgs_epoch_loss.update(age_loss_rgs_l1.item(), 1)
        loss = age_loss_cls
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
    losses = [total_loss.avg, age_cls_epoch_loss.avg, age_l1_rgs_epoch_loss.avg]
    LOG("[" + pharse +"] [Loss  ], [total, cls, l1 ]: " + str(losses), logFile)
    LOG("[" + pharse +"] , MAE                  : " + str(age_epoch_mae.avg), logFile)
    try:
        lr = float(str(optimizer).split("\n")[3].split(" ")[-1])
    except:
        lr = 100
    LOG("lr: " + str(lr), logFile)
    
    return losses, lr, model