import torch
import torch.nn as nn

from utils.helper_2 import LOG
from utils.helper_3 import AverageMeter

CLASSES_NUM_IS_100 = "100_classes"
CLASSES_NUM_IS_20 = "20_classes"
CLASSES_NUM_IS_10 = "10_classes"
CLASSES_NUM_IS_5 = "5_classes"


def age_mae_criterion_encapsulation(age_criterion, age_out_cls, age_label):
    _, pred_ages = torch.max(age_out_cls, 1)
    pred_ages = pred_ages.type(torch.cuda.FloatTensor)
    age_label = age_label.type(torch.cuda.FloatTensor)
    age_loss_mae = age_criterion(pred_ages, age_label)
    return age_loss_mae


def age_mapping_function(origin_value, age_divide):
    origin_value = origin_value.cpu()
    y_true_rgs = torch.div(origin_value, age_divide)
    return y_true_rgs


def age_rgs_criterion_encapsulation(age_criterion, pred_ages_rgs, age_label, classification_type):
    if classification_type == CLASSES_NUM_IS_100:
        age_divide = 1
    elif classification_type == CLASSES_NUM_IS_20:
        age_divide = 5
    elif classification_type == CLASSES_NUM_IS_10:
        age_divide = 10
    elif classification_type == CLASSES_NUM_IS_5:
        age_divide = 20
    else:
        print("ground truth age classes type value is wrong")
        raise ValueError
    mapped_age_label = age_mapping_function(age_label, age_divide)
    mapped_age_label = mapped_age_label.type(torch.cuda.FloatTensor)
    age_rgs_loss = age_criterion(pred_ages_rgs, mapped_age_label)
    return age_rgs_loss


def age_cls_criterion_encapsulation(age_criterion, age_out_cls, age_label, classification_type):
    if classification_type == CLASSES_NUM_IS_100:
        age_divide = 1
    elif classification_type == CLASSES_NUM_IS_20:
        age_divide = 5
    elif classification_type == CLASSES_NUM_IS_10:
        age_divide = 10
    elif classification_type == CLASSES_NUM_IS_5:
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
    age_mae_rgs_epoch_loss = AverageMeter()
    age_epoch_loss = AverageMeter()
    age_epoch_mae = AverageMeter()
    total_loss_scalar = AverageMeter()
    if pharse == "train":
        model.train()
    elif pharse == "valid":
        model.eval()
    else:
        raise NotImplementedError
    torch.cuda.empty_cache()
    age_cls_criterion = criterion[0]
    age_mse_criterion = criterion[1]

    age_loader = loader[0]
    for j_batch_idx, (age_img, age_label) in enumerate(age_loader):
        if pharse == "train":
            optimizer.zero_grad()
        # age
        age_img = age_img.cuda()
        age_label = age_label.cuda().type(torch.cuda.FloatTensor)

        # # [age_pred_100_classes, age_pred_20_classes, age_pred_10_classes, age_pred_5_classes], [age_pred_rgs_100_classes, age_pred_rgs_20_classes] = model(age_img)
        # temp1, temp2 = model(age_img)
        # print(temp1)
        # print()
        # print(temp2)

        [age_pred_100_classes, age_pred_20_classes, age_pred_10_classes, age_pred_5_classes], [age_pred_rgs_100_classes,
                                                                                               age_pred_rgs_20_classes,
                                                                                               age_pred_rgs_10_classes] = model(
            age_img)

        if args.age_classification_combination == [1, 1, 1, 1]:
            age_100_classes_CE_loss = age_cls_criterion_encapsulation(age_cls_criterion, age_pred_100_classes,
                                                                      age_label, CLASSES_NUM_IS_100)
            age_20_classes_CE_loss = age_cls_criterion_encapsulation(age_cls_criterion, age_pred_20_classes, age_label,
                                                                     CLASSES_NUM_IS_20)
            age_10_classes_CE_loss = age_cls_criterion_encapsulation(age_cls_criterion, age_pred_10_classes, age_label,
                                                                     CLASSES_NUM_IS_10)
            age_5_classes_CE_loss = age_cls_criterion_encapsulation(age_cls_criterion, age_pred_5_classes, age_label,
                                                                    CLASSES_NUM_IS_5)

            age_loss_cls = torch.add(
                torch.add(torch.add(age_100_classes_CE_loss, age_20_classes_CE_loss), age_10_classes_CE_loss),
                age_5_classes_CE_loss)
            # print("age loss cls 100 classes: ", age_100_classes_CE_loss)
            # print("age_5_classes_CE_loss: ", age_5_classes_CE_loss)
            # print("age ")

        else:
            print("validate the regression branch function based on the four classification branches")
            raise ValueError

        # print("age_pred_rgs_100_classes.size(): ", age_pred_rgs_100_classes.size())
        # print("age_label.size()   : ", age_label.size())
        # age mse regrssion loss

        # print("age_pred_rgs_100_classes: ", age_pred_rgs_100_classes)

        # print("age pred rgs: ", age_pred_rgs_100_classes)
        # print("age label: ", age_label)

        age_loss_rgs_100_class_mse = age_mse_criterion(age_pred_rgs_100_classes, age_label)
        age_loss_rgs_20_class_mse = age_rgs_criterion_encapsulation(age_mse_criterion, age_pred_rgs_20_classes,
                                                                    age_label, CLASSES_NUM_IS_20)
        age_loss_rgs_10_class_mse = age_rgs_criterion_encapsulation(age_mse_criterion, age_pred_rgs_10_classes,
                                                                    age_label, CLASSES_NUM_IS_10)
        age_loss_rgs_mse = age_loss_rgs_100_class_mse + age_loss_rgs_20_class_mse + age_loss_rgs_10_class_mse

        # print("age mean squared error: ", age_loss_rgs_mse)
        # print("four cross entropy losses: ", age_loss_cls)

        total_loss = age_loss_cls + age_loss_rgs_mse

        total_loss_scalar.update(total_loss.item(), 1)

        age_cls_epoch_loss.update(age_loss_cls.item(), 1)
        age_mae_rgs_epoch_loss.update(age_loss_rgs_mse.item(), 1)

        # print("age_loss_cls    : ", age_loss_cls.item())
        # print("age loss rgs mse: ", age_loss_rgs_mse.item())

        # age l1 regrssion to calculate the age MAE value
        age_mae = age_mae_criterion_encapsulation(nn.L1Loss(), age_pred_100_classes, age_label)
        age_epoch_mae.update(age_mae.item(), 1)

        if pharse == "train":
            total_loss.backward()
            optimizer.step()
        elif pharse == "valid":
            continue
        else:
            print("pharse should be in [train, valid]")
            raise NotImplementedError

    losses = [total_loss_scalar.avg, age_cls_epoch_loss.avg, age_mae_rgs_epoch_loss.avg]
    LOG("[" + pharse + "] [Loss  ], [total, cls, mse ]: " + str(losses), logFile)
    LOG(" 3 age regression, 100 20  and 10 age classes: ", logFile)
    LOG("[" + pharse + "] , MAE                  : " + str(age_epoch_mae.avg), logFile)
    try:
        lr = float(str(optimizer).split("\n")[3].split(" ")[-1])
    except:
        lr = 100
    LOG("lr: " + str(lr), logFile)

    return losses, lr, model