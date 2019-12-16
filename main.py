import datetime
import os
from sacred import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from data_load.cvpr_16_chalearn_dataloader import get_cvpr_age_data
from train_valid.train_valid import train_valid
from utils.helper import save_checkpoint
from utils.helper_2 import LOG
from utils.utils_1 import get_model
from parse_config import ConfigParser

ex = Experiment('age_estimation_classification_regression_combination')

ENTER_TRAIN_PHRASE = "...enter train phrase..."
ENTER_VALID_PHRASE = "...enter valid phrase..."
TRAIN = "train"
VALID = "valid"

global writer
global logFile

@ex.config
def cfg():
  C = 1.0
  gamma = 0.


def parse_args(args):
    folder_sub_name = "_" + args.model + "_" + "mse_regression_classification_loss"
    return folder_sub_name

@ex.automain
def main(**kwargs):
    global args
    for arg, v in kwargs.items():
        args.__setattr__(arg, v)

    # parse loss weight to sub folder name
    args.folder_sub_name = parse_args(args)

    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')

    path = "./results" + os.sep + args.model + "_" + args.dataset + \
           os.sep + args.folder_sub_name + os.sep + ts_str

    tensorboard_folder = os.path.join(path, "Graph")
    os.makedirs(path)
    print("path: ", path)
    logFile = path + os.sep + "log.txt"
    LOG(args, logFile)
    writer = SummaryWriter(tensorboard_folder)
    age_train_loader, age_test_loader = get_cvpr_age_data(args)
    model = get_model(args, logFile)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), momentum=0.9,
                          lr=args.lr_rate, weight_decay=args.weight_decay)

    age_cls_criterion = nn.CrossEntropyLoss()
    age_regression_criterion = nn.MSELoss()  # this is for age regression loss - mean squared error

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=1e-5, patience=10)
    LOG(model, logFile)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.cuda.empty_cache()
        model = model.cuda()

    # epochs_train_age_rgs_mae_loss, epochs_valid_age_rgs_mae_loss = [], []

    epochs_train_lr = []

    lowest_loss = 100000

    for epoch in range(0, args.epoch):

        print(ENTER_TRAIN_PHRASE)

        train_losses, lr, model = train_valid(model, [age_train_loader], [age_cls_criterion, age_regression_criterion],
                                              optimizer, epoch, logFile, args, TRAIN)

        # log_variables_to_board([epochs_train_age_rgs_l1_loss],
        #                         train_losses,
        #                         'train_age_mae_loss',
        #                         "Train", tensorboard_folder, epoch, log_file, writer)

        print(ENTER_VALID_PHRASE)
        val_losses, lr, model = train_valid(model, [age_test_loader], [age_cls_criterion, age_regression_criterion],
                                            optimizer, epoch, logFile, args, VALID)

        # log_variables_to_board([epochs_valid_age_rgs_l1_loss],
        #                         val_losses,
        #                         ['val_age_mae_loss'],
        #                         ['val_total_loss', 'val_age_cls_loss', 'val_age_l1_mae_loss'],
        #                         "Valid", tensorboard_folder, epoch, log_file, writer)

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
    config = ConfigParser.from_args(args, options)

    main()
