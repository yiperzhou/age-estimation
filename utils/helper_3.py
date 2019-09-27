import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_age_loss(age_criterion, age_out, age_label):
    # age_label format is: [0,7,6,2] with batch_size = 4
    _, pred_age = torch.max(age_out, 1)
    pred_age = pred_age.view(-1).cpu().numpy()
    pred_ages = []
    for p in pred_age:
        pred_ages.append([p])

    pred_ages = torch.FloatTensor(pred_ages)
    pred_ages = pred_ages.cuda()


    age_label = age_label.cuda()
    age_label = np.reshape(age_label, (len(age_label),1))
    age_label = age_label.type(torch.cuda.FloatTensor)
    age_label = torch.autograd.Variable(age_label)

    # print("age out: ", pred_ages)
    # print("age label: ", age_label)

    age_loss = age_criterion(pred_ages, age_label)

    return age_loss



def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr