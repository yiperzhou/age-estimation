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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    try:
        target = target.type(torch.cuda.LongTensor)
    
    except ValueError as identifier:
        try:
            target = target.type(torch.LongTensor)
        except ValueError as verr:
            pass

    # print("target: ", target)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



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

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        #print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)
    step_time = cur_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()
