import torch
from utils.helper import indexes_to_one_hot

def calculate_epsilon_loss(y_true,y_pred):
    mu = 0
    sigma = 0
    epsilon = 1 - torch.exp(- torch.pow((y_pred- mu), 2)/(2 * sigma^2))
    return epsilon

def pure_age_l1_loss():
    '''
    use l1_loss to calcualte age loss, but after many experiments, the result is not good.

    $M A E=\frac{1}{N} \sum_{n=1}^{N}\left|\hat{y}_{n}-y_{n}\right|$
    '''

    loss = 0

    return loss


def all_in_one_model_loss():
    '''
    use age loss applied in all-in-one model paper. Gaussian loss


    $L_{A}=(1-\lambda) \frac{1}{2}(y-a)^{2}+\lambda\left(1-\exp \left(-\frac{(y-a)^{2}}{2 \sigma^{2}}\right)\right)$

    '''
    loss = 0

    return loss


def apply_label_smoothing(y_true, epsilon=0.1):
    # https://github.com/keras-team/keras/pull/4723
    # https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/tools/loss.py#L6
    # https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf

    # convert the y_true from torch cuda variable to cpu variable.
    y_true = y_true.type(torch.IntTensor)

    y_true = indexes_to_one_hot(y_true, n_dims=100)

    # convert back from cpu variable to gpu variable
    y_true = y_true.type(torch.cuda.FloatTensor)

    y_true = (1 - epsilon) * y_true + epsilon / y_true.shape[1]

    # print("y_true:  ", )
    return y_true
