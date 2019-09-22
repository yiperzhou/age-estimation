import torch
import torch.nn as nn

# test different age estimation loss function.

from utils.helper import indexes_to_one_hot




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


def elastic_neural_network_loss():
    '''
    use Yue Bai's paper -- , loss function in "elastic neural network for age estiamtion"
    '''
    loss = 0

    return loss

def Age_Gender_Pred_repo_loss():
    '''
    use the Age-Gender-Pred github repository loss to train the model
    '''

def age_crossentropy_loss(age_out_1, age_label):
    loss = 0

    return loss


def age_regression_classification_loss(age_out, age_label):

    loss = 0

    cls_loss = 0
    rgs_loss = 0

    loss = cls_loss + rgs_loss

    return loss


class Age_rgs_loss(nn.Module):
    '''
    age regression loss, reference, the github repository, Age-Gender-Pred, repository
    '''
    def __init__(self):
        super().__init__()
        self.age_divde = torch.tensor(10)
        
        
    def forward(self, y_pred, y_true, age_rgs_criterion):

        y_true_rgs = torch.div(y_true, self.age_divde)
        y_true_rgs = torch.ceil(y_true_rgs)

        # _, y_pred = torch.max(y_pred, 1)
        age_loss_rgs = age_rgs_criterion(y_pred, y_true_rgs)

        return age_loss_rgs
