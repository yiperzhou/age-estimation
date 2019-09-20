import torch
import torch.nn as nn

# test different age estimation loss function.

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


class Euclidean_age_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.LAMDA = LAMDA
        self.SIGMOID = SIGMOID

        
        
    def forward(self, y_pred, y_true):

        _, y_pred = torch.max(y_pred, 1)

        y_true = y_true.type(torch.cuda.FloatTensor)
        y_pred = y_pred.type(torch.cuda.FloatTensor)

        temp_1 = y_pred - y_true

        loss1 = (1.0/2.0) * torch.pow(temp_1, 2)

        loss1 = torch.mean(loss1)

        return loss1


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