import torch
import torch.nn as nn

# test different age estimation loss function.

def calculate_epsilon_loss(y_true,y_pred):
    mu = 0
    sigma = 0

    epsilon = 1 - torch.exp(- torch.pow((y_pred- mu), 2)/(2 * sigma^2))

    return epsilon


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