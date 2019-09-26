import torch
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

def age_crossentropy_loss(age_out_1, age_label):
    loss = 0
    return loss