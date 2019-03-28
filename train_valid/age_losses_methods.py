# test different age estimation loss function.

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