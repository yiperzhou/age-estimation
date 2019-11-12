import os
import torch


def save_checkpoint(state, savedir):
    
    model_dir = os.path.join(savedir, 'save_models')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    torch.save(state, best_filename)
    print("=> saved checkpoint '{}'".format(best_filename))

    return


def load_model_weights(initial_model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    initial_model.load_state_dict(state_dict, strict=True)
    # for k, v in initial_model.parameters():
    #     print("k, v", k)

    return initial_model


def indexes_to_one_hot(indexes, n_dims=None):
    """Converts a vector of indexes to a batch of one-hot vectors. """
    indexes = indexes.type(torch.int64).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(indexes)) + 1
    one_hots = torch.zeros(indexes.size()[0], n_dims).scatter_(1, indexes, 1)
    # one_hots = one_hots.view(*indexes.shape, -1)
    # print(one_hots)
    return one_hots



