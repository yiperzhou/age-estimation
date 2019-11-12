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
    state_dict = checkpoint["state_dict"]

    initial_model.load_state_dict(state_dict, strict=True)
    # for k, v in initial_model.parameters():
    #     print("k, v", k)

    return initial_model





