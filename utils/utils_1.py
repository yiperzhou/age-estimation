import os
import sys

# # from os.path import dirname
# sys.path.append("/models/")

# print(dirname(__file__))
from models.MTL_AlexNet_model import MTL_AlexNet_model
from models.MTL_DenseNet_121_model import MTL_DenseNet_121_model
# from models.Elastic_MTL_DenseNet_121_model import Elastic_MTL_DenseNet_121_model
from models.MTL_ResNet_50_model import MTL_ResNet_50_model
from models.MTL_ResNet_18_model import MTL_ResNet_18_model
from models.agegenpredmodel import AgeGenPredModel
from models.MTL_VGG11_bn_model import MTL_VGG11_bn_model

from .helper_2 import LOG

def get_pretrained_model_weights_path(args):

    pretrained_model_weight_path = ""

    #load pretrained model 
    if args.model == "MTL_ResNet_18":
        # thinkstation path
        if args.working_machine == "thinkstation":
            pretrained_model_weight_path = "/media/yi/harddrive/codes/MultitaskLearningFace/results/2019-03-24-12-56-49--Age_Gender--IMDB_WIKI/save_models/model_best.pth.tar"

        elif args.working_machine == "narvi":
            pretrained_model_weight_path = ""

        else:
            print("working machine should be  [thinkstation, narvi]")
            NotImplementedError

    elif args.model == "res18_cls70":
        # thinkstation path
        if args.working_machine == "thinkstation":
            pretrained_model_weight_path = ""

        elif args.working_machine == "narvi":
            pretrained_model_weight_path = ""

        else:
            print("working machine should be  [thinkstation, narvi]")
            NotImplementedError
    
    elif args.model == "MTL_ResNet_50":
        if args.working_machine == "thinkstation": # thinkstation path
            pretrained_model_weight_path = ""

        elif args.working_machine == "narvi": # narvi
            pretrained_model_weight_path = ""

        else:
            print("working machine should be  [thinkstation, narvi]")
            NotImplementedError

    elif args.model == "MTL_DenseNet_121_model":
        if args.working_machine == "thinkstation": # thinkstation path
            pretrained_model_weight_path = ""

        elif args.working_machine == "narvi": # narvi
            pretrained_model_weight_path = ""

        else:
            print("working machine should be  [thinkstation, narvi]")
            NotImplementedError

    elif args.model == "MTL_AlexNet_model":
        if args.working_machine == "thinkstation": # thinkstation path
            pretrained_model_weight_path = "/home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/pretrained_MTL_AlexNet_model/_gender_1_age_1_CVPR_16_IMDB_WIKI/2019-04-08-21-29-56/save_models/model_best.pth.tar"

        elif args.working_machine == "narvi": # narvi
            pretrained_model_weight_path = "/home/zhouy/projects/MultitaskLearningFace/results/pretrained_MTL_AlexNet_model/_gender_1_age_1_CVPR_16_IMDB_WIKI/2019-04-08-21-29-56/save_models/model_best.pth.tar"

        else:
            print("working machine should be  [thinkstation, narvi]")
            NotImplementedError

    elif args.model == "MTL_VGG11_bn_model":
        if args.working_machine == "thinkstation": # thinkstation path
            pretrained_model_weight_path = ""

        elif args.working_machine == "narvi": # narvi
            pretrained_model_weight_path = ""

        else:
            print("working machine should be  [thinkstation, narvi]")
            NotImplementedError

    else:
        NotImplementedError

    return pretrained_model_weight_path




def get_model(args, logFile):

    if args.model == "MTL_ResNet_18_model":
        model = MTL_ResNet_18_model()

    elif args.model == "res18_cls70":
        model = AgeGenPredModel()
        # replace the last fully connected layer.
        model.age_cls_pred = nn.Linear(in_features = 512, out_features=100, bias=True)
        LOG("modify the age prediction range from [0, 60] -> [0, 99]", logFile)

    elif args.model == "MTL_ResNet_50_model":
        model = MTL_ResNet_50_model()

    elif args.model == "MTL_DenseNet_121_model":
        model = MTL_DenseNet_121_model()

    # elif args.model == "Elastic_MTL_DenseNet_121_model":
    #     model = Elastic_MTL_DenseNet_121_model(args, logFile)

    elif args.model == "MTL_AlexNet_model":
        model = MTL_AlexNet_model(args, logFile)

    elif args.model == "MTL_VGG11_bn_model":
        model = MTL_VGG11_bn_model()

    else:
        NotImplementedError

    return model

