import os
import sys

# # from os.path import dirname
# sys.path.append("/models/")

# print(dirname(__file__))
from models.Multi_Classification_AlexNet_model import Multi_Classification_AlexNet_Model

from models.Multi_Classification_MobileNet_V1_model import Multi_Classification_MobileNet_V1_Model
from models.Multi_Classification_ResNet_50_model import Multi_Classification_ResNet_50_model
from models.Multi_Classification_VGG16_bn_model import Multi_Classification_VGG16_bn_model
from models.Multi_Classification_DenseNet_121_model import Multi_Classification_DenseNet_121_model
from models.Multi_Classification_ResNet_18_model import Multi_Classification_ResNet_18_model 
# from models.Multi_Classification_Inception_v3_model import Multi_Classification_InceptionV3 

from .helper_2 import LOG

Multi_Classification_ResNet_18_model_name    = "Multi_Classification_ResNet_18_model"
Multi_Classification_ResNet_50_model_name    = "Multi_Classification_ResNet_50"
Multi_Classification_AlexNet_model_name      = "Multi_Classification_AlexNet"
Multi_Classification_MobileNet_V1_model_name = "Multi_Classification_MobileNet_V1"

Multi_Classification_VGG16_bn_model_name = "Multi_Classification_VGG16_bn_model"
Multi_Classification_DenseNet_121_model_name = "Multi_Classification_DenseNet_121_model"
Multi_Classification_InceptionV3_model_name = "Multi_Classification_InceptionV3"

def get_pretrained_model_weights_path(args):

    pretrained_model_weight_path = ""

    if args.model == Multi_Classification_AlexNet_model_name:
        if args.working_machine == "thinkstation": # thinkstation path
            pretrained_model_weight_path = "/home/yi/Narvi_yi_home/projects/MultitaskLearningFace/results/pretrained_MTL_AlexNet_model/_gender_1_age_1_CVPR_16_IMDB_WIKI/2019-04-08-21-29-56/save_models/model_best.pth.tar"

        elif args.working_machine == "Narvi": # Narvi
            pretrained_model_weight_path = "/home/zhouy/projects/MultitaskLearningFace/results/pretrained_MTL_AlexNet_model/_gender_1_age_1_CVPR_16_IMDB_WIKI/2019-04-08-21-29-56/save_models/model_best.pth.tar"

        else:
            print("working machine should be  [thinkstation, Narvi]")
            NotImplementedError

    elif args.model == Multi_Classification_MobileNet_V1_model_name:
        if args.working_machine == "thinkstation": # thinkstation path
            pretrained_model_weight_path = "results/pretrained_MTL_MobileNet_V1_model/_gender_1_age_1_IMDB_WIKI/2019-04-21-11-32-30/save_models/model_best.pth.tar"

        elif args.working_machine == "Narvi": # Narvi
            pretrained_model_weight_path = "/home/zhouy/projects/MultitaskLearningFace/results/pretrained_MTL_MobileNet_V1_model/_gender_1_age_1_IMDB_WIKI/2019-04-21-11-32-30/save_models/model_best.pth.tar"

        else:
            print("working machine should be  [thinkstation, Narvi]")
            NotImplementedError            

    elif args.model == Multi_Classification_ResNet_50_model_name:
        if args.working_machine == "thinkstation": # thinkstation path
            pretrained_model_weight_path = "/home/yi/Narvi_yi_home/projects/MultitaskLearningFace/results/pretrained_MTL_ResNet_50_model/_gender_0_age_1_IMDB_WIKI/2019-04-13-21-03-03/save_models/model_best.pth.tar"

        elif args.working_machine == "Narvi": # Narvi
            pretrained_model_weight_path = ""

        else:
            print("working machine should be  [thinkstation, Narvi]")
            NotImplementedError

    elif args.model == Multi_Classification_VGG16_bn_model_name:
        if args.working_machine == "thinkstation": # thinkstation path
            pretrained_model_weight_path = ""

        elif args.working_machine == "Narvi": # Narvi
            pretrained_model_weight_path = ""

        else:
            print("working machine should be  [thinkstation, Narvi]")
            NotImplementedError
            
    elif args.model == Multi_Classification_DenseNet_121_model_name:
        if args.working_machine == "thinkstation": # thinkstation path
            pretrained_model_weight_path = ""

        elif args.working_machine == "Narvi": # Narvi
            pretrained_model_weight_path = ""

        else:
            print("working machine should be  [thinkstation, Narvi]")
            NotImplementedError

    elif args.model == Multi_Classification_ResNet_18_model:
        # thinkstation path
        if args.working_machine == "thinkstation":
            pretrained_model_weight_path = "/media/yi/harddrive/codes/MultitaskLearningFace/results/2019-03-24-12-56-49--Age_Gender--IMDB_WIKI/save_models/model_best.pth.tar"

        elif args.working_machine == "Narvi":
            pretrained_model_weight_path = "/home/zhouy/projects/MultitaskLearningFace/results/pretrained_MTL_ResNet_18_model/2019-03-24-12-56-49--Age_Gender--IMDB_WIKI/save_models/model_best.pth.tar"

        else:
            print("working machine should be  [thinkstation, Narvi]")
            NotImplementedError    

    elif args.model == Multi_Classification_InceptionV3_model_name:
        # thinkstation path
        if args.working_machine == "thinkstation":
            pretrained_model_weight_path = "/media/yi/harddrive/codes/MultitaskLearningFace/results/2019-03-24-12-56-49--Age_Gender--IMDB_WIKI/save_models/model_best.pth.tar"

        elif args.working_machine == "Narvi":
            pretrained_model_weight_path = "/home/zhouy/projects/MultitaskLearningFace/results/pretrained_MTL_ResNet_18_model/2019-03-24-12-56-49--Age_Gender--IMDB_WIKI/save_models/model_best.pth.tar"

        else:
            print("working machine should be  [thinkstation, Narvi]")
            NotImplementedError                    

    else:
        NotImplementedError

    return pretrained_model_weight_path




def get_model(args, logFile):

    if args.model == Multi_Classification_AlexNet_model_name:
        model = Multi_Classification_AlexNet_Model(args, logFile)

    elif args.model == Multi_Classification_MobileNet_V1_model_name:
        model = Multi_Classification_MobileNet_V1_Model(args)

    elif args.model == Multi_Classification_ResNet_50_model_name:
        model = Multi_Classification_ResNet_50_model(args)

    elif args.model == Multi_Classification_VGG16_bn_model_name:
        model = Multi_Classification_VGG16_bn_model(args)

    elif args.model == Multi_Classification_DenseNet_121_model_name:
        model = Multi_Classification_DenseNet_121_model(args)
        
    # elif args.model == Multi_Classification_InceptionV3_model_name:
    #     model = Multi_Classification_InceptionV3(args)

    else:
        NotImplementedError

    return model

