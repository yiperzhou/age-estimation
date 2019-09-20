import os
import sys

from models.Multi_Regression_AlexNet_model import Multi_Regression_AlexNet_Model
from models.Multi_Regression_MobileNet_V1 import Multi_Regression_MobileNet_V1_Model

from models.Multi_Regression_ResNet_50_model import Multi_Regression_ResNet_50_model
from models.Multi_Regression_VGG16_bn_model import Multi_Regression_VGG16_bn_model
from models.Multi_Regression_DenseNet_121 import Multi_Regression_DenseNet_121

from .helper_2 import LOG

Multi_Regression_ResNet_18_model_name    = "Multi_Regression_ResNet_18_model"
Multi_Regression_ResNet_50_model_name    = "Multi_Regression_ResNet_50"
Multi_Regression_AlexNet_model_name      = "Multi_Regression_AlexNet"
Multi_Regression_MobileNet_V1_model_name = "Multi_Regression_MobileNet_V1"

Multi_Regression_VGG16_bn_model_name = "Multi_Regression_VGG16_bn_model"
Multi_Regression_DenseNet_121_model_name = "Multi_Regression_DenseNet_121_model"
Multi_Regression_InceptionV3_model_name = "Multi_Regression_InceptionV3"

def get_pretrained_model_weights_path(args):

    pretrained_model_weight_path = ""

    if args.model == Multi_Regression_AlexNet_model_name:
        if args.working_machine == "thinkstation": # thinkstation path
            pretrained_model_weight_path = "/home/yi/Narvi_yi_home/projects/MultitaskLearningFace/results/pretrained_MTL_AlexNet_model/_gender_1_age_1_CVPR_16_IMDB_WIKI/2019-04-08-21-29-56/save_models/model_best.pth.tar"

        elif args.working_machine == "Narvi": # Narvi
            pretrained_model_weight_path = "/home/zhouy/projects/MultitaskLearningFace/results/pretrained_MTL_AlexNet_model/_gender_1_age_1_CVPR_16_IMDB_WIKI/2019-04-08-21-29-56/save_models/model_best.pth.tar"

        else:
            print("working machine should be  [thinkstation, Narvi]")
            NotImplementedError       

    elif args.model == Multi_Regression_MobileNet_V1_model_name:
        if args.working_machine == "thinkstation": # thinkstation path
            pretrained_model_weight_path = "results/pretrained_MTL_MobileNet_V1_model/_gender_1_age_1_IMDB_WIKI/2019-04-21-11-32-30/save_models/model_best.pth.tar"

        elif args.working_machine == "Narvi": # Narvi
            pretrained_model_weight_path = "/home/zhouy/projects/MultitaskLearningFace/results/pretrained_MTL_MobileNet_V1_model/_gender_1_age_1_IMDB_WIKI/2019-04-21-11-32-30/save_models/model_best.pth.tar"

        else:
            print("working machine should be  [thinkstation, Narvi]")
            NotImplementedError            

    elif args.model == Multi_Regression_ResNet_50_model_name:
        if args.working_machine == "thinkstation": # thinkstation path
            pretrained_model_weight_path = "/home/yi/Narvi_yi_home/projects/MultitaskLearningFace/results/pretrained_MTL_ResNet_50_model/_gender_0_age_1_IMDB_WIKI/2019-04-13-21-03-03/save_models/model_best.pth.tar"

        elif args.working_machine == "Narvi": # Narvi
            pretrained_model_weight_path = ""

        else:
            print("working machine should be  [thinkstation, Narvi]")
            NotImplementedError

    elif args.model == Multi_Regression_VGG16_bn_model_name:
        if args.working_machine == "thinkstation": # thinkstation path
            pretrained_model_weight_path = ""

        elif args.working_machine == "Narvi": # Narvi
            pretrained_model_weight_path = ""

        else:
            print("working machine should be  [thinkstation, Narvi]")
            NotImplementedError
            
    elif args.model == Multi_Regression_DenseNet_121_model_name:
        if args.working_machine == "thinkstation": # thinkstation path
            pretrained_model_weight_path = ""

        elif args.working_machine == "Narvi": # Narvi
            pretrained_model_weight_path = ""

        else:
            print("working machine should be  [thinkstation, Narvi]")
            NotImplementedError

    elif args.model == Multi_Regression_ResNet_18_model:
        # thinkstation path
        if args.working_machine == "thinkstation":
            pretrained_model_weight_path = "/media/yi/harddrive/codes/MultitaskLearningFace/results/2019-03-24-12-56-49--Age_Gender--IMDB_WIKI/save_models/model_best.pth.tar"

        elif args.working_machine == "Narvi":
            pretrained_model_weight_path = "/home/zhouy/projects/MultitaskLearningFace/results/pretrained_MTL_ResNet_18_model/2019-03-24-12-56-49--Age_Gender--IMDB_WIKI/save_models/model_best.pth.tar"

        else:
            print("working machine should be  [thinkstation, Narvi]")
            NotImplementedError    

    elif args.model == Multi_Regression_InceptionV3_model_name:
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

    if args.model == Multi_Regression_AlexNet_model_name:
        model = Multi_Regression_AlexNet_Model(args, logFile)

    elif args.model == Multi_Regression_MobileNet_V1_model_name:
        model = Multi_Regression_MobileNet_V1_Model(args)

    elif args.model == Multi_Regression_ResNet_50_model_name:
        model = Multi_Regression_ResNet_50_model(args)

    elif args.model == Multi_Regression_VGG16_bn_model_name:
        model = Multi_Regression_VGG16_bn_model(args)

    elif args.model == Multi_Regression_DenseNet_121_model_name:
        model = Multi_Regression_DenseNet_121_model(args)
        
    elif args.model == Multi_Regression_InceptionV3_model_name:
        model = Multi_Regression_InceptionV3(args)

    else:
        NotImplementedError

    return model

