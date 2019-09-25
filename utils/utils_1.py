import os
import sys

from models.Multi_Regression_AlexNet_model import Multi_Regression_AlexNet_Model
# from models.Multi_Regression_MobileNet_V1 import Multi_Regression_MobileNet_V1_Model

# from models.Multi_Regression_ResNet_50_model import Multi_Regression_ResNet_50_model
# from models.Multi_Regression_VGG16_bn_model import Multi_Regression_VGG16_bn_model
# from models.Multi_Regression_DenseNet_121 import Multi_Regression_DenseNet_121

from .helper_2 import LOG

# Multi_Regression_ResNet_18_model_name    = "Multi_Regression_ResNet_18_model"
# Multi_Regression_ResNet_50_model_name    = "Multi_Regression_ResNet_50"
Multi_Regression_AlexNet_model_name      = "Multi_Regression_AlexNet"
# Multi_Regression_MobileNet_V1_model_name = "Multi_Regression_MobileNet_V1"

# Multi_Regression_VGG16_bn_model_name = "Multi_Regression_VGG16_bn_model"
# Multi_Regression_DenseNet_121_model_name = "Multi_Regression_DenseNet_121_model"
# Multi_Regression_InceptionV3_model_name = "Multi_Regression_InceptionV3"



def get_model(args, logFile):

    if args.model == Multi_Regression_AlexNet_model_name:
        model = Multi_Regression_AlexNet_Model(args, logFile)

    # elif args.model == Multi_Regression_MobileNet_V1_model_name:
    #     model = Multi_Regression_MobileNet_V1_Model(args)

    # elif args.model == Multi_Regression_ResNet_50_model_name:
    #     model = Multi_Regression_ResNet_50_model(args)

    # elif args.model == Multi_Regression_VGG16_bn_model_name:
    #     model = Multi_Regression_VGG16_bn_model(args)

    # elif args.model == Multi_Regression_DenseNet_121_model_name:
    #     model = Multi_Regression_DenseNet_121_model(args)
        
    # elif args.model == Multi_Regression_InceptionV3_model_name:
    #     model = Multi_Regression_InceptionV3(args)

    else:
        model = None
        print("input correct model name")
        NotImplementedError

    return model

