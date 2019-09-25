import os
import sys

# # from os.path import dirname
# sys.path.append("/models/")

from models.Multi_Classification_AlexNet_model import Multi_Classification_AlexNet_model

# from models.Multi_Classification_MobileNet_V1_model import Multi_Classification_MobileNet_V1_Model
# from models.Multi_Classification_ResNet_50_model import Multi_Classification_ResNet_50_model
# from models.Multi_Classification_VGG16_bn_model import Multi_Classification_VGG16_bn_model
# from models.Multi_Classification_DenseNet_121_model import Multi_Classification_DenseNet_121_model
# from models.Multi_Classification_ResNet_18_model import Multi_Classification_ResNet_18_model 
# from models.Multi_Classification_Inception_v3_model import Multi_Classification_InceptionV3 

Multi_Classification_AlexNet_model_name      = "Multi_Classification_AlexNet"
# Multi_Classification_ResNet_18_model_name    = "Multi_Classification_ResNet_18_model"
# Multi_Classification_ResNet_50_model_name    = "Multi_Classification_ResNet_50"

# Multi_Classification_MobileNet_V1_model_name = "Multi_Classification_MobileNet_V1"

# Multi_Classification_VGG16_bn_model_name = "Multi_Classification_VGG16_bn_model"
# Multi_Classification_DenseNet_121_model_name = "Multi_Classification_DenseNet_121_model"
# Multi_Classification_InceptionV3_model_name = "Multi_Classification_InceptionV3"


def get_model(args, logFile):

    if args.model == Multi_Classification_AlexNet_model_name:
        model = Multi_Classification_AlexNet_model(args, logFile)

    # elif args.model == Multi_Classification_MobileNet_V1_model_name:
    #     model = Multi_Classification_MobileNet_V1_Model(args)

    # elif args.model == Multi_Classification_ResNet_50_model_name:
    #     model = Multi_Classification_ResNet_50_model(args)

    # elif args.model == Multi_Classification_VGG16_bn_model_name:
    #     model = Multi_Classification_VGG16_bn_model(args)

    # elif args.model == Multi_Classification_DenseNet_121_model_name:
    #     model = Multi_Classification_DenseNet_121_model(args)
        
    # elif args.model == Multi_Classification_InceptionV3_model_name:
    #     model = Multi_Classification_InceptionV3(args)

    else:
        NotImplementedError

    return model

