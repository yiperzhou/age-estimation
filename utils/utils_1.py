# sys.path.append("/models/")

<<<<<<< HEAD
from models.regression_resnet50 import RegressionAndClassificationResNet50
from models.regression_vgg16_bn import RegressionAndClassificationVGG16bn

regression_and_classification_vgg16bn_model = "RegressionAndClassificationVGG16bn"
regression_and_classification_resnet50_model = "RegressionAndClassificationResNet50"


def get_model(args, logFile):

<<<<<<< HEAD
    if args.model == regression_and_classification_resnet50_model:
        model = RegressionAndClassificationResNet50(args)

    elif args.model == regression_and_classification_vgg16bn_model:
        model = RegressionAndClassificationVGG16bn(args)
=======
    if args.model == Multi_Classification_AlexNet_model:
        model = MultiClassificationAlexNet(args)

    elif args.model == Multi_Classification_MobileNet_V1_model:
        model = MultiClassificationMobileNetv1(args)

    elif args.model == Multi_Classification_ResNet50_model:
        model = MultiClassificationResNet50(args)

    elif args.model == Multi_Classification_VGG16_bn_model:
        model = MultiClassificationVGG16bn(args)

    elif args.model == Multi_Classification_DenseNet_121_model:
        model = MultiClassificationDenseNet121(args)
        
    elif args.model == Multi_Classification_InceptionV3_model:
        model = MultiClassificationInceptionv3(args)
>>>>>>> remotes/origin/classification_combination

    else:
        NotImplementedError

    return model

