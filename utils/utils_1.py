# sys.path.append("/models/")

from models.multi_classification_alexnet import MultiClassificationAlexNet
from models.multi_classification_mobilenet_v1 import MultiClassificationMobileNetv1
from models.multi_classification_resnet50 import MultiClassificationResNet50
from models.multi_classification_vgg16_bn import MultiClassificationVGG16bn
from models.multi_classification_densenet121 import MultiClassificationDenseNet121
from models.multi_classification_resnet18 import MultiClassificationResNet18
from models.multi_classification_inception_v3 import MultiClassificationInceptionv3

Multi_Classification_AlexNet_model      = "MultiClassificationAlexNet"
Multi_Classification_MobileNet_V1_model = "MultiClassificationMobileNetv1"
Multi_Classification_VGG16_bn_model = "MultiClassificationVGG16bn"
Multi_Classification_DenseNet_121_model = "MultiClassificationDenseNet121"
Multi_Classification_InceptionV3_model = "MultiClassificationInceptionv3"
Multi_Classification_ResNet50_model = "MultiClassificationResNet50"


def get_model(args, logFile):

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

    else:
        NotImplementedError

    return model

