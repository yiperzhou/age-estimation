# sys.path.append("/models/")

from models.classification_regression_resnet50 import RegressionAndClassificationResNet50
from models.classification_regression_vgg16_bn import RegressionAndClassificationVGG16bn

regression_and_classification_vgg16bn_model = "RegressionAndClassificationVGG16bn"
regression_and_classification_resnet50_model = "RegressionAndClassificationResNet50"


def get_model(args, logFile):

    if args.model == regression_and_classification_resnet50_model:
        model = RegressionAndClassificationResNet50(args)

    elif args.model == regression_and_classification_vgg16bn_model:
        model = RegressionAndClassificationVGG16bn(args)

    else:
        NotImplementedError

    return model

