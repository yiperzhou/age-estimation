from models.classification_regression_resnet50 import RegressionAndClassificationResNet50
from models.classification_regression_vgg16_bn import RegressionAndClassificationVGG16bn
from models.classification_regression_densenet121 import RegressionAndClassificationDenseNet121

regression_and_classification_vgg16bn_model = "RegressionAndClassificationVGG16bn"
regression_and_classification_resnet50_model = "RegressionAndClassificationResNet50"
regression_and_classification_densenet121_model = "RegressionAndClassificationDenseNet121"

def get_model(args, logFile):

    if args.model == regression_and_classification_resnet50_model:
        model = RegressionAndClassificationResNet50(args)

    elif args.model == regression_and_classification_vgg16bn_model:
        model = RegressionAndClassificationVGG16bn(args)

    elif args.model == regression_and_classification_densenet121_model:
        model = RegressionAndClassificationDenseNet121(args)

    else:
        NotImplementedError

    return model

