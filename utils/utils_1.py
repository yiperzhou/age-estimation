# sys.path.append("/models/")

from models.multi_classification_alexnet import MultiClassificationAlexNet

Multi_Classification_AlexNet_model      = "MultiClassificationAlexNet"


def get_model(args, logFile):

    if args.model == Multi_Classification_AlexNet_model:
        model = MultiClassificationAlexNet(args)

    else:
        NotImplementedError

    return model

