import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.nn.functional as F


class Multi_Regression_AlexNet_Model(nn.Module):
    def __init__(self, args, logFile, age_classes = 100):
        super(Multi_Regression_AlexNet_Model, self).__init__()
        self.AlexNet_features = models.alexnet(pretrained=True).features
        
        self.features_length = 9216
        self.args = args
        self.age_divide = self.get_age_rgs_number_class()

        self.age_rgs_clf = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, self.age_divide)
        )        

    def get_age_rgs_number_class(self):
        if self.args.l1_regression_loss == "0-100_age_rgs":
            age_divide = 100

        elif self.args.l1_regression_loss == "0-20_age_rgs":
            age_divide = 5
            # print("20_age_cls_loss")

        elif self.args.l1_regression_loss == "0-10_age_rgs":
            age_divide = 10
            # print("10_age_cls_loss")

        elif self.args.l1_regression_loss == "0-5_age_rgs":
            age_divide = 20
            # print("5_age_cls_loss")

        else:
            print("0-100_age_rgs, 0-20_age_rgs, 0-10_age_rgs, 0-5_age_rgs")
            ValueError

        return age_divide
    
    def forward(self, x):
        x = self.AlexNet_features(x)
        x = x.view(x.size(0), -1)

        age_pred_rgs = self.age_rgs_clf(x)

        return age_pred_rgs 



if __name__ == "__main__":
    # alexModel = Multi_Regression_AlexNet_Model()

    # # print_model_dimension_summary(alexModel)
    # print("AlexModel: ")
    # print(alexModel)

    print("done")

