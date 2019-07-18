import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.nn.functional as F


class Multi_loss_AlexNet_Model(nn.Module):
    def __init__(self, args, logFile, age_classes = 100):
        super(Multi_loss_AlexNet_Model, self).__init__()
        self.MTL_AlexNet_features = models.alexnet(pretrained=True).features
        
        self.features_length = 9216
        self.args = args
        # self.age_divide = self.get_age_rgs_number_class()

        self.age_clf = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, age_classes)
        )
        # self.age_rgs_clf = nn.Sequential(
        #     nn.Linear(self.features_length, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(256, self.age_divide)
        # )

    # def get_age_rgs_number_class(self):
    #
    #     if self.args.l1_regression_loss == "0_20_age_rgs":
    #         age_divide = 5
    #         # print("20_age_cls_loss")
    #
    #     if self.args.l1_regression_loss == "0_10_age_rgs":
    #         age_divide = 10
    #         # print("10_age_cls_loss")
    #     elif self.args.l1_regression_loss == "0_5_age_rgs":
    #         age_divide = 20
    #         # print("5_age_cls_loss")
    #     else:
    #         print("0_20_age_rgs, 0_10_age_rgs, 0_5_age_rgs")
    #         ValueError
    #
    #     return age_divide
    #
    # def get_age_cls_class(self):
    #
    #     if self.args.classification_loss == "100_classes":
    #         age_divide = 1
    #     if self.args.classification_loss == "20_classes":
    #         age_divide = 5
    #         # print("20_age_cls_loss")
    #     elif self.args.classification_loss == "10_classes":
    #         age_divide = 10
    #         # print("10_age_cls_loss")
    #     elif self.args.classification_loss == "5_classes":
    #         age_divide = 20
    #         # print("5_age_cls_loss")
    #     else:
    #         print("100_age_cls_loss, 20_age_cls_loss, 10_age_cls_loss, 5_age_cls_loss")
    #         ValueError
    #
    #     return age_divide

    
    def forward(self, x):
        x = self.MTL_AlexNet_features(x)
        x = x.view(x.size(0), -1)

        age_pred  = self.age_clf(x)

        return age_pred



if __name__ == "__main__":
    # alexModel = MTL_AlexNet_model()

    # # print_model_dimension_summary(alexModel)
    # print("AlexModel: ")
    # print(alexModel)

    print("done")

