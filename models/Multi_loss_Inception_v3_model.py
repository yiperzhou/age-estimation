import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

# from helper import LOG

class Multi_loss_InceptionV3(nn.Module):
    def __init__(self, args):
        super(Multi_loss_InceptionV3, self).__init__()
        self.inceptionv3 = self.get_Inception_V3()
        
        self.features_length = 9216
        self.args = args

        self.age_divide_100_classes, self.age_divide_20_classes, self.age_divide_10_classes, self.age_divide_5_classes = self.get_age_cls_class()

        self.age_clf_100_classes = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 100)            
        )

        self.age_clf_20_classes = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 20)            
        )

        self.age_clf_10_classes = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 10)            
        )        

        self.age_clf_5_classes = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 5)                        
        )

    def get_Inception_V3(self):
        
        # Inception_v3_feature = nn.Sequential(
        #     models.Inception3.Conv2d_1a_3x3,
        #     models.Inception3.Conv2d_2a_3x3,
        #     models.Inception3.Conv2d_2b_3x3,
        #     models.Inception3.Conv2d_3b_1x1,
        #     models.Inception3.Conv2d_4a_3x3,
        #     models.Inception3.Mixed_5b,
        #     models.Inception3.Mixed_5c,
        #     models.Inception3.Mixed_5d,
        #     models.Inception3.Mixed_6a,
        #     models.Inception3.Mixed_5c,
        # )


        removed = list(models.inception_v3.children())[:-1]
        Inception_v3_feature = nn.Sequential(*self.removed)

        return Inception_v3_feature


    def get_age_cls_class(self):

        age_divide_100_classes = False
        age_divide_20_classes = False
        age_divide_10_classes = False
        age_divide_5_classes = False               

        if self.args.age_classification_combination == [1, 0, 0, 0]:
            age_divide_100_classes = True
            age_divide_20_classes = False
            age_divide_10_classes = False
            age_divide_5_classes = False       
                
        elif self.args.age_classification_combination == [1, 1, 0, 0]:
            age_divide_100_classes = True
            age_divide_20_classes = True
            age_divide_10_classes = False
            age_divide_5_classes = False        
            
        elif self.args.age_classification_combination == [1, 1, 1, 0]:
            age_divide_100_classes = True
            age_divide_20_classes = True
            age_divide_10_classes = True
            age_divide_5_classes = False                    

        elif self.args.age_classification_combination == [1, 1, 1, 1]:
            age_divide_100_classes = True
            age_divide_20_classes = True
            age_divide_10_classes = True
            age_divide_5_classes = True
                            
        else:
            print("age_divide_100_classes, age_divide_20_classes, age_divide_10_classes, age_divide_5_classes")
            ValueError

        return age_divide_100_classes, age_divide_20_classes, age_divide_10_classes, age_divide_5_classes

    
    def forward(self, x):
        x = self.inceptionv3(x)

        x = x.view(x.size(0), -1)

        age_pred_100_classes, age_pred_20_classes, age_pred_10_classes, age_pred_5_classes = None, None, None, None

        if self.age_divide_100_classes == True:
            age_pred_100_classes = self.age_clf_100_classes(x)

        if self.age_divide_20_classes == True:
            age_pred_20_classes = self.age_clf_20_classes(x)

        if self.age_divide_10_classes == True:
            age_pred_10_classes = self.age_clf_10_classes(x)

        if self.age_divide_5_classes == True:
            age_pred_5_classes = self.age_clf_5_classes(x)

        return age_pred_100_classes, age_pred_20_classes, age_pred_10_classes, age_pred_5_classes




# if __name__ == "__main__":
#     print("test model")
#     model = Multi_loss_InceptionV3