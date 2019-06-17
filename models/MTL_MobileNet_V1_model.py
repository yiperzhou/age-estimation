from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch
import os

from models.Elastic_MobileNet_V1 import Elastic_MobileNet

# from helper import LOG

# not official model weights
model_urls = {
    # 'mobilenetV1': '/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/elastic/pytorch_code/models/mobilenet_sgd_68.848.pth.tar'
    'mobilenetV1': '/home/yi/narvi_yi_home/projects/MultitaskLearningFace/resources/mobilenet_sgd_68.848.pth.tar'
}



class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        # self.intermediate_CLF = []
        # self.add_intermediate_layers = add_intermediate_layers
        # self.num_categories = num_categories
        # self.num_outputs = num_outputs

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3,  32, 2),
            conv_dw(32,  64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )

        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):

        x = self.model(x)

        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class MTL_MobileNet_V1_model(torch.nn.Module):
    def __init__(self, args, gen_classes=2, smile_classes=2, emo_classes=7, age_classes=100):
        super(MTL_MobileNet_V1_model, self).__init__()
        
        mobilenet_v1_model = Elastic_MobileNet()
        # mobilenet_v1_model = self.load_MobileNet_V1_ImageNet_pretrained_weight(mobilenet_v1_model)

        # print(mobilenet_v1_model)
        
        self.MTL_MobileNet_V1_model_feature = mobilenet_v1_model.model

        # self.MTL_MobileNet_V1_model_feature = mobilenet_v1_model

        self.features_length = 1024

        self.use_gpu = torch.cuda.is_available()

        self.args = args
        self.age_divide = self.get_age_rgs_number_class()

        # gender branch
        self.gender_clf = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, gen_classes)
        )

        # smile branch
        self.smile_clf = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, smile_classes)
        )

        # emotion branch
        self.emotion_clf = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, emo_classes)
        )

        self.age_clf = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, age_classes)
        )

        self.age_rgs_clf = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, self.age_divide)
        )        

    def load_MobileNet_V1_ImageNet_pretrained_weight(self, mobilenet_v1_model):
        tar = torch.load(model_urls['mobilenetV1'])
        state_dict = tar['state_dict']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        mobilenet_v1_model = mobilenet_v1_model.load_state_dict(new_state_dict)

        # model.load_state_dict(model_zoo.load_url(model_urls['mobilenetV1']))
        # LOG("loaded ImageNet pretrained weights", logfile)
        print("[MobileNet_v1] loaded ImageNet pretrained weights")

        return mobilenet_v1_model

    def get_age_rgs_number_class(self):

        if self.args.age_loss_type == "10_age_cls_loss":
            age_divide = 10
            # print("10_age_cls_loss")
        elif self.args.age_loss_type == "5_age_cls_loss":
            age_divide = 20
            # print("5_age_cls_loss")
        elif self.args.age_loss_type == "20_age_cls_loss":
            age_divide = 5
            # print("20_age_cls_loss")
        else:
            print("10_age_cls_loss, 5_age_cls_loss, 20_age_cls_loss")
            ValueError

        return age_divide


    def forward(self, x):
        x = self.MTL_MobileNet_V1_model_feature(x)

        # print(x.size())  # torch.Size([8, 1280, 7, 7])
        x = x.view(x.size(0), -1)
        # print(x.size())   # torch.Size([8, 62720])

        gen_pred = self.gender_clf(x)
        smile_pred = self.smile_clf(x)
        emo_pred = self.emotion_clf(x)
        age_pred = self.age_clf(x)
        age_pred_rgs = self.age_rgs_clf(x)

        return gen_pred, smile_pred, emo_pred, age_pred, age_pred_rgs


# if __name__ == "__main__":
#     model = MTL_MobileNet_V1_model()
#     print("done")
