import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.nn.functional as F

# from utils.helper import print_model_dimension_summary

# __all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model




class MTL_AlexNet_model(nn.Module):
    def __init__(self):
        super(MTL_AlexNet_model, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        
        self.features_length = 9216


        # gender branch
        self.gender_clf = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 2)
        )

        # smile branch
        self.smile_clf = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 2)
        )
        
        # emotion branch
        self.emotion_clf = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 7)
        )

        self.age_clf = nn.Sequential(
            nn.Linear(self.features_length, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 100)
        )

        self.fc1 = nn.Linear(self.features_length, 256)
        # self.age_cls_pred = nn.Linear(256, 100)
        self.age_cls_pred = nn.Linear(256, 88)

        self.dropout    = nn.Dropout(p=0.5, inplace=False)

    def get_convs_out(self, x):
        """
        get outputs from feature block of alexnet
        :param x: image input
        :return: middle ouput from feature block
        """
        x = self.alexnet.features(x)
        
        return x

    def get_face_attribute(self, last_conv_out):
        # print("1 last_conv_out: ", last_conv_out.size())
        # last_conv_out = nn.AvgPool2d((6, 6))(last_conv_out)
        
        # print("2: ", last_conv_out.size())
        # # x = x.view(x.size(0), 256 * 6 * 6)
        last_conv_out = last_conv_out.view(last_conv_out.size(0), -1)

        # print("3: last_conv_out: ", last_conv_out.size())



        # gen_pred = F.relu(self.dropout(self.fc2(last_conv_out)))
        # gen_pred = F.softmax(self.gen_cls_pred(gen_pred))

        # smile_pred   = F.relu(self.dropout(self.fc4(last_conv_out)))
        # smile_pred   = F.softmax(self.smile_cls_pred(smile_pred))

        # emo_pred = F.relu(self.dropout(self.fc3(last_conv_out)))
        # emo_pred = F.softmax(self.emo_cls_pred(emo_pred))

        age_pred = self.dropout(F.relu(self.fc1(last_conv_out)))
        age_pred = F.softmax(self.age_cls_pred(age_pred), 1)
        
        gen_pred = self.gender_clf(last_conv_out)
        smile_pred = self.smile_clf(last_conv_out)
        emo_pred = self.emotion_clf(last_conv_out)
        # # age_pred = F.softmax(self.age_clf(last_conv_out), 1)
        # age_pred = self.age_clf(last_conv_out)

        return gen_pred, smile_pred, emo_pred, age_pred

    
    def forward(self, x, return_atten_info = False):
        last1 = self.get_convs_out(x)
        gen_pred, smile_pred, emo_pred, age_pred = self.get_face_attribute(last1)
        return gen_pred, smile_pred, emo_pred, age_pred



if __name__ == "__main__":
    alexModel = MTL_AlexNet_model()

    # print_model_dimension_summary(alexModel)
    print("AlexModel: ")
    print(alexModel)

    print("done")

