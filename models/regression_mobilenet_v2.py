import torch
from torch import nn



class Multi_Regression_MobileNet_V2_model(torch.nn.Module):
  def __init__(self, gen_classes= 2, smile_classes = 2, emo_classes = 7, age_classes = 100):
    super(Multi_Regression_MobileNet_V2_model, self).__init__()

    backbone_model = mobilenet_v2(pretrained=True)

    self.MTL_MobileNet_V2_model_feature = nn.Sequential(
        backbone_model.features,
        torch.nn.AvgPool2d(kernel_size=(7, 7))
    )
    self.features_length = 1280

    self.use_gpu = torch.cuda.is_available()


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

  def forward(self, x):
    x = self.MTL_MobileNet_V2_model_feature(x)
    
    # print(x.size())  # torch.Size([8, 1280, 7, 7])
    x = x.view(x.size(0), -1)
    # print(x.size())   # torch.Size([8, 62720])


    gen_pred  = self.gender_clf(x)
    smile_pred  = self.smile_clf(x)
    emo_pred  = self.emotion_clf(x)
    age_pred  = self.age_clf(x)

    return gen_pred, smile_pred, emo_pred, age_pred 
