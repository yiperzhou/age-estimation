import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim

# not official model weights
model_urls = {
    'mobilenetV1': 'resources/mobilenet_sgd_68.848.pth.tar'
}

class MobileNet(nn.Module):
    def __init__(self, num_categories, add_intermediate_layers, num_outputs=1):
        super(MobileNet, self).__init__()
        
        self.intermediate_CLF = []
        self.add_intermediate_layers = add_intermediate_layers
        self.num_categories = num_categories
        self.num_outputs = num_outputs

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
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
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
        if self.add_intermediate_layers == 2:                
            self.intermediate_CLF.append(IntermediateClassifier(112, 64, self.num_categories))
            self.num_outputs += 1     

            self.intermediate_CLF.append(IntermediateClassifier(56, 128, self.num_categories))
            self.num_outputs += 1

            self.intermediate_CLF.append(IntermediateClassifier(56, 128, self.num_categories))
            self.num_outputs += 1     
            
            self.intermediate_CLF.append(IntermediateClassifier(28, 256, self.num_categories))
            self.num_outputs += 1     

            self.intermediate_CLF.append(IntermediateClassifier(28, 256, self.num_categories))
            self.num_outputs += 1                     

            self.intermediate_CLF.append(IntermediateClassifier(14, 512, self.num_categories))
            self.num_outputs += 1        

            self.intermediate_CLF.append(IntermediateClassifier(14, 512, self.num_categories))
            self.num_outputs += 1     

            self.intermediate_CLF.append(IntermediateClassifier(14, 512, self.num_categories))
            self.num_outputs += 1        

            self.intermediate_CLF.append(IntermediateClassifier(14, 512, self.num_categories))
            self.num_outputs += 1     

            self.intermediate_CLF.append(IntermediateClassifier(14, 512, self.num_categories))
            self.num_outputs += 1        

            self.intermediate_CLF.append(IntermediateClassifier(14, 512, self.num_categories))
            self.num_outputs += 1     

            self.intermediate_CLF.append(IntermediateClassifier(7, 1024, self.num_categories))
            self.num_outputs += 1     

        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        intermediate_outputs = []

        
        if self.add_intermediate_layers == 2:
            x0 = self.model[:2](x)
            intermediate_outputs.append(self.intermediate_CLF[0](x0))

            x1 = self.model[2](x0)
            intermediate_outputs.append(self.intermediate_CLF[1](x1))

            x2 = self.model[3](x1)
            intermediate_outputs.append(self.intermediate_CLF[2](x2))

            x3 = self.model[4](x2)
            intermediate_outputs.append(self.intermediate_CLF[3](x3)) 

            x4 = self.model[5](x3)
            intermediate_outputs.append(self.intermediate_CLF[4](x4))

            x5 = self.model[6](x4)
            intermediate_outputs.append(self.intermediate_CLF[5](x5))

            x6 = self.model[7](x5)
            intermediate_outputs.append(self.intermediate_CLF[6](x6))

            x7 = self.model[8](x6)
            intermediate_outputs.append(self.intermediate_CLF[7](x7))

            x8 = self.model[9](x7)
            intermediate_outputs.append(self.intermediate_CLF[8](x8))

            x9 = self.model[10](x8)
            intermediate_outputs.append(self.intermediate_CLF[9](x9))

            x10 = self.model[11](x9)
            intermediate_outputs.append(self.intermediate_CLF[10](x10))

            x11 = self.model[12](x10)
            intermediate_outputs.append(self.intermediate_CLF[11](x11))

            x = self.model[13:](x11)

        elif self.add_intermediate_layers == 0:
            x = self.model(x)
        else:
            NotImplementedError

        x = x.view(-1, 1024)
        x = self.fc(x)
        return intermediate_outputs + [x]

class IntermediateClassifier(nn.Module):

    def __init__(self, global_pooling_size, num_channels, num_classes):
        """
        Classifier of a cifar10/100 image.

        :param num_channels: Number of input channels to the classifier
        :param num_classes: Number of classes to classify
        """
        super(IntermediateClassifier, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        # self.residual_block_type = residual_block_type
        self.device = 'cuda'
        # if self.residual_block_type == 2: # basicblock type, ResNet-18, ResNet-34
        #     kernel_size = int(3584/self.num_channels)
        # elif self.residual_block_type == 3: # bottleneck block, ResNet-50, ResNet-101, ResNet-152
        #     kernel_size = int(14336/self.num_channels)
        # else:
        #     NotImplementedError
        
        kernel_size = global_pooling_size

        print("kernel_size for global pooling: ", kernel_size)

        self.features = nn.Sequential(
            nn.AvgPool2d(kernel_size=(kernel_size, kernel_size)),
            nn.Dropout(p=0.2, inplace=False)
        ).to(self.device)
        # print("num_channels: ", num_channels, "\n")
        self.classifier = torch.nn.Sequential(nn.Linear(num_channels, num_classes)).to(self.device)

    def forward(self, x):
        """
        Drive features to classification.

        :param x: Input of the lowest scale of the last layer of
                  the last block
        :return: Cifar object classification result
        """
        x = self.features(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



def Elastic_MobileNet():
    """
    based on MobileNet Version1 and ImageNet pretrained weight, https://github.com/marvis/pytorch-mobilenet
    但是这里并没有实现 alpha 乘子和width 乘子
    """
    num_categories = 100
    add_intermediate_layers = 0
    pretrained_weight = 1

    model = MobileNet(num_categories, add_intermediate_layers)

    if pretrained_weight == 1:
        tar = torch.load(model_urls['mobilenetV1'])
        state_dict = tar['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        # model.load_state_dict(model_zoo.load_url(model_urls['mobilenetV1']))
        # LOG("loaded ImageNet pretrained weights", logfile)
        print("loaded ImageNet pretrained weights")
        
    elif pretrained_weight == 0:
        # LOG("not loading ImageNet pretrained weights", logfile)
        pass

    else:
        # LOG("parameter--pretrained_weight, should be 0 or 1", logfile)
        NotImplementedError

    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_categories)

    # # for param in model.parameters():
    # #     param.requires_grad = False
    
    # if add_intermediate_layers == 2:
    #     # LOG("set all intermediate classifiers and final classifiers parameter as trainable.", logfile)

    #     # get all extra classifiers params and final classifier params
    #     for inter_clf in model.intermediate_CLF:
    #         for param in inter_clf.parameters():
    #             param.requires_grad = True

    #     for param in model.fc.parameters():
    #         param.requires_grad = True     

    # elif add_intermediate_layers == 0:
    #     # LOG("only set final classifiers parameter as trainable.", logfile)

    #     for param in model.fc.parameters():
    #         param.requires_grad = True         
    # else:
    #     NotImplementedError

    return model


if __name__ == "__main__":
    model = Elastic_MobileNet()
    print("Elastic_MobileNet", model)
    # pass