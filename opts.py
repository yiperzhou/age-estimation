import argparse

parser = argparse.ArgumentParser(description='Multitask Learning Face Attribute')

# general parameters
parser.add_argument('--log_dir', type=str, help='log directory')

parser.add_argument('--loading_jobs', type=int, default = 4, help="the number of cpu to load job")
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--weight_decay', type=float, default=1e-6)


parser.add_argument('--dataset', type=str, default="CVPR_16_ChaLearn", help="CVPR_16_ChaLearn, IMDB_WIKI")
parser.add_argument('--model', type=str, default = "MTL_DenseNet_121_model",
                     help="[MTL_ResNet_18_model, res18_cls70, MTL_ResNet_50_model, MTL_DenseNet_121_model, MTL_AlexNet_model, MTL_VGG11_bn_model, MTL_MobileNet_V2_model， MTL_MobileNet_V1_model]")
parser.add_argument('--lr_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
# neural network hyperparamter
parser.add_argument('--lr_schedule', type=float, default=8, help='learning rate schedule')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',help='input batch size for training (default: 32)')
parser.add_argument('--epoch', type=int, help="epoch number, default 1", default=80)
parser.add_argument('--load_IMDB_WIKI_pretrained_model', type=bool, default=False, help="[False, True]")


# [multitask learning traing method hyperparameter]
# "multitask learning training type, \
#                          3: losses = loss[0], losses = loss[1], \
#                          2: losses = loss[0] + loss[1], \
#                             losses.backpropagation()"

parser.add_argument('--multitask_training_type', type=str, default = "Train_Valid",
                    help="Train_Valid, Train_Valid_2")

parser.add_argument('--loss_weights', type=list, default = [0,0,0,1], help="[1,1,1,1], [0,0,0,1], multitask learning weight type, _gender_, _smile_,_emotion_, _age_ ")
parser.add_argument('--subtasks', type=list, default = ["gender", "smile", "emotion", "age"])


# working machine environment
parser.add_argument('--working_machine', type=str, default="narvi", help="[thinkstation, narvi]")
# parser.add_argument('--store_folder', type=str, default=["", ""], help="["store_folder_in_thinkstation","store_folder_in_narvi"]")

parser.add_argument('--debug', type=bool, default=True, help="[True, False]")
parser.add_argument('--age_loss_type', type=str, default="5_age_cls_loss", help="[5_age_cls_loss, 10_age_cls_loss, 20_age_cls_loss]")
parser.add_argument('--no_age_rgs_loss', type=bool, default=True, help="default false")
parser.add_argument('--age_rgs_loss_weight', type=float, default=0, help="gaussian_loss_weight")

parser.add_argument('--age_loss_gaussian', type=str, default="age_loss_gaussian", help="age_loss_gaussian")

parser.add_argument('--no_age_loss_gaussian', type=bool, default=True, help="default false")
parser.add_argument('--age_gaussian_loss_weight', type=float, default=0, help="age_gaussian_loss_weight")












# Init Environment
args = parser.parse_args()



print("Multitask learning Face Attribute")
