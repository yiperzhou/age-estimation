import argparse

parser = argparse.ArgumentParser(description='Multitask Learning Face Attribute')

# general parameters
parser.add_argument('--log_dir', type=str, help='log directory')

parser.add_argument('--loading_jobs', type=int, default = 4, help="the number of cpu to load job")
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--weight_decay', type=float, default=1e-6)


parser.add_argument('--dataset', type=str, default="CVPR_16_ChaLearn", help="CVPR_16_ChaLearn, IMDB_WIKI")
parser.add_argument('--model', type=str, default = "MTL_ResNet_50_model",
                     help="[MTL_ResNet_18_model, res18_cls70, MTL_ResNet_50_model, MTL_DenseNet_121_model, MTL_AlexNet_model, MTL_VGG11_bn_model]", )
parser.add_argument('--lr_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
# neural network hyperparamter
parser.add_argument('--lr_schedule', type=float, default=8, help='learning rate schedule')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',help='input batch size for training (default: 32)')
parser.add_argument('--epoch', type=int, help="epoch number, default 1", default=40)
parser.add_argument('--load_pretrained_model', type=bool, default=True, help="[False, True]")

# [multitask learning traing method hyperparameter]
# "multitask learning training type, \
#                          3: losses = loss[0], losses = loss[1], \
#                          2: losses = loss[0] + loss[1], \
#                             losses.backpropagation()"

parser.add_argument('--multitask_training_type', type=str, default = "Train_Valid",
                    help="Train_Valid, Train_Valid_2")

parser.add_argument('--loss_weights', type=list, default = [0,0,0,1], help="[0,0,0,1], [0,0,0,1], multitask learning weight type, _gender_, _smile_,_emotion_, _age_ ")
parser.add_argument('--subtasks', type=list, default = ["gender", "smile", "emotion", "age"])


# working machine environment
parser.add_argument('--working_machine', type=str, default="thinkstation", help="[thinkstation, narvi]")
# parser.add_argument('--store_folder', type=str, default=["", ""], help="["store_folder_in_thinkstation","store_folder_in_narvi"]")

parser.add_argument('--debug', type=bool, default=True, help="[False, True]")
parser.add_argument('--age_loss_type', type=str, default="age_cls_loss", help="[age_cls_loss, age_mae_loss]")




# Init Environment
args = parser.parse_args()



print("Multitask learning Face Attribute")
