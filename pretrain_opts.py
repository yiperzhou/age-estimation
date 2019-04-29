import argparse

parser = argparse.ArgumentParser(description='Multitask Learning Face Attribute')

# general parameters
parser.add_argument('--log_dir', type=str, help='log directory')

parser.add_argument('--loading_jobs', type=int, default = 4, help="the number of cpu to load job")
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--load_pretrained_model', type=bool, default=False)


parser.add_argument('--dataset', type=str, default="IMDB_WIKI", help="CVPR_16_ChaLearn, IMDB_WIKI")
parser.add_argument('--model', type=str, help="MTL_ResNet_18, res18_cls70, MTL_MobileNet_V1_model, MTL_MobileNet_V2_model", default = "MTL_MobileNet_V2_model")
parser.add_argument('--lr_rate', type=float, default=0.01, help='learning rate (default: 0.001)')
# neural network hyperparamter
parser.add_argument('--lr_schedule', type=float, default=8, help='learning rate schedule')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, help="epoch number, default 1", default=20)

# [multitask learning traing method hyperparameter]
parser.add_argument('--multitask_training_type', type=int, default = 2,
                    help="multitask learning training type, \
                         3: losses = loss[0], losses = loss[1], \
                         2: losses = loss[0] + loss[1], \
                            losses.backpropagation()")

parser.add_argument('--loss_weights', type=list, default = [1,1], help="multitask learning weight type, _gender_, _age_ ")
parser.add_argument('--subtasks', type=list, default = ["gender", "age"])


# working machine environment
parser.add_argument('--working_machine', type=str, default="narvi", help="thinkstation, narvi")


# Init Environment
args = parser.parse_args()



print("Multitask learning Face")
