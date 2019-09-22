import argparse

parser = argparse.ArgumentParser(description='label smoothing branch of the age estimation')

# general parameters
parser.add_argument('--log_dir', type=str, help='log directory')

parser.add_argument('--loading_jobs', type=int, default = 4, help="the number of cpu to load job")
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--weight_decay', type=float, default=1e-6)


parser.add_argument('--dataset', type=str, default="CVPR_16_ChaLearn", help="CVPR_16_ChaLearn, IMDB_WIKI")
parser.add_argument('--model', type=str, default = "AlexNet",
                     help="[AlexNet, MobileNet_V1]")
parser.add_argument('--lr_rate', type=float, default=0.001, help='learning rate (default: 0.001)')

# neural network hyperparamter
parser.add_argument('--lr_schedule', type=float, default=8, help='learning rate schedule')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',help='input batch size for training (default: 32)')
parser.add_argument('--epoch', type=int, help="epoch number, default 1", default=10)

parser.add_argument('--load_IMDB_WIKI_pretrained_model', type=bool, default=False, help="[False, True]")


parser.add_argument('--loss_weights', type=list, default = [1, 0, 0, 0],
                        help="[1,1,1], multi-loss on age estimation, [classification_loss, l1_regression_loss, euclidean_regression_loss, gaussian_loss]")


parser.add_argument('--classification_loss', type=str, default ="100_classes", 
                        help="age classification loss type, two options, (100_classes, 20_classes, 10_classes, 5_classes)")
                        
# working machine environment
parser.add_argument('--working_machine', type=str, default="thinkstation", help="[thinkstation, narvi]")
# parser.add_argument('--store_folder', type=str, default=["", ""], help="["store_folder_in_thinkstation","store_folder_in_narvi"]")

parser.add_argument('--debug', type=bool, default=True, help="[True, False]")


parser.add_argument('--description', type=str, default="", help="SoftMarginLoss function")

# test different technique

parser.add_argument('--label_smoothing', type=bool, default=True, help="label smoothing technique")




# Init Environment
args = parser.parse_args()

