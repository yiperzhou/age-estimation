import argparse

parser = argparse.ArgumentParser(description='Multi-Regression age estimation')

# general parameters
parser.add_argument('--log_dir', type=str, help='log directory')

parser.add_argument('--loading_jobs', type=int, default = 4, help="the number of cpu to load job")
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--weight_decay', type=float, default=1e-6)

parser.add_argument('--dataset', type=str, default="CVPR_16_ChaLearn", help="CVPR_16_ChaLearn")
parser.add_argument('--model', type=str, default="MultiClassificationAlexNet",
                    help="[MultiClassificationAlexNet, MultiClassificationMobileNetv1,MultiClassificationVGG16bn, MultiClassificationDenseNet121, MultiClassificationInceptionv3]")
parser.add_argument('--lr_rate', type=float, default=0.001, help='learning rate (default: 0.001)')

# neural network hyperparamter
parser.add_argument('--lr_schedule', type=float, default=8, help='learning rate schedule')
parser.add_argument('--batch_size', type=int, default=2, metavar='N',help='input batch size for training (default: 32)')

parser.add_argument('--epoch', type=int, help="epoch number, default 1", default=60)

parser.add_argument('--load_IMDB_WIKI_pretrained_model', type=bool, default=False, help="[False, True]")


parser.add_argument('--l1_regression_loss', type=str, default ="0-100_age_rgs",
                    help="4 age regression loss options, "
                         "[0-5_age_rgs, 0-10_age_rgs, 0-20_age_rgs, 0-100_age_rgs] regression")

# working machine environment
parser.add_argument('--working_machine', type=str, default="Narvi", help="[thinkstation, Narvi]")

# Init Environment
args = parser.parse_args()

