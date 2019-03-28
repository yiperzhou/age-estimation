import argparse

parser = argparse.ArgumentParser(description='Multitask Learning Face')

# general parameters
parser.add_argument('--log_dir', type=str, help='log directory')
parser.add_argument('--lr_rate', type=float, default=0.01, help='learning rate (default: 0.001)')
parser.add_argument('--verify-images', type=str, help='verify 2 images of face belong to one person,')
parser.add_argument('--img_path', type=str, help='[ThinkStation] CelebA aligned Image path', default="/media/yi/harddrive/data/face_data/celeba/224_by_224_img_align_celeba")
# parser.add_argument('--img_path', type=str, help='[Narvi] CelebA aligned Image path', default="/home/zhouy/data/224_by_224_img_align_celeba")
# parser.add_argument('--img_path', type=str, help='[XPS] CelebA aligned Image path', default="/media/yi/harddrive/data/face_data/celeba/224_by_224_img_align_celeba")


# neural network hyperparamter
parser.add_argument('--lr_schedule', type=float, default=5, help='learning rate schedule')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, help="epoch number, default 1", default=1)

# [multitask learning traing method hyperparameter]
parser.add_argument('--multitask_training_type', type=int, default = 2,
                    help="multitask learning training type, \
                         3: losses = loss[0], losses = loss[1], \
                         2: losses = loss[0] + loss[1], \
                            losses.backpropagation()")

parser.add_argument('--multitask_weight_type', type=int, default = 4, help="multitask learning weight type, 3 = [1,1,1]")

# training models
parser.add_argument('--loading_jobs', type=int, default = 4, help="the number of cpu to load job")
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--load_pretrained_model', type=bool, default=True)




parser.add_argument('--dataset', type=str, default="CVPR_16_ChaLearn")
parser.add_argument('--model', type=str, help="MTL_ResNet_18, res18_cls70", default = "MTL_ResNet_50")
parser.add_argument('--num_workers', type=int, default=4)
# Init Environment
args = parser.parse_args()

print("Multitask learning Face")
