import os
import re
import glob
import math
import torch
import numpy as np

from skimage import io
from torchvision import transforms
from torch.utils.data import Dataset
import torch

#from config import parser

# FG_Net_data, IMDB-WIKI dataset data_loader

# IMDB_WIKI dataset has 101 age classes. However, the last two classes are treated as one class. Therefore, age_cls_unit = 100, instead of 101
age_cls_unit = 100

age_cls_unit_fg_net = 70

def load_FG_Net_data():
    # fg_net_data_path = ""

    # distribution of test dataset: FG-NET, len(fg_distr) = 70, sum(fg_distr) = 376, check the excel table for data visualization
    fg_distr = [10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                9, 8, 8, 9, 9, 5, 7, 6, 6, 7, 6, 9, 5, 4, 6, 5, 7, 6, 3, 3, 5, 5, 4, 4, 2,
                3, 5, 2, 2, 2, 3, 2, 3, 3, 2, 2, 2, 0, 0, 1, 0, 1, 3, 1, 1, 0, 0, 0, 1, 0, 0]

    fg_distr[age_cls_unit_fg_net - 1] = sum(fg_distr[age_cls_unit_fg_net - 1:])
    fg_distr = fg_distr[:age_cls_unit_fg_net]
    fg_distr = np.array(fg_distr, dtype='float') + 1

    return fg_distr


# distribution of IMDB-WIKi dataset I: IMDB-Wiki, sum(imdb_distr) = 393405, len(imdb_distr) = 10,000
imdb_distr = [25, 63, 145, 54, 46, 113, 168, 232, 455, 556,
                752, 1089, 1285, 1654, 1819, 1844, 2334, 2828,
                3346, 4493, 6279, 7414, 7706, 9300, 9512, 11489,
                10481, 12483, 11280, 13096, 12766, 14346, 13296,
                12525, 12568, 12278, 12694, 11115, 12089, 11994,
                9960, 9599, 9609, 8967, 7940, 8267, 7733, 6292,
                6235, 5596, 5129, 4864, 4466, 4278, 3515, 3644,
                3032, 2841, 2917, 2755, 2853, 2380, 2169, 2084,
                1763, 1671, 1556, 1343, 1320, 1121, 1196, 949,
                912, 710, 633, 581, 678, 532, 491, 428, 367,
                339, 298, 203, 258, 161, 136, 134, 121, 63, 63,
                82, 40, 37, 24, 16, 18, 11, 4, 9]

# the classes prediction here is class, age-100 and class age-101 are treated as 1 class; age_cls_unit = 100
imdb_distr[age_cls_unit - 1] = sum(imdb_distr[age_cls_unit - 1:])

imdb_distr = imdb_distr[:age_cls_unit]
imdb_distr = np.array(imdb_distr, dtype='float')


# # distribution of test dataset: FG-NET, len(fg_distr) = 70, sum(fg_distr) = 376
# fg_distr = load_FG_Net_data()

# # step 1: correct different distribution between datasets
# loss_weight = fg_distr / imdb_distr

# # step 2: normalize the weight so that the expected weight for a random sample
# #         from training dataset equals to 1, i.e. sum(weight * 1/imdb_distr ) = 1
# loss_weight = loss_weight / sum(loss_weight / imdb_distr)

# # >>> (loss_weight * 100).astype('int')
# # array([1398,  554,  241,  647,  760,  309,  208,  150,   76,   57,   46,
# #          32,   27,   21,   19,   18,   14,   12,   10,    7,    4,    3,
# #           4,    3,    2,    2,    2,    1,    2,    1,    2,    1,    1,
# #           1,    1,    2,    1,    1,    1,    1,    1,    1,    1,    1,
# #           1,    2,    1,    1,    1,    2,    1,    2,    2,    2,    2,
# #           2,    1,    1,    2,    0])


# loss_weight = torch.from_numpy(np.array(loss_weight, dtype='float'))
# loss_weight = loss_weight.type(torch.FloatTensor)


class FaceDataset(Dataset):
  """
  read images from disk dynamically, read IMDB-WIKI dataset
  """

  def __init__(self, datapath, transformer):
    """
    init function
    :param datapath: datapath to aligned folder  
    :param transformer: image transformer
    """
    if datapath[-1] != '/':
      print("[WARNING] PARAM: datapath SHOULD END WITH '/'")
      datapath += '/'
    self.datapath     = datapath
    self.pics         = [f[len(datapath) : ] for f in
                         glob.glob(datapath + "*.jpg")]
    self.transformer  = transformer
    self.age_divde = float(10)
    self.age_cls_unit = int(age_cls_unit)

    #　这里将数据转化维标准的正太分布
    self.age_cls = {x: self.GaussianProb(x) for x in range(1, self.age_cls_unit + 1)}
    self.age_cls_zeroone = {x: self.ZeroOneProb(x) for x in range(1, self.age_cls_unit + 1)}

  def __len__(self):
    return len(self.pics)

# 高斯概率密度函数; gaussian density probality function.
  def GaussianProb(self, true, var = 2.5):
    x = np.array(range(1, self.age_cls_unit + 1), dtype='float')
    probs = np.exp(-np.square(x - true) / (2 * var ** 2)) / (var * (2 * np.pi ** .5))
    return probs / probs.max()

  def ZeroOneProb(self, true):
    x = np.zeros(shape=(self.age_cls_unit, ))
    x[true - 1] = 1
    return x


  def __getitem__(self, idx):
    """
    get images and labels
    :param idx: image index 
    :return: image: transformed image, gender: torch.LongTensor, age: torch.FloatTensor
    """
    # read image and labels
    img_name = self.datapath + self.pics[idx]
    img = io.imread(img_name)
    if len(img.shape) == 2: # gray image
      img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    #解析下，图片名称为：7_0_DakotaFanning2614.jpg，　其中，　７代表 年龄，０　代表性别女
    (age, gender) = re.findall(r"([^_]*)_([^_]*)_[^_]*.jpg", self.pics[idx])[0]
    age = max(1., min(float(age), float(self.age_cls_unit)))

    # preprcess images
    if self.transformer:
      img = transforms.ToPILImage()(img)
      image = self.transformer(img)
    else:
      image = torch.from_numpy(img)

    # preprocess labels
    gender = float(gender)
    gender = torch.from_numpy(np.array([gender], dtype='float'))
    gender = gender.type(torch.LongTensor)

    # through the age_divide, convert the age from [0, 100] to [0, 10], age/ 10 = age_regression

    age_rgs_label = torch.from_numpy(np.array([age / self.age_divde], dtype='float'))
    age_rgs_label = age_rgs_label.type(torch.FloatTensor)

    # 这里的age_cls_label 并不是一个 single label, 而是一个数组, 下面是打印出来
    # print(image_datasets["train"].age_cls[1])
    # [1.00000000e+000 9.23116346e-001 7.26149037e-001 4.86752256e-001
    #  2.78037300e-001 1.35335283e-001 5.61347628e-002 1.98410947e-002
    #  5.97602290e-003 1.53381068e-003 3.35462628e-004 6.25215038e-005
    #  9.92950431e-006 1.34381228e-006 1.54975314e-007 1.52299797e-008
    #  1.27540763e-009 9.10147076e-011 5.53461007e-012 2.86797501e-013
    #  1.26641655e-014 4.76530474e-016 1.52797997e-017 4.17501006e-019
    #  9.72098502e-021 1.92874985e-022 3.26102718e-024 4.69835486e-026
    #  5.76832996e-028 6.03486081e-030 5.38018616e-032 4.08733497e-034
    #  2.64603779e-036 1.45970379e-038 6.86193048e-041 2.74878501e-043
    #  9.38313827e-046 2.72940726e-048 6.76552418e-051 1.42905006e-053
    #  2.57220937e-056 3.94528221e-059 5.15659133e-062 5.74328327e-065
    #  5.45093048e-068 4.40853133e-071 3.03829615e-074 1.78434636e-077
    #  8.92978691e-081 3.80816639e-084 1.38389653e-087 4.28552670e-091
    #  1.13088300e-094 2.54298634e-098 4.87285246e-102 7.95674389e-106
    #  1.10713449e-109 1.31273862e-113 1.32638325e-117 1.14201729e-121
    #  8.37894253e-126 5.23864087e-130 2.79100700e-134 1.26711522e-138
    #  4.90212262e-143 1.61608841e-147 4.54003234e-152 1.08684011e-156
    #  2.21709986e-161 3.85405344e-166 5.70904011e-171 7.20644934e-176
    #  7.75161983e-181 7.10520275e-186 5.54974933e-191 3.69388307e-196
    #  2.09510504e-201 1.01260798e-206 4.17051577e-212 1.46369664e-217
    #  4.37749104e-223 1.11560985e-228 2.42277060e-234 4.48358217e-240
    #  7.07051186e-246 9.50144065e-252 1.08803019e-257 1.06170856e-263
    #  8.82841157e-270 6.25565382e-276 3.77724997e-282 1.94353170e-288
    #  8.52158565e-295 3.18391952e-301 1.01371677e-307 2.75032531e-314
    #  6.34874355e-321 0.00000000e+000 0.00000000e+000 0.00000000e+000]
    # age_cls_label = self.age_cls[int(age)]

    # age_cls_label = self.age_cls_zeroone[int(age)]

    # age_cls_label = torch.from_numpy(np.array([age_cls_label], dtype='float'))
    # age_cls_label = age_cls_label.type(torch.FloatTensor)
    
    # the index starts from 0 instead of 1.
    age = age-1
    
    age_cls_label = torch.tensor(int(age))
    age_cls_label = age_cls_label.type(torch.LongTensor)

    # image of shape [256, 256]
    # gender of shape [,1] and value in {0, 1}
    # age of shape [,1] and value in [0 ~ 10)
    # 这里的age_rgs_label 还是非常难生成的；注意看这里也用到了高斯分布，age_divide, 这里的数学原理其实我并没有懂
    return image, gender, age_rgs_label, age_cls_label


# # image transformaer for IMDB dataset.
# def imdb_image_transformer():
#   return {
#       'train': transforms.Compose([
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#       ]),
#       'val': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#       ]),
#     }



def image_transformer():
  """
  :return:  A transformer to convert a PIL image to a tensor image
            ready to feed into a neural network, for IMDB-WIKI cropped Face Images
  """
  return {
      'train': transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
    }


def load_IMDB_WIKI_dataset(args):

    if args.working_machine == "thinkstation":
      processed_imdb_wiki_dataset = "/media/yi/harddrive/codes/Age-Gender-Pred/pics/"

      # imdb_crop_path = "/media/yi/harddrive/data/IMDB_only_face/imdb_crop"
      # wiki_crop_path = "/media/yi/harddrive/data/WIKI_only_face/wiki_crop"

    elif args.working_machine == "Narvi":
      processed_imdb_wiki_dataset = "/home/zhouy/projects/Age-Gender-Pred/pics/"
      
    else:
        print("working machine should be  [thinkstation, Narvi]")
        NotImplementedError



    
    transformer = image_transformer()

    # image_datasets = {x: FaceDataset(processed_imdb_wiki_dataset + x + '/', transformer[x])
    #                 for x in ['train', 'val']}
        
    print("[IMDB_WIKI] dataset: start loading...")
    image_datasets = {x: FaceDataset(processed_imdb_wiki_dataset + x + '/', transformer[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size= args.batch_size, shuffle=True, num_workers=args.num_workers) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print("[IMDB_WIKI] load_data: Done! Get {} for train and {} for test!"
        .format(dataset_sizes['train'],
                dataset_sizes['val']))
    # print("[IMDB_WIKI] dataset: Done  ...")


    # # train age task
    # for i, input_data in enumerate(dataloaders["val"]):
    #     print("i: ", i)
    #     img = input_data[0]
    #     label = input_data[1]
        
    #     print("input_img: ", img)
    #     print("label: ", label)

    return dataloaders["train"], dataloaders["val"]


# if __name__ == "__main__":
#     dataloaders["train"], dataloaders["val"] = load_IMDB_WIKI_dataset()
