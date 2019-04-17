import os
import cv2
import re
import glob
import math
import torch
import numpy as np

from skimage import io
from torchvision import transforms
from torch.utils.data import Dataset

#from config import parser

# FG_Net_data, IMDB-WIKI dataset data_loader

age_cls_unit = 100
age_cls_unit_fg_net = 70

def load_FG_Net_data():
    fg_net_data_path = ""

    # distribution of test dataset: FG-NET, len(fg_distr) = 70, sum(fg_distr) = 376
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
    self.age_cls_unit = int(100)

    self.age_cls = {x: self.GaussianProb(x) for x in range(1, self.age_cls_unit + 1)}
    self.age_cls_zeroone = {x: self.ZeroOneProb(x) for x in range(1, self.age_cls_unit + 1)}

  def __len__(self):
    return len(self.pics)

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

    age_rgs_label = torch.from_numpy(np.array([age / self.age_divde], dtype='float'))
    age_rgs_label = age_rgs_label.type(torch.FloatTensor)

    age_cls_label = self.age_cls[int(age)]
    # age_cls_label = self.age_cls_zeroone[int(age)]

    age_cls_label = torch.from_numpy(np.array([age_cls_label], dtype='float'))
    age_cls_label = age_cls_label.type(torch.FloatTensor)

    # image of shape [256, 256]
    # gender of shape [,1] and value in {0, 1}
    # age of shape [,1] and value in [0 ~ 10)
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

    elif args.working_machine == "narvi":
      processed_imdb_wiki_dataset = "/home/zhouy/projects/Age-Gender-Pred/pics/"
      
    else:
        print("working machine should be  [thinkstation, narvi]")
        NotImplementedError



    
    transformer = image_transformer()

    image_datasets = {x: FaceDataset(processed_imdb_wiki_dataset + x + '/', transformer[x])
                    for x in ['train', 'val']}
        
    print("[AgePredModel] load_data: start loading...")
    image_datasets = {x: FaceDataset(processed_imdb_wiki_dataset + x + '/', transformer[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size= args.batch_size, shuffle=True, num_workers=args.num_workers) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print("[AgePredModel] load_data: Done! Get {} for train and {} for test!"
        .format(dataset_sizes['train'],
                dataset_sizes['val']))
    print("[AgePredModel] load_data: loading finished !")


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
