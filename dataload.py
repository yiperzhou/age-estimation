import os
import cv2
import re
import glob
import math
import torch
import numpy as np
import pandas as pd
from PIL import Image

from skimage import io
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision

# from config import parser
import h5py
# from config import config, parser


# age_cls_unit = int(parser['RacNet']['age_cls_unit'])
# # age_cls_unit = 60

# # distribution of IMDB-WIKi dataset I: IMDB-Wiki
# imdb_distr = [25, 63, 145, 54, 46, 113, 168, 232, 455, 556,
#               752, 1089, 1285, 1654, 1819, 1844, 2334, 2828,
#               3346, 4493, 6279, 7414, 7706, 9300, 9512, 11489,
#               10481, 12483, 11280, 13096, 12766, 14346, 13296,
#               12525, 12568, 12278, 12694, 11115, 12089, 11994,
#               9960, 9599, 9609, 8967, 7940, 8267, 7733, 6292,
#               6235, 5596, 5129, 4864, 4466, 4278, 3515, 3644,
#               3032, 2841, 2917, 2755, 2853, 2380, 2169, 2084,
#               1763, 1671, 1556, 1343, 1320, 1121, 1196, 949,
#               912, 710, 633, 581, 678, 532, 491, 428, 367,
#               339, 298, 203, 258, 161, 136, 134, 121, 63, 63,
#               82, 40, 37, 24, 16, 18, 11, 4, 9]
# imdb_distr[age_cls_unit - 1] = sum(imdb_distr[age_cls_unit - 1:])
# imdb_distr = imdb_distr[:age_cls_unit]
# imdb_distr = np.array(imdb_distr, dtype='float')

# # distribution of test dataset: FG-NET
# fg_distr = [10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10,
#             9, 8, 8, 9, 9, 5, 7, 6, 6, 7, 6, 9, 5, 4, 6, 5, 7, 6, 3, 3, 5, 5, 4, 4, 2,
#             3, 5, 2, 2, 2, 3, 2, 3, 3, 2, 2, 2, 0, 0, 1, 0, 1, 3, 1, 1, 0, 0, 0, 1, 0, 0]
# fg_distr[age_cls_unit - 1] = sum(fg_distr[age_cls_unit - 1:])
# fg_distr = fg_distr[:age_cls_unit]
# fg_distr = np.array(fg_distr, dtype='float') + 1

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


# class FaceDataset(Dataset):

#   def __init__(self, datapath, transformer):
#     """
#     init function
#     :param datapath: datapath to aligned folder
#     :param transformer: image transformer
#     """
#     if datapath[-1] != '/':
#       print("[WARNING] PARAM: datapath SHOULD END WITH '/'")
#       datapath += '/'
#     self.datapath = datapath
#     self.pics = [f[len(datapath):] for f in
#                  glob.glob(datapath + "*.jpg")]
#     self.transformer = transformer
#     self.age_divde = float(parser['DATA']['age_divide'])
#     self.age_cls_unit = int(parser['RacNet']['age_cls_unit'])

#     self.age_cls = {x: self.GaussianProb(x)
#                     for x in range(1, self.age_cls_unit + 1)}
#     self.age_cls_zeroone = {x: self.ZeroOneProb(
#         x) for x in range(1, self.age_cls_unit + 1)}

#   def __len__(self):
#     return len(self.pics)

#   def GaussianProb(self, true, var=2.5):
#     x = np.array(range(1, self.age_cls_unit + 1), dtype='float')
#     probs = np.exp(-np.square(x - true) / (2 * var ** 2)) / \
#         (var * (2 * np.pi ** .5))
#     return probs / probs.max()

#   def ZeroOneProb(self, true):
#     x = np.zeros(shape=(self.age_cls_unit, ))
#     x[true - 1] = 1
#     return x

#   def __getitem__(self, idx):
#     """
#     get images and labels
#     :param idx: image index
#     :return: image: transformed image, gender: torch.LongTensor, age: torch.FloatTensor
#     """
#     # read image and labels
#     img_name = self.datapath + self.pics[idx]
#     img = io.imread(img_name)
#     if len(img.shape) == 2:  # gray image
#       img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
#     (age, gender) = re.findall(r"([^_]*)_([^_]*)_[^_]*.jpg", self.pics[idx])[0]
#     age = max(1., min(float(age), float(self.age_cls_unit)))

#     # preprcess images
#     if self.transformer:
#       img = transforms.ToPILImage()(img)
#       image = self.transformer(img)
#     else:
#       image = torch.from_numpy(img)

#     # preprocess labels
#     gender = float(gender)
#     gender = torch.from_numpy(np.array([gender], dtype='float'))
#     gender = gender.type(torch.LongTensor)

#     age_rgs_label = torch.from_numpy(
#         np.array([age / self.age_divde], dtype='float'))
#     age_rgs_label = age_rgs_label.type(torch.FloatTensor)

#     age_cls_label = self.age_cls[int(age)]
#     # age_cls_label = self.age_cls_zeroone[int(age)]

#     age_cls_label = torch.from_numpy(np.array([age_cls_label], dtype='float'))
#     age_cls_label = age_cls_label.type(torch.FloatTensor)

#     # image of shape [256, 256]
#     # gender of shape [,1] and value in {0, 1}
#     # age of shape [,1] and value in [0 ~ 10)
#     return image, gender, age_rgs_label, age_cls_label
#     # return image, gender


class Mix_Age_Gender_Emotion_Dataset(Dataset):

    def __init__(self, age_folder, age_describe_data, gender_folder, emotion_folder, transform=None, train=True):
        self.train = train
        self.transform = transform

        # for age
        self.age_folder = age_folder
        self.age_describe_data = age_describe_data
        self.age_img_labels = []
        self.age_img_paths = []

        # for gender
        self.gender_folder = gender_folder
        self.gender_img_labels = []
        self.gender_img_paths = []

        # for emotion
        self.emotion_folder = emotion_folder
        self.emotion_img_labels = []
        self.emotion_imgs = []

        if self.train:
            print("train")
            # parse train img paths
            self.age_img_folder = self.age_folder + os.sep + "train"
            self.parse_age(self.age_img_folder)

            # parse train gender img path
            self.gender_img_folder = self.gender_folder + os.sep + "train"
            self.parse_gender(self.gender_img_folder)

            # parse emotion
            self.emotion_img_folder = self.emotion_folder + os.sep + "train.csv"
            self.parse_emotion(self.emotion_img_folder)

        else:
            print("valid")
            # parse test age img paths
            self.age_img_folder = self.age_folder + os.sep + "valid"
            self.parse_age(self.age_img_folder)

            # parse test gender img path
            self.gender_img_folder = self.gender_folder + os.sep + "test"
            self.parse_gender(self.gender_img_folder)

            # parse emotion
            self.emotion_img_folder = self.emotion_folder + os.sep + "val.csv"
            self.parse_emotion(self.emotion_img_folder)


    def __getitem__(self, index):
        # get age image
        # print("index: ", index)
        
        age_img = Image.open(self.age_img_paths[index])
        age_label = float(self.age_img_labels[index])

        # get gender image
        gender_img = Image.open(self.gender_img_paths[index])
        gender_label = int(self.gender_img_labels[index])
        
        pixel = np.asarray(self.emotion_imgs[index]).reshape(48,48)

        emotion_img = Image.fromarray(pixel).convert('L')
        emotion_label = self.emotion_img_labels[index]

        if self.transform:
          age_img = self.transform(age_img)
          gender_img = self.transform(gender_img)
          
          # convert grey image to RGB image, one channel to three channel
          emotion_img = emotion_img.convert('RGB')
          emotion_img = self.transform(emotion_img)
        
        # print("age_img size: ", age_img.size())   # age_img size:  torch.Size([3, 224, 224])
        # print("gender_img size: ", gender_img.size()) # gender_img size:  torch.Size([3, 224, 224])
        # print("emotion_img size: ", emotion_img.size()) # emotion_img size:  torch.Size([1, 224, 224]),　注意看这里，这里是灰度图


        # print("gender label: ", gender_label)

        return (age_img, gender_img, emotion_img), (age_label, gender_label, emotion_label)


    def __len__(self):
        # 将三个数据集中的最少的那个作为 len()
        
        # length = min(len(self.age_img_labels), len(self.gender_img_labels), len(self.emotion_img_labels))
        length = 1000
        print("length: ", length)
        return length


    def parse_age(self, data_folder):
      # csv_file = "/media/yi/harddrive/data/appa-real-release/gt_train.csv"
      csv_file = self.age_describe_data

      df = pd.read_csv(csv_file)
      for index, row in df.iterrows():
        img = row['file_name']
        age_label = row["apparent_age_avg"]

        # check the path(img file) existing
        img_path = data_folder + os.sep + img+"_face.jpg"
        # print(img_path)
        if os.path.exists(img_path):
          self.age_img_paths.append(img_path)
          self.age_img_labels.append(int(age_label))

    def parse_gender(self, data_folder):

        # for each img in train/val folder, we parse 
        self.gender_img_paths = glob.glob(data_folder + os.sep + "*.jpg")
        for idx in range(len(self.gender_img_paths)):
          (age, gender) = re.findall(r"([^_]*)_([^_]*)_[^_]*.jpg", self.gender_img_paths[idx])[0]
          # print demo: /media/yi/harddrive/codes/MultitaskingFace/pics/train/22 0
          self.gender_img_labels.append(gender)
          # 0:  Female , 1: Male
        assert len(self.gender_img_paths) == len(self.gender_img_labels)


    def parse_emotion(self, data_path):
      # parse csv file
      df = pd.read_csv(data_path)
      for index, row in df.iterrows():
          pixel = row['pixels']
          img = [float(p) for p in pixel.split()]

          emotion_label = row["emotion"]
          self.emotion_img_labels.append(emotion_label)
          self.emotion_imgs.append(img)

      assert len(self.emotion_imgs) == len(self.emotion_img_labels)





class Mix_Age_Gender_Emotion_Yue_Dataset(Dataset):

    def __init__(self, age_folder, age_describe_data, gender_folder, emotion_folder, transform=None, train=True):
        self.train = train
        self.transform = transform

        # CVPR Age dataset
        self.age_folder = age_folder
        self.age_describe_data = age_describe_data
        self.age_img_labels = []
        self.age_img_paths = []

        # Gender dataset
        self.gender_folder = gender_folder
        self.gender_img_labels = []
        self.gender_img_paths = []

        # Smile Dataset
        self.emotion_folder = emotion_folder
        self.emotion_img_labels = []
        self.emotion_imgs = []

        if self.train:
            print("train")
            # parse train img paths
            self.age_img_folder = self.age_folder + os.sep + "train"
            self.parse_age(self.age_img_folder)

            # parse train gender img path
            self.gender_img_folder = self.gender_folder + os.sep + "train"
            self.parse_gender(self.gender_img_folder)

            # parse emotion
            self.emotion_img_folder = self.emotion_folder + os.sep + "train.csv"
            self.parse_emotion(self.emotion_img_folder)

        else:
            print("valid")
            # parse test age img paths
            self.age_img_folder = self.age_folder + os.sep + "valid"
            self.parse_age(self.age_img_folder)

            # parse test gender img path
            self.gender_img_folder = self.gender_folder + os.sep + "test"
            self.parse_gender(self.gender_img_folder)

            # parse emotion
            self.emotion_img_folder = self.emotion_folder + os.sep + "val.csv"
            self.parse_emotion(self.emotion_img_folder)


    def __getitem__(self, index):
        # get age image
        # print("index: ", index)
        
        age_img = Image.open(self.age_img_paths[index])
        age_label = float(self.age_img_labels[index])

        # get gender image
        gender_img = Image.open(self.gender_img_paths[index])
        gender_label = int(self.gender_img_labels[index])
        
        pixel = np.asarray(self.emotion_imgs[index]).reshape(48,48)

        emotion_img = Image.fromarray(pixel).convert('L')
        emotion_label = self.emotion_img_labels[index]

        if self.transform:
          age_img = self.transform(age_img)
          gender_img = self.transform(gender_img)
          
          # convert grey image to RGB image, one channel to three channel
          emotion_img = emotion_img.convert('RGB')
          emotion_img = self.transform(emotion_img)
        
        # print("age_img size: ", age_img.size())   # age_img size:  torch.Size([3, 224, 224])
        # print("gender_img size: ", gender_img.size()) # gender_img size:  torch.Size([3, 224, 224])
        # print("emotion_img size: ", emotion_img.size()) # emotion_img size:  torch.Size([1, 224, 224]),　注意看这里，这里是灰度图


        # print("gender label: ", gender_label)

        return (age_img, gender_img, emotion_img), (age_label, gender_label, emotion_label)


    def __len__(self):
        # 将三个数据集中的最少的那个作为 len()
        
        # length = min(len(self.age_img_labels), len(self.gender_img_labels), len(self.emotion_img_labels))
        length = 1000
        print("length: ", length)
        return length


    def parse_age(self, data_folder):
      # csv_file = "/media/yi/harddrive/data/appa-real-release/gt_train.csv"
      csv_file = self.age_describe_data

      df = pd.read_csv(csv_file)
      for index, row in df.iterrows():
        img = row['file_name']
        age_label = row["apparent_age_avg"]

        # check the path(img file) existing
        img_path = data_folder + os.sep + img+"_face.jpg"
        # print(img_path)
        if os.path.exists(img_path):
          self.age_img_paths.append(img_path)
          self.age_img_labels.append(int(age_label))

    def parse_gender(self, data_folder):

        # for each img in train/val folder, we parse 
        self.gender_img_paths = glob.glob(data_folder + os.sep + "*.jpg")
        for idx in range(len(self.gender_img_paths)):
          (age, gender) = re.findall(r"([^_]*)_([^_]*)_[^_]*.jpg", self.gender_img_paths[idx])[0]
          # print demo: /media/yi/harddrive/codes/MultitaskingFace/pics/train/22 0
          self.gender_img_labels.append(gender)
          # 0:  Female , 1: Male
        assert len(self.gender_img_paths) == len(self.gender_img_labels)


    def parse_emotion(self, data_path):
      # parse csv file
      df = pd.read_csv(data_path)
      for index, row in df.iterrows():
          pixel = row['pixels']
          img = [float(p) for p in pixel.split()]

          emotion_label = row["emotion"]
          self.emotion_img_labels.append(emotion_label)
          self.emotion_imgs.append(img)

      assert len(self.emotion_imgs) == len(self.emotion_img_labels)



def CVPE_AGE_load_dataset(data_path, transforms):
    # data_path = '/home/yi/narvi_MLG/AGE_ESTIMATION/CVPR_AGE_5_points/TRAIN/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transforms
    )
    train_dataset.class_to_idx = {'24': 23, '25': 24, '26': 25, '27': 26, '20': 19, '21': 20, '22': 21, '23': 22, '28': 27, '29': 28, '4': 3, '8': 7, '59': 58, '58': 57, '55': 54, '54': 53, '57': 56, '56': 55, '51': 50, '50': 49, '53': 52, '52': 51, '88': 87, '89': 88, '82': 81, '83': 82, '80': 79, '81': 80, '86': 85, '87': 86, '84': 83, '85': 84, '3': 2, '7': 6, '100': 99, '39': 38, '38': 37, '33': 32, '32': 31, '31': 30, '30': 29, '37': 36, '36': 35, '35': 34, '34': 33, '60': 59, '61': 60, '62': 61, '63': 62, '64': 63, '65': 64, '66': 65, '67': 66, '68': 67, '69': 68, '2': 1, '6': 5, '99': 98, '98': 97, '91': 90, '90': 89, '93': 92, '92': 91, '95': 94, '94': 93, '97': 96, '96': 95, '11': 10, '10': 9, '13': 12, '12': 11, '15': 14, '14': 13, '17': 16, '16': 15, '19': 18, '18': 17, '48': 47, '49': 48, '46': 45, '47': 46, '44': 43, '45': 44, '42': 41, '43': 42, '40': 39, '41': 40, '1': 0, '5': 4, '9': 8, '77': 76, '76': 75, '75': 74, '74': 73, '73': 72, '72': 71, '71': 70, '70': 69, '79': 78, '78': 77}
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=64,
    #     num_workers=0,
    #     shuffle=True
    # )
    return train_dataset


def get_model_data(args):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    age_train_dataloader = CVPE_AGE_load_dataset("/home/yi/narvi_MLG/AGE_ESTIMATION/CVPR_AGE_5_points/TRAIN/", transform)
    age_test_dataloader = CVPE_AGE_load_dataset("/home/yi/narvi_MLG/AGE_ESTIMATION/CVPR_AGE_5_points/VALID/", transform)
    # age train set images: 3707; test image: 1356

    gender_train_dataloader = torchvision.datasets.ImageFolder("/home/yi/narvi_MLG/AGE_ESTIMATION/CVPR_GENDER_5_points/TRAIN/", transform)
    gender_test_dataloader = torchvision.datasets.ImageFolder("/home/yi/narvi_MLG/AGE_ESTIMATION/CVPR_GENDER_5_points/VALID/", transform)
    # gender train set images: 4548; test image: 2250


    Smile_train_dataloader = torchvision.datasets.ImageFolder("/home/yi/narvi_MLG/AGE_ESTIMATION/CVPR_SMILE_5_points/TRAIN/", transform)
    Smile_test_dataloader = torchvision.datasets.ImageFolder("/home/yi/narvi_MLG/AGE_ESTIMATION/CVPR_SMILE_5_points/VALID/", transform)

    # smile train set images: 4548, test image: 2250

    age_train_loader = torch.utils.data.DataLoader(age_train_dataloader, batch_size=args.batch_size, shuffle=True, num_workers=args.loading_jobs)
    age_test_loader = torch.utils.data.DataLoader(age_test_dataloader, batch_size=args.batch_size, shuffle=True, num_workers=args.loading_jobs)

    gender_train_loader = torch.utils.data.DataLoader(gender_train_dataloader, batch_size=args.batch_size, shuffle=True, num_workers=args.loading_jobs)
    gender_test_loader = torch.utils.data.DataLoader(gender_test_dataloader, batch_size=args.batch_size, shuffle=True, num_workers=args.loading_jobs)

    smile_train_loader = torch.utils.data.DataLoader(Smile_train_dataloader, batch_size=args.batch_size, shuffle=True, num_workers=args.loading_jobs)
    smile_test_loader = torch.utils.data.DataLoader(Smile_test_dataloader, batch_size=args.batch_size, shuffle=True, num_workers=args.loading_jobs)
    
    # model_dataloader
    return [age_train_loader, age_test_loader, gender_train_loader, gender_test_loader, smile_train_loader, smile_test_loader]


