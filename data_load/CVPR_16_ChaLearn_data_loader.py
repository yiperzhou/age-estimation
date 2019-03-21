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
import accimage
import h5py
from torchvision.datasets import DatasetFolder
# from config import parser

# from config import config, parser
# FER2013 Data, resize image from 48*48 to 224*224

emotion_transform = {
    "emotion_transform_train": transforms.Compose([
        # transforms.RandomCrop(44),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
    ]),
    "emotion_transform_valid": transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ]),
    "emotion_transform_test": transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
}

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


def get_CVPR_Age_Gender_Smile_data(args):

    augment = False

    if augment:
        transform = transforms.Compose([
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    else:
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



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class AGE_ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(AGE_ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples


def load_chalearn_dataset(data_dir,resize=(224,224)):

    data_transforms = {
        'TRAIN': transforms.Compose([
            transforms.RandomSizedCrop(max(resize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'VALID': transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'TEST': transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])            

    }
    dsets = {x: AGE_ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['TRAIN', 'VALID', 'TEST']}

    return dsets


def getstd(csv_path):
    std_output={}
    for ann_file in [csv_path]:
        with open(ann_file, "r") as fp:
            for k, line in enumerate(fp):
                if k == 0:  # skip header
                    continue
                name, age, std = line.split(",")
                std_output[name] = float(std)
    return std_output


def load_raw_chalearn_age_dataset():
    # load cvpr age dataset
    age_folder = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/data/ChaLearn_AGE_CVPR_16"
    age_valid_folder = age_folder+ os.sep + "validation_data/valid"
    age_train_folder = age_folder+ os.sep + "training_data/train"
    age_test_folder = age_folder+ os.sep + "test_data/test"

    age_csv_train_file = age_folder + os.sep + 'training_data/train_gt.csv'
    age_csv_valid_file = age_folder + os.sep + 'validation_data/valid_gt.csv'
    age_csv_test_file = age_folder + os.sep + 'test_data/test_gt.csv'

    STD_VALID = getstd(age_csv_valid_file)
    STD_TEST = getstd(age_csv_test_file)


    return age_train_folder, age_valid_folder, age_test_folder, age_csv_train_file, age_csv_valid_file, age_csv_test_file




