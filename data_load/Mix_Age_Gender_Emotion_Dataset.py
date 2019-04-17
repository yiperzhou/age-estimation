import os
import cv2
import re
import glob
import math
import torch
import numpy as np
import pandas as pd
from PIL import Image
import sys

# from skimage import io
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision
# import accimage
import h5py
from torchvision.datasets import DatasetFolder
# from config import parser

# from config import config, parser
# FER2013 Data, resize image from 48*48 to 224*224
from torch.utils.data import RandomSampler



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

