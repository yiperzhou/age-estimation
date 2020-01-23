import os
import numpy as np

RAW_CHALEARN_AGE_DATASET_AGE_FOLDER = "/home/yi/Narvi_yi_home/data/ChaLearn_CVPR_2016_age"
RAW_CHALEARN_AGE_DATASET_AGE_FOLDER_TRAIN = RAW_CHALEARN_AGE_DATASET_AGE_FOLDER + os.sep + "train"
RAW_CHALEARN_AGE_DATASET_AGE_FOLDER_VALID = RAW_CHALEARN_AGE_DATASET_AGE_FOLDER + os.sep + "valid"
RAW_CHALEARN_AGE_DATASET_AGE_FOLDER_TEST  = RAW_CHALEARN_AGE_DATASET_AGE_FOLDER + os.sep + "test"


def getstd(csv_path):
    std_output = {}
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
    age_folder = RAW_CHALEARN_AGE_DATASET_AGE_FOLDER
    age_valid_folder = RAW_CHALEARN_AGE_DATASET_AGE_FOLDER_VALID
    age_train_folder = RAW_CHALEARN_AGE_DATASET_AGE_FOLDER_TRAIN
    age_test_folder = RAW_CHALEARN_AGE_DATASET_AGE_FOLDER_TEST

    age_csv_train_file = RAW_CHALEARN_AGE_DATASET_AGE_FOLDER_TRAIN + os.sep + 'train_gt.csv'
    age_csv_valid_file = RAW_CHALEARN_AGE_DATASET_AGE_FOLDER_VALID + os.sep + 'valid_gt.csv'
    age_csv_test_file = RAW_CHALEARN_AGE_DATASET_AGE_FOLDER_TEST + os.sep + 'test_gt.csv'

    STD_TRAIN = getstd(age_csv_train_file)
    STD_VALID = getstd(age_csv_valid_file)
    STD_TEST = getstd(age_csv_test_file)

    return age_train_folder, age_valid_folder, age_test_folder, age_csv_train_file, age_csv_valid_file, age_csv_test_file