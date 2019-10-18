import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler
from torchvision import transforms

from data_load.cvpr_16_chalearn_dataset import cvpr_age_load_dataset_imagefolder
from utils.helper_4 import plot_images

def CVPR_AGE_load_dataset(data_path, transforms):
    # # data_path = '/home/yi/Narvi_MLG/AGE_ESTIMATION/CVPR_AGE_5_points/TRAIN/'
    # train_dataset = torchvision.datasets.ImageFolder(
    #     root=data_path,
    #     transform=transforms
    # )
    print("data_path: ", data_path)
    folder_to_classes = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '10': 9, '11': 10, '12': 11, '13': 12, '14': 13, '15': 14, '16': 15, '17': 16, '18': 17, '19': 18, '20': 19, '21': 20, '22': 21, '23': 22, '24': 23, '25': 24, '26': 25, '27': 26, '28': 27, '29': 28, '30': 29, '31': 30, '32': 31, '33': 32, '34': 33, '35': 34, '36': 35, '37': 36, '38': 37, '39': 38, '40': 39, '41': 40, '42': 41, '43': 42, '44': 43, '45': 44, '46': 45, '47': 46, '48': 47, '49': 48, '50': 49, '51': 50, '52': 51, '53': 52, '54': 53, '55': 54, '56': 55, '57': 56, '58': 57, '59': 58, '60': 59, '61': 60, '62': 61, '63': 62, '64': 63, '65': 64, '66': 65, '67': 66, '68': 67, '69': 68, '70': 69, '71': 70, '72': 71, '73': 72, '74': 73, '75': 74, '76': 75, '77': 76, '78': 77, '79': 78, '80': 79, '81': 80, '82': 81, '83': 82, '84': 83, '85': 84, '86': 85, '87': 86, '88':87, '89':88, '90':89, '91':90, '92':91, '93':92, '94':93, '95':94, '96':95, '97':96, '98':97, '99':98, '100':99}
    
    age_classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100']

    train_dataset =  cvpr_age_load_dataset_imagefolder(folder_to_classes, age_classes,
                                                        root=data_path,
                                                        transform=transforms)

    # train_dataset = CVPE_AGE_load_dataset_ImageFolder_class(root=data_path, transform=transforms)

    

    # # # CVPR_AGE_5_POINT, train dataset has 87 classes; validation has 88 classes
    
    # # train_dataset.class_to_idx = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '10': 9, '11': 10, '12': 11, '13': 12, '14': 13, '15': 14, '16': 15, '17': 16, '18': 17, '19': 18, '20': 19, '21': 20, '22': 21, '23': 22, '24': 23, '25': 24, '26': 25, '27': 26, '28': 27, '29': 28, '30': 29, '31': 30, '32': 31, '33': 32, '34': 33, '35': 34, '36': 35, '37': 36, '38': 37, '39': 38, '40': 39, '41': 40, '42': 41, '43': 42, '44': 43, '45': 44, '46': 45, '47': 46, '48': 47, '49': 48, '50': 49, '51': 50, '52': 51, '53': 52, '54': 53, '55': 54, '56': 55, '57': 56, '58': 57, '59': 58, '60': 59, '61': 60, '62': 61, '63': 62, '64': 63, '65': 64, '66': 65, '67': 66, '68': 67, '69': 68, '70': 69, '71': 70, '72': 71, '73': 72, '74': 73, '75': 74, '76': 75, '77': 76, '78': 77, '79': 78, '80': 79, '81': 80, '82': 81, '83': 82, '84': 83, '85': 84, '86': 85, '87': 86, '88':87}
    
    
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=64,
    #     num_workers=0,
    #     shuffle=True
    # )
    return train_dataset



def dataset_augmentation_sampler(origin_dataset, target_num_samples):
    
    # randomSampler to increase the number of dataset into the target_num_samples
    augmentation_sampler = RandomSampler(origin_dataset, replacement=True, num_samples=target_num_samples)

    return augmentation_sampler



def get_cvpr_age_data(args, show_sample=False):

    # augment = True
    augment = False

    resize = 224
    if args.model == "Multi_loss_InceptionV3":
        resize = 299
        print("Inception V3: ", resize, " image size")


    if augment == True:

        transform = transforms.Compose([
            transforms.RandomCrop(resize, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    else:
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    if args.working_machine == "thinkstation":
        # tut thinkstation

        age_train_dataset = CVPR_AGE_load_dataset("/home/yi/Narvi_MLG/AGE_ESTIMATION/CVPR_AGE_5_points/TRAIN/", transform)
        age_test_dataset = CVPR_AGE_load_dataset("/home/yi/Narvi_MLG/AGE_ESTIMATION/CVPR_AGE_5_points/VALID/", transform)
        # age train set images: 3707; test image: 1356

    elif args.working_machine == "Narvi":
        # Narvi
        age_train_dataset = CVPR_AGE_load_dataset("/sgn-data/MLG/AGE_ESTIMATION/CVPR_AGE_5_points/TRAIN/", transform)
        age_test_dataset = CVPR_AGE_load_dataset("/sgn-data/MLG/AGE_ESTIMATION/CVPR_AGE_5_points/VALID/", transform)
        # age train set images: 3707; test image: 1356

    else:
        print("working machine should be  [thinkstation, Narvi]")
        NotImplementedError
        



    resampling = False

    if resampling == False:
        age_train_loader = torch.utils.data.DataLoader(age_train_dataset, batch_size=args.batch_size, 
                                                        shuffle=True, num_workers=args.loading_jobs)

        age_test_loader = torch.utils.data.DataLoader(age_test_dataset, batch_size=args.batch_size,
                                                        shuffle=True, num_workers=args.loading_jobs)

    else:
        age_train_loader = torch.utils.data.DataLoader(age_train_dataset, batch_size=args.batch_size, 
                                                        shuffle=False, num_workers=args.loading_jobs, 
                                                        sampler=dataset_augmentation_sampler(age_train_dataset, 28710))

        age_test_loader = torch.utils.data.DataLoader(age_test_dataset, batch_size=args.batch_size,
                                                        shuffle=False, num_workers=args.loading_jobs,
                                                        sampler=dataset_augmentation_sampler(age_test_dataset, 3590))


    show_sample = False

    if show_sample == True:
        sample_loader = torch.utils.data.DataLoader(age_train_dataset, 
                                                    batch_size=36, 
                                                    shuffle=True, 
                                                    num_workers=args.loading_jobs)

        show_sample_times = 0
        show_sampel_max_times = 5
        
        data_iter = iter(sample_loader)
        for images, labels in data_iter:
            X = images.numpy()
            X = np.transpose(X, [0, 2, 3, 1])
            plot_images(X, labels, "Cha_Learn_2016_dataset")

            show_sample_times = show_sample_times + 1
            if show_sample_times > show_sampel_max_times:
                break
            else:
                continue

    # model_dataloader
    return [age_train_loader, age_test_loader]


# def merge_multi_data_loader():
#     return 0


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




