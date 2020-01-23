import cv2
import os

import cv2
import dlib
import numpy as np
import pandas as pd
from helper import aligner_path
from helper import load_raw_chalearn_age_dataset

from mtcnn.mtcnn import MTCNN

# path="/media/yi/e7036176-287c-4b18-9609-9811b8e33769/MTL_FACE/TUT-live-age-estimator"

SAVE_PATH = "/home/yi/Narvi_yi_home/data/ChaLearn_CVPR_2016_age/preprocess"
FACE_ALIGNMENT_68 = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/MTL_FACE/TUT-live-age-estimator/recognizers/alignment/targets_symm.txt"
ALIGNER_TARGETS_PATH = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/MTL_FACE/TUT-live-age-estimator/recognizers/alignment/targets_symm_224_5_landmarks.txt"

timestamp = datetime.datetime.now()
ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')


aligner_targets = np.loadtxt(ALIGNER_TARGETS_PATH).astype(np.float32).reshape(1,5,2)
target_img_size = (224, 224)

aligner = dlib.shape_predictor(aligner_path)
detector = MTCNN()


def face_detect_by_MTCNN(img):
    """
    face detection using MTCNN model
    """
    detected_result = detector.detect_faces(img)

    return detected_result


def face_alignment(img):
    """
    afffine transformation
    """
    landmarks = np.array(landmarks).astype(np.float32).reshape(1,5,2)
    # aligner_targets =aligner_targets.reshape(1,5,2)
    # dlib_box = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)
    
    # dlib_img = img_copy[..., ::-1].astype(np.uint8)

    # s = aligner(dlib_img, dlib_box)
    # landmarks = [[s.part(k).x, s.part(k).y] for k in range(s.num_parts)]
    # landmarks = np.array(landmarks)

    # M,R = cv2.estimateRigidTransform(landmarks, aligner_targets)

    # landmarks = [   [landmarks[0][0], landmarks[0][5]], [landmarks[0][1], landmarks[0][6]], [landmarks[0][2], landmarks[0][7]], [landmarks[0][3], landmarks[0][8]], [landmarks[0][4], landmarks[0][9]]    ]
    M = cv2.estimateRigidTransform(landmarks, aligner_targets, True)

    # here, we first use 68 face landmark template instead of using MTCNN 5 landmarks
    img2 = img.copy()
    crop = cv2.warpAffine(img2, M, target_img_size)


def process_raw_image(csv_file, image_folder, phase="train"):

    dataframe = pd.read_csv(csv_file)

    print("phase: ", phase)

    for index, row in dataframe.iterrows():
        img = cv2.imread(image_folder + os.sep + row["image"])
        img_copy = cv2.imread(image_folder + os.sep + row["image"])

        detected_result = face_detect_by_MTCNN(img)

        aligned_result = face_alignment(detected_result)

        # save the image to folder
        

        return 




def chelearn_age_dataset_preprocess():
    train_folder, valid_folder, test_folder, csv_train_file, csv_valid_file, csv_test_file = load_raw_chalearn_age_dataset()


    train_image_processed = process_raw_image(csv_train_file, train_folder)
    valid_image_processed = process_raw_image(csv_valid_file, valid_folder)
    test_image_processed = process_raw_image(csv_test_file, test_folder)




    # train_pd = pd.read_csv(csv_train_file)
    # valid_pd = pd.read_csv(csv_valid_file)
    test_pd  = pd.read_csv(csv_test_file)







    # for index, train_row in train_pd.iterrows():
    #     if 3499 < index < 4200:
    #         print(train_row["image"])
    #         img = cv2.imread(train_folder + os.sep + train_row["image"])
    #         img_copy = cv2.imread(train_folder + os.sep + train_row["image"])
            
    #         # bbox[0],bbox[1],bbox[2],bbox[3], confidence, [landmark[0][0], landmark[0][1], ...]
    #         result = detector.detect_faces(img)
            
    #         if len(result) > 0:
    #             bounding_boxes = result[0]['box']
    #             landmarks = [list(result[0]['keypoints']["left_eye"]), list(result[0]['keypoints']["right_eye"]), list(result[0]['keypoints']["nose"]), list(result[0]['keypoints']["mouth_left"]), list(result[0]['keypoints']["mouth_right"])]
    #             x = int(bounding_boxes[0])
    #             y  = int(bounding_boxes[1])
    #             w  = int(bounding_boxes[2]-x)
    #             h  = int(bounding_boxes[3]-y)
    #             landmarks = np.array(landmarks).astype(np.float32).reshape(1,5,2)
    #             # aligner_targets =aligner_targets.reshape(1,5,2)
    #             # dlib_box = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)
                
    #             # dlib_img = img_copy[..., ::-1].astype(np.uint8)

    #             # s = aligner(dlib_img, dlib_box)
    #             # landmarks = [[s.part(k).x, s.part(k).y] for k in range(s.num_parts)]
    #             # landmarks = np.array(landmarks)

    #             M = cv2.estimateRigidTransform(landmarks, aligner_targets, True)
    #             # M,R = estimateRigidTransform(landmarks, aligner_targets)

    #             # here, we first use 68 face landmark template instead of using MTCNN 5 landmarks
    #             img2 = img.copy()
    #             crop = cv2.warpAffine(img2, M, target_img_size)
    #             # cv2.warpAffine(img2, transmat, (imgSize[1], imgSize[0]))
    #             # if R < 1.5:
    #             #     crop = cv2.warpAffine(img_copy, M, target_img_size, borderMode=2)
    #             # else:
    #             #     # Seems to distort too much, probably error in landmarks, then take center crop.
    #             #     # Take center crop and scale to 128x128
    #             #     w_crop, h_crop, d = img_copy.shape
                    
    #             #     if w_crop > h_crop:
    #             #         c = w_crop // 2
    #             #         x1 = c - h_crop // 2
    #             #         x2 = x1 + h_crop
    #             #         img_copy = img_copy[:, x1:x2, :]
    #             #     elif w_crop < h_crop:
    #             #         c = h_crop // 2
    #             #         y1 = c - w_crop // 2
    #             #         y2 = y1 + w_crop
    #             #         img_copy = img_copy[y1:y2, :, :]

    #             #     crop = cv2.resize(img_copy, target_img_size)
    #             # print("there are images with R < 2: ", train_row["image"])
    #             cv2.imwrite(SAVE_PATH + os.sep + "train" + os.sep + train_row["image"], crop)

    #             del M, crop, img2, landmarks, bounding_boxes

    #         else:
    #             w_crop, h_crop, d = img_copy.shape
                
    #             if w_crop > h_crop:
    #                 c = w_crop // 2
    #                 x1 = c - h_crop // 2
    #                 x2 = x1 + h_crop
    #                 img_copy = img_copy[:, x1:x2, :]
    #             elif w_crop < h_crop:
    #                 c = h_crop // 2
    #                 y1 = c - w_crop // 2
    #                 y2 = y1 + w_crop
    #                 img_copy = img_copy[y1:y2, :, :]

    #             crop = cv2.resize(img_copy, target_img_size)                

    #             print("there are images without detecting face: ", train_row["image"])
    #             cv2.imwrite(SAVE_PATH + os.sep + "train" + os.sep + train_row["image"], crop)

    #             del crop
                
    #         del result
    #     else:
    #         pass

    # print("train done")

    # for index, valid_row in valid_pd.iterrows():
    #     if 1599 < index < 2400:

    #         print(valid_folder + os.sep + valid_row["image"])
    #         # img = Image.open(train_folder + os.sep + train_row["image"])
    #         img = cv2.imread(valid_folder + os.sep + valid_row["image"])
    #         img_copy = cv2.imread(valid_folder + os.sep + valid_row["image"])
            
    #         # bbox[0],bbox[1],bbox[2],bbox[3], confidence, [landmark[0][0], landmark[0][1], ...]
    #         # bounding_boxes, landmarks = detect_faces(img)
    #         result = detector.detect_faces(img)
    #         # print(result)
    #         # print(result[0]['box'])
            
    #         if len(result) > 0:
    #             bounding_boxes = result[0]['box']
    #             landmarks = [list(result[0]['keypoints']["left_eye"]), list(result[0]['keypoints']["right_eye"]), list(result[0]['keypoints']["nose"]), list(result[0]['keypoints']["mouth_left"]), list(result[0]['keypoints']["mouth_right"])]
    #             x = int(bounding_boxes[0])
    #             y  = int(bounding_boxes[1])
    #             w  = int(bounding_boxes[2]-x)
    #             h  = int(bounding_boxes[3]-y)
    #             landmarks = np.array(landmarks).astype(np.float32).reshape(1,5,2)
    #             # aligner_targets =aligner_targets.reshape(1,5,2)
    #             # dlib_box = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)
                
    #             # dlib_img = img_copy[..., ::-1].astype(np.uint8)

    #             # s = aligner(dlib_img, dlib_box)
    #             # landmarks = [[s.part(k).x, s.part(k).y] for k in range(s.num_parts)]
    #             # landmarks = np.array(landmarks)

    #             # M,R = cv2.estimateRigidTransform(landmarks, aligner_targets)

    #             # landmarks = [   [landmarks[0][0], landmarks[0][5]], [landmarks[0][1], landmarks[0][6]], [landmarks[0][2], landmarks[0][7]], [landmarks[0][3], landmarks[0][8]], [landmarks[0][4], landmarks[0][9]]    ]
    #             M = cv2.estimateRigidTransform(landmarks, aligner_targets, True)
    #             # M,R = estimateRigidTransform(landmarks, aligner_targets)

    #             # here, we first use 68 face landmark template instead of using MTCNN 5 landmarks
    #             img2 = img.copy()
    #             crop = cv2.warpAffine(img2, M, target_img_size)
    #             # cv2.warpAffine(img2, transmat, (imgSize[1], imgSize[0]))
    #             # if R < 1.5:
    #             #     crop = cv2.warpAffine(img_copy, M, target_img_size, borderMode=2)
    #             # else:
    #             #     # Seems to distort too much, probably error in landmarks, then take center crop.
    #             #     # Take center crop and scale to 128x128
    #             #     w_crop, h_crop, d = img_copy.shape
                    
    #             #     if w_crop > h_crop:
    #             #         c = w_crop // 2
    #             #         x1 = c - h_crop // 2
    #             #         x2 = x1 + h_crop
    #             #         img_copy = img_copy[:, x1:x2, :]
    #             #     elif w_crop < h_crop:
    #             #         c = h_crop // 2
    #             #         y1 = c - w_crop // 2
    #             #         y2 = y1 + w_crop
    #             #         img_copy = img_copy[y1:y2, :, :]

    #             #     crop = cv2.resize(img_copy, target_img_size)
    #             # print("there are images with R < 2: ", train_row["image"])
    #             cv2.imwrite(SAVE_PATH + os.sep + "valid" + os.sep + valid_row["image"], crop)

    #             del M, crop, img2, landmarks, bounding_boxes

    #         else:
    #             w_crop, h_crop, d = img_copy.shape
                
    #             if w_crop > h_crop:
    #                 c = w_crop // 2
    #                 x1 = c - h_crop // 2
    #                 x2 = x1 + h_crop
    #                 img_copy = img_copy[:, x1:x2, :]
    #             elif w_crop < h_crop:
    #                 c = h_crop // 2
    #                 y1 = c - w_crop // 2
    #                 y2 = y1 + w_crop
    #                 img_copy = img_copy[y1:y2, :, :]

    #             crop = cv2.resize(img_copy, target_img_size)                

    #             print("there are images without detecting face: ", valid_row["image"])
    #             cv2.imwrite(SAVE_PATH + os.sep + "valid" + os.sep + valid_row["image"], crop)
                
    #             del crop
    #     else:
    #         pass

    # print("validation done")

    for index, test_row in test_pd.iterrows():
        if index < 800:

            print(test_folder + os.sep + test_row["image"])
            # img = Image.open(train_folder + os.sep + train_row["image"])
            img = cv2.imread(test_folder + os.sep + test_row["image"])
            img_copy = cv2.imread(test_folder + os.sep + test_row["image"])
            
            # bbox[0],bbox[1],bbox[2],bbox[3], confidence, [landmark[0][0], landmark[0][1], ...]
            # bounding_boxes, landmarks = detect_faces(img)
            result = detector.detect_faces(img)
            # print(result)
            # print(result[0]['box'])
            
            if len(result) > 0:
                bounding_boxes = result[0]['box']
                landmarks = [list(result[0]['keypoints']["left_eye"]), list(result[0]['keypoints']["right_eye"]), list(result[0]['keypoints']["nose"]), list(result[0]['keypoints']["mouth_left"]), list(result[0]['keypoints']["mouth_right"])]
                x = int(bounding_boxes[0])
                y  = int(bounding_boxes[1])
                w  = int(bounding_boxes[2]-x)
                h  = int(bounding_boxes[3]-y)
                landmarks = np.array(landmarks).astype(np.float32).reshape(1,5,2)
                # aligner_targets =aligner_targets.reshape(1,5,2)
                # dlib_box = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)
                
                # dlib_img = img_copy[..., ::-1].astype(np.uint8)

                # s = aligner(dlib_img, dlib_box)
                # landmarks = [[s.part(k).x, s.part(k).y] for k in range(s.num_parts)]
                # landmarks = np.array(landmarks)

                # M,R = cv2.estimateRigidTransform(landmarks, aligner_targets)

                # landmarks = [   [landmarks[0][0], landmarks[0][5]], [landmarks[0][1], landmarks[0][6]], [landmarks[0][2], landmarks[0][7]], [landmarks[0][3], landmarks[0][8]], [landmarks[0][4], landmarks[0][9]]    ]
                M = cv2.estimateRigidTransform(landmarks, aligner_targets, True)

                # here, we first use 68 face landmark template instead of using MTCNN 5 landmarks
                img2 = img.copy()
                crop = cv2.warpAffine(img2, M, target_img_size)
                # cv2.warpAffine(img2, transmat, (imgSize[1], imgSize[0]))
                # if R < 1.5:
                #     crop = cv2.warpAffine(img_copy, M, target_img_size, borderMode=2)
                # else:
                #     # Seems to distort too much, probably error in landmarks, then take center crop.
                #     # Take center crop and scale to 128x128
                #     w_crop, h_crop, d = img_copy.shape
                    
                #     if w_crop > h_crop:
                #         c = w_crop // 2
                #         x1 = c - h_crop // 2
                #         x2 = x1 + h_crop
                #         img_copy = img_copy[:, x1:x2, :]
                #     elif w_crop < h_crop:
                #         c = h_crop // 2
                #         y1 = c - w_crop // 2
                #         y2 = y1 + w_crop
                #         img_copy = img_copy[y1:y2, :, :]

                #     crop = cv2.resize(img_copy, target_img_size)
                # print("there are images with R < 2: ", train_row["image"])
                cv2.imwrite(SAVE_PATH + os.sep + "test" + os.sep + test_row["image"], crop)

                del M, crop, img2, landmarks, bounding_boxes

            else:
                w_crop, h_crop, d = img_copy.shape
                
                if w_crop > h_crop:
                    c = w_crop // 2
                    x1 = c - h_crop // 2
                    x2 = x1 + h_crop
                    img_copy = img_copy[:, x1:x2, :]
                elif w_crop < h_crop:
                    c = h_crop // 2
                    y1 = c - w_crop // 2
                    y2 = y1 + w_crop
                    img_copy = img_copy[y1:y2, :, :]

                crop = cv2.resize(img_copy, target_img_size)                

                print("there are images without detecting face: ", test_row["image"])
                cv2.imwrite(SAVE_PATH + os.sep + "test" + os.sep + test_row["image"], crop)
                
                del crop
        else:
            pass

    print("test done")


if __name__ == "__main__":
    print("including face detection, face alignment")
    chelearn_age_dataset_preprocess()



