import dlib
import os

import dlib
import imutils
import numpy as np
from PIL import Image
from imutils.face_utils import FaceAligner as FA
from imutils.face_utils import rect_to_bb

path="/media/yi/e7036176-287c-4b18-9609-9811b8e33769/MTL_FACE/MultitaskingFace"
os.chdir(path)

from config import config, parser
from mtcnn.src import detect_faces

aligner_targets_path = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/MTL_FACE/TUT-live-age-estimator/recognizers/alignment/targets_symm.2.txt"
aligner_targets = np.loadtxt(aligner_targets_path)

class FaceAligner:

  def __init__(self):
    self.desiredFaceWidth = int(parser['DATA']['aligned_out_size']) # output size
    self.face_threshold   = float(parser['DATA']['face_threshold'])
    self.expand_margin    = float(parser['DATA']['expand_margin'])

    self.Path2ShapePred   = config.model + "shape_predictor_68_face_landmarks.dat"
    # self.Path2Detecor     = config.model + "mmod_human_face_detector.dat"

    self.detector         = dlib.get_frontal_face_detector()
    
    self.predictor        = dlib.shape_predictor(self.Path2ShapePred)
    self.fa               = FA(self.predictor, desiredFaceWidth=self.desiredFaceWidth)

  def getAligns(self,
                img,
                savepath = None,
                return_info = False):
    """
    get face alignment picture
    :param img: original BGR image or a path to it
    :param savepath: savepath, format "xx/xx/xx.png"
    :param return_info: if set, return face positinos [(x, y, w, h)] 
    :return: aligned faces, (opt) rects
    """
    if type(img) == str:
      img = cv2.imread(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    unsample = 1 if img.shape[0] * img.shape[1] < 512 * 512 else 0

  # 这里我应该使用image pil 来读取，否则的话这个函数没有用
    dets, landmarks = detect_faces(img)
    confidence = [d[4] for d in dets if d[4] > self.face_threshold]
    # filter face detection result according to confidence
    rects = [d[:4] for d in dets if d[4] > self.face_threshold]
    landmarks = [   [landmarks[0][0], landmarks[0][5]], [landmarks[0][1], landmarks[0][6]], [landmarks[0][2], landmarks[0][7]], [landmarks[0][3], landmarks[0][8]], [landmarks[0][4], landmarks[0][9]]    ]
    M,R = cv2.estimateRigidTransform(landmarks, aligner_targets)
    
    # expand rects by some margin
    exp_rects = []
    for rect in rects:
      x, y, w, h = rect_to_bb(rect[:4])

      # make sure bounds are within the image
      x = max(0, x)
      y = max(0, y)
      w = min(img.shape[1] - x, w)
      h = min(img.shape[0] - y, h)

      exp = min(int(w * self.expand_margin), x, img.shape[1] - x - w,
                int(h * self.expand_margin), y, img.shape[0] - y - h)
      exp = max(0, exp)

      x, y = x - exp, y - exp
      w, h = w + 2 * exp, h + 2 * exp

      exp_rects.append(dlib.rectangle(x, y, x + w, y + h))

    aligned = [self.fa.align(img, gray, rect) for rect in exp_rects]

    if savepath:
      if len(aligned) == 1:
        cv2.imwrite(savepath, aligned)
      else:
        for i, al in enumerate(aligned):
          cv2.imwrite("{}_{}.{}".format(savepath[:-4], i, savepath[-3:]), aligned)

    if return_info:
      return aligned, [rect_to_bb(rect) for rect in exp_rects], scores
    return aligned # BGR faces, cv2.imshow("Aligned", faceAligned)


  def example(self):
    # image = cv2.imread("/media/yi/e7036176-287c-4b18-9609-9811b8e33769/MTL_FACE/MultitaskingFace/example/example_02.png")
    # image = imutils.resize(image, width=self.resizeWidth)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale image
    # cv2.imshow("Input", image)
    # rects = self.detector(gray, 2)
    img = Image.open("/media/yi/e7036176-287c-4b18-9609-9811b8e33769/MTL_FACE/MultitaskingFace/example/example_02.png")
    
    rects, landmarks = detect_faces(img)

    # loop over the face detections
    for rect in rects:
      # extract the ROI of the *original* face, then align the face
      # using facial landmarks
      (x, y, w, h) = rect_to_bb(rect[:4])
      faceOrig = imutils.resize(image[y:y + h, x:x + w], width=self.desiredFaceWidth)
      faceAligned = self.fa.align(image, gray, rect)

      import uuid
      f = str(uuid.uuid4())
      cv2.imwrite("foo/" + f + ".png", faceAligned)

      # display the output images
      cv2.imshow("Original", faceOrig)
      cv2.imshow("Aligned", faceAligned)
      cv2.waitKey(0)


if __name__ == "__main__":
  # ts = FaceAligner()
  # ts.example()
  # pass

  from mtcnn.mtcnn import MTCNN
  import cv2
  img = cv2.imread("./data/94b47c67c779945fda0ff8fc87f1c545.jpg")

  detector = MTCNN()

  img_detect_result = detector.detect_faces(img)

  x = img_detect_result[0]["box"][0]
  y = img_detect_result[0]["box"][1]
  w = img_detect_result[0]["box"][2]
  h = img_detect_result[0]["box"][3]

  crop_img = img[y:y+h, x:x+w]

  cv2.imshow('crop_img', crop_img)


  #cv2.imshow('image',img)
  #cv2.waitKey(0)

  print(img_detect_result)  




