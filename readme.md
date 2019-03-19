# MultitaskingFace

multitask learning on face analysis

# Environment
Ubuntu 18.04(TUT thinkstation)    
Python Virtual Environment: xx, concrete library, xxx. file

## export virtual environment
conda env export > tut_thinkstation_caffe.yml  

## import virtual environment

# example

![Example](https://raw.githubusercontent.com/yipersevere/MultitaskingFace/master/example/example_03.png)


# run 
## run 3 multitask learning on age, gender, emotion
python main3_age_gender_emotion.py  

## run 2 multitask learning on age, gender
python main2_age_gender.py  


configuration file is in: config.ini  

# preprocess
## dataset
emotion dataset: FER2013  
age dataset: ChaLEARN CVPR 2016  
gender dataset: IMDB  

## face detection and face alignment
use MTCNN to do face detection,   
then perform a 2-D face alignment using the 5 facial points obtained by MTCNN;   
cropped and resize to 224 * 224 pixels  

## clasification
ResNet-18 with imagenet pretrained weights



## reference
1. hyperface, link, https://github.com/shashanktyagi/HyperFace-TensorFlow-implementation; 
2. TUT-live-age-estimator, link, https://github.com/mahehu/TUT-live-age-estimator;
3. Recursive Spatial Transformer (ReST) for Alignment-Free Face Recognition, http://vipl.ict.ac.cn/uploadfile/upload/2017122111505619.pdf
4. Spatial Transformer Networks, https://arxiv.org/abs/1506.02025
5. An All-In-One Convolutional Neural Network for Face Analysis, https://arxiv.org/abs/1611.00851
6. Zhang, Kaipeng, et al. "Joint face detection and alignment using multitask cascaded convolutional networks." IEEE Signal Processing Letters 23.10 (2016): 1499-1503.

