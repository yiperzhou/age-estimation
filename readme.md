# MultitaskingFace

multitask learning on face analysis

# Environment
Ubuntu 18.04(TUT thinkstation)    
Python Virtual Environment: xx, concrete library, xxx. file


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
1. age, gender, emotion, Age and gender prediction of multiple faces from an image using pytorch, https://github.com/adamzjk/Age-Gender-Pred

