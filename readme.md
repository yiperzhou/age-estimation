# age_estimation_multi_loss, branch - 

multitask learning on face analysis

### experimental environment

tools                              |  value
---------------------------------- | -----------------------------
OS                                 |  Ubuntu 18.04 (TUT thinkstation)    
virtual environment                | Python Virtual Environment: xx, concrete library, xxx. file



### example

<img src="https://raw.githubusercontent.com/yipersevere/MultitaskingFace/master/example/example_03.png" alt="age_gender_emotion_recognition_demo" width="700"/>


### run 

```
# run 3 multitask learning on age, gender, emotion
python main3_age_gender_emotion.py  
```

```
# run 2 multitask learning on age, gender
python main2_age_gender.py  
```

configuration file is in: config.ini  

### preprocess

#### dataset

tasks                                                 | datasets     
----------------------------------------------------- |-------------------------
age                                                   |  ChaLEARN CVPR 2016           
gender                                                |  IMDB                  
emotion                                               |  FER2013                  




### face detection and face alignment

1. detect face: MTCNN model;
2. perform a 2-D face alignment using the 5 facial points obtained by MTCNN;
3. crop and resize the image to 224 * 224 pixels.

### clasification

ResNet-18 with imagenet pretrained weights



### reference

1. age, gender, emotion, Age and gender prediction of multiple faces from an image using pytorch, https://github.com/adamzjk/Age-Gender-Pred 
2. ChaLearn Looking at People and Faces of the World 2016, https://competitions.codalab.org/competitions/7511#learn_the_details-evaluation 