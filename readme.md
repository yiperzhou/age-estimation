# Multi-loss function for age estimation


## example

![Example](./master/example/example_03.png)


### run experiments

#### run multi-loss on age estimation 

python main3_age_gender_emotion.py  


configuration file is in: config.ini  

#### age dataset

age dataset: cleaned ChaLEARN CVPR 2016  

#### experimental result

                  model |                     age MAE 
----------------------- | -------------------------------------
AlexNet                 |                                      
MobileNet-V1            |                                      
                  


#### face detection and face alignment

using Yue's processed images on ChaLearn CVPR 2016 dataset.

#### clasification

AlexNet model as the backbone




## reference
1. ChaLearn Looking at People and Faces of the World 2016, https://competitions.codalab.org/competitions/7511#learn_the_details-evaluation 