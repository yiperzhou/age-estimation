# Multi-loss function for age estimation

### branch

1. multiple_losses_on_age_estimation  
2.   

<!-- ## example

![Example](./master/example/example_03.png) -->


### to do task

[] clean the multitask learning source code to multi-loss age estimation task, 
        1. set the flag of different idea as the flag in the main file,
        2. for multi-task learning as one branch,
        3. for multi-loss learning as one branch. 
[x] reference ni xingyang's repository.


### run experiments

#### run multi-loss experiments on age estimation 

python main.py  

####  configuration file

config.ini  

#### age dataset

age dataset: cleaned ChaLEARN CVPR 2016  

#### experimental result

                  model |                     age MAE 
----------------------- | -------------------------------------
AlexNet                 |                                      
MobileNet-V1            |                                      
                  


#### face detection and face alignment

using Yue's processed images on ChaLearn CVPR 2016 dataset.




## reference

1. ChaLearn Looking at People and Faces of the World 2016, https://competitions.codalab.org/competitions/7511#learn_the_details-evaluation 