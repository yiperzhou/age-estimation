## Multi-loss function for age estimation

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

1. [cleaned ChaLEARN CVPR 2016](http://chalearnlap.cvc.uab.es/dataset/19/description/)           
2. [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)  

<!--
#### experimental result

               model    |    age MAE 
----------------------- | -------------------  
AlexNet                 |                                        
MobileNet-V1            |                                        
                  
 -->

#### face detection and face alignment

using Yue's processed images on ChaLearn CVPR 2016 dataset.


#### The state of the art result on age estimation from the [BridgeNet](https://arxiv.org/abs/1904.03358) paper

![Example](related_materials/state-of-the-art-result-age-estimation-on-chalearn-2016.png)



## reference

1. the current state of the art approach, [BridgeNet](https://arxiv.org/abs/1904.03358) CVPR 2019
2. the demo paper for writing, [SAF- BAGE](https://arxiv.org/abs/1803.05719), it was accepted by WACV 2019.