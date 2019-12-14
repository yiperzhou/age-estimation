# age_estimation

<!-- TOC -->

- [age_estimation](#ageestimation)
  - [Git Network graph](#git-network-graph)
  - [datasets](#datasets)
  - [face detection and alignment](#face-detection-and-alignment)
    - [face detection](#face-detection)
    - [face alignment](#face-alignment)
  - [model](#model)
  - [experiments](#experiments)
    - [environment](#environment)
    - [the STOA on age estimation](#the-stoa-on-age-estimation)
  - [references](#references)

<!-- /TOC -->


## Git Network graph

```sh
age-estimation
        |----------- classification_combination 
        |----------- regression_classification_combination
        |----------- label_smoothing  
```


## datasets

1. [cleaned ChaLEARN CVPR 2016](http://chalearnlap.cvc.uab.es/dataset/19/description/) 
<!-- 2. [IMDB-WIKI – 500k+ face images with age and gender labels](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) -->
- [ ] [AgeDB: the first manually collected, in-the-wild age database](https://ibug.doc.ic.ac.uk/resources/agedb/)


## face detection and alignment

- using Yue's processed images on ChaLearn CVPR 2016 dataset

### face detection
- [MTCNN - Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://github.com/TropComplique/mtcnn-pytorch)
- [ ] check [Tiny Face Detector](https://github.com/peiyunh/tiny)

### face alignment
- Dlib face alignment method is applied.
- [ ] find out which exact algorithm it is.
- other related resource
 [2D and 3D Face alignment library build using pytorch](https://github.com/1adrianb/face-alignment)
 

## model
 
 
## experiments


1. **classification_combination** branch experimental result, same, **classification_combination** page  
2. **regression_classification_combination** branch  
3. **label_smoothing** branch experimental result  




### environment

  - virtual environment: `pytorch`

```sh
# run experiments for the age estimation
$ python main.py  
```


### the STOA on age estimation

- from the [BridgeNet](https://arxiv.org/abs/1904.03358)

## references

1. the current state of the art approach, 
   1. [Li, W., Lu, J., Feng, J., Xu, C., Zhou, J., & Tian, Q. (2019). BridgeNet: A Continuity-Aware Probabilistic Network for Age Estimation. ArXiv, abs/1904.03358](https://arxiv.org/abs/1904.03358), [CVPR 2019](http://cvpr2019.thecvf.com/)
2. the demo paper for instrucing the writing process
   1. [SAF- BAGE: Salient Approach for Facial Soft-Biometric Classification - Age, Gender, and Facial Expression](https://arxiv.org/abs/1803.05719)  
   2. [WACV 2019](https://wacv19.wacv.net).  
3. the similar idea from the head pose estimation, 
   1. [Fine-Grained Head Pose Estimation Without Keypoints](https://arxiv.org/abs/1710.00925)  
   2. [GitHub repository](https://github.com/natanielruiz/deep-head-pose)  
   3. [Computer Vision and Pattern Recognition Workshops (CVPRW) 2018](http://cvpr2018.thecvf.com/program/workshops)  
