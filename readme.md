# Age Estimation

<!-- TOC -->

- [Age Estimation](#age-estimation)
  - [Git Network graph](#git-network-graph)
  - [to do](#to-do)
  - [Writing in Overleaf](#writing-in-overleaf)
  - [datasets](#datasets)
  - [face detection and alignment](#face-detection-and-alignment)
    - [face detection](#face-detection)
    - [face alignment](#face-alignment)
  - [experiments](#experiments)
    - [results](#results)
    - [environment](#environment)
    - [the STOA on age estimation](#the-stoa-on-age-estimation)
  - [references](#references)

<!-- /TOC -->


## Git Network graph

```sh
age-estimation
        |----------- master
        |----------- label_smoothing
        |----------- classification_combination 
        |----------- multiple_losses_on_age_estimation
```



## to do
- [ ] test function, 
$l_{n}=-w_{n}\left[t_{n} \cdot \log \sigma\left(x_{n}\right)+\left(1-t_{n}\right) \cdot \log \left(1-\sigma\left(x_{n}\right)\right)\right]$
- [ ] elaborate the method of the face detection and face alignment, currently I can not remember.
- [ ] continue implementing the bar chart drawing function, the reference link is [here](https://pythonspot.com/matplotlib-bar-chart/)
- [ ] implement epsilon $\epsilon$ error function
- [ ] ~~learn LaTex~~
- [x] create new branch - regression_loss_combination, to experiments all possible experiments, $15 = 2^{4}-1$
- [x] ~~reply Yanlin that I am interested in the barcode detection project, the [news here](https://businesstampere.com/scandit-opens-rd-office-in-tampere-to-strengthen-leadership-in-mobile-computer-vision-and-augmented-reality/)~~
- [ ] **solve the Gaussian loss calculation in the age classification branch**
- [ ] ~~use the notion application, which intergates markdown, to do list in one tool. I remember that I have browsed the website.~~
- [x] experimented the **label smoothing** idea, it improves.
- [x] ~~Gaussian Loss function does not work, no gradient problem~~
- [x] ~~clean the multitask learning source code to multi-loss age estimation task,~~  
- [x] ~~reference ni xingyang's repository.~~


## Writing in Overleaf

- [manuscript in Overleaf](https://www.overleaf.com/project/5d2310338e2b2d7e89e37358)


## datasets

1. [cleaned ChaLEARN CVPR 2016](http://chalearnlap.cvc.uab.es/dataset/19/description/) 
2. [IMDB-WIKI – 500k+ face images with age and gender labels](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
3. [AgeDB: the first manually collected, in-the-wild age database](https://ibug.doc.ic.ac.uk/resources/agedb/)


## face detection and alignment

using Yue's processed images on ChaLearn CVPR 2016 dataset

### face detection
- [MTCNN - Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://github.com/TropComplique/mtcnn-pytorch)
- [ ] check [Tiny Face Detector](https://github.com/peiyunh/tiny)

### face alignment
- Dlib face alignment method is applied.
- [ ] find out which exact algorithm it is.
- 

## experiments

### results

- Original experimental result in Google Drive, [Multitask learning](https://drive.google.com/drive/folders/1JSRQxQfCnNyKONFnrRL7D_sDituPLR73?usp=sharing), [Multiloss on age estimation loss](https://drive.google.com/drive/folders/1BNY4DsRx3oGBibo3Xi8oLNVaaYFMRAjl?usp=sharing) 

### environment
- ThinkStation
  - virtual environment: `pytorch`
  - PyTorch 1.1.0
- Narvi
  - virtual environment: `dl`
  - PyTorch 1.0.1

```sh
# run experiments for the age estimation
$ python main.py  
```


### the STOA on age estimation

- from the [BridgeNet](https://arxiv.org/abs/1904.03358) paper
- ![Example](related_materials/state-of-the-art-result-age-estimation-on-chalearn-2016.png)

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
