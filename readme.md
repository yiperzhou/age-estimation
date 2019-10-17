# Age Estimation

<!-- TOC -->

- [Age Estimation](#age-estimation)
  - [Git Network graph](#git-network-graph)
  - [datasets](#datasets)
  - [face detection and alignment](#face-detection-and-alignment)
    - [face detection](#face-detection)
    - [face alignment](#face-alignment)
  - [experiments](#experiments)
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
        |----------- regression_combination  
```

- [manuscript in Overleaf](https://www.overleaf.com/project/5d2310338e2b2d7e89e37358)


## datasets

1. [cleaned ChaLEARN CVPR 2016](http://chalearnlap.cvc.uab.es/dataset/19/description/) 
2. [IMDB-WIKI – 500k+ face images with age and gender labels](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
3. [AgeDB: the first manually collected, in-the-wild age database](https://ibug.doc.ic.ac.uk/resources/agedb/)


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
 
## experiments

1. **label_smoothing** branch experimental result
  
   [TUNI Onedrive, excel table, label_smoothing page](https://tuni-my.sharepoint.com/:x:/r/personal/yi_zhou_tuni_fi/_layouts/15/Doc.aspx?sourcedoc=%7B0FAA15DB-6F0E-4794-8F72-F58B4E6E970A%7D&file=experimental%20result%20of%20the%20combination%20on%20different%20classification%20losses.xlsx&action=default&mobileredirect=true)
   
2. **classification_combination** branch experimental result
  
   same, **classification_combination** page
   
3. **regression_combination** branch experimental result
  
  same, **regression_combination** page

### environment

- ThinkStation
  - virtual environment: `pytorch`
- Narvi
  - virtual environment: `dl`

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
4. [Python 风格指南 - 内容目录](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/contents/)
