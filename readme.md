## Multi-loss function for age estimation

### the gitflow of this repository

```bash
|--- master
       |----------- label_smoothing
       |----------- regression_loss_combination 
    
% label_smoothing branch
git checkout label_smoothing

% regression_loss_combination branch
git checkout regression_loss_combination
```
    

### task
- [ ] create new branch - regression_loss_combination, to experiments all possible experiments, $2^{4}-1$
- [ ] reply Yanlin that I am interested in the barcode detection project, the [news here](https://businesstampere.com/scandit-opens-rd-office-in-tampere-to-strengthen-leadership-in-mobile-computer-vision-and-augmented-reality/)
- [ ] solve the Gaussian loss calculation in the age classification branch. 
- [ ] use the notepad application, which intergates markdown, to do list in one tool. I remember that I have browsed the website.
- [x] experimented the **label smoothing** idea, it improves.
- [x] ~~Gaussian Loss function does not work, no gradient problem~~
- [x] ~~clean the multitask learning source code to multi-loss age estimation task,~~  
- [x] ~~reference ni xingyang's repository.~~


****
### contents
* [task](#task)
* [experiments](#experiments)
* [age dataset](#age-dataset)
* [face detection and alignment](#face-detection-and-alignment)
* [reference and other materials](#reference-and-other-materials)
****


#### other materials

* pdf report, [overleaf](https://www.overleaf.com/project/5d2310338e2b2d7e89e37358)
* [Multitask learning on Google Drive](https://drive.google.com/drive/folders/1JSRQxQfCnNyKONFnrRL7D_sDituPLR73?usp=sharing), [Multiloss on age estimation loss](https://drive.google.com/drive/folders/1BNY4DsRx3oGBibo3Xi8oLNVaaYFMRAjl?usp=sharing)

#### run experiments for the age estimation 

```sh
$ python main.py  
```

#### age dataset

1. [cleaned ChaLEARN CVPR 2016](http://chalearnlap.cvc.uab.es/dataset/19/description/) 
2. [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
3. [AgeDB](https://ibug.doc.ic.ac.uk/resources/agedb/)


#### face detection and alignment

* using Yue's processed images on ChaLearn CVPR 2016 dataset; 
* - [ ] elaborate the method of the face detection and face alignment, currently I can not remember.



#### The state of the art result on age estimation from the [BridgeNet](https://arxiv.org/abs/1904.03358) paper

![Example](related_materials/state-of-the-art-result-age-estimation-on-chalearn-2016.png)



## reference and other materials

1. the current state of the art approach, [BridgeNet](https://arxiv.org/abs/1904.03358) CVPR 2019
2. the demo paper for writing, [SAF- BAGE](https://arxiv.org/abs/1803.05719), it was accepted by WACV 2019.
3. the similar idea as the head pose estimation, [hopenet](https://arxiv.org/abs/1710.00925), the GitHub repository is [here](https://github.com/natanielruiz/deep-head-pose)
