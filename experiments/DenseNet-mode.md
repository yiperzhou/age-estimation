# common 

## dataset
1. IMDB + WIKI, 153004 for train and 8829 for test



# pretrained

## experiment result

1. pretraining model 
2. IMDB_WIKI

model                                   | Age MAE (IMDB_WIKI) | Gender Acc (IMDB_WIKI)
----------------------------------------| ------------------- | ----------------------


reference models                        |                     |                       
SSR-Net[3]                              | 

### referecne


model                                            | Age MAE (ChaLearn_2016) | Gender Acc(Chalearn_2016) | Emotion Acc (FER_2013) |               weight               | initial lr    | learning method 
------------------------------------------------ |------------------------ | ------------------------- | ---------------------- |----------------------------------- | --------------| -----------------                   
model                                            |        Age  MAE         | Gender Acc (%)            |    Emotion  Acc (%)    |Smile Acc (%)    |     weight      | initial lr # common 

## dataset
1. IMDB + WIKI, 153004 for train and 8829 for test

# pretrained

## experiment result

1. pretraining model 
2. IMDB_WIKI

model                                   | Age MAE (IMDB_WIKI) | Gender Acc (IMDB_WIKI)
----------------------------------------| ------------------- | ----------------------

----------------------------------------| ------------------- | ----------------------
reference models                        |                     |                       
SSR-Net[3]                              | 

### referecne
[1] Pretrained-ResNet-18-Age-Gender, IMDB_WIKI train data, initial_lr=0.01, weight=_gender_1_age_1_IMDB_WIKI,
[2] Pretrained_MTL_AlexNet_model, IMDB_WIKI train data, initial_lr=0.01, weight=_gender_1_age_1_IMDB_WIKI,
[3] SSR-Net, 


# trained
1. load pretrained model
2. load the pretrained model, load the CVPR Age, Gender dataset, FER 2013 emotion dataset, sampling the age, gender images from around 4500 to 28710, 
3. set gender, emotion weight as 0, use the same data, 
4. set age emotion loss weight as 0, use the same data,
5. set age, gender weight as 0, use the same data
4. CVPR_ChaLearn_2016


model                                            | Age MAE (ChaLearn_2016) | Gender Acc(Chalearn_2016) | Emotion Acc (FER_2013) |               weight               | initial lr    | learning method 
------------------------------------------------ |------------------------ | ------------------------- | ---------------------- |----------------------------------- | --------------| -----------------                   
DenseNet-121-Age-Gender_Emotion_Smile            | 15.71                   |                           |  41.82                 | 83.43           |  1, 1, 1, 1     |    0.01          
DenseNet-121-Age-Gender_Emotion_Smile            | 15.46                   |  76.46                    |  33.80                 | 83.98           |  1, 1, 1, 1     |    0.01          
MTL_DenseNet_121_model                           | 15.83                   |     xx                    |                        | _gender_0_age_1_IMDB_WIKI          |   age_cls_loss, gen_loss    

-------------------------------------------------| ----------------------- | ------------------------- | ---------------------- |---------------- | --------------- | ---------------            
 reference models                                |                         |                           |                        |                 |               |                                 
DEX model                                        |                         |                           |                        |                 |               |                                 



### reference






-  [ ] check CVPR 2016 我们生成的测试数据集使用其他的模型测试时的 age mae, gender acc 分别是多少
* 使用 Age-gender-pred,的训练了3 epoch的模型然后再修改 last fully connected layer ,再在 cvpr 2016 数据集上做训练和测试，运行了 12　ｅｐｏｃｈ的结果如下：　说明也是 age 12.48; gender 87%


[25.03.2019 20:56:57] lr : 0.01
[25.03.2019 20:56:57] Train epoch    12:
[25.03.2019 20:56:57]           gender_loss: 0.004286306880591927
[25.03.2019 20:56:57]           age_loss: 4.351883343876301
[25.03.2019 20:56:57]           emotion_loss: 0
[25.03.2019 20:56:57] 

[25.03.2019 20:56:57]           gender_acc: 99.83629397422501
[25.03.2019 20:56:57]           emotion_acc: 0
[25.03.2019 20:59:29] [Valid] age mae: 12.487859796083074
[25.03.2019 20:57:06]           val_gender_loss: 0.7626899842100223
[25.03.2019 20:57:06]           val_age_loss: 4.574302972458863
[25.03.2019 20:57:06]           val_emotion_loss: 0
[25.03.2019 20:57:06] 

[25.03.2019 20:57:06]           val_gender_acc: 87.21448467116502
[25.03.2019 20:57:06]           val_emotion_acc: 0
[25.03.2019 20:57:06] ---------------


use the 30 epochs pretrained model

[26.03.2019 08:53:39]           val_gender_acc: 83.59331475473049
[26.03.2019 08:53:39] [Valid] age mae: 11.789214832337786

* Age_gender_pred Github repository web page, declare, 94% acc for gender and MAE of 4.2 for age can be achieved after just 32 epochs of training for FGNet dataset

但是我这里实现的时候却是 age mae, 9.856, gender acc, 88.798. 



[references]
[1]. SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation
-------------------------------------------------| ----------------------- | ------------------------- | ---------------------- |---------------- | --------------- | ---------------            
DenseNet-121-Age-Gender_Emotion_Smile            | 15.71                   |                           |  41.82                 | 83.43           |  1, 1, 1, 1     |    0.01          
DenseNet-121-Age-Gender_Emotion_Smile            | 15.46                   |  76.46                    |  33.80                 | 83.98           |  1, 1, 1, 1     |    0.01          
MTL_DenseNet_121_model                           | 15.83                   |     xx                    |                        | _gender_0_age_1_IMDB_WIKI          |   age_cls_loss, gen_loss    

-------------------------------------------------| ----------------------- | ------------------------- | ---------------------- |---------------- | --------------- | ---------------            
 reference models                                |                         |                           |                        |                 |               |                                 
DEX model                                        |                         |                           |                        |                 |               |                                 



### reference
[1] AlexNet-model, train_valid_1, pretrained_IMDB_WIKI_weight, _gender_1_smile_1_emotion_1_age_1, 
[2] AlexNet-model, train_valid_2, pretrained_IMDB_WIKI_weight, _gender_1_smile_1_emotion_1_age_1, 0.001, CVPR_16_ChaLearn
[3] ResNet-18-model, train_valid_2, pretrained_IMDB_WIKI_weight






-  [ ] check CVPR 2016 我们生成的测试数据集使用其他的模型测试时的 age mae, gender acc 分别是多少
* 使用 Age-gender-pred,的训练了3 epoch的模型然后再修改 last fully connected layer ,再在 cvpr 2016 数据集上做训练和测试，运行了 12　ｅｐｏｃｈ的结果如下：　说明也是 age 12.48; gender 87%


[25.03.2019 20:56:57] lr : 0.01
[25.03.2019 20:56:57] Train epoch    12:
[25.03.2019 20:56:57]           gender_loss: 0.004286306880591927
[25.03.2019 20:56:57]           age_loss: 4.351883343876301
[25.03.2019 20:56:57]           emotion_loss: 0
[25.03.2019 20:56:57] 

[25.03.2019 20:56:57]           gender_acc: 99.83629397422501
[25.03.2019 20:56:57]           emotion_acc: 0
[25.03.2019 20:59:29] [Valid] age mae: 12.487859796083074
[25.03.2019 20:57:06]           val_gender_loss: 0.7626899842100223
[25.03.2019 20:57:06]           val_age_loss: 4.574302972458863
[25.03.2019 20:57:06]           val_emotion_loss: 0
[25.03.2019 20:57:06] 

[25.03.2019 20:57:06]           val_gender_acc: 87.21448467116502
[25.03.2019 20:57:06]           val_emotion_acc: 0
[25.03.2019 20:57:06] ---------------


use the 30 epochs pretrained model

[26.03.2019 08:53:39]           val_gender_acc: 83.59331475473049
[26.03.2019 08:53:39] [Valid] age mae: 11.789214832337786

* Age_gender_pred Github repository web page, declare, 94% acc for gender and MAE of 4.2 for age can be achieved after just 32 epochs of training for FGNet dataset

但是我这里实现的时候却是 age mae, 9.856, gender acc, 88.798. 



[references]
[1]. SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation