1. use IMDB-WIKI dataset to pretrain the ResNet-18 model, 
2. load the pretrained model, load the CVPR Age, Gender dataset, FER 2013 emotion dataset, sampling the age, gender images from around 4500 to 28710, 
3. set gender, emotion weight as 0, use the same data, 
4. set age emotion loss weight as 0, use the same data,
5. set age, gender weight as 0, use the same data
4. the training result is show below.


model                                   |        Age  MAE    | Gender Acc (%)     |    Emotion  Acc (%)  |     weight      | initial lr     
----------------------------------------| ------------------ | ------------------ | ------------------- |---------------- | ----------------                    
ResNet-18-Age-Gender_Emotion            |    12.95           |  84.15             |  48.20              |  1, 0.1, 1      |   0.01           
ResNet-18-Age-Gender_Emotion            |    12.23           |  86.63             |  52.91              |    1, 1, 1       |      0.01  
ResNet-18-Age                           |  12.30             |                    |                     |                 |              
ResNet-18-Gender                        |                    |  88.12             |                      |                |                 
ResNet-18-Emotion                       |                    |                    |   55.56              |                 |                    


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


--------------------------------------------------------------------------------------------------
