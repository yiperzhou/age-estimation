## dataset
IMDB + WIKI, 153004 for train and 8829 for test

## experiment result



model                                   |      Age  MAE    |     Gender Acc       |  initial lr    |      weight       |       note        
----------------------------------------| ---------------- | -------------------- | -------------- | ----------------- | ---------------------- | 
Pretrained-ResNet-18-Age-Gender         |  9.856           |     88.798           |      0.01      |       1, 1        |   age_cls_loss, gen_loss      




* Age_gender_pred Github repository web page, declare, 94% acc for gender and MAE of 4.2 for age can be achieved after just 32 epochs of training for FGNet dataset

但是我这里实现的时候却是 age mae, 9.856, gender acc, 88.798. 