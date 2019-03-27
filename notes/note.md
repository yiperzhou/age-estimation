## Multitask Learning to do
实验设计
1. 使用 三个数据集先训练一个多任务学习的网络，然后将两个其他任务的分支去除，训练一个模型，完成下面的表格
[] find the clean test set, or use the validation set (age, gender, smile) to test the model, after finishing training, then fill the following table



model                                   |        Age  MAE    | Gender Acc (%)     |    Smile  Acc (%)          
----------------------------------------| ------------------ | ------------------ | ------------------
Multitask-ResNet-18                     |    15              |  71.38             |  71.82
ResNet-18-Age                           |   22.20            |                    |  
ResNet-18-Gender                        |                    |   55.46            |  
ResNet-18-Smile                         |                    |                    |  82.97


  
<br>

                                                                            

 task   | train set(images)  | test set 
------- | ---------- | ---------  
age     |  3707      | 1356  
gender  |  4548      | 2250  
smile   |  4548      | 2250  