# Multitask learning & Single task learning on ResNet-18-model

# pretrained

## experiment result

1. pretraining model 
2. IMDB_WIKI

model                                   | Age MAE (IMDB_WIKI) | Gender Acc (IMDB_WIKI)
----------------------------------------| ------------------- | ----------------------
Pretrained-ResNet-18-Age-Gender [1]     |  9.856              |     88.798            
----------------------------------------| ------------------- | ----------------------
reference models                        |                     |                       
SSR-Net[3]                              | 

### referecne
[1] Pretrained-ResNet-18-Age-Gender, IMDB_WIKI train data, initial_lr=0.01, weight=_gender_1_age_1_IMDB_WIKI,
[2] Pretrained_MTL_AlexNet_model, IMDB_WIKI train data, initial_lr=0.01, weight=_gender_1_age_1_IMDB_WIKI,
[3] SSR-Net, 



4. set age emotion loss weight as 0, use the same data,



model                                            | Age MAE (ChaLearn_2016) | Gender Acc(Chalearn_2016) | Emotion Acc (FER_2013) |               weight         
------------------------------------------------ |------------------------ | ------------------------- | ---------------------- |-----------------------------
ResNet-18-Age-Gender_Emotion                     |         12.95           |        84.15              |    48.20               |                1, 0.1, 1      
ResNet-18-Age-Gender_Emotion                     |        12.23            |         86.63             |    52.91               |                1, 1, 1         
ResNet-18-Age                                    |       12.30             |                           |                        |                                
ResNet-18-Gender                                 |                         |         88.12             |                        |                       
ResNet-18-Emotion                                |                         |                           |     55.56              |                                
ResNet-18-model[3]                               |                         |                           |                        |                               

model                                            |        Age  MAE         | Gender Acc (%)            |    Emotion  Acc (%)    |Smile Acc (%)   
-------------------------------------------------| ----------------------- | ------------------------- | ---------------------- |---------------- 
ResNet-18-Age-Gender_Emotion_Smile               |       11.90             |  87.33                    |  48.45                 | 88.05        
ResNet-50-Age-Gender_Emotion_Smile               |   13.18                 |   85.24                   |  47.81                 | 87.55        
-------------------------------------------------| ----------------------- | ------------------------- | ---------------------- |---------------
DEX model                                        |                         |                           |                        |                


SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation


### reference
[1] ResNet-18-model, train_valid_2, pretrained_IMDB_WIKI_weight


## experiment results, load_IMDB_WIKI_pretrained_model = False

model                                                                  | Gender(%)(Chalearn_2016) | Smile(%)(Chalearn_2016) | Emotion Acc(%)(FER_2013) | Age Acc(%)                 | Age MAE (ChaLearn_2016)
---------------------------------------------------------------------- |------------------------- | ----------------------- | ------------------------ |--------------------------- | ---------------------
ResNet-18-Gender_Smile_Emotion_Age                                     |  90.2                    |   87.0                  |    55.2                  |                            | 11.8
ResNet-18-Gender                                                       |  89.1                    |                         |                          |                            |     
ResNet-18-Smile                                                        |                          |    86.2                 |                          |                            | 
ResNet-18-Emotion                                                      |                          |                         |          51.5            |                            |
ResNet-18-Age                                                          |                          |                         |                          |                            | 11.4
ResNet-18-Age-100-class-10-classes-crossentropy [12]                   |                          |                         |                          |                            |   10.23
ResNet-18-Age-10-classes-crossentropy  [11]                            |                          |                         |                          |                            |  20.47                                
ResNet-18-Age-100-class-10-classes-crossentropy-gaussian-loss [10]     |                          |                         |                          |                            |   9.85
ResNet-18-Age-100-class-5-classes-crossentropy [13]                    |                         |                          |                          |                            |   11.17       
ResNet-18-Age-100-class-20-classes-crossentropy [14]                   |                          |                         |                          |                            |   10.56
ResNet-18-Age-100-class-5-classes-crossentropy-gaussian-losss [15]     |                         |                          |                          |                            |  11.35        
ResNet-18-Age-100-class-20-classes-crossentropy-gaussian-losses [16]   |                          |                         |                          |                            |   10.84
ResNet-18-Age-100-class-gaussian-losses [17]                           |                          |                         |                          |                            |   11.83






### reference 

[10] /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_ResNet_18_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-07-14-35-05_100-class-10-classes-crossentropy-gaussian-loss

[11] /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_ResNet_18_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-07-14-24-47_10-classes-crossentropy

[12] /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_ResNet_18_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-07-14-13-05_100-class-10-classes-crossentropy

[13] /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_ResNet_18_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-28-23-36-08; Namespace(age_loss_type='20_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_ResNet_18_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='thinkstation')

[14] Namespace(age_loss_type='5_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_ResNet_18_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi'); /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_ResNet_18_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-29-00-24-03

[15] Namespace(age_loss_gaussian='age_loss_gaussian', age_loss_type='20_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_ResNet_18_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi'); /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_ResNet_18_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-29-09-26-19

[16] Namespace(age_loss_gaussian='age_loss_gaussian', age_loss_type='5_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_ResNet_18_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='thinkstation'); /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_ResNet_18_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-29-09-26-54

[17] Namespace(age_loss_gaussian='age_loss_gaussian', age_loss_type='5_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_ResNet_18_model', multitask_training_type='Train_Valid', no_age_rgs_loss=True, num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi'); /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_ResNet_18_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-29-10-50-40

