# Multitask learning & Single task learning on MobileNet-V1-model

## experiment results, load_IMDB_WIKI_pretrained_model = False

model                                                 | Gender(%)(Chalearn_2016) | Smile(%)(Chalearn_2016) | Emotion Acc(%)(FER_2013) | Age Acc(%)                 | Age MAE (ChaLearn_2016)
----------------------------------------------------- |------------------------- | ----------------------- | ------------------------ |--------------------------- | ---------------------------
MobileNet-V1-Gender_Smile_Emotion_Age [1][2]          |   88.2                   |  85.5                   |     52.7                 |   10.3                     |     5.2           
MobileNet-V1-Gender [21][22]                          |  86.0                    |                         |                          |                            |     
MobileNet-V1-Smile [23][24]                           |                          |     85.5                |                          |                            | 
MobileNet-V1-Emotion [25][26]                         |                          |                         |     48.9                 |                            |                     
MobileNet-V1-Age  [27][28]                            |                          |                         |                          |   8.3                      |  5.5      


### reference

[1] results/MTL_MobileNet_V1_model/_gender_1_smile_1_emotion_1_age_1_CVPR_16_ChaLearn/2019-04-21-14-31-06   
[2] Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_1_smile_1_emotion_1_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[1, 1, 1, 1], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi')   

[21] results/MTL_MobileNet_V1_model/_gender_1_smile_0_emotion_0_age_0_CVPR_16_ChaLearn/2019-04-22-10-48-23  
[22] Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_1_smile_0_emotion_0_age_0', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[1, 0, 0, 0], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi')  

[23] results/MTL_MobileNet_V1_model/_gender_0_smile_1_emotion_0_age_0_CVPR_16_ChaLearn/2019-04-22-10-48-32  
[24] Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_0_smile_1_emotion_0_age_0', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 1, 0, 0], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi')  

[25] results/MTL_MobileNet_V1_model/_gender_0_smile_0_emotion_1_age_0_CVPR_16_ChaLearn/2019-04-22-10-48-49  
[26] Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_0_smile_0_emotion_1_age_0', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 1, 0], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi')  

[27] results/MTL_MobileNet_V1_model/_gender_1_smile_0_emotion_0_age_0_CVPR_16_ChaLearn/2019-04-22-10-48-23  
[28] Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_1_smile_0_emotion_0_age_0', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[1, 0, 0, 0], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi')  






## pretrained MobileNet-V1 on the IMDB-WIKI dataset

model                                                 | Gender(%)(IMDB-WIKI)     |  Age Acc(%)(IMDB-WIKI)    | Age MAE (IMDB-WIKI)
----------------------------------------------------- |------------------------- | ------------------------- | ------------------------ 
MobileNet-V1-Gender_Age[3][4]                         |   91.7                   |   8.4                     |   3.3                       
MobileNet-V1-Gender                                   |                          |                           |                          
MobileNet-V1-Age                                      |                          |                           |                          

### reference

[3] results/pretrained_MTL_MobileNet_V1_model/_gender_1_age_1_IMDB_WIKI/2019-04-21-11-32-30  
[4] Namespace(batch_size=64, dataset='IMDB_WIKI', epochs=20, folder_sub_name='_gender_1_age_1', load_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[1, 1], lr_rate=0.01, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type=2, num_workers=4, subtasks=['gender', 'age'], weight_decay=1e-06, working_machine='narvi')  






## experiment results, load_IMDB_WIKI_pretrained_model = True 

model                                                 | Gender(%)(Chalearn_2016) | Smile(%)(Chalearn_2016) | Emotion Acc(%)(FER_2013) | Age Acc(%)                 | Age MAE (ChaLearn_2016)
----------------------------------------------------- |------------------------- | ----------------------- | ------------------------ |--------------------------- | ---------------------------
MobileNet-V1-Gender_Smile_Emotion_Age [5][6]          |   88.2                   |    86.6                 |    45.4                  |  11.3                      |   4.6             
MobileNet-V1-Gender [7][8]                            |   88.9                   |                         |                          |                            |     
MobileNet-V1-Smile [9][10]                            |                          |   86.1                  |                          |                            | 
MobileNet-V1-Emotion [11][12]                         |                          |                         |      37.2                |                            |                     
MobileNet-V1-Age [13][14]                             |                          |                         |                          |   11.2                     |   4.5                      


### reference

[5] results/loaded_pretrained-MTL_MobileNet_V1_model/_gender_1_smile_1_emotion_1_age_1_CVPR_16_ChaLearn/2019-04-21-17-11-42  
[6] Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_1_smile_1_emotion_1_age_1', load_IMDB_WIKI_pretrained_model=True, loading_jobs=4, log_dir=None, loss_weights=[1, 1, 1, 1], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi')  
[7] results/loaded_pretrained-MTL_MobileNet_V1_model/_gender_1_smile_0_emotion_0_age_0_CVPR_16_ChaLearn/2019-04-22-07-38-06  
[8] Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_1_smile_0_emotion_0_age_0', load_IMDB_WIKI_pretrained_model=True, loading_jobs=4, log_dir=None, loss_weights=[1, 0, 0, 0], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi')  
[9] results/loaded_pretrained-MTL_MobileNet_V1_model/_gender_0_smile_1_emotion_0_age_0_CVPR_16_ChaLearn/2019-04-22-07-38-26  
[10] Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_0_smile_1_emotion_0_age_0', load_IMDB_WIKI_pretrained_model=True, loading_jobs=4, log_dir=None, loss_weights=[0, 1, 0, 0], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi')  
[11] results/loaded_pretrained-MTL_MobileNet_V1_model/_gender_0_smile_0_emotion_1_age_0_CVPR_16_ChaLearn/2019-04-22-07-40-08  
[12] Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_0_smile_0_emotion_1_age_0', load_IMDB_WIKI_pretrained_model=True, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 1, 0], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi')  
[13] results/loaded_pretrained-MTL_MobileNet_V1_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-04-22-07-43-02  
[14] Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=True, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi')  




### Multitask learning & single task learning without IMDB-WIKI pretrained weights

model                                                           | Gender(%)(Chalearn_2016) | Smile(%)(Chalearn_2016) | Emotion Acc(%)(FER_2013) | Age Acc(%)                 | Age MAE (ChaLearn_2016)
--------------------------------------------------------------- |------------------------- | ----------------------- | ------------------------ |--------------------------- | ---------------------------
MobileNet-V1-Gender_Smile_Emotion_Age_No_IMDB_WIKI_pretrained   |   88.2                   |  85.5                   |     52.7                 |   10.3                     |     5.2           
MobileNet-V1-Gender                                             |  86.0                    |                         |                          |                            |     
MobileNet-V1-Smile                                              |                          |     85.5                |                          |                            | 
MobileNet-V1-Emotion                                            |                          |                         |     48.9                 |                            |                     
MobileNet-V1-Age                                                |                          |                         |                          |   8.3                      |  5.5                      


### Multitask learning & single task learning with loading IMDB-WIKI pretrained weights

model                                                           | Gender(%)(Chalearn_2016) | Smile(%)(Chalearn_2016) | Emotion Acc(%)(FER_2013) | Age Acc(%)                 | Age MAE (ChaLearn_2016)  
--------------------------------------------------------------- |------------------------- | ----------------------- | ------------------------ |--------------------------- | ---------------------------
MobileNet-V1-Gender_Smile_Emotion_Age_IMDB_WIKI_pretrained      |   88.2                   |    86.6                 |    45.4                  |  11.3                      |   4.6             
MobileNet-V1-Gender                                             |   88.9                   |                         |                          |                            |     
MobileNet-V1-Smile                                              |                          |   86.1                  |                          |                            | 
MobileNet-V1-Emotion                                            |                          |                         |      37.2                |                            |                     
MobileNet-V1-Age                                                |                          |                         |                          |   11.2                     |   4.5                      