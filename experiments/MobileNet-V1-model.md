# Multitask learning & Single task learning on MobileNet-V1-model

## experimental results, load_IMDB_WIKI_pretrained_model = False

model                                                                  | Gender(%)(Chalearn_2016) | Smile(%)(Chalearn_2016) | Emotion Acc(%)(FER_2013) | Age Acc(%)                 | Age MAE (ChaLearn_2016)
---------------------------------------------------------------------- |------------------------- | ----------------------- | ------------------------ |--------------------------- | ---------------------------
MobileNet-V1-Gender_Smile_Emotion_Age [1]                              |   88.2                   |  85.5                   |     52.7                 |   10.3                     |     5.2           
MobileNet-V1-Gender [21]                                               |  86.0                    |                         |                          |                            |     
MobileNet-V1-Smile [23]                                                |                          |     85.5                |                          |                            | 
MobileNet-V1-Emotion [25]                                              |                          |                         |     48.9                 |                            |                     
MobileNet-V1-Age  [27]                                                 |                          |                         |                          |   8.3                      |  5.5      
MobileNet-V1-Age-100-class-10-classes-crossentropy [31]                |                          |                         |                          |                            |   4.9






### reference

[1] results/MTL_MobileNet_V1_model/_gender_1_smile_1_emotion_1_age_1_CVPR_16_ChaLearn/2019-04-21-14-31-06; Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_1_smile_1_emotion_1_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[1, 1, 1, 1], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='Narvi')   

[21] results/MTL_MobileNet_V1_model/_gender_1_smile_0_emotion_0_age_0_CVPR_16_ChaLearn/2019-04-22-10-48-23; Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_1_smile_0_emotion_0_age_0', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[1, 0, 0, 0], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='Narvi')  

[23] results/MTL_MobileNet_V1_model/_gender_0_smile_1_emotion_0_age_0_CVPR_16_ChaLearn/2019-04-22-10-48-32; Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_0_smile_1_emotion_0_age_0', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 1, 0, 0], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='Narvi')  

[25] results/MTL_MobileNet_V1_model/_gender_0_smile_0_emotion_1_age_0_CVPR_16_ChaLearn/2019-04-22-10-48-49; Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_0_smile_0_emotion_1_age_0', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 1, 0], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='Narvi')  

[27] results/MTL_MobileNet_V1_model/_gender_1_smile_0_emotion_0_age_0_CVPR_16_ChaLearn/2019-04-22-10-48-23; Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_1_smile_0_emotion_0_age_0', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[1, 0, 0, 0], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='Narvi')  

[31] Namespace(age_loss_type='5_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='Narvi'); /home/yi/Narvi_yi_home/projects/MultitaskLearningFace/results/MTL_MobileNet_V1_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-28-23-26-40




## pretrained MobileNet-V1 on the IMDB-WIKI dataset

model                                                 | Gender(%)(IMDB-WIKI)     |  Age Acc(%)(IMDB-WIKI)    | Age MAE (IMDB-WIKI)
----------------------------------------------------- |------------------------- | ------------------------- | ------------------------ 
MobileNet-V1-Gender_Age[3]                            |   91.7                   |   8.4                     |   3.3                       
MobileNet-V1-Gender                                   |                          |                           |                          
MobileNet-V1-Age                                      |                          |                           |                          

### reference

[3] results/pretrained_MTL_MobileNet_V1_model/_gender_1_age_1_IMDB_WIKI/2019-04-21-11-32-30; Namespace(batch_size=64, dataset='IMDB_WIKI', epochs=20, folder_sub_name='_gender_1_age_1', load_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[1, 1], lr_rate=0.01, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type=2, num_workers=4, subtasks=['gender', 'age'], weight_decay=1e-06, working_machine='Narvi')  






## experiment results, load_IMDB_WIKI_pretrained_model = True 

model                                                 | Gender(%)(Chalearn_2016) | Smile(%)(Chalearn_2016) | Emotion Acc(%)(FER_2013) | Age Acc(%)                 | Age MAE (ChaLearn_2016)
----------------------------------------------------- |------------------------- | ----------------------- | ------------------------ |--------------------------- | ---------------------------
MobileNet-V1-Gender_Smile_Emotion_Age [5]             |   88.2                   |    86.6                 |    45.4                  |  11.3                      |   4.6             
MobileNet-V1-Gender [7]                               |   88.9                   |                         |                          |                            |     
MobileNet-V1-Smile [9]                                |                          |   86.1                  |                          |                            | 
MobileNet-V1-Emotion [11]                             |                          |                         |      37.2                |                            |                     
MobileNet-V1-Age [13]                                 |                          |                         |                          |   11.2                     |   4.5                      


### reference

[5] results/loaded_pretrained-MTL_MobileNet_V1_model/_gender_1_smile_1_emotion_1_age_1_CVPR_16_ChaLearn/2019-04-21-17-11-42; Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_1_smile_1_emotion_1_age_1', load_IMDB_WIKI_pretrained_model=True, loading_jobs=4, log_dir=None, loss_weights=[1, 1, 1, 1], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='Narvi')  

[7] results/loaded_pretrained-MTL_MobileNet_V1_model/_gender_1_smile_0_emotion_0_age_0_CVPR_16_ChaLearn/2019-04-22-07-38-06; Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_1_smile_0_emotion_0_age_0', load_IMDB_WIKI_pretrained_model=True, loading_jobs=4, log_dir=None, loss_weights=[1, 0, 0, 0], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='Narvi')  

[9] results/loaded_pretrained-MTL_MobileNet_V1_model/_gender_0_smile_1_emotion_0_age_0_CVPR_16_ChaLearn/2019-04-22-07-38-26; Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_0_smile_1_emotion_0_age_0', load_IMDB_WIKI_pretrained_model=True, loading_jobs=4, log_dir=None, loss_weights=[0, 1, 0, 0], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='Narvi')  

[11] results/loaded_pretrained-MTL_MobileNet_V1_model/_gender_0_smile_0_emotion_1_age_0_CVPR_16_ChaLearn/2019-04-22-07-40-08; Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_0_smile_0_emotion_1_age_0', load_IMDB_WIKI_pretrained_model=True, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 1, 0], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='Narvi')  

[13] results/loaded_pretrained-MTL_MobileNet_V1_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-04-22-07-43-02; Namespace(age_loss_type='age_cls_loss', batch_size=16, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=True, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='Narvi')  




# weekly report 

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