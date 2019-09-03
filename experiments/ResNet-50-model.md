# Multitask learning & Single task learning on ResNet-50-model



## experiment results, load_IMDB_WIKI_pretrained_model = False

model                                              | Gender(%)(Chalearn_2016) | Smile(%)(Chalearn_2016) | Emotion Acc(%)(FER_2013) | Age Acc(%)                 | Age MAE (ChaLearn_2016)
-------------------------------------------------- |------------------------- | ----------------------- | ------------------------ |--------------------------- | ---------------------
ResNet-50-Gender_Smile_Emotion_Age [1][2]          |  88.0                    |   89.9                  |    54.1                  |  11.5                      | 4.5
ResNet-50-Gender [3][4]                            |  87.1                    |                         |                          |                            |     
ResNet-50-Smile [5][6]                             |                          |    88.3                 |                          |                            | 
ResNet-50-Emotion [7][8]                           |                          |                         |          50.8            |                            |
ResNet-50-Age  [9][10]                             |                          |                         |                          |                            | 4.7


### reference  

[1] results/MTL_ResNet_50_model/_gender_1_smile_1_emotion_1_age_1_CVPR_16_ChaLearn/2019-04-16-19-13-31  
[2] Namespace(age_loss_type='age_cls_loss', batch_size=8, dataset='CVPR_16_ChaLearn', debug=False, epoch=40, folder_sub_name='_gender_1_smile_1_emotion_1_age_1', load_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[1, 1, 1, 1], lr_rate=0.001, lr_schedule=8, model='MTL_ResNet_50_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='thinkstation')  

[3] results/MTL_ResNet_50_model/_gender_1_smile_0_emotion_0_age_0_CVPR_16_ChaLearn/2019-04-16-19-48-36  
[4] Namespace(age_loss_type='age_cls_loss', batch_size=8, dataset='CVPR_16_ChaLearn', debug=False, epoch=40, folder_sub_name='_gender_1_smile_0_emotion_0_age_0', load_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[1, 0, 0, 0], lr_rate=0.001, lr_schedule=8, model='MTL_ResNet_50_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='Narvi')

[5] results/MTL_ResNet_50_model/_gender_0_smile_1_emotion_0_age_0_CVPR_16_ChaLearn/2019-04-16-19-55-33
[6] Namespace(age_loss_type='age_cls_loss', batch_size=8, dataset='CVPR_16_ChaLearn', debug=False, epoch=40, folder_sub_name='_gender_0_smile_1_emotion_0_age_0', load_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 1, 0, 0], lr_rate=0.001, lr_schedule=8, model='MTL_ResNet_50_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='Narvi')

[7] results/MTL_ResNet_50_model/_gender_0_smile_0_emotion_1_age_0_CVPR_16_ChaLearn/2019-04-16-19-53-46
[8] Namespace(age_loss_type='age_cls_loss', batch_size=8, dataset='CVPR_16_ChaLearn', debug=False, epoch=40, folder_sub_name='_gender_0_smile_0_emotion_1_age_0', load_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 1, 0], lr_rate=0.001, lr_schedule=8, model='MTL_ResNet_50_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='Narvi')

[9] results/MTL_ResNet_50_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-04-16-19-52-33
[10] Namespace(age_loss_type='age_cls_loss', batch_size=8, dataset='CVPR_16_ChaLearn', debug=False, epoch=40, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_ResNet_50_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='Narvi')



## experiment results, load_IMDB_WIKI_pretrained_model = True 

model                                                 | Gender(%)(Chalearn_2016) | Smile(%)(Chalearn_2016) | Emotion Acc(%)(FER_2013) | Age Acc(%)                 | Age MAE (ChaLearn_2016)
----------------------------------------------------- |------------------------- | ----------------------- | ------------------------ |--------------------------- | ---------------------------
ResNet-50-Gender_Smile_Emotion_Age                    |   
ResNet-50-Gender                                      |   
ResNet-50-Smile                                       |    
ResNet-50-Emotion                                     |    
ResNet-50-Age                                         |   


### reference
