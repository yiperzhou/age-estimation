# Multitask learning & Single task learning on AlexNet model



## pretrained AlexNet on the IMDB-WIKI dataset

model                                                 | Gender(%)(IMDB-WIKI)     |  Age Acc(%)(IMDB-WIKI)    | Age MAE (IMDB-WIKI)
----------------------------------------------------- |------------------------- | ------------------------- | ------------------------ 
AlexNet-Gender_Age                                    |                          |   　　                    |                     
AlexNet-Gender                                        |                          |                           |                          
AlexNet-Age                                           |                          |                           |                          

### reference

[1] IMDB + WIKI, 153004 for train and 8829 for test








## experiment results, load_IMDB_WIKI_pretrained_model = False

model                                                                 | Gender(%)(Chalearn_2016) | Smile(%)(Chalearn_2016) | Emotion Acc(%)(FER_2013) |   Age Acc(%)                 | Age MAE (ChaLearn_2016)
--------------------------------------------------------------------- |------------------------- | ----------------------- | ------------------------ | ---------------------------  | ---------------------
AlexNet-Gender_Smile_Emotion_Age [1]                                  |       84.3               |      85.0               |       58.3               |                              |  7.0
AlexNet-Gender [3]                                                    |       84.2               |                         |                          |                              | 
AlexNet-Smile  [5]                                                    |                          |     86.5                |                          |                              |
AlexNet-Emotion [7]                                                   |                          |                         |      61.3                |                              |
AlexNet-Age [9]                                                       |                          |                         |                          |                              | 7.7
AlexNet-Age-100-class-10-classes-crossentropy [12]                    |                          |                         |                          |                              | 6.9   
AlexNet-Age-10-classes-crossentropy                                   |                          |                         |                          |                              |                              



### reference

[1] results/MTL_AlexNet_model/_gender_1_smile_1_emotion_1_age_1_CVPR_16_ChaLearn/2019-04-16-20-47-31; Namespace(age_loss_type='age_cls_loss', batch_size=8, dataset='CVPR_16_ChaLearn', debug=False, epoch=40, folder_sub_name='_gender_1_smile_1_emotion_1_age_1', load_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[1, 1, 1, 1], lr_rate=0.001, lr_schedule=8, model='MTL_AlexNet_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='thinkstation')

[3] results/MTL_AlexNet_model/_gender_1_smile_0_emotion_0_age_0_CVPR_16_ChaLearn/2019-04-16-22-15-56; Namespace(age_loss_type='age_cls_loss', batch_size=8, dataset='CVPR_16_ChaLearn', debug=False, epoch=40, folder_sub_name='_gender_1_smile_0_emotion_0_age_0', load_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[1, 0, 0, 0], lr_rate=0.001, lr_schedule=8, model='MTL_AlexNet_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='Narvi')

[5] results/MTL_AlexNet_model/_gender_0_smile_1_emotion_0_age_0_CVPR_16_ChaLearn/2019-04-16-22-16-24; Namespace(age_loss_type='age_cls_loss', batch_size=8, dataset='CVPR_16_ChaLearn', debug=False, epoch=40, folder_sub_name='_gender_0_smile_1_emotion_0_age_0', load_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 1, 0, 0], lr_rate=0.001, lr_schedule=8, model='MTL_AlexNet_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='Narvi')

[7] results/MTL_AlexNet_model/_gender_0_smile_0_emotion_1_age_0_CVPR_16_ChaLearn/2019-04-16-22-16-56; Namespace(age_loss_type='age_cls_loss', batch_size=8, dataset='CVPR_16_ChaLearn', debug=False, epoch=40, folder_sub_name='_gender_0_smile_0_emotion_1_age_0', load_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 1, 0], lr_rate=0.001, lr_schedule=8, model='MTL_AlexNet_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='Narvi')

[9] results/MTL_AlexNet_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-04-16-22-17-15; Namespace(age_loss_type='age_cls_loss', batch_size=8, dataset='CVPR_16_ChaLearn', debug=False, epoch=40, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_AlexNet_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='Narvi')  


[12] /home/yi/Narvi_yi_home/projects/MultitaskLearningFace/results/MTL_AlexNet_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-07-15-25-26_100-class-10-classes-crossentropy

