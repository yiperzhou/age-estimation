# experiments on 100-classes crossentropy loss, 10-classes crossentropy loss, gaussian loss for AlexNet model

## experimental results, multiple-losses for age estimation, load_imagenet_pretrained_weights = True, May 29th 2019

model                                                                 | Gender(%)(Chalearn_2016) | Smile(%)(Chalearn_2016) | Emotion Acc(%)(FER_2013) |   Age Acc(%)                 | Age MAE (ChaLearn_2016)
--------------------------------------------------------------------- |------------------------- | ----------------------- | ------------------------ | ---------------------------  | ---------------------
AlexNet-Age-100-class-10-classes-crossentropy-gaussian-loss [11]      |                          |                         |                          |                              |  6.4
AlexNet-Age-100-class-20-classes-crossentropy [13]                    |                          |                         |                          |                              |  6.77                
AlexNet-Age-100-class-5-classes-crossentropy [14]                     |                          |                         |                          |                              |  6.60          
AlexNet-Age-100-class-5-classes-crossentropy-gaussian-loss [15]       |                          |                         |                          |                              |  6.21                 
AlexNet-Age-100-class-20-classes-crossentropy-gaussian-loss [16]      |                          |                         |                          |                              |  6.94                
AlexNet-Age-100-class-gaussian-loss [17]                              |                          |                         |                          |                              |  7.26              



### reference

[11] /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_AlexNet_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-07-15-24-16_100-class-10-classes-crossentropy-gaussian-loss

[13] Namespace(age_loss_type='5_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_AlexNet_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi'); /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_AlexNet_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-28-23-07-17

[14] Namespace(age_loss_type='20_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_AlexNet_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi'); /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_AlexNet_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-28-23-29-08

[15] Namespace(age_loss_gaussian='age_loss_gaussian', age_loss_type='20_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_AlexNet_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi'); /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_AlexNet_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-29-00-54-12

[16] Namespace(age_loss_gaussian='age_loss_gaussian', age_loss_type='5_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_AlexNet_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi'); /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_AlexNet_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-29-00-54-43

[17] Namespace(age_loss_gaussian='age_loss_gaussian', age_loss_type='5_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_AlexNet_model', multitask_training_type='Train_Valid', no_age_rgs_loss=True, num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06,working_machine='thinkstation'); /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_AlexNet_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-29-10-45-20