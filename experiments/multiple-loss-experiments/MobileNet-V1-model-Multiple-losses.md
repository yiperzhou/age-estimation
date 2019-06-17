# experiments on 100-classes crossentropy loss, 10-classes crossentropy loss, gaussian loss for MobileNet-V1 model

## experimental results, multiple-losses for age estimation, load_imagenet_pretrained_weights = True, May 29th 2019

model                                                                  | Gender(%)(Chalearn_2016) | Smile(%)(Chalearn_2016) | Emotion Acc(%)(FER_2013) | Age Acc(%)                 | Age MAE (ChaLearn_2016)
---------------------------------------------------------------------- |------------------------- | ----------------------- | ------------------------ |--------------------------- | ---------------------------
MobileNet-V1-Age-100-class-10-classes-crossentropy [31]                |                          |                         |                          |                            |   4.9
MobileNet-V1-Age-10-classes-crossentropy                               |                          |                         |                          |                            |                              
MobileNet-V1-Age-100-class-10-classes-crossentropy-gaussian-loss [30]  |                          |                         |                          |                            |   4.8
MobileNet-V1-Age-100-class-20-classes [31]                             |                          |                         |                          |                            |   4.92                  
MobileNet-V1-Age-100-class-5-classes-crossentropy [32]                 |                          |                         |                          |                            |   4.94                  
MobileNet-V1-Age-100-class-5-classes-crossentropy-gaussian-loss [33]   |                          |                         |                          |                            |   4.79
MobileNet-V1-Age-100-class-20-classes-crossentropy-gaussian-loss [34]  |                          |                         |                          |                            |   4.81 
MobileNet-V1-Age-100-class-gaussian-loss [35]                          |                          |                         |                          |                            |   5.25


### reference

[30] /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_MobileNet_V1_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-07-14-55-44_100-class-10-classes-crossentropy-gaussian-loss

[31] Namespace(age_loss_type='5_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi'); /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_MobileNet_V1_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-28-23-26-40

[32] Namespace(age_loss_type='20_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi'); /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_MobileNet_V1_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-29-00-18-26

[33] Namespace(age_loss_gaussian='age_loss_gaussian', age_loss_type='5_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi');/home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_MobileNet_V1_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-29-01-21-34

[34] Namespace(age_loss_gaussian='age_loss_gaussian', age_loss_type='20_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi'); /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_MobileNet_V1_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-29-01-21-09

[35] Namespace(age_loss_gaussian='age_loss_gaussian', age_loss_type='5_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V1_model', multitask_training_type='Train_Valid', no_age_rgs_loss=True, num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi'); /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_MobileNet_V1_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-29-12-36-38