# experiments on 100-classes crossentropy loss, 10-classes crossentropy loss, gaussian loss for ResNet-18 model


## experimental results, multiple-losses for age estimation, load_imagenet_pretrained_weights = True, May 29th 2019

model                                                                  | Gender(%)(Chalearn_2016) | Smile(%)(Chalearn_2016) | Emotion Acc(%)(FER_2013) | Age Acc(%)                 | Age MAE (ChaLearn_2016)
---------------------------------------------------------------------- |------------------------- | ----------------------- | ------------------------ |--------------------------- | ---------------------
ResNet-18-Age-100-class-10-classes-crossentropy-gaussian-loss [10]     |                          |                         |                          |                            |   9.85
ResNet-18-Age-100-class-5-classes-crossentropy [13]                    |                          |                         |                          |                            |   11.17       
ResNet-18-Age-100-class-20-classes-crossentropy [14]                   |                          |                         |                          |                            |   10.56
ResNet-18-Age-100-class-5-classes-crossentropy-gaussian-losss [15]     |                          |                         |                          |                            |  11.35        
ResNet-18-Age-100-class-20-classes-crossentropy-gaussian-losses [16]   |                          |                         |                          |                            |   10.84
ResNet-18-Age-100-class-gaussian-losses [17]                           |                          |                         |                          |                            |   11.83




### reference

[10] /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_ResNet_18_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-07-14-35-05_100-class-10-classes-crossentropy-gaussian-loss

[13] /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_ResNet_18_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-28-23-36-08; Namespace(age_loss_type='20_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_ResNet_18_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='thinkstation')

[14] Namespace(age_loss_type='5_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_ResNet_18_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi'); /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_ResNet_18_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-29-00-24-03

[15] Namespace(age_loss_gaussian='age_loss_gaussian', age_loss_type='20_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_ResNet_18_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi'); /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_ResNet_18_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-29-09-26-19

[16] Namespace(age_loss_gaussian='age_loss_gaussian', age_loss_type='5_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_ResNet_18_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='thinkstation'); /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_ResNet_18_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-29-09-26-54

[17] Namespace(age_loss_gaussian='age_loss_gaussian', age_loss_type='5_age_cls_loss', batch_size=32, dataset='CVPR_16_ChaLearn', debug=True, epoch=80, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_IMDB_WIKI_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_ResNet_18_model', multitask_training_type='Train_Valid', no_age_rgs_loss=True, num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi'); /home/yi/narvi_yi_home/projects/MultitaskLearningFace/results/MTL_ResNet_18_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-05-29-10-50-40