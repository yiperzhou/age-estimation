# Multitask learning & Single task learning on MobileNet-V2-model

## experiment results, load_IMDB_WIKI_pretrained_model = False

model                                                 | Gender(%)(Chalearn_2016) | Smile(%)(Chalearn_2016) | Emotion Acc(%)(FER_2013) | Age Acc(%)                 | Age MAE (ChaLearn_2016)
----------------------------------------------------- |------------------------- | ----------------------- | ------------------------ |--------------------------- | ---------------------------
MobileNet-V2-Gender_Smile_Emotion_Age [4][5]          |   82.5                   |   87.6                  |   33.7                   |     7.2                    |   6.7             
MobileNet-V2-Gender [1][2]                            |   77.8                   |                         |                          |                            |     
MobileNet-V2-Smile [6][7]                             |                          |    84.0                 |                          |                            | 
MobileNet-V2-Emotion [8][9]                           |                          |                         |  25.1                    |                            |                     
MobileNet-V2-Age [10][11]                             |                          |                         |                          |     7.5                    |  7.1


## reference

<!-- [2] 40 epochs, other 4 models, 60 epochs -->
[4] data source: "results/MTL_MobileNet_V2_model/_gender_1_smile_1_emotion_1_age_1_CVPR_16_ChaLearn/2019-04-17-12-57-39"
[5] hyper-parameters: Namespace(age_loss_type='age_cls_loss', batch_size=8, dataset='CVPR_16_ChaLearn', debug=False, epoch=40, folder_sub_name='_gender_1_smile_1_emotion_1_age_1', load_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[1, 1, 1, 1], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V2_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='thinkstation')

[1] results/MTL_MobileNet_V2_model/_gender_1_smile_0_emotion_0_age_0_CVPR_16_ChaLearn/2019-04-18-07-46-04
[2] Namespace(age_loss_type='age_cls_loss', batch_size=8, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_1_smile_0_emotion_0_age_0', load_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[1, 0, 0, 0], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V2_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi')

[6] results/MTL_MobileNet_V2_model/_gender_0_smile_1_emotion_0_age_0_CVPR_16_ChaLearn/2019-04-17-14-31-26
[7] Namespace(age_loss_type='age_cls_loss', batch_size=8, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_0_smile_1_emotion_0_age_0', load_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 1, 0, 0], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V2_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi')

[8] results/MTL_MobileNet_V2_model/_gender_0_smile_0_emotion_1_age_0_CVPR_16_ChaLearn/2019-04-17-14-31-42
[9] Namespace(age_loss_type='age_cls_loss', batch_size=8, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_0_smile_0_emotion_1_age_0', load_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 1, 0], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V2_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi')

[10] results/MTL_MobileNet_V2_model/_gender_0_smile_0_emotion_0_age_1_CVPR_16_ChaLearn/2019-04-17-14-39-00
[11] Namespace(age_loss_type='age_cls_loss', batch_size=8, dataset='CVPR_16_ChaLearn', debug=False, epoch=60, folder_sub_name='_gender_0_smile_0_emotion_0_age_1', load_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[0, 0, 0, 1], lr_rate=0.001, lr_schedule=8, model='MTL_MobileNet_V2_model', multitask_training_type='Train_Valid', num_workers=4, subtasks=['gender', 'smile', 'emotion', 'age'], weight_decay=1e-06, working_machine='narvi')







## pretrained MobileNet-V2 on the IMDB-WIKI dataset

model                                                 | Gender(%)(IMDB-WIKI)     |  Age Acc(%)(IMDB-WIKI)    | Age MAE (IMDB-WIKI)
----------------------------------------------------- |------------------------- | ------------------------- | ------------------------ 
MobileNet-V2-Gender_Age                               |   90.9                   |    5.7                    | 3.5                     
MobileNet-V2-Gender                                   |                          |                           |                         
MobileNet-V2-Age                                      |                          |                           |                         

### notes

[1] data source: "results/pretrained_MTL_MobileNet_V2_model/_gender_1_age_1_IMDB_WIKI/2019-04-21-14-00-09"
[2] hyper-parameters: Namespace(batch_size=64, dataset='IMDB_WIKI', epochs=20, folder_sub_name='_gender_1_age_1', load_pretrained_model=False, loading_jobs=4, log_dir=None, loss_weights=[1, 1], lr_rate=0.01, lr_schedule=8, model='MTL_MobileNet_V2_model', multitask_training_type=2, num_workers=4, subtasks=['gender', 'age'], weight_decay=1e-06, working_machine='narvi')
