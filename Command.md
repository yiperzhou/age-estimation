## conda virtual envs

caffe  
source activate caffe  


## narvi zhouy computer environment

user name: zhouy  
ssh zhouy@narvi.tut.fi  


source activate dl
module load CUDA


# mount remote folder into local folder

mkdir narvi  
sshfs zhouy@narvi.tut.fi:/home/zhouy/local_elasticnn narvi/


## mount remote folder permantely

sshfs zhouy@narvi.tut.fi:/home/zhouy/local_elasticnn narvi/ xxx



## conda export environment yaml

conda env export > tut_thinkstation_virtual_env_caffe.yml


## conda import environment yaml

conda env create -f tut_thinkstation_virtual_env_caffe.yml


## umount folder

sudo umount narvi/
sudo umount -l narvi_yi_home
sudo umount -f narvi_yi_home/

## mount folder since computer starts


# frequent command used in this project based on TUT's lab computer

tensorboard folder path:  
/home/yi/anaconda3/lib/python3.6/site-packages/tensorboard


## run tensorboard:  

python /home/yi/anaconda3/lib/python3.6/site-packages/tensorboard/main.py --port=8008 --logdir=Graph/ 
tensorboard --logdir=Graph/


## test whether tensorflow-gpu works

'''
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
'''

# request allocation resource

## srun

srun --pty -J torch --gres=gpu:1 --partition=gpu --time=5-23:59:00 --mem=40960 --ntasks=1 --cpus-per-task=6 /bin/bash -i

module load CUDA
source activate dl

## batch runing, batch processing

/bin/bash MobileNets_alpha_0_75.sh

## run in background

screen  
ctrl + z (stop current job, but not kill it)  
bg (restore suspend job and run it background)  

## run in foreground

ctrl + z
fg (foreground)

## submit own job

sbatch -J resnet --partition=gpu --gres=gpu:1 --mem=40960 --ntasks=1 --cpus-per-task=4 --time=6-23:59:00 ResNet_script.sh



#!/bin/bash
#
# At first let's give job some descriptive name to distinct
# from other jobs running on cluster
#SBATCH -J elasticNN
#
# Let's redirect job's out some other file than default slurm-%jobid-out
#SBATCH --output=log/output.txt
#SBATCH --error=log/error.txt
#
# We'll want to allocate one CPU core
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#
# We'll want to reserve 2GB memory for the job
# and 3 days of compute time to finish.
# Also define to use the GPU partition.
#SBATCH --mem=40960
#SBATCH --time=7:59:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#
# These commands will be executed on the compute node:

# Load all modules you need below. Edit these to your needs.

#module load matlab
module load CUDA
source activate learning

# Finally run your job. Here's an example of a python script.
python train.py





## check job status
squeue
squeue -u zhouy

## cancel job
scancel 2865293  

## send file or folder to Narvi
scp -r keras-2.1.5 zhouy@narvi.tut.fi:~  

# Jakko's template code folder on narvi
/sgn-data/VisionGrp/Landmarks/Landmark_recognition  

#slurm  
https://wiki.eduuni.fi/display/tutsgn/TUT+Narvi+Cluster    



## flush cuda memory, GPU memory
sudo fuser -v /dev/nvidia*

sudo kill -9 PID
