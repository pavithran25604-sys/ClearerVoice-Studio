# ClearerVoice-Studio: Online Target Speaker Extraction Algorithms


## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Usage](#2-usage)
- [3. Task: Audio-visual Speaker Extraction Conditioned on Face (Lip) Recording](#4-audio-visual-speaker-extraction-conditioned-on-face-or-lip-recording)


## 1. Introduction

This repository provides training scripts for online audio-visual target speaker extraction algorithms.

## 2. Usage

### Step-by-Step Guide

1. **Clone the Repository**

``` sh
git clone https://github.com/modelscope/ClearerVoice-Studio.git
```

2. **Create Conda Environment**

``` sh
cd ClearerVoice-Studio/train/target_speaker_extraction_online/
conda create -n clear_voice_tse python=3.9
conda activate clear_voice_tse
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

3. **Download Dataset**
> Please download LRS3 dataset [here](https://mmai.io/datasets/lip_reading/) 

4. **Modify Dataset Paths** 
> Update the paths to your datasets in the configuration files.

5. **Modify Train Configuration** 
> Adjust the settings in the "train.sh" file. For example, set "n_gpu=1" for single-GPU training, or "n_gpu=2" for two-GPU distributed training

6. **Start Training**

``` sh
bash train.sh
```

7. **Visualize Training Progress using Tensorboard**

``` sh
tensorboard --logdir ./checkpoints/
```

8. **Optionally Evaluate Checkpoints**

``` sh
bash evaluate_only.sh
```




## 3. Audio-visual speaker extraction conditioned on face or lip recording

### Support datasets for training: 

* LRS3 [[Download]([https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html](https://mmai.io/datasets/lip_reading/))]

### Support models for training: 

* AV-SkiM (Causal/Non-causal) [[Paper: Online Audio-Visual Autoregressive Speaker Extraction]([https://arxiv.org/abs/1904.03760](https://arxiv.org/abs/2506.01270))]





### Non-causal (Offline) LRS2-mix benchmark: 

 Dataset | Speakers | Model| Config | Checkpoint | SI-SDRi (dB) | SDRi (dB) 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LRS2 | 2-mix | [AV-ConvTasNet](https://arxiv.org/abs/1904.03760) | [This repo](./config/config_LRS2_lip_convtasnet_2spk.yaml)| [This repo](https://huggingface.co/alibabasglab/log_LRS2_lip_convtasnet_2spk/) | 11.6 | 11.9
| LRS2 | 2-mix | [AV-DPRNN](https://ieeexplore.ieee.org/document/9887809) | [This repo](./config/config_LRS2_lip_dprnn_2spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_LRS2_lip_dprnn_2spk/) | 12.0 | 12.4 
| LRS2 | 2-mix | [AV-TFGridNet](https://arxiv.org/abs/2310.19644) | [This repo](./config/config_LRS2_lip_tfgridnet_2spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_LRS2_lip_tfgridnet_2spk/)| 15.1 | 15.4 
| LRS2 | 2-mix | [AV-Mossformer2](https://arxiv.org/abs/2506.19398)| [This repo](./config/config_LRS2_lip_mossformer2_2spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_LRS2_lip_mossformer2_2spk/)| 15.5 | 15.8 
| LRS2 | 3-mix | [AV-ConvTasNet](https://arxiv.org/abs/1904.03760) | [This repo](./config/config_LRS2_lip_convtasnet_3spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_LRS2_lip_convtasnet_3spk/)| 10.8 | 11.3
| LRS2 | 3-mix | [AV-DPRNN](https://ieeexplore.ieee.org/document/9887809) | [This repo](./config/config_LRS2_lip_dprnn_3spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_LRS2_lip_dprnn_3spk/)| 10.6 | 11.1 
| LRS2 | 3-mix | [AV-TFGridNet](https://arxiv.org/abs/2310.19644) | [This repo](./config/config_LRS2_lip_tfgridnet_3spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_LRS2_lip_tfgridnet_3spk/)| 15.0 | 15.4 
| LRS2 | 3-mix | [AV-Mossformer2](https://arxiv.org/abs/2506.19398) | [This repo](./config/config_LRS2_lip_mossformer2_3spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_LRS2_lip_mossformer2_3spk/)| 16.2 | 16.6 

