# this repositroy is a working repository of my AI and cybersecurity project at the university of luxembourg, while the oroginal repository is focusing on Botnet detection and malware, our project is limited only to Botenet detection for PGD attack and fence.

# Realistic adversarial hardening
This repository contains the data and the code used for S&amp;P 2023 paper <em>[On The Empirical Effectiveness of Unrealistic Adversarial Hardening Against Realistic Adversarial Attacks](https://arxiv.org/abs/2202.03277)</em>. 
The paper investigates whether "cheap" unrealistic adversarial attacks can be used to harden ML models against computationally expensive realistic attacks. 
The paper studies 3 use cases (text, botnet, malware) and at least one realistic and unrealistic attack for each of them. 


### Downloading the data
All the data used for the results analysis can be downloaded in this [link](https://uniluxembourg-my.sharepoint.com/:f:/g/personal/salijona_dyrmishi_uni_lu/Eo3LPuU7nVJBs5UyHEBadU8BkOOJHnCFXdGE55dNbCETow?e=YTyCXn). 


### 1. Text classification
We consider two attacks: [Deepwordbug](https://arxiv.org/abs/1801.04354)(unrealistic) and [TextFooler](https://arxiv.org/abs/1907.11932)(realistic).

We use [Textattack](https://textattack.readthedocs.io/en/latest/) library from the command line to train the models and evaluate their adversarial robustness. 
You will find all the models presented in the paper in this [link](https://uniluxembourg-my.sharepoint.com/:u:/g/personal/salijona_dyrmishi_uni_lu/EShqUwlR76xNttj4_BPuOU0BrY854BrNlkm84adN3wRpqQ?e=rrncfb).
In the config files of each model you can see the training parameters, including the attack used for the hardening. This [file](https://uniluxembourg-my.sharepoint.com/:x:/g/personal/salijona_dyrmishi_uni_lu/ES71tvRvV1pBl9thHzz0AA8BpVDzoTXv4yVMURyQuVs9uw?e=ct4uwh)
provides an easy way to identify the models. 


### 2. Botnet detection
As an unrealistic attack we use the PGD implementation of [ART](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/) library. <br>
For the realistic attack we use the [FENCE](https://arxiv.org/pdf/1909.10480.pdf) attack by Chernikova et al. Our implementation in this repo includes updates from TensorFlow 1
to TensorFlow 2, code refactoring and as well as bug fixes. However you can access the original repository by authors [here](https://github.com/achernikova/cybersecurity_evasion) if needed.

If you need to do feature engineering for new datasets not included in this repo check the tool from [Talha Ongun](https://github.com/tongun/ctu13-botnet-detection).
