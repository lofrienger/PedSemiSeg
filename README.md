# PedSemiSeg: 

## Introduction

This is the pytorch implementation of the paper entitled "PedSemiSeg: **Ped**agogy-inspired **Semi**-supervised Polyp **Seg**mentation", whcih was submitted to *ICRA2024*.

Recent advancements in deep learning techniques have contributed to developing improved polyp segmentation methods, thereby aiding in the diagnosis of colorectal cancer and facilitating endoscopic submucosal dissection (ESD) surgery. However, the scarcity of well-annotated data poses challenges by increasing the annotation burden and diminishing the performance of fully-supervised learning approaches. To address this challenge, we present *PedSemiSeg*, a pedagogy-inspired semi-supervised learning framework designed to enhance polyp segmentation performance with limited labeled training data. In particular, we take inspiration from the pedagogy used in real-world educational settings, where teacher feedback and peer learning are crucial in influencing the overall learning outcome. Expanding upon this concept, our approach involves supervising the outputs of the strongly augmented input (the students) using the pseudo and complementary labels crafted from the output of the weakly augmented input (the teacher) in a positive and negative learning manner. Additionally, we incorporate entropy-guided reciprocal peer learning among the students. With these holistic learning processes, we aim to ensure consistent predictions for various versions of the same input. The experimental results illustrate the superiority of our method in polyp segmentation across various ratios of labeled data. Furthermore, our approach also generalizes well on external datasets, which are unseen during training, highlighting its broader clinical significance in practice. 

![PedSemiSeg](Image/PedSemiSeg.png?raw=true "PedSemiSeg")
*Overall diagram of our proposed PedSemiSeg, a pedagogy-inspired semi-supervised approach for label-efficient polyp segmentation.*

## Environment
- NVIDIA RTX3090
- Python 3.8
- Pytorch 1.10 with CUDA 11.3
- Check [environment.yml](code/environment.yml) for the detailed conda environment.

## Usage
1. Dataset
    - SUN-SEG: Download from [SUN-SEG](https://github.com/GewelsJI/VPS), then follow the json files in [data/polyp](data/polyp/) for our splits. 
    - Kvasir-SEG: Download from [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/).
    - PolypGen: Download from [PolypGen](https://www.synapse.org/#!Synapse:syn26376615/wiki/613312).

2. Training
 
- Command:
  
    ```
    CUDA_VISIBLE_DEVICES=1 python train_pedsemiseg.py
    ```
- Some essential hyperparameters:
  
    - *mps* and *mps_type*: Whether apply **Mutual Teaching** and its type (*entropy* is our entropy-guided fusion);
    - *cps*: Apply **Ensemble Learning** or not;
    - Refer to [train_pedsemiseg.py](code/train_s2me.py) for more explanation on other hyperparameters.



## Acknowledgement
Some of the codes are borrowed/refer from below repositories:
- [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)
- [PyMIC](https://github.com/HiLab-git/PyMIC)
- [VPS](https://github.com/GewelsJI/VPS)


