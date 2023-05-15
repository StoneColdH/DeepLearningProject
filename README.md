<<<<<<< HEAD
# GTAN
implementation of paper:
****.

## Overview

<p align="center">
    <br>
    <a href="https://github.com/StoneColdH/DeepLearningProject">
        <img src="https://github.com/StoneColdH/DeepLearningProject/blob/main/figures/framework_tgtn.png" width="900"/>
    </a>
    <br>
<p>



## Setup

You can download the project and install the required packages using the following commands:

```bash
git clone https://github.com/StoneColdH/DeepLearningProject
cd DeepLearningProject
conda create -n antifraud -c nvidia python=3.7 cudatoolkit=11.3
conda activate antifraud
pip3 install numpy pandas sklearn
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
```

## Usage

1. In GTAN directory, run `unzip /data/data.zip` to unzip the datasets if you want to train model on these three datasets or pass this step;
2. Run `python data_process.py` to generate adjacency lists and additional features on FFSD dataset used by GTAN;
3. Run `python train.py` to run GTAN-GNN with default settings.

For other dataset and parameter settings, please refer to the arg parser in `train.py`. For example,you can run `python train.py --dataset yelp` to use YelpChi dataset. 



## Repo Structure
The repository is organized as follows:
- `data/`: dataset files;
- `data_process.py`: handle dataset and do some feature engineering;
- `layers.py`: GTAN-GNN layers implementations;
- `model.py`: GTAN-GNN model implementations;
- `train.py`: training and testing models;
- `utils.py`: utility functions for data i/o and model early stopper.

