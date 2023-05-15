# GTAN
implementation of paper:
****.

## Overview

<p align="center">
    <br>
    <a href="https://github.com/StoneColdH/DeepLearningProject">
        <img src="https://github.com/StoneColdH/DeepLearningProject/blob/main/figures/framework_gtan.png" width="900"/>
    </a>
    <br>
<p>

Gated Temporal Attention Network
**G**ated-**T**emporal **A**ttention **N**etwork **(GTAN)** is a semi-supervised GNN-based fraud detector 
via attribute-driven graph representation.


Three contributions of our work are:
- **Semi-supervised task** We model credit card behaviors as a temporal transaction graph and formulate a credit card fraud detection problem as a semi-supervised node classification task;
- **Attribute-driven temporal GNN** We present a novel attribute-driven temporal graph neural network for credit card fraud detection;
- **New dataset and superiority on fraud detection** We contribute a new large-scale semi-supervised credit card fraud detection dataset. Extensive experiments conducted on three fraud detection datasets show the superiority of our proposed GTAN;

GTAN has following advantages:
- **Superiority.** GTAN gets the best metric with all comparison methods on the fraud detection task;
- **Robust.** GTAN allows stable detection with very limited data labels;
- **Practicality.** The real-world case studies demonstrate our proposed method’s effectiveness in detecting real-world fraud patterns;


## Setup

You can download the project and install the required packages using the following commands:

```bash
git clone https://github.com/StoneColdH/DeepLearningProject
cd GTAN
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

**Please note that our FFSD dataset has millions of transactions, such that we put a sampled dataset (10k transactions) in this repository.**

## Repo Structure
The repository is organized as follows:
- `data/`: dataset files;
- `data_process.py`: handle dataset and do some feature engineering;
- `layers.py`: GTAN-GNN layers implementations;
- `model.py`: GTAN-GNN model implementations;
- `train.py`: training and testing models;
- `utils.py`: utility functions for data i/o and model early stopper.
