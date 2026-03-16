# Heart Disease Classification with a Simple Neural Network

## Introduction

I intended to built a neural network from scratch using NumPy, largely for the purpose of exploring the concepts behind how neural networks operation; and in addition, see how powerful of a tool it really is. For the purposes of training and validating, I used the UCI Heart Disease dataset to predict 
presence of heart disease using a 3-layer neural network with ReLU activations 
and binary cross-entropy loss.

Although there are wonderful tools like pytorch, tensorflow, and sklearn, I thought that experiencing what is going on in a neural network on a fundamental level is a worthwhile experience. This project also serves as a test of my python programming ability, data-related skills, and linear algebra knowledge.

The core of this repository is simpleNN.py, where the code for a basic neural net module that applies to this project is written. In addtion, all exploration and learning can be found in testSNN.ipynb.

## Data
The data I used can be found in the repo as HeartDiseaseTrain-Test.csv, I downloaded this data from Ketan Gangal's public post on kaggle found at https://www.kaggle.com/datasets/ketangangal/heart-disease-dataset-uci.

The csv file I found is directly from the credited source, any and all changes (for example data wrangling) can be found in testingSNN.ipynb. 