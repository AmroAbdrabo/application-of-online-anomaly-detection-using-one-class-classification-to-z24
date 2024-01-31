# Master Thesis of Amro
## Overview

This repository contains the code and resources for the "Amro's Student Thesis" project. The project aims to investigate and implement various techniques for assessing the damage state of bridge.

## Table of Contents
## Introduction

The only two files required to be run are eda.ipynb in Chapter2 folder and river_experiments_occ.ipynb in Chapter3 folder.\
Download Z24 data files for Chapter2-Z24-dataset from https://polybox.ethz.ch/index.php/s/8T6Lu8Hi8VqJcze and extract them so that folder /data/ exists, whose content is folders 01 to 17

## Requirements 
- Python 3.11.5

Set system environment variable. Replace path with the path in your computer of folder data. \
![System variable](https://drive.usercontent.google.com/download?id=1GjgFIP7-BKzdv5xZ_BG8s1A3C_Arkjcf&export=view&authuser=0) \
For Chapter2-Z24-dataset\eda.ipynb 

<div align="center">

| Package     | Version              |
|-------------|----------------------|
| numpy       | 1.26.1               |
| matplotlib  | 3.7.2                |
| sklearn     | 1.3.0                |
| os          | version not available|
| random      | version not available|
| warnings    | version not available|
| xgboost     | 1.7.6                |
| pandas      | 2.0.3                |
| scipy       | 1.11.2               |
| seaborn     | 0.12.2               |
| tsfresh     | 0.20.1               |
| skrebate    | 0.62                 |
| river       | 0.19.0               |
</div>

\
Once the eda.ipynb is ran, you should have as output X_train_new.npy, labels_train_new.npy, X_test_new.npy, and labels_test_new.npy. Save these inside a folder called /features 
inside the directed referenced by environment variable Z24_DATA. Next, run the river_experiments_occ_new.ipynb (inside Chapter3-ActiveLearning) notebook to get the outputs for the online learning segment of the project. 

### Ansys Simulated Vibration Data Generation

To get the geenrated data in Chapter4-PhysicalSimulation, follow this tutorial video https://polybox.ethz.ch/index.php/s/iAWIzudH6P3gF8K.
Note, in the tutorial, I only do it for a few of the channels, though doing it for all channels is self-explanatory. 


## Installation

## Usage
## Contributing
## Acknowledgments
Dr. Cyprien Hoelzl, Dr. Yves Reuland, Dr. Christos Lataionitis, Dr. Panagiotis Martakis, Prof. Dr. Eleni Chatzi
## Contact
amro.abdrabo@gmail.com
