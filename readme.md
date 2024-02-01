# Master Thesis of Amro
## Overview

This repository contains the code and resources for the Amro's Master Thesis project. The project aims to investigate and implement various techniques for assessing the damage state of bridge.

## Introduction
Modes of the Z24 bridge. All supports are chosen as fixed supports. Fixed supports are chosen at pier bases, as well as the two ends of the bridge that merge with the road. 
![Modal shapes](https://drive.usercontent.google.com/download?id=12GrWRltz42P_b4djttd1gvW2yIQ7KqEy&export=view&authuser=0)

## Requirements (Chapter 2 and 3)
- <b> Python 3.11.5 </b>

<div align="center">

| Package     | Version              |
|-------------|----------------------|
| numpy       | 1.26.1               |
| matplotlib  | 3.7.2                |
| sklearn     | 1.3.0                |
| xgboost     | 1.7.6                |
| pandas      | 2.0.3                |
| scipy       | 1.11.2               |
| seaborn     | 0.12.2               |
| tsfresh     | 0.20.1               |
| skrebate    | 0.62                 |
| river       | 0.19.0               |
| plotly      | 5.17.0               |

</div>

- <b> Path system variable (Windows) </b> 
<div align="center">

![System variable](https://drive.usercontent.google.com/download?id=1GjgFIP7-BKzdv5xZ_BG8s1A3C_Arkjcf&export=view&authuser=0) 

</div>

For MacOS users, refer to https://phoenixnap.com/kb/set-environment-variable-mac.

The only two files required to be run are eda.ipynb in Chapter2 folder and river_experiments_occ.ipynb in Chapter3-ActiveLearning folder. Download Z24 data files for Chapter2-Z24-dataset from https://polybox.ethz.ch/index.php/s/8T6Lu8Hi8VqJcze and extract them so that folder /data/ exists, whose content is folders 01 to 17. Set system environment variable to point to path of /data folder. Replace path with the path in your computer of folder data. 


Once the eda.ipynb has ran, you should have as output X_train_new.npy, labels_train_new.npy, X_test_new.npy, and labels_test_new.npy. Save these inside a folder called /features inside the directory referenced by environment variable Z24_DATA. You can also find these numpy files here https://polybox.ethz.ch/index.php/s/IdGWA8OKFVE0lfa. Next, run the river_experiments_occ_new.ipynb (inside Chapter3-ActiveLearning) notebook to get the outputs for the online learning segment of the project. 

## Ansys Simulated Vibration Data Generation (Chapter 4)

To get the generated Excel data of the frequency response in Chapter4-PhysicalSimulation, follow this tutorial video https://polybox.ethz.ch/index.php/s/iAWIzudH6P3gF8K.
Note, in the tutorial, I only do it for a few of the channels, though doing it for all channels is self-explanatory. 
![Ansys screenshot](https://drive.usercontent.google.com/download?id=1Ig5SJIwKs5HkKpB3Jd53_PWp2A9bHNTi&export=view&authuser=0)

## Installation
The notebooks were run inside Visual Studio Code, where the working directory contains the folders for all the chapters. For Ansys 2023 WB, the installation along with instructions can be found at the ETH IT Shop https://itshop.ethz.ch/ (requires VPN connection to run)
## Usage
## Contributing
## Acknowledgments
Dr. Cyprien Hoelzl, Dr. Yves Reuland, Dr. Christos Lataionitis, Dr. Panagiotis Martakis, Prof. Dr. Eleni Chatzi
## Contact
amro.abdrabo@gmail.com
