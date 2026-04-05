# lqcd-density-interpolation-maf
This repository contains the code and plot data for [arXiv:2604.xxxxx]. As lattice MC data is not public at this stage, we provide the setup for the Gaussian Mixture Model (GMM) provided in Appendix B of the manuscript. This includes the input training data, input file, training script and evaluation script for generating the data. We further provide plotting scripts to obtain the figures. 

**Note**: Although we don't provide the lQCD training data, we are providing trained models for the cases shown in the manuscript. Using this the users can readily evaltuate the densities or cumulants as they wish and succesfully generate the data from the manuscript.

Before describe the file structure, we would like to point out a peculiarity of the ML library used in this work - TensorFlow[cite]. The scripts provided in this setup will only work with the older version of TF because of how Keras (used to specify the architecture of the Neural Network (NN)) interacts with TF. After a lot of gymnastics, the following set-up has been independently been shown to be reproduced on three independent machines - includng two HPC clusters and one local mac-based laptop.

## TensorFlow / Keras Compatibility

This project depends on a **specific and non-trivial TensorFlow–Keras setup**.

### Core Environment
- **Python**: 3.10  
- **TensorFlow**: 2.11.0  
- **Keras**: 2.11.0 (via `tf.keras`)  
- **TensorFlow Probability**: 0.19.0  

### !! Important

TensorFlow versions **≥ 2.12 changed how Keras is handled**  
(`keras` vs `tf.keras` separation), which can break this code.

To ensure compatibility, you must include the following **at the very beginning of your script**:

```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
```

## Respository file structure
This describes the set of files you will find in this repository. 

```
GMM/
├── GaussianMixtureModel.ipynb       # Jupyter notebook containing generation of input training data, computing MMD on MAF model generated data and comparison of distribution - with this one can reproduce Figs. 12,13 and 14 of the manuscript.
├── input_eval.txt    # Takes as input the csv file containing training data (for the purpose of normalisation), number of samples needed, minimum and maximum values of parameters and the range of parameters one needs to evaluate the model (for interpolation purposes). Also specified are the directory paths for the trained models and evaluated output data.
├── eval_MAF.py       # Evaluation script using as input the trained models and an input_eval.txt file, the output is the npz files to be used in the Jupyter notebook
├── input_train.txt  # Takes as input the csv file containing training data, rangle of values of parameters for the normalisation, the number of epochs and batch size
└── train_MAF.py     # Training script taking as input the csv files generated from the Jupyter notebook and the input_train.txt file
```

```
Lattice_Nt4/
├── trained_models/
    ├── contains trained models used in Figs. 4,5,6,7,8,9 for Nt_4 ,names of files are descriptive
├── Loss_curves/
    ├──  loss_ep800_bs{xxxx}_nummade8_loop{y}.csv : Each file contains three columns in the order epoch_number, training loss and validation loss. The                                                       names of the files have xxxx as the batch sizes 1024, 2048 and 4096. The loop number y runs over 1, 2
    ├──  loss.ipynb : Usiung the files above and this notebook one can reproduce the left plot in Fig. 10
├── input_eval.txt    # same description as GMM/input_eval.txt
├── eval_MAF.py       # Only change from GMM/eval_MAF.py is the quantities evaluated: here we evaluate the first 4 cumulants of the chiral condensate: mean, variance, skewness and kurtosis
├── input_train.txt  # same description as GMM/input_train.txt
└── train_MAF.py     # same as GMM/train_MAF.py
```
