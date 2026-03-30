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

## Files in this repo
This describes the set of files you will find in this repository. 

