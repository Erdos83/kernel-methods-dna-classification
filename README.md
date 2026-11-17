# Kernel Methods for DNA Sequence Classification

## Overview

This project explores the application of kernel methods to DNA sequence classification using Kernel Ridge Regression (KRR) with a Gaussian Radial Basis Function (RBF) kernel.

The objective is to classify DNA fragments originating from COVID-19 genomes, human genomes, or bacterial genomes using similarity-based machine learning methods.

## Dataset

The dataset consists of short DNA sequencing reads, approximately 100–300 base pairs long, obtained from sequencing experiments or simulated from complete genomes.

The challenge provides:

- **2,000 labeled training samples**
- **1,000 unlabeled test samples**

Since the test labels were unavailable, the original training dataset was divided into **1,500 training samples and 500 validation samples** to evaluate model performance and tune hyperparameters before fitting the final model on all 2,000 labeled samples.

## Mathematical Formulation

### Gaussian (RBF) Kernel

The Gaussian kernel measures similarity between two feature vectors:

```math
k(x_i,x_j)=\exp\left(-\frac{\|x_i-x_j\|^2}{2\sigma^2}\right)
```

where $\sigma$ is the kernel bandwidth controlling the influence of distance on similarity.

### Kernel Ridge Regression

Kernel Ridge Regression combines kernel-based similarity with regularization. The model coefficients are obtained by solving:

```math
(K+\lambda I)\alpha=y
```

where $K$ is the kernel matrix, $\lambda$ is the regularization parameter, and $y$ contains the training labels.

For a new observation $x$, the prediction is:

```math
f(x)=\sum_{i=1}^{n}\alpha_i k(x_i,x)
```

## Implementation and Evaluation

The project implements Gaussian kernel computation and Kernel Ridge Regression using Python and NumPy.

Model development includes training-validation splitting, hyperparameter tuning, prediction, and classification accuracy evaluation.

Exploratory visualizations are used to examine the feature distributions and kernel-based similarity relationships.

## Technologies

Python · NumPy · Pandas · Matplotlib · Jupyter Notebook

## Conclusion

This project demonstrates how kernel-based learning can be applied to DNA sequence classification, connecting nonlinear similarity measures, regularization, and statistical learning with a real-world biological classification problem.