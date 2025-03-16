# Exploring-Machine-Learning-Techniques-for-Classification-and-Regression
This project explores various machine learning models for classification and regression tasks, including a Multilayer Perceptron for MNIST digit classification and a Random Forest Classifier for census data analysis. It also compares LDA, QDA, Logistic Regression, and SVMs.
Here's an updated README file that includes results for each phase of your project:

---
## Phase 1: Introduction to Machine Learning Basics

This phase focuses on foundational concepts of machine learning, including data preprocessing and feature selection techniques. It lays the groundwork for understanding how machine learning models are developed and evaluated.

- **Key Concepts**: Data preprocessing, feature selection, and introduction to classification and regression problems.
- **Tools Used**: Python libraries such as Pandas, NumPy, and Scikit-learn.
- **Outcomes**: Understanding of basic machine learning workflows and data preparation techniques.

## Phase 2: Handwritten Digit Classification with Neural Networks

In this phase, a Multilayer Perceptron (MLP) is implemented for handwritten digit classification using the MNIST dataset. The project involves designing and training a neural network with custom functions for key tasks such as weight initialization and optimization.

- **Key Features**:
  - **Neural Network Architecture**: A simple MLP with one hidden layer.
  - **Optimization Technique**: Conjugate Gradient optimization.
  - **Hyperparameter Tuning**: Experimentation with different numbers of hidden units and regularization parameters.
- **Results**:
  - **Training Accuracy**: Up to 94.01% with 20 hidden units and lambda=10.
  - **Validation Accuracy**: Up to 92.88% with similar settings.
  - **Test Accuracy**: Up to 93.26% with optimal hyperparameters.
- **Tools Used**: Python with TensorFlow or PyTorch for neural network implementation.

## Phase 3: Classification and Regression with Gaussian Discriminators

This phase explores the application of Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) for classification tasks. Additionally, it includes experiments with linear regression models, comparing the performance with and without an intercept term.

- **Key Features**:
  - **LDA and QDA**: Tested on a synthetic dataset, highlighting their strengths in handling different covariance structures.
  - **Linear Regression**: Comparison of models with and without an intercept term.
- **Results**:
  - **LDA Accuracy**: 0.97
  - **QDA Accuracy**: 0.94
  - **Linear Regression MSE**:
    - Without Intercept: High MSE (19099.4468 on training, 106775.36 on test).
    - With Intercept: Lower MSE (2187.1603 on training, 3707.84 on test).
- **Tools Used**: Python with Scikit-learn for LDA, QDA, and linear regression.

## Phase 4: Classification with Logistic Regression and SVMs

The final phase involves comparing the performance of Logistic Regression, Multi-class Logistic Regression, and Support Vector Machines (SVMs) on the MNIST dataset.

- **Key Features**:
  - **Logistic Regression**: Implemented using both One-vs-All and Multi-class strategies.
  - **SVMs**: Tested with linear and RBF kernels, varying the penalty parameter C.
- **Results**:
  - **One-vs-All Logistic Regression**:
    - Training Accuracy: 92.67%
    - Validation Accuracy: 91.45%
    - Test Accuracy: 91.99%
  - **Multi-Class Logistic Regression**:
    - Training Accuracy: 93.45%
    - Validation Accuracy: 92.48%
    - Test Accuracy: 92.55%
  - **SVMs**:
    - Linear Kernel: High accuracy on testing (93.78%).
    - RBF Kernel with Default Gamma: High accuracy on testing (97.87%).
- **Tools Used**: Python with Scikit-learn for logistic regression and SVM implementation.

