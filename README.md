# BINF5507 Machine Learning and AI for Bioinformatics

This is a repository containing the laboratory and assignment code for the Machine Learning and AI for Bioinformatics course. Listed below are short insights on the ongoing Assignments for the course, as well as how to use them.

## Assignment 1: Object-Oriented Data Cleaning and Preprocessing

### Project Overview

This project focuses on cleaning a messy dataset and evaluating how preprocessing affects the performance of a simple logistic regression model. It includes:

- Handling missing values through imputation
- Removing duplicate rows
- Normalizing numerical features
- Encoding categorical features
- Removing redundant/highly correlated features
- Training and evaluating a logistic regression model

### How to Run the Project
This project used the following Python packages:
- pandas = ">=2.2.3,<3"
- numpy = ">=2.0.2,<3"
- scikit-learn = ">=1.6.1,<2"

An pixi.toml file has also been included for use with the package manager Pixi by prefix.dev.

To run the code, use the main.ipynb notebook loacted within BINF5507/Assignment1/Scripts. 

The first cell will allow you to import the necessary modules, load the dataset, preprocess using the abovementioned methods and train the model with different preprocessing and model arguments if needed. 

The second cell will allow you to check the summary statistics of the cleaned vs messy datasets, as well as the removed rows and features.

## Assignment 2: Regression and Classification on Heart Disease Data

### Project Overview

This project applies both regression and classification techniques on a heart disease dataset obtained from UCI. The goals are to predict cholesterol levels using an ElasticNet regression model and to classify the presence of heart disease using logistic regression and k-Nearest Neighbors (k-NN). The project demonstrates:

- Preprocessing the dataset using a pipeline that handles imputation, scaling, and encoding.
- Splitting the data into training and test sets, with a focus on preventing data leakage.
- Tuning hyperparameters for a linear regression model with ElasticNet, including exploring a range of alpha values and L1 ratios, and visualizing model performance through heatmaps of R² and RMSE.
- Evaluating classification performance by using GridSearchCV for logistic regression and k-NN, and comparing metrics such as accuracy, F1 Score, AUROC, and AUPRC.
- Plotting ROC and Precision-Recall curves with AUC/AUPRC annotations to compare model performance.

### How to Run the Project
This project uses several Python packages namely:
- pandas = ">=2.2.3,<3"
- numpy = ">=2.0.2,<3"
- scikit-learn = ">=1.6.1,<2"
- matplotlib = ">=3.9.4,<4"
- seaborn = ">=0.13.2,<0.14"

A pixi.toml file is also provided for use with package management using Pixi by prefix.dev.

To run the code, navigate to the corresponding project folder (e.g., BINF5507/Assignment2/Scripts) and open the Jupyter notebook that contains the main pipeline. The notebook is organized into sections for data loading, preprocessing (with an object-oriented pipeline), model training, hyperparameter tuning, and visualization of evaluation metrics.

The code begins by loading the heart disease dataset, checking for missing values, and then applying a preprocessing pipeline that imputes missing values, scales numeric features, and encodes categorical features. The project then splits the data into training and test sets, trains a linear regression model to predict cholesterol levels, and visualizes the effect of different hyperparameters using heatmaps. Moreover, classification models (Logistic Regression and K-NN classifiers) are then tuned and evaluated using GridSearchCV, with ROC and Precision-Recall curves generated to assess performance of said models.
