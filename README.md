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
