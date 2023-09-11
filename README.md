# UsoDeFramework_Uresti

This repository includes an implementation of Gradient Boosting Classifier on a stroke predicition dataset, the implementation is based on the scikit-learn implementaiton of the algorith. In the repository, there is also an analysis and report adressing common problems with machine learning models, such as bias, variance, overfitting (or underfitting) and some good practices to make a better algorithm such as division of the data set in train, validation and test sets, adjustment of hyperparameters and use of regularization techniques.


The main files in this repository are:
- README.md: This file describing the contents and function of the repository
- modelo.py: Is the file containing the model implementation and analysis of the model. It is mandatory to run de file in the same directory as the dataset, otherwise it will fail it's execution. This file, when ran, automatically iniciates the algortithm using the dataset determined in the code (stroke_data.csv). You only need to run it and it will automatically run and fit the model, return and print all the parts of the analysis, for a better undestanding of the analysis please refer to the report. CAUTION: Patience is key in the execution of the file; if a figure is shown in screen it must be closed to continue the code execution.

This file, when ran, automatically iniciates the algorith usign the dataset determined in the code (stroke_data.csv). You only need to run it and it will automatically make the predicitons using said dataset and will print the predicitions, the accuracy and the total of correct predictions. This is an updated version that includes: minor fixes, more evaluation metrics, deeper code documentation, comparison charts and a brief conclusion.

