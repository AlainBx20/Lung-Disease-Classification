# Lung-Disease-Classification
Lung Disease Classification using SVM



This project is a machine learning-based lung disease classification system that uses images of normal and diseased lungs to predict lung health. 
The system utilizes a Support Vector Machine (SVM) with a linear kernel for binary classification.

# Project Overview

The goal of this project is to classify lung images into two categories:

Normal: Images showing healthy lungs[0].

Lung Opacity: Images showing signs of lung opacity (a common symptom of lung diseases such as pneumonia)[1].

The model processes images from two folders (Normal and Lung_Opacity), preprocesses them, and trains a classification model using the Support Vector Machine (SVM).

Installation
To run the project locally, you need to install the following dependencies:

                                                 pip install scikit-learn scikit-image numpy joblib
                                                 
# Dataset

The dataset consists of images categorized into two folders:

Normal: Contains images of healthy lungs.

Lung_Opacity: Contains images of lungs with opacity (disease signs).

both can be found in (lung.rar) make sure to extract them in the same folder as the source codes !


Usage
1) Preprocessing and Model Training

 
The code provided (lung.py) loads images, resizes them, normalizes them, and then flattens them into 1D arrays. The dataset is split into training and testing sets, and an SVM model is trained on the images.

2) Model Inference:

Once the model is trained and saved, you can use it to predict on new images by loading the model and running inference with ( run_test.py )


Model Evaluation

The model is evaluated on the testing dataset, and the performance is reported using:

Classification Report: Displays precision, recall, and F1-score for each class.

Confusion Matrix: A matrix showing the number of true positives, true negatives, false positives, and false negatives


# License
This project is licensed under the MIT License - see the LICENSE file for details.

