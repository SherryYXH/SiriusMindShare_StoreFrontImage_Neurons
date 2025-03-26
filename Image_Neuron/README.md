# Store Front Image Attention & Aesthetics Analysis With Neuron Flatform
Table of Contents
1. Overview
2. Data Collection
3. Basic EDA
4. Prediction Model
5. Model Performance

## Overview
This phase of the research focuses on validating the label prediction models used to classify storefront neuron improvement suggestions. By evaluating multiple classification models, the goal is to determine the most effective approach for predicting enhancement recommendations based on image and text features.
### Methodology
Model Training & Prediction:
Implementing three classification models:
1. Logistic Regression
2. Random Forest
3. XGBoost
Training models on extracted image features and text features.

Model Optimization：
Identify the best hyperparameters for different classification models using Grid Search with Cross-Validation. 

Performance Evaluation:
1. ROC-AUC Analysis: Measuring model discrimination ability for multi-class classification.
2. Confusion Matrix & Classification Report: Identifying misclassification patterns and precision-recall trade-offs.
3. Feature Analysis: Interpreting key contributors to model decisions.

