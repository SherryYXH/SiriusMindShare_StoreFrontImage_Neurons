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

### Expected Outcomes
1. Validation of model effectiveness in predicting neuron improvement suggestions.
2. Identification of the best-performing model based on evaluation metrics.
3. Insights into how image aesthetics and attention-based factors influence prediction accuracy.

By systematically comparing these models, this validation step ensures the reliability of AI-driven recommendations, supporting small businesses in optimizing their storefront designs through data-driven insights.

## Data Collection
### Data Source
144 images were collected, covering three store types: restaurant, retail, and Salon.
### Location Information
These stores are located in the Bay Area of ​​California, including San Jose, Santa Clara, Sunnyvale, Milpitas, Cupertino, Sonoma, Campbell, etc.
![Plot1_Store Distribution](https://github.com/user-attachments/assets/6c3709dc-b042-4bef-9cda-f26d41662155)
### Metadata
The metadata contains additional image information (image name, store name) and store attributes (such as store name, category, geographical location, etc.), image quality scores (such as NIMA Score, Paq2piq Scores), and related image keywords (Strength, Human Eyes Focus, Suggestions) for subsequent analysis.
![image](https://github.com/user-attachments/assets/e5275bbd-e96e-4a9b-ba18-0cfc8dbbf30e)


