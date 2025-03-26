# Store Front Image Attention & Aesthetics Analysis With Neuron Flatform
Table of Contents
1. Overview
2. Necessary Libraries
3. Data Collection
4. Basic EDA
5. Label Prediction Model
6. Model Performance

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

## Necessary Libraries
### Front Door Image Info_EDA.ipynb
```python
import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
import missingno as msno
import pyogrio 
from collections import Counter
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
from matplotlib.colors import LinearSegmentedColormap
import nltk
```
### Prior_Image_Label_Prediction.ipynb & Post_Image _Label_Prediction_Neuron.ipynb
```python
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputert
from imblearn.over_sampling import SMOTE 
```

## Data Collection
### Data Source
144 images were collected, covering three store types: restaurant, retail, and Salon.

### Location Information
These stores are located in the Bay Area of ​​California, including San Jose, Santa Clara, Sunnyvale, Milpitas, Cupertino, Sonoma, Campbell, etc.
![image](https://github.com/user-attachments/assets/f8b9c2f5-13d7-4f9f-bf75-0b3bfd098525)

### Metadata
The metadata contains additional image information (image name, store name) and store attributes (such as store name, category, geographical location, etc.), image quality scores (such as NIMA Score, Paq2piq Scores), and related image keywords (Strength, Human Eyes Focus, Suggestions) for subsequent analysis. 

![image](https://github.com/user-attachments/assets/e5275bbd-e96e-4a9b-ba18-0cfc8dbbf30e)
### Neuron AI Processed Image
![IMG_7515](https://github.com/user-attachments/assets/c5b47e22-9a7b-48a9-892d-e7ba4444f79f)
Examples:  Storefront Image Prior Version(left) and Post Neurons Version(right)

## Basic EDA
### Distribution of Image Quality Class
![image](https://github.com/user-attachments/assets/e25c0312-6455-4f74-a8ee-8a79d2863d21)

### Distribution of Image Score
![image](https://github.com/user-attachments/assets/71a5e5c5-ce64-4b3b-a57f-edcfba3fa2a0)

### Prior Store Front Image Keywords
![image](https://github.com/user-attachments/assets/a74d47f8-0fd9-4866-ae65-9df6c45c7673)
![image](https://github.com/user-attachments/assets/0140e563-85a9-42e8-bd0d-2bd44d5a4596)

### Manual Hyperparameter Improved Store Front Image Keywords
![image](https://github.com/user-attachments/assets/01b431b9-fb62-4b12-baf7-b2b7c07b0373)
![image](https://github.com/user-attachments/assets/b447400f-ebb1-4121-afd9-9cbd5c125075)

### Post Neuron Processed Store Front Image Keywords
![image](https://github.com/user-attachments/assets/db0b047e-004f-442e-8ff0-174693567c5c)
![image](https://github.com/user-attachments/assets/0921a4a5-677b-4ad6-8608-70017728ec0d)

### Suggestion Keywords Venn Diagram
![image](https://github.com/user-attachments/assets/6d338c29-855d-46ab-b5b8-0d14cd02ee2f)

### Keywords Among Different Store Categories
![image](https://github.com/user-attachments/assets/b5427df6-911f-4be3-8bc7-a55261a5c6f5)
![image](https://github.com/user-attachments/assets/60d3886f-98f1-4c27-bd61-fdb068c9f283)

### Keywords Among Different Image Quality Classes
![image](https://github.com/user-attachments/assets/15679fb8-6670-4427-aa2d-8211c4ec40ee)
![image](https://github.com/user-attachments/assets/3e46622c-63ae-4ee7-a9a8-a042b2f02af6)

## Label Prediction Model
### Define Label Topics
![image](https://github.com/user-attachments/assets/daf75d25-6c80-4896-8f1e-e76047aa9f19)

### Label Topics Results
![image](https://github.com/user-attachments/assets/695fa23b-892e-4487-9e7c-77161d62047d)

### Fit Label Prediction Model
![image](https://github.com/user-attachments/assets/47374c81-dd63-4daa-8554-1af8100d4a2c)

## Model Performance
### Prior Topics Prediction Model
![image](https://github.com/user-attachments/assets/82da199a-6270-46e3-a091-3e66b27f356c)

### Post Topics Prediction Model
#### Mutual Information Scores Of Four Neuron_* Variables
![image](https://github.com/user-attachments/assets/6127cabc-e239-4689-b618-9b0591ed9656)

#### Post Topics Prediction Model Without All Four Neuron_* Variables
![image](https://github.com/user-attachments/assets/9e6094a1-7ce7-422c-80c1-4bb2c584da74)
#### Post Topics Prediction Model Without Clarity & Cognitive Demand
![image](https://github.com/user-attachments/assets/6041986c-caca-446a-9486-0e32dd00367c)
#### Post Topics Prediction Model With All Four Neuron_* Variables
![image](https://github.com/user-attachments/assets/ec2c54b2-d773-44db-9153-d125a668b366)













