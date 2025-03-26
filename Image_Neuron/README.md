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
```python
#  Define Label Topics
#  Convert text to numerical format using CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(df['Keywords'].dropna())
#  Convert data into Gensim format for coherence scoring
texts = [text.split(",") for text in df['Keywords'].dropna()]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
#  Function to compute coherence score dynamically
def compute_coherence_score(n_topics):
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(X)
    words = vectorizer.get_feature_names_out()
    topics = [[words[j] for j in topic.argsort()[-10:]] for topic in lda_model.components_]
    topic_word_ids = [[dictionary.token2id[word] for word in topic if word in dictionary.token2id] for topic in topics]
    if any(len(topic) == 0 for topic in topic_word_ids):
        return 0  # Prevent empty topics from affecting coherence
    coherence_model = CoherenceModel(topics=topic_word_ids, texts=texts, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()
#  Compute coherence scores for a range of topics
topic_range = range(3, 12)  # Testing topics from 3 to 11
coherence_scores = [compute_coherence_score(n) for n in topic_range]
#  Get the top 3 best topic counts based on coherence scores
top_3_indices = np.argsort(coherence_scores)[-3:][::-1]  # Get indices of top 3 scores
top_3_topics = [topic_range[i] for i in top_3_indices]
print(f" Best 3 topic counts based on coherence score: {top_3_topics}")
#  Function to apply LDA and extract topics dynamically
def apply_lda(n_components, X, vectorizer):
    """Train LDA model and return topic assignments and extracted topics"""
    lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
    lda.fit(X)
    words = vectorizer.get_feature_names_out()
    topics_dict = {}
    # Extract Top Words for Each Topic
    print(f"\n LDA with {n_components} topics:")
    for i, topic in enumerate(lda.components_):
        top_words = [words[j] for j in topic.argsort()[-10:]]  # Get top 10 words per topic
        topics_dict[f"Topic {i+1}"] = top_words
        print(f" Topic {i+1}: {', '.join(top_words)}")
    # Assign topics to each row in the dataset
    topic_assignments = lda.transform(X).argmax(axis=1)
    return topic_assignments, topics_dict
#  Apply LDA for the **top 3 best coherence scores**
topics_summary = {}
for n_topics in top_3_topics:
    df[f'Topic_{n_topics}'], topics_summary[f'Topics_{n_topics}'] = apply_lda(n_topics, X, vectorizer)
```

### Label Topics Results
![image](https://github.com/user-attachments/assets/695fa23b-892e-4487-9e7c-77161d62047d)

### Fit Label Prediction Model
![image](https://github.com/user-attachments/assets/47374c81-dd63-4daa-8554-1af8100d4a2c)

```python
# Define The Label Prediction Model
def train_evaluate_and_visualize(X, y, topic_name):
    print(f"\n Processing {topic_name}...\n{'-' * 50}")

    #  Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    #  Apply SMOTE for balancing classes
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    #  Models and Hyperparameters
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=2000, random_state=42),
            "params": {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            "params": {
                'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2]
            }
        }
    }

    #  Train and Tune Models
    best_models = {}
    for name, details in models.items():
        print(f"\n🔍 Finding best parameters for {name}...")
        grid_search = GridSearchCV(
            details["model"], details["params"],
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_

        print(f"Best parameters for {name}: {grid_search.best_params_}")

    #  Evaluate Models
    performance = {}
    for name, model in best_models.items():
        print(f"\n {name} Classification Report:")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        # Store accuracy and AUC scores
        performance[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'AUC': roc_auc_score(pd.get_dummies(y_test), model.predict_proba(X_test), multi_class='ovr', average='macro')
        }

    #  Save Performance Metrics
    performance_df = pd.DataFrame(performance).T
    print("\n Performance Metrics:")
    print(performance_df)

    #  Plot ROC Curves
    plt.figure(figsize=(10, 8))
    for name, model in best_models.items():
        y_pred_prob = model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(pd.get_dummies(y_test).values.ravel(), y_pred_prob.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multiclass ROC Curve for {topic_name}')
    plt.legend()
    plt.grid()
    plt.show()

    #  Confusion Matrices
    for name, model in best_models.items():
        y_pred_test = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_test)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"{name} - Test Set Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    return performance_df
```

## Model Performance
### Prior Topics Prediction Model
![image](https://github.com/user-attachments/assets/82da199a-6270-46e3-a091-3e66b27f356c)

### Post Topics Prediction Model
#### Mutual Information Scores Of Four Neuron_* Variables
![image](https://github.com/user-attachments/assets/6127cabc-e239-4689-b618-9b0591ed9656)

#### Post Topics Prediction Model Without All Four Neuron_* Variables
![image](https://github.com/user-attachments/assets/9acd551e-cd0f-4063-a071-f11520411859)
#### Post Topics Prediction Model Without Clarity & Cognitive Demand
![image](https://github.com/user-attachments/assets/b13d0e83-3469-4d45-9e12-8f6c7e2d8424)
#### Post Topics Prediction Model With All Four Neuron_* Variables
![image](https://github.com/user-attachments/assets/72b8300f-38ce-4d0d-9e6a-df133a7b5f1f)

### Model Performance Comparison
![image](https://github.com/user-attachments/assets/1d3d52d5-2f6b-4ed4-8873-5598c09813d0)
Based on the label predictions results above and the mutual information score analysis,  had already highlighted the predictive power of Neuron_Focus and Neuron_Engagement, suggesting their strong association with consumer attention. Neuron_Focus and Neuron_Engagement Features have demonstrated strong predictive value and should be prioritized in feature selection. Using neuron-processed image features is crucial for improving predictive accuracy and should be incorporated into future models. This label prediction study demonstrated that incorporating neuron-based features enhances the accuracy of front door image label prediction models. By using machine learning techniques and feature engineering, we identified key variables that contribute to effective classification. 















