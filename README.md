# LATENT DISPLAY TOPIC PREDICTIVE MODELS
## 🔍 Table of Contents
1. Overview
2. Necessary Libraries
3. Data Collection
4. Basic EDA
5. Label Prediction Model
6. Model Performance

## 📌 Overview
This phase focuses on **validating AI models** that predict storefront improvement suggestions using **neuron-processed image insights** and textual features. Designed to empower small businesses with data-driven design optimizations.

## 🧰 Key Features
- **Cross-Category Analysis**: Retail/Restaurant/Salon comparisons  
- **Image Quality Tier Evaluation**: Poor → Excellent performance grading  
- **Neuron-Enhanced Predictions**: Attention heatmap-driven features  
- **Model Benchmarking**: XGBoost vs. Random Forest vs. Logistic Regression  

## 🔧 Methodology
### Model Implementation
| Model | Strength | Use Case |
|-------|----------|----------|
| **XGBoost** | High accuracy | Final deployment |
| **Random Forest** | Feature interpretability | Diagnostic analysis | 
| **Logistic Regression** | Baseline performance | Quick prototyping |

### ⚙️ Optimization
- **Grid Search** with 5-fold cross-validation  
- **Hyperparameter Tuning**: max_depth (XGBoost), n_estimators (Random Forest)  
- **Class Weight Balancing** for imputed suggestion categories

### 🏆 Performance Evaluation:
- **ROC-AUC Analysis**: Measuring model discrimination ability for multi-class classification.
- **Confusion Matrix & Classification Report**: Identifying misclassification patterns and precision-recall trade-offs.
- **Feature Analysis**: Interpreting key contributors to model decisions.

### Expected Outcomes
- Validation of model effectiveness in predicting neuron improvement suggestions.
- Validation of model effectiveness in predicting neuron improvement suggestions.Identification of the best-performing model based on evaluation metrics.
- Validation of model effectiveness in predicting neuron improvement suggestions.Insights into how image aesthetics and attention-based factors influence prediction accuracy.

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
### Image Quality Visualization

**Distribution of Image Quality Class**
![image](https://github.com/user-attachments/assets/e25c0312-6455-4f74-a8ee-8a79d2863d21)
**Distribution of Image Score**
![image](https://github.com/user-attachments/assets/71a5e5c5-ce64-4b3b-a57f-edcfba3fa2a0)
🔍 **Image Quality & Aesthetic Benchmarking**  

**Quality Distribution**:  
- **High-Performance Majority**: 65% of images fall into "Good" (50) or "Excellent" (50) categories  
- **Improvement Targets**: 27% require upgrades ("Fair"=25, "Poor"=19)  

**Cross-Category Analysis**:  
- **Salon Excellence**:  
  - 15% higher **local aesthetic scores** compared to retail/restaurants  
  - Stronger median global scores, indicating superior visual cohesion  

**NIMA Score Insights**:  
- Scores strongly correlate with human perception of visual appeal (r=0.82)  
- **Strategic Thresholds**:  
  - Baseline: Maintain >6.5 NIMA score for core visual competence  
  - Competitive Edge: Target >7.8 for standout storefronts  

**Design Implications**:  
Salons demonstrate how **targeted local enhancements** (e.g., decor density, focal point contrast) can elevate both technical quality and perceived attractiveness—a transferable strategy for other retail categories.

### Image Keywords Visualization
**Code**
```Python
# Generate Word Cloud
def generate_wordcloud_subplot(ax, text, title, colormap):
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(text)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=14)

# Generate Bar Chart
def plot_bar_subplot(ax, text, title, colormap):
    words = text.split()
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(20)
    keywords, counts = zip(*most_common_words)
    
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=min(counts), vmax=max(counts))
    colors = [cmap(norm(value)) for value in counts]
    
    ax.barh(keywords, counts, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_title(title, fontsize=14)
```
**Initial & Enhanced Keyword Visualization**: Identifies strengths/focus areas using traditional methods.

**Initial Keywords**
![image](https://github.com/user-attachments/assets/a74d47f8-0fd9-4866-ae65-9df6c45c7673)
![image](https://github.com/user-attachments/assets/0140e563-85a9-42e8-bd0d-2bd44d5a4596)

**Manual Enhanced Keywords**
![image](https://github.com/user-attachments/assets/01b431b9-fb62-4b12-baf7-b2b7c07b0373)
![image](https://github.com/user-attachments/assets/b447400f-ebb1-4121-afd9-9cbd5c125075)

**Neuron-Processed Image Keyword Visualization**: Leverages AI attention mapping for deep visual insights.
**Neuron Processed Keywords**
![image](https://github.com/user-attachments/assets/db0b047e-004f-442e-8ff0-174693567c5c)
![image](https://github.com/user-attachments/assets/0921a4a5-677b-4ad6-8608-70017728ec0d)

🔍 **Key Insights from Multi-Method Analysis**  

Three analytical approaches (Initial, Manual Enhanced, Neuron) consistently identify **bold signage**, **clear branding**, and **service clarity** as foundational strengths for customer engagement. Enhanced methods reveal additional aesthetic drivers like "inviting signage" and "appealing branding," emphasizing the role of visual appeal in creating welcoming environments.  

**Customer Attention Patterns**:  
- Universal focus on **signage** and **logos** across all analyses  
- Neuron/HyperParameter models uncover functional priorities: **entrance design** and **seating areas**  

**Optimization Opportunities**:  
- Universal: Improved **lighting** and **visibility**  
- Neuron-specific: **Evening visibility** adjustments and **product presentation** enhancements  

**Progressive Insights**:  
Transitioning from basic to advanced analyses reveals refined priorities like **neon elements** for nighttime appeal and targeted lighting strategies. These findings validate that while strong signage remains critical, modern storefront success requires harmonizing core branding with context-aware design adaptations.  

*Recommendation*: Prioritize lighting upgrades and signage boldness while tailoring functional areas (entrance/seating) to your business category.

### Suggestion Keywords Venn Diagram
![image](https://github.com/user-attachments/assets/6d338c29-855d-46ab-b5b8-0d14cd02ee2f)
Then use **Venn diagrams** to illustrate the overlap and uniqueness of Strengths, Focus, and Suggestion keywords identified by the initial, enhanced, and Neuron keywords. The Neuron model consistently captures the largest number of unique keywords across all categories.

### Keywords Among Different Store Categories
![image](https://github.com/user-attachments/assets/b5427df6-911f-4be3-8bc7-a55261a5c6f5)
![image](https://github.com/user-attachments/assets/60d3886f-98f1-4c27-bd61-fdb068c9f283)
🔍 **Cross-Category Insights & Optimization Strategies**  

**Universal Success Factors**:  
- **Bold signage** and **clear branding** emerge as foundational strengths across *all* store types (Retail, Restaurant, Salon)  
- Customer attention consistently prioritizes **signage visibility** and **brand recognition**  

**Category-Specific Differentiators**:  
- *Restaurants*: Benefit from **outdoor seating** and **entrance zone optimization**  
- *Salons*: Gain advantage through **bilingual signage** and **streamlined window displays**  
- *Retail*: Require emphasis on **exterior lighting strategies**  

**Actionable Improvements**:  
- **All Categories**: Implement lighting upgrades and visibility enhancements  
- **Targeted Adjustments**:  
  - Simplify complex window displays (Salons)  
  - Enhance pathway lighting (Retail)  
  - Optimize entrance flow (Restaurants)  

This analysis reveals a **dual strategy** for storefront success: maintain industry-standard visual clarity while implementing *category-tailored* design elements that address unique customer expectations and operational contexts.

### Keywords Among Different Image Quality Classes
![image](https://github.com/user-attachments/assets/15679fb8-6670-4427-aa2d-8211c4ec40ee)
![image](https://github.com/user-attachments/assets/3e46622c-63ae-4ee7-a9a8-a042b2f02af6)
🔍 **Quality-Level Insights: From Poor to Excellence**  

**Universal Foundations**:  
- All stores, regardless of quality tier, succeed with **bold signage** and **clear branding**  
- **Lighting upgrades** and **visibility improvements** remain critical for all  

**Performance Gradients**:  
- *Low-Performing Stores (Poor/Fair)*:  
  - Strengths: Rely on generic elements ("minimalistic design")  
  - Focus: Limited to basic signage visibility  
  - Priorities: **Curb appeal** enhancements and **window simplification**  

- *High-Performing Stores (Good/Excellent)*:  
  - Strengths: Add "service clarity" and aesthetic "signage appeal"  
  - Focus: Diversify attention to **entrance flow** and **seating experience**  
  - Optimization: Refine **promotional displays** and **window organization**  

**Progression Strategy**:  
The analysis reveals an evolution path:  
1. **Base Layer**: Master signage clarity and lighting fundamentals  
2. **Mid-Tier**: Improve functional areas (entrances/windows)  
3. **Excellence**: Perfect experiential elements (seating zones, promotional synergy)  

*Key Takeaway*: Storefront excellence requires maintaining core visual standards while strategically layering sophistication – from visibility basics to experience design – as performance improves.

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















