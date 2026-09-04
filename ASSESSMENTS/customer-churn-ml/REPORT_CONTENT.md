# Academic Report: Customer Churn Prediction and Segmentation System

---

## 1. Abstract
Customer attrition (churn) is a critical challenge in the telecommunications industry, where acquiring new customers costs significantly more than retaining existing ones. This project develops a complete, end-to-end Machine Learning system that combines supervised classification models to predict customer churn probability and unsupervised K-Means clustering to discover distinct customer service segments. Using the IBM Telco Customer Churn dataset (7,043 customer records), three supervised algorithms (**Logistic Regression**, **Decision Tree Classifier**, and **Support Vector Machine**) were trained and evaluated on an 80/20 stratified train/test split. **Logistic Regression** achieved the best classification performance with an **F1-Score of 0.6040**, **Accuracy of 80.55%**, **Precision of 65.72%**, and **Recall of 55.88%**. For unsupervised customer segmentation, K-Means clustering (evaluated across K = 2 to 10 using the Elbow Method and Silhouette Analysis) identified **K = 4** as the optimal cluster count. Principal Component Analysis (PCA) was applied to project the high-dimensional customer feature space onto 2D coordinates for visual interpretation. **Cluster 1** emerged as the **High-Risk Segment**, exhibiting a **57.03% churn rate** (compared to the baseline dataset average of 26.54%) characterized by short tenure (~15.6 months) and high monthly charges ($84.60/month). Targeted customer retention strategies were formulated based directly on empirical segment characteristics.

---

## 2. Introduction
In competitive service industries such as telecommunications, subscriber retention is paramount to long-term profitability. Machine Learning techniques offer powerful predictive tools to identify patterns associated with customer dissatisfaction before cancellation occurs. This project integrates both supervised machine learning for individual churn prediction and unsupervised machine learning for strategic customer cohort profiling.

---

## 3. Problem Statement
A telecommunications provider is experiencing significant subscriber churn. The company requires an integrated Machine Learning system that:
1. Accurately predicts whether a customer will discontinue service (`Churn = Yes/1` or `No/0`).
2. Benchmarks three supervised learning algorithms (**Logistic Regression**, **Decision Tree**, **SVM**).
3. Selects the optimal classification model using **F1-score** to balance Precision and Recall.
4. Profiles customer behavioural patterns using **K-Means Clustering** without target leakage.
5. Utilizes **PCA** to visualize customer segments in 2D space.
6. Flags high-risk customer groups and prescribes data-driven retention interventions.

---

## 4. Objectives
- Preprocess and clean real-world customer attributes (handling missing values, encoding categoricals, scaling numerical features).
- Perform Exploratory Data Analysis (EDA) to highlight key risk factors (tenure, contract type, payment method).
- Build and evaluate supervised classification models on an 80/20 stratified train/test split.
- Determine optimal K-Means cluster count via Elbow Method and Silhouette Scores.
- Map cluster profiles to business risk categories and formulate retention recommendations.

---

## 5. Pseudocode / Workflow

```text
START

1. Load customer churn dataset ('WA_Fn-UseC_-Telco-Customer-Churn.csv').
2. Inspect dataset dimensions, data types, and missing values.
3. Convert TotalCharges from object/string to numeric (coercing blank strings to NaN).
4. Impute missing TotalCharges values with column median.
5. Remove uninformative identifier column 'customerID'.
6. Map target variable Churn (Yes -> 1, No -> 0).
7. Construct ColumnTransformer:
   - Apply StandardScaler to numerical features (tenure, MonthlyCharges, TotalCharges).
   - Apply OneHotEncoder to categorical features.
8. Perform Exploratory Data Analysis (EDA) and save visual plots.
9. Split data into Training Set (80%) and Testing Set (20%) using stratified sampling (random_state=42).

10. FOR EACH classification algorithm (Logistic Regression, Decision Tree, SVM):
      a. Fit model on preprocessed training data.
      b. Predict churn labels on test dataset.
      c. Compute Accuracy, Precision, Recall, and F1-Score.
      d. Generate and save Confusion Matrix.
    END FOR

11. Construct classification model comparison table.
12. Select best-performing model based on highest F1-Score (Logistic Regression).

13. Prepare clustering feature matrix:
      - EXCLUDE customerID and Churn target variable to prevent data leakage.
      - Fit StandardScaler on clustering features.

14. FOR K = 2 to 10:
      a. Train K-Means model (random_state=42).
      b. Calculate inertia (within-cluster sum of squares).
      c. Calculate Silhouette Score.
    END FOR

15. Plot Elbow curve and Silhouette analysis graph; select optimal K = 4.
16. Fit final K-Means model with K = 4 and assign cluster labels.
17. Calculate cluster summary statistics (Avg tenure, MonthlyCharges, TotalCharges, Churn Rate).
18. Apply Principal Component Analysis (PCA, n_components=2) for 2D visual cluster scatter plot.
19. Identify High-Risk Customer Segments (Churn Rate > baseline 26.54%).
20. Formulate targeted customer retention recommendations.
21. Export results to CSV tables and figures to output directory.

END
```

---

## 6. Dataset Description
The IBM Telco Customer Churn dataset contains 7,043 rows and 21 features:
- **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Services**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Account Details**: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Target**: `Churn` (`Yes` / `No`)

---

## 7. Dataset Source
- **Public Reference**: IBM Telco Customer Churn Sample Dataset.
- **Repository URL**: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`

---

## 8. Data Preprocessing
- **Missing Value Handling**: Inspection revealed 11 blank string entries (`" "`) in `TotalCharges` corresponding to new customers with `tenure = 0`. These were converted to `NaN` via `pd.to_numeric(errors='coerce')` and imputed using the median `TotalCharges` ($1,397.47).
- **Identifier Removal**: `customerID` was dropped prior to training and clustering.
- **Scaling & Encoding**: A `ColumnTransformer` applied `StandardScaler()` to continuous numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) and `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to categorical attributes.

---

## 9. Exploratory Data Analysis (EDA)
Key findings from visual analysis:
1. **Overall Churn Rate**: 26.54% of customers churned (1,869 out of 7,043).
2. **Tenure Relationship**: Churn is heavily concentrated among new customers (tenure < 12 months).
3. **Contract Impact**: Month-to-month contract holders have a significantly higher churn rate compared to one-year and two-year contract subscribers.
4. **Internet Service & Payment Method**: Fiber optic internet subscribers and customers paying via Electronic Check demonstrate elevated churn rates.

---

## 10. Machine Learning Algorithms
Three supervised classifiers were selected for comparison:
1. **Logistic Regression**: Linear classifier optimized with `max_iter=1000`, offering high interpretability and robust performance on normalized feature spaces.
2. **Decision Tree Classifier**: Non-linear tree model (`max_depth=5`, `random_state=42`) providing rule-based decision boundaries.
3. **Support Vector Machine (SVM)**: Kernel classifier (`kernel="rbf"`, `probability=True`, `random_state=42`) mapping data into higher-dimensional feature space.

---

## 11. Implementation / Code

### Data Preprocessing Pipeline (`src/preprocessing.py`)
```python
def clean_data(df):
    df_clean = df.copy()
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median())
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])
    return df_clean
```

### Classification Training (`src/classification.py`)
```python
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}
```

---

## 12. Model Evaluation
The models were evaluated on the 20% test dataset (1,409 samples: 1,035 No Churn, 374 Churn).

### Confusion Matrix Definitions
- **True Positive (TP)**: Correctly predicted churners.
- **True Negative (TN)**: Correctly predicted non-churners.
- **False Positive (FP)**: Non-churners incorrectly flagged as churners.
- **False Negative (FN)**: Actual churners missed by the model.

### Empirical Confusion Matrices
- **Logistic Regression**: TN = 926, FP = 109, FN = 165, TP = 209
- **Decision Tree**: TN = 913, FP = 122, FN = 162, TP = 212
- **SVM**: TN = 932, FP = 103, FN = 192, TP = 182

---

## 13. Performance Comparison Table

| Model Algorithm | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8055** | **0.6572** | **0.5588** | **0.6040** |
| Decision Tree Classifier | 0.7984 | 0.6347 | 0.5668 | 0.5989 |
| Support Vector Machine (SVM) | 0.7906 | 0.6386 | 0.4866 | 0.5524 |

---

## 14. Best Model Selection
**Logistic Regression** is selected as the winning model based on the primary criterion of **F1-Score (0.6040)**. It also achieved the highest overall Accuracy (**80.55%**) and Precision (**65.72%**). F1-score ensures a balanced trade-off between identifying true churners (Recall) and minimizing false alarms (Precision).

---

## 15. K-Means Clustering
Customer segmentation was conducted on scaled behavioural features while strictly omitting `customerID` and `Churn`. Target churn statistics were computed only post-clustering for risk interpretation.

---

## 16. Elbow Method
Inertia values calculated across K = 2 to 10:
- K=2: 63,693.03
- K=3: 50,630.49
- **K=4: 46,885.07** (Elbow inflection point)
- K=5: 44,529.45
- K=6: 42,755.80

---

## 17. Silhouette Analysis
Silhouette scores across K values:
- K=2: 0.2510
- K=3: 0.2498
- **K=4: 0.2049**
- K=5: 0.1870
- K=6: 0.1496

**Selection Rationale**: K = 4 balances mathematical separation (Elbow inflection & distinct silhouette score) with clear business interpretability.

---

## 18. Principal Component Analysis (PCA)
2D PCA was applied to project the scaled feature space. Component 1 (PC1) captures feature variance primarily driven by tenure and total charges, while Component 2 (PC2) reflects monthly charge levels.

---

## 19. Customer Segment Interpretation

| Cluster ID | Customer Count | % of Total | Avg Tenure | Avg Monthly Charges | Avg Total Charges | Churn Rate | Segment Profile |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cluster 0** | 1,526 | 21.67% | 30.55 mos | $21.08 | $668.10 | **7.41%** | Low-Cost / Basic Service |
| **Cluster 1** | **1,878** | **26.66%** | **15.56 mos** | **$84.60** | **$1,344.48** | **57.03%** | **High-Risk New High-Cost** |
| **Cluster 2** | 1,985 | 28.18% | 59.49 mos | $91.25 | $5,418.73 | **13.40%** | Loyal High-Value |
| **Cluster 3** | 1,654 | 23.48% | 20.61 mos | $50.75 | $1,070.68 | **25.33%** | Moderate-Cost Short-Tenure |

---

## 20. High-Risk Customer Groups
**Cluster 1** represents the highest risk segment with a **57.03% churn rate** (1,071 out of 1,878 customers). High risk stems from short tenure (~15.6 months) combined with high monthly charges ($84.60/month) and month-to-month contract terms.

---

## 21. Retention Recommendations
1. **Focus on Cluster 1 (High-Risk)**: Provide introductory rate locks, onboarding guidance, and contract conversion discounts.
2. **Improve Service Quality**: Address Fiber Optic internet issues and promote automatic payment methods.
3. **Protect Cluster 2 (High-Value)**: Offer loyalty rewards to maintain retention among long-term subscribers.

---

## 22. Final Observations
- Supervised classification effectively identifies individual churn risks (Logistic Regression F1 = 0.6040).
- Unsupervised clustering cleanly separates high-risk new subscribers from loyal long-term customers.

---

## 23. Conclusion
The project successfully delivered a complete Machine Learning workflow, meeting all technical and business objectives for churn prediction and customer segmentation.

---

## 24. Individual Contributions

### Team Member 1 — Dataset Preparation and Preprocessing
Responsible for data acquisition, dataset inspection, missing value handling for `TotalCharges`, feature encoding (`OneHotEncoder`), numerical normalization (`StandardScaler`), and pipeline construction (`preprocessing.py`).

---

### Team Member 2 — Exploratory Data Analysis and Visualization
Responsible for EDA visual design, distribution analysis (tenure, monthly charges, churn), correlation heatmaps, contract/service categorical breakdown, and formatting figures (`visualization.py`).

---

### Team Member 3 — Classification and Model Evaluation
Responsible for implementing Logistic Regression, Decision Tree Classifier, and SVM, managing train/test splits, evaluating confusion matrices, constructing metric benchmark comparisons, and model selection (`classification.py`).

---

### Team Member 4 — Clustering, PCA, Testing and Documentation
Responsible for K-Means clustering implementation, Elbow method, Silhouette score analysis, PCA 2D visualization, segment profiling, high-risk group identification, pipeline testing (`main.py`), and documentation (`README.md`, `REPORT_CONTENT.md`).

---

## 25. GitHub Repository Information
- **Repository Name**: `customer-churn-ml`
- **Main Files**: `main.py`, `notebooks/customer_churn_analysis.ipynb`, `src/`, `outputs/`, `requirements.txt`, `README.md`, `REPORT_CONTENT.md`.

---

## 26. References
1. IBM Telco Customer Churn Sample Dataset.
2. Scikit-Learn Documentation: Machine Learning in Python.
3. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825-2830, 2011.
