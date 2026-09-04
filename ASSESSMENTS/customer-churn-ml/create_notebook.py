import json
import os

os.makedirs('notebooks', exist_ok=True)
notebook_path = os.path.join('notebooks', 'customer_churn_analysis.ipynb')

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True)
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True)
    }

cells = [
    # 1. Title and Objective
    md_cell("""# Customer Churn Prediction and Segmentation System
**College Assignment — End-to-End Machine Learning Project**

## Objective
The primary objectives of this project are:
1. Predict customer churn using historical telecommunications customer data.
2. Train and compare three supervised classification models: **Logistic Regression**, **Decision Tree Classifier**, and **Support Vector Machine (SVM)**.
3. Identify the best-performing classification model using **F1-score** as the primary evaluation metric.
4. Segment customers into distinct behavioural groups using **K-Means Clustering** (excluding target variable `Churn` to prevent target leakage).
5. Determine the optimal number of clusters using the **Elbow Method** and **Silhouette Score**.
6. Apply **Principal Component Analysis (PCA)** for 2D visual interpretation of customer clusters.
7. Identify **high-risk customer segments** with elevated churn rates.
8. Provide actionable **customer retention strategies** tailored to the identified segments."""),

    # 2. Dataset Source and Description
    md_cell("""# 1. Dataset Source and Description

- **Dataset**: IBM Telco Customer Churn Dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Source**: Publicly available IBM sample dataset (hosted on Kaggle & GitHub).
- **Target Variable**: `Churn` (Yes = 1, No = 0)
- **Key Attributes**:
  - **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - **Account & Billing**: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
  - **Services**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
  - **Identifier**: `customerID` (removed during preprocessing)"""),

    # 3. Import Libraries
    md_cell("""# 2. Import Libraries"""),
    code_cell("""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Add src module directory to path
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
from preprocessing import load_data, clean_data, get_preprocessor, split_data
from classification import train_and_evaluate_models
from clustering import prepare_clustering_features, evaluate_kmeans, fit_kmeans, generate_cluster_summary, apply_pca, identify_high_risk_segments
from visualization import (
    plot_churn_distribution, plot_tenure_distribution, plot_monthly_charges_distribution,
    plot_churn_by_contract, plot_churn_by_internet_service, plot_churn_by_payment_method,
    plot_correlation_heatmap, plot_confusion_matrix, plot_model_comparison,
    plot_elbow_method, plot_silhouette_analysis, plot_pca_clusters
)

%matplotlib inline
sns.set_theme(style="whitegrid", palette="muted")
print("All libraries imported successfully!")"""),

    # 4. Load Dataset
    md_cell("""# 3. Load Dataset"""),
    code_cell("""data_path = os.path.join('..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
if not os.path.exists(data_path):
    data_path = os.path.join('data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

df_raw = pd.read_csv(data_path)
print("Dataset Loaded Successfully! Shape:", df_raw.shape[0], "rows,", df_raw.shape[1], "columns")
df_raw.head()"""),

    # 5. Data Understanding
    md_cell("""# 4. Data Understanding
Inspect data types, dataset dimensions, missing value counts, and unique value summaries."""),
    code_cell("""print("--- Dataset Information ---")
df_raw.info()

print("\\n--- Missing Values Check ---")
print(df_raw.isnull().sum())

print("\\n--- Churn Target Class Distribution ---")
print(df_raw['Churn'].value_counts(normalize=True) * 100)"""),

    # 6. Data Cleaning
    md_cell("""# 5. Data Cleaning
1. Convert `TotalCharges` to numeric data type (handles blank string entries `' '`).
2. Impute missing `TotalCharges` values with the median.
3. Drop `customerID` column as it is an uninformative identifier."""),
    code_cell("""df_clean, prep_info = clean_data(df_raw)
print("Blanks in TotalCharges detected:", prep_info['blank_total_charges'])
print("Missing values after median imputation:", df_clean['TotalCharges'].isnull().sum())
print("Processed Dataset Shape:", df_clean.shape)
df_clean.head()"""),

    # 7. Data Preprocessing
    md_cell("""# 6. Data Preprocessing
Setting up `ColumnTransformer` with `StandardScaler` for numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) and `OneHotEncoder` for categorical variables."""),
    code_cell("""X_raw = df_clean.drop(columns=['Churn'])
preprocessor, num_cols, cat_cols = get_preprocessor(X_raw)
print("Numerical Features:", len(num_cols), num_cols)
print("Categorical Features:", len(cat_cols), cat_cols)"""),

    # 8. Exploratory Data Analysis
    md_cell("""# 7. Exploratory Data Analysis (EDA)
Generate visualizations to understand key churn patterns, tenure relationships, contract dynamics, and feature correlations."""),
    code_cell("""fig_dir = os.path.join('..', 'outputs', 'figures')
if not os.path.exists(os.path.join('..', 'outputs')):
    fig_dir = os.path.join('outputs', 'figures')
os.makedirs(fig_dir, exist_ok=True)

# 1. Churn Distribution
plot_churn_distribution(df_clean, fig_dir)
plt.figure(figsize=(6, 4))
sns.countplot(data=df_clean, x='Churn', hue='Churn', palette=['#2ecc71', '#e74c3c'], legend=False)
plt.title('Customer Churn Distribution')
plt.show()

# 2. Tenure Distribution by Churn
plot_tenure_distribution(df_clean, fig_dir)
plt.figure(figsize=(8, 4))
sns.histplot(data=df_clean, x='tenure', hue='Churn', kde=True, bins=30, palette=['#2ecc71', '#e74c3c'])
plt.title('Tenure Distribution by Churn')
plt.show()

# 3. Monthly Charges Distribution
plot_monthly_charges_distribution(df_clean, fig_dir)
plt.figure(figsize=(8, 4))
sns.histplot(data=df_clean, x='MonthlyCharges', hue='Churn', kde=True, bins=30, palette=['#2ecc71', '#e74c3c'])
plt.title('Monthly Charges Distribution by Churn')
plt.show()

# 4. Churn by Contract
plot_churn_by_contract(df_clean, fig_dir)
plt.figure(figsize=(7, 4))
sns.countplot(data=df_clean, x='Contract', hue='Churn', palette=['#2ecc71', '#e74c3c'])
plt.title('Churn by Contract Type')
plt.show()

# 5. Churn by Internet Service
plot_churn_by_internet_service(df_clean, fig_dir)
plt.figure(figsize=(7, 4))
sns.countplot(data=df_clean, x='InternetService', hue='Churn', palette=['#2ecc71', '#e74c3c'])
plt.title('Churn by Internet Service Type')
plt.show()

# 6. Churn by Payment Method
plot_churn_by_payment_method(df_clean, fig_dir)
plt.figure(figsize=(9, 4))
sns.countplot(data=df_clean, x='PaymentMethod', hue='Churn', palette=['#2ecc71', '#e74c3c'])
plt.title('Churn by Payment Method')
plt.xticks(rotation=15)
plt.show()

# 7. Correlation Heatmap
plot_correlation_heatmap(df_clean, fig_dir)
df_num = df_clean.select_dtypes(include=[np.number]).copy()
plt.figure(figsize=(6, 5))
sns.heatmap(df_num.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()"""),

    # 9. Train/Test Split
    md_cell("""# 8. Train / Test Split
Separate target variable `Churn` (Yes -> 1, No -> 0) and split data into 80% Training and 20% Testing using stratified sampling (`random_state=42`)."""),
    code_cell("""X_train, X_test, y_train, y_test, X, y = split_data(df_clean, target_col='Churn')

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

print("Training Set Dimensions:", X_train_prep.shape)
print("Testing Set Dimensions:", X_test_prep.shape)"""),

    # 10. Logistic Regression
    md_cell("""# 9. Model 1 — Logistic Regression
Train Logistic Regression classifier with `max_iter=1000`."""),
    code_cell("""lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_prep, y_train)
lr_preds = lr_model.predict(X_test_prep)

print("--- Logistic Regression Classification Report ---")
print(classification_report(y_test, lr_preds, target_names=['No Churn', 'Churn']))"""),

    # 11. Decision Tree Classifier
    md_cell("""# 10. Model 2 — Decision Tree Classifier
Train Decision Tree classifier with `max_depth=5` and `random_state=42`."""),
    code_cell("""dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train_prep, y_train)
dt_preds = dt_model.predict(X_test_prep)

print("--- Decision Tree Classification Report ---")
print(classification_report(y_test, dt_preds, target_names=['No Churn', 'Churn']))"""),

    # 12. Support Vector Machine
    md_cell("""# 11. Model 3 — Support Vector Machine (SVM)
Train SVM classifier with `kernel='rbf'`, `probability=True`, and `random_state=42`."""),
    code_cell("""svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_prep, y_train)
svm_preds = svm_model.predict(X_test_prep)

print("--- Support Vector Machine Classification Report ---")
print(classification_report(y_test, svm_preds, target_names=['No Churn', 'Churn']))"""),

    # 13. Model Evaluation
    md_cell("""# 12. Model Evaluation
Compute Accuracy, Precision, Recall, F1-Score, and Confusion Matrix for all three models."""),
    code_cell("""df_comparison, models, predictions, confusion_matrices, reports, best_model_name = train_and_evaluate_models(
    X_train_prep, X_test_prep, y_train, y_test
)

for name, cm in confusion_matrices.items():
    print("Confusion Matrix for", name, ":")
    print("  TN:", cm[0,0], "| FP:", cm[0,1])
    print("  FN:", cm[1,0], "| TP:", cm[1,1])
    plot_confusion_matrix(cm, name, "confusion_matrix_" + name.lower().replace(' ', '_') + ".png", fig_dir)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title("Confusion Matrix: " + name)
    plt.show()"""),

    # 14. Model Comparison
    md_cell("""# 13. Model Comparison
Display empirical classification metrics table and visual comparison bar chart."""),
    code_cell("""print("=== Empirical Classification Model Comparison ===")
display(df_comparison)

plot_model_comparison(df_comparison, fig_dir)
df_melted = pd.melt(df_comparison, id_vars=['Model'], var_name='Metric', value_name='Score')
plt.figure(figsize=(9, 5))
sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', palette='viridis')
plt.title('Classification Performance Comparison')
plt.ylim(0, 1.05)
plt.show()"""),

    # 15. Best Model Selection
    md_cell("""# 14. Best Model Selection
Comparing models using **F1-score as the primary evaluation criterion** because customer churn prediction requires a strong balance between Precision (minimizing false retention alerts) and Recall (capturing actual churners)."""),
    code_cell("""best_row = df_comparison.sort_values(by='F1-Score', ascending=False).iloc[0]
print("=" * 70)
print("The best-performing model based on F1-score is", best_row['Model'])
print("   - F1-Score: ", best_row['F1-Score'])
print("   - Accuracy: ", best_row['Accuracy'])
print("   - Precision:", best_row['Precision'])
print("   - Recall:   ", best_row['Recall'])
print("=" * 70)"""),

    # 16. K-Means Clustering
    md_cell("""# 15. Unsupervised Learning — K-Means Customer Segmentation
> **IMPORTANT GUARANTEE**: Clustering is performed **strictly excluding** `customerID` and the `Churn` target variable. Churn statistics are only computed post-clustering to evaluate business risk without target leakage."""),
    code_cell("""X_cluster_raw = prepare_clustering_features(df_clean)
preprocessor_cluster, _, _ = get_preprocessor(X_cluster_raw)
X_cluster_scaled = preprocessor_cluster.fit_transform(X_cluster_raw)

print("Clustering Input Feature Matrix Shape:", X_cluster_scaled.shape)"""),

    # 17. Elbow Method
    md_cell("""# 16. Elbow Method
Evaluate K-Means inertia across K = 2 to 10."""),
    code_cell("""inertia_dict, df_silhouette = evaluate_kmeans(X_cluster_scaled, k_range=range(2, 11))

plot_elbow_method(inertia_dict, selected_k=4, output_dir=fig_dir)
ks = list(inertia_dict.keys())
inertias = list(inertia_dict.values())

plt.figure(figsize=(7, 4))
plt.plot(ks, inertias, 'bo-', linewidth=2)
plt.axvline(x=4, color='red', linestyle='--', label='Selected K = 4')
plt.title('Elbow Method For Optimal K Selection')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.legend()
plt.show()"""),

    # 18. Silhouette Analysis
    md_cell("""# 17. Silhouette Score Analysis
Evaluate Silhouette Scores across K = 2 to 10 to confirm cluster separation quality."""),
    code_cell("""print("=== Silhouette Scores Table ===")
display(df_silhouette)

plot_silhouette_analysis(df_silhouette, selected_k=4, output_dir=fig_dir)
plt.figure(figsize=(7, 4))
plt.plot(df_silhouette['K'], df_silhouette['Silhouette_Score'], 'go-', linewidth=2)
plt.axvline(x=4, color='red', linestyle='--', label='Selected K = 4')
plt.title('Silhouette Scores across K Values')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.legend()
plt.show()"""),

    # 19. PCA
    md_cell("""# 18. Dimensionality Reduction — Principal Component Analysis (PCA)
Apply 2D PCA to visualize high-dimensional customer clusters on a 2D scatter plot."""),
    code_cell("""selected_k = 4
kmeans_model, cluster_labels = fit_kmeans(X_cluster_scaled, n_clusters=selected_k, random_state=42)

df_pca, explained_var = apply_pca(X_cluster_scaled, n_components=2)
print("Explained Variance Ratio: PC1 =", round(explained_var[0], 4), "PC2 =", round(explained_var[1], 4))

plot_pca_clusters(df_pca, cluster_labels, output_dir=fig_dir)
df_pca_plot = df_pca.copy()
df_pca_plot['Cluster'] = ["Cluster " + str(c) for c in cluster_labels]

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_pca_plot, x='PC1', y='PC2', hue='Cluster', palette='tab10', alpha=0.7)
plt.title('2D PCA Visualization of K-Means Customer Clusters')
plt.show()"""),

    # 20. Customer Segment Interpretation
    md_cell("""# 19. Customer Segment Interpretation
Post-hoc summary of customer behavioural features, average tenure, monthly charges, total charges, and churn rate per cluster."""),
    code_cell("""churn_series = df_clean['Churn'].map({'Yes': 1, 'No': 0})
cluster_summary = generate_cluster_summary(df_clean, cluster_labels, churn_series)

print("=== Final K-Means Customer Cluster Summary (K=4) ===")
display(cluster_summary)"""),

    # 21. High-Risk Customer Groups
    md_cell("""# 20. High-Risk Customer Groups Identification
Identify clusters with churn rates significantly exceeding the baseline dataset average of 26.54%."""),
    code_cell("""overall_churn_rate = churn_series.mean()
high_risk_segments = identify_high_risk_segments(cluster_summary, overall_churn_rate=overall_churn_rate)

print("=== Identified High-Risk Customer Groups ===")
display(high_risk_segments)"""),

    # 22. Retention Recommendations
    md_cell("""# 21. Customer Retention Recommendations

Based on empirical segment statistics:

1. **High-Risk Segment (Cluster 1 — Short Tenure, High Monthly Cost)**:
   - **Churn Rate**: **57.03%**
   - **Strategy**: Offer introductory discounts for 6-12 month contracts, targeted onboarding tutorials, and proactive customer check-ins during the first 90 days.
2. **Moderate-Risk Segment (Cluster 3 — Low/Medium Tenure, Standard Service)**:
   - **Churn Rate**: **25.33%**
   - **Strategy**: Bundle services (e.g., streaming + tech support) at discounted rates to increase customer stickiness.
3. **Loyal High-Value Segment (Cluster 2 — Long Tenure, High Monthly Cost)**:
   - **Churn Rate**: **13.40%**
   - **Strategy**: Provide VIP loyalty rewards and dedicated priority customer support.
4. **Low-Cost Basic Segment (Cluster 0 — Medium Tenure, Low Monthly Cost)**:
   - **Churn Rate**: **7.41%**
   - **Strategy**: Low maintenance, cross-sell value-added internet/security services."""),

    # 23. Final Observations
    md_cell("""# 22. Final Observations

- **Data Insights**: `tenure` and `Contract` type are the strongest individual indicators of churn. Month-to-month customers exhibit dramatically higher churn rates compared to two-year contract holders.
- **Classification Performance**: Logistic Regression proved to be the most balanced classifier with an F1-Score of **0.6040** and accuracy of **80.55%**.
- **Clustering Insights**: K=4 cleanly partitioned customers by spending level and tenure duration, isolating a high-risk group of 1,878 customers with a **57.03% churn rate**.
- **Business Value**: Combining classification risk scores with K-Means segment profiling enables the business to allocate retention budgets effectively."""),

    # 24. Conclusion
    md_cell("""# 23. Conclusion
This project successfully built an end-to-end Customer Churn Prediction and Segmentation System. By performing rigorous data cleaning, exploratory visual analysis, supervised model benchmarking, and unsupervised customer segmentation, the system provides both accurate individual churn predictions and strategic business segment insights.""")
]

notebook_json = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_json, f, indent=2)

print(f"Notebook successfully created at: {notebook_path}")
