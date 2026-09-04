# Customer Churn Prediction and Segmentation System

An end-to-end Machine Learning project that predicts telecommunications customer churn using supervised classification algorithms and performs customer segmentation using unsupervised K-Means clustering.

---

## 1. Problem Statement
A service-based telecommunications provider is facing customer attrition. To maintain profitability and customer loyalty, the company needs a machine learning system that:
1. Predicts whether an individual customer will churn (`Yes = 1`, `No = 0`).
2. Evaluates and compares three classification algorithms (**Logistic Regression**, **Decision Tree Classifier**, and **Support Vector Machine**).
3. Selects the optimal classification model based on **F1-score**.
4. Groups customers into distinct behavioural segments using **K-Means Clustering** (without target leakage).
5. Reduces high-dimensional service features using **Principal Component Analysis (PCA)** for 2D visualization.
6. Identifies high-risk customer groups and recommends targeted retention strategies.

---

## 2. Dataset Description & Source

- **Dataset Name**: IBM Telco Customer Churn Dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Source**: Publicly hosted on Kaggle and IBM Sample Data Sets repositories.
- **Dimensions**: 7,043 rows x 21 columns
- **Target Variable**: `Churn` (Binary: `Yes` / `No`)
- **Attributes**:
  - **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - **Account Details**: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
  - **Services**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
  - **Identifier**: `customerID` (removed during data cleaning)

---

## 3. Technologies Used

- **Programming Language**: Python 3.13
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn` (Classification, Clustering, PCA, Preprocessing Pipelines)
- **Interactive Notebook**: `Jupyter Notebook`

---

## 4. Project Structure

```
customer-churn-ml/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # IBM Telco Customer Churn dataset
│
├── notebooks/
│   └── customer_churn_analysis.ipynb          # Self-contained 24-section Jupyter Notebook
│
├── src/
│   ├── __init__.py                            # Package marker
│   ├── preprocessing.py                        # Data loading, cleaning, ColumnTransformer
│   ├── classification.py                     # Logistic Regression, Decision Tree, SVM
│   ├── clustering.py                         # K-Means, Elbow method, Silhouette score, PCA
│   └── visualization.py                      # Matplotlib/Seaborn plot generation
│
├── outputs/
│   ├── figures/                               # 14 generated high-resolution PNG plots
│   └── results/                               # 4 generated CSV output tables
│
├── main.py                                    # Command-line workflow execution entry point
├── requirements.txt                           # Project Python dependencies
├── README.md                                  # Comprehensive project documentation
├── REPORT_CONTENT.md                          # Full academic report (26 sections)
└── .gitignore                                 # Git ignore rules
```

---

## 5. Installation & Setup

### Prerequisites
- Python 3.9+ installed on your system.

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/customer-churn-ml.git
   cd customer-churn-ml
   ```

2. **Create and activate a virtual environment**:
   - **Windows**:
     ```powershell
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - **Linux / macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Dataset**:
   Ensure `WA_Fn-UseC_-Telco-Customer-Churn.csv` is present in the `data/` directory.

---

## 6. How to Run the Project

### Running the End-to-End Execution Script (`main.py`)
Run the following command from the root directory:
```bash
python main.py
```
This script automatically executes data cleaning, EDA figure generation, train/test split, classification benchmarking, K-Means clustering, PCA, and summary CSV exports.

### Running the Interactive Jupyter Notebook
Start Jupyter Notebook and open `notebooks/customer_churn_analysis.ipynb`:
```bash
jupyter notebook notebooks/customer_churn_analysis.ipynb
```
Select **Kernel -> Restart & Run All** to execute all 24 sections from start to finish.

---

## 7. Algorithms & Evaluation Results

### Supervised Classification Benchmark (80/20 Stratified Split)

| Classification Algorithm | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8055** | **0.6572** | **0.5588** | **0.6040** |
| Decision Tree Classifier | 0.7984 | 0.6347 | 0.5668 | 0.5989 |
| Support Vector Machine (SVM) | 0.7906 | 0.6386 | 0.4866 | 0.5524 |

> **Best Model Selection**: **Logistic Regression** achieved the highest **F1-Score (0.6040)** and overall Accuracy (80.55%). F1-score is chosen as the primary metric to ensure a balanced trade-off between precision and recall in churn detection.

---

## 8. Customer Segmentation & PCA Findings

### K-Means Clustering (Selected K = 4)
Clustering was performed on preprocessed behavioural features while strictly excluding `customerID` and the `Churn` target variable.

| Cluster ID | Customer Count | % of Total | Avg Tenure (Months) | Avg Monthly Charges ($) | Avg Total Charges ($) | Churn Rate | Segment Label |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cluster 0** | 1,526 | 21.67% | 30.55 | $21.08 | $668.10 | **7.41%** | Low-Cost / Basic Service |
| **Cluster 1** | **1,878** | **26.66%** | **15.56** | **$84.60** | **$1,344.48** | **57.03%** | **High-Risk New High-Cost** |
| **Cluster 2** | 1,985 | 28.18% | 59.49 | $91.25 | $5,418.73 | **13.40%** | Loyal High-Value |
| **Cluster 3** | 1,654 | 23.48% | 20.61 | $50.75 | $1,070.68 | **25.33%** | Moderate-Cost Short-Tenure |

### High-Risk Customer Segment Analysis
**Cluster 1** exhibits the highest churn rate at **57.03%** (more than double the overall dataset churn average of 26.54%). Key risk drivers include short tenure (~15.6 months), high monthly bills ($84.60/month), and month-to-month contracts.

---

## 9. Customer Retention Strategies

1. **Target Cluster 1 (High-Risk)**: Provide 6-month onboarding discounts, proactive technical setup support, and long-term contract conversion incentives.
2. **Target Month-to-Month Fiber Optic Subscribers**: Offer bundled value packages (free online security & backup) to improve service satisfaction.
3. **Reward Cluster 2 (Loyal High-Value)**: Implement a VIP loyalty points program to maintain high retention.

---

## 10. Generated Outputs

### Figures (`outputs/figures/`)
1. `churn_distribution.png`
2. `tenure_distribution.png`
3. `monthly_charges_distribution.png`
4. `churn_by_contract.png`
5. `churn_by_internet_service.png`
6. `churn_by_payment_method.png`
7. `correlation_heatmap.png`
8. `confusion_matrix_logistic_regression.png`
9. `confusion_matrix_decision_tree.png`
10. `confusion_matrix_svm.png`
11. `model_comparison.png`
12. `elbow_method.png`
13. `silhouette_analysis.png`
14. `pca_customer_clusters.png`

### Results (`outputs/results/`)
1. `model_comparison.csv`
2. `cluster_summary.csv`
3. `silhouette_scores.csv`
4. `high_risk_segments.csv`

---

## 11. GitHub Repository Upload Instructions

```bash
git init
git add .
git commit -m "Add complete Customer Churn Prediction and Segmentation System project"
git branch -M main
git remote add origin https://github.com/your-username/customer-churn-ml.git
git push -u origin main
```
