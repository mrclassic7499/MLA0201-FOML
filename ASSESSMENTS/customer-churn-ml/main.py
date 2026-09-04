import os
import sys
import pandas as pd
import numpy as np

# Add src to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from preprocessing import load_data, clean_data, get_preprocessor, split_data
from classification import train_and_evaluate_models
from clustering import (
    prepare_clustering_features, evaluate_kmeans, fit_kmeans,
    generate_cluster_summary, apply_pca, identify_high_risk_segments
)
from visualization import (
    plot_churn_distribution, plot_tenure_distribution, plot_monthly_charges_distribution,
    plot_churn_by_contract, plot_churn_by_internet_service, plot_churn_by_payment_method,
    plot_correlation_heatmap, plot_confusion_matrix, plot_model_comparison,
    plot_elbow_method, plot_silhouette_analysis, plot_pca_clusters
)

def main():
    print("=" * 70)
    print(" CUSTOMER CHURN PREDICTION AND SEGMENTATION SYSTEM")
    print(" College Assignment - End-to-End Machine Learning Pipeline")
    print("=" * 70)
    
    # Define directories
    data_path = os.path.join('data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    fig_dir = os.path.join('outputs', 'figures')
    res_dir = os.path.join('outputs', 'results')
    
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. DATA LOADING & INSPECTION
    # -------------------------------------------------------------------------
    print("\n[1/7] Loading and Inspecting Dataset...")
    try:
        df_raw = load_data(data_path)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please download the IBM Telco Customer Churn dataset CSV file")
        print("and place it inside the 'data/' directory as 'WA_Fn-UseC_-Telco-Customer-Churn.csv'.")
        sys.exit(1)
        
    print(f" Dataset successfully loaded from: {data_path}")
    print(f" Dimensions: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")
    
    # -------------------------------------------------------------------------
    # 2. DATA PREPROCESSING
    # -------------------------------------------------------------------------
    print("\n[2/7] Preprocessing Data...")
    df_clean, prep_info = clean_data(df_raw)
    
    print(f" Blank TotalCharges converted to numeric (Coerced missing count: {prep_info['blank_total_charges']})")
    print(" Missing values in TotalCharges imputed with median.")
    print(" Irrelevant identifier 'customerID' removed.")
    
    # -------------------------------------------------------------------------
    # 3. EXPLORATORY DATA ANALYSIS & VISUALIZATION
    # -------------------------------------------------------------------------
    print("\n[3/7] Performing Exploratory Data Analysis & Generating Figures...")
    
    fig1 = plot_churn_distribution(df_clean, fig_dir)
    fig2 = plot_tenure_distribution(df_clean, fig_dir)
    fig3 = plot_monthly_charges_distribution(df_clean, fig_dir)
    fig4 = plot_churn_by_contract(df_clean, fig_dir)
    fig5 = plot_churn_by_internet_service(df_clean, fig_dir)
    fig6 = plot_churn_by_payment_method(df_clean, fig_dir)
    fig7 = plot_correlation_heatmap(df_clean, fig_dir)
    
    print(" EDA Visualizations generated and saved to 'outputs/figures/':")
    print("   - churn_distribution.png")
    print("   - tenure_distribution.png")
    print("   - monthly_charges_distribution.png")
    print("   - churn_by_contract.png")
    print("   - churn_by_internet_service.png")
    print("   - churn_by_payment_method.png")
    print("   - correlation_heatmap.png")
    
    # -------------------------------------------------------------------------
    # 4. SUPERVISED LEARNING CLASSIFICATION
    # -------------------------------------------------------------------------
    print("\n[4/7] Training & Evaluating Classification Models...")
    
    X_train, X_test, y_train, y_test, X, y = split_data(df_clean, target_col='Churn')
    print(f" Train/Test Split (80/20): Train shape={X_train.shape}, Test shape={X_test.shape}")
    
    preprocessor, num_cols, cat_cols = get_preprocessor(X)
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    df_comparison, models, predictions, confusion_matrices, reports, best_model_name = train_and_evaluate_models(
        X_train_prep, X_test_prep, y_train, y_test
    )
    
    print("\n--- Classification Models Performance Summary ---")
    print(df_comparison.to_string(index=False))
    
    # Save model comparison table
    comp_csv_path = os.path.join(res_dir, 'model_comparison.csv')
    df_comparison.to_csv(comp_csv_path, index=False)
    print(f"\n Model comparison results saved to: {comp_csv_path}")
    
    # Save confusion matrices & comparison plots
    plot_confusion_matrix(confusion_matrices['Logistic Regression'], 'Logistic Regression', 'confusion_matrix_logistic_regression.png', fig_dir)
    plot_confusion_matrix(confusion_matrices['Decision Tree'], 'Decision Tree', 'confusion_matrix_decision_tree.png', fig_dir)
    plot_confusion_matrix(confusion_matrices['SVM'], 'Support Vector Machine', 'confusion_matrix_svm.png', fig_dir)
    plot_model_comparison(df_comparison, fig_dir)
    
    print(" Classification Visualizations generated and saved to 'outputs/figures/':")
    print("   - confusion_matrix_logistic_regression.png")
    print("   - confusion_matrix_decision_tree.png")
    print("   - confusion_matrix_svm.png")
    print("   - model_comparison.png")
    
    print(f"\n The best-performing model based on F1-score is {best_model_name}.")
    
    # -------------------------------------------------------------------------
    # 5. UNSUPERVISED LEARNING: K-MEANS CLUSTERING
    # -------------------------------------------------------------------------
    print("\n[5/7] Performing Customer Segmentation via K-Means Clustering...")
    print(" (Excluding 'customerID' and 'Churn' target variable to prevent target leakage)")
    
    X_cluster_raw = prepare_clustering_features(df_clean)
    preprocessor_cluster, _, _ = get_preprocessor(X_cluster_raw)
    X_cluster_scaled = preprocessor_cluster.fit_transform(X_cluster_raw)
    
    # Evaluate K=2..10
    inertia_dict, df_silhouette = evaluate_kmeans(X_cluster_scaled, k_range=range(2, 11))
    
    sil_csv_path = os.path.join(res_dir, 'silhouette_scores.csv')
    df_silhouette.to_csv(sil_csv_path, index=False)
    
    plot_elbow_method(inertia_dict, selected_k=4, output_dir=fig_dir)
    plot_silhouette_analysis(df_silhouette, selected_k=4, output_dir=fig_dir)
    
    print(" Clustering Evaluation Visualizations generated:")
    print("   - elbow_method.png")
    print("   - silhouette_analysis.png")
    print(f" Silhouette scores saved to: {sil_csv_path}")
    
    # Fit final K-Means (K=4)
    selected_k = 4
    print(f"\n Selected Optimal Number of Clusters: K = {selected_k}")
    kmeans_model, cluster_labels = fit_kmeans(X_cluster_scaled, n_clusters=selected_k, random_state=42)
    
    # Calculate Cluster Summary
    churn_series = df_clean['Churn'].map({'Yes': 1, 'No': 0})
    cluster_summary = generate_cluster_summary(df_clean, cluster_labels, churn_series)
    
    cluster_csv_path = os.path.join(res_dir, 'cluster_summary.csv')
    cluster_summary.to_csv(cluster_csv_path, index=False)
    
    print("\n--- K-Means Customer Cluster Summary ---")
    print(cluster_summary.to_string(index=False))
    print(f" Cluster summary saved to: {cluster_csv_path}")
    
    # -------------------------------------------------------------------------
    # 6. PCA VISUALIZATION & HIGH-RISK SEGMENT IDENTIFICATION
    # -------------------------------------------------------------------------
    print("\n[6/7] Applying Dimensionality Reduction (PCA) & Identifying High-Risk Segments...")
    
    df_pca, explained_var = apply_pca(X_cluster_scaled, n_components=2)
    plot_pca_clusters(df_pca, cluster_labels, fig_dir)
    print(f" PCA Explained Variance Ratio: PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f} (Total: {sum(explained_var)*100:.2f}%)")
    print(" PCA Cluster Scatter Plot saved to: outputs/figures/pca_customer_clusters.png")
    
    overall_churn_rate = churn_series.mean()
    high_risk_segments = identify_high_risk_segments(cluster_summary, overall_churn_rate=overall_churn_rate)
    
    high_risk_csv_path = os.path.join(res_dir, 'high_risk_segments.csv')
    high_risk_segments.to_csv(high_risk_csv_path, index=False)
    
    print("\n--- High-Risk Customer Segments (Churn Rate > Overall Average of 26.54%) ---")
    print(high_risk_segments.to_string(index=False))
    print(f" High-risk segment summary saved to: {high_risk_csv_path}")
    
    # -------------------------------------------------------------------------
    # 7. BUSINESS RECOMMENDATIONS & FINAL SUMMARY
    # -------------------------------------------------------------------------
    print("\n[7/7] Customer Retention Recommendations & Execution Summary")
    print("-" * 70)
    top_high_risk = high_risk_segments.iloc[0]
    print(f" Highest Risk Group: Cluster {top_high_risk['Cluster']}")
    print(f"   - Churn Rate: {top_high_risk['Churn_Rate']*100:.2f}% ({top_high_risk['Customer_Count']} customers)")
    print(f"   - Characteristics: Short tenure (~{top_high_risk['Avg_Tenure']} months), High monthly charges (${top_high_risk['Avg_MonthlyCharges']})")
    print("\n Actionable Retention Recommendations:")
    print("   1. Target Cluster 1 (New, High-Monthly-Cost Users) with onboarding discounts and long-term contract conversion incentives.")
    print("   2. Upgrade tech support and service reliability for Fiber Optic internet subscribers.")
    print("   3. Promote automated payment methods to reduce friction in electronic check payments.")
    
    print("\n======================================================================")
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(" All 14 figures saved in 'outputs/figures/'")
    print(" All 4 CSV results saved in 'outputs/results/'")
    print("======================================================================\n")

if __name__ == '__main__':
    main()
