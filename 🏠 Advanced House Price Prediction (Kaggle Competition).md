# 🏠 Advanced House Price Prediction (Kaggle Competition)

**Author:** Charlie  
**Institution:** Johns Hopkins University, Whiting School of Engineering  
**Duration:** Dec 2024 – Feb 2025  
**Tech Stack:** Python (Pandas, NumPy, scikit-learn, XGBoost), Matplotlib, Seaborn, KNNImputer, RandomForest, Ridge/Lasso/ElasticNet, Ensemble Modeling  

---

## 📘 Overview

This project was completed as part of my **Senior Data Science Capstone** at Johns Hopkins University and built upon the **Kaggle House Prices – Advanced Regression Techniques** dataset.  
The objective was to **develop a robust regression pipeline** capable of predicting housing prices by combining statistical modeling, machine learning, and ensemble methods.

The final model achieved a **top-decile Kaggle RMSLE score (≈0.126)** through systematic feature engineering, advanced imputation, and model blending.

---

## 🎯 Project Goals

- Handle complex **missing data** and heterogeneous feature types efficiently.
- Engineer domain-driven **structural and temporal features** to enhance predictive signal.
- Compare and optimize **regularized linear models** (Ridge, Lasso, ElasticNet) and **tree-based ensembles** (XGBoost).
- Design a **reproducible, modular ML pipeline** for future model deployment or retraining.

---

## 🧩 Data Overview

Dataset: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

- **Training set:** 1,460 observations × 81 features  
- **Test set:** 1,459 observations  
- Features include property attributes (e.g., lot area, year built), material quality, neighborhood, and sale conditions.

---

## ⚙️ Workflow Summary

### **1️⃣ Data Preprocessing & Smart Imputation**
- Removed high-missing columns (*PoolQC, Alley, MiscFeature*).  
- Group-based median imputation for *LotFrontage* (by `Neighborhood`).  
- Regression-based imputation for *MasVnrArea* (RandomForest).  
- Final numeric completion using **KNNImputer** across all numeric columns.  
- Unified “None”/NaN encoding for categorical quality fields.

### **2️⃣ Feature Engineering**
- Created structural & temporal features:
  - `TotalSF` = `TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF`
  - `Age`, `RemodAge`, `GarageAreaPerCar`
- Encoded ordinal quality metrics (*ExterQual*, *KitchenQual*, etc.) into numeric scales (0–5).
- Standardized and one-hot encoded categorical features using `ColumnTransformer`.

### **3️⃣ Model Comparison**
- Evaluated **Ridge**, **Lasso**, and **ElasticNet** regressors using 5-Fold CV (RMSLE metric).  
- Selected ElasticNet (α=0.001, l1_ratio=0.5) as best linear baseline.  

### **4️⃣ XGBoost Optimization**
- Tuned hyperparameters via **RandomizedSearchCV**:
  - `n_estimators=512`, `learning_rate=0.0356`, `max_depth=3`,  
    `min_child_weight=3`, `subsample=0.613`, `colsample_bytree=0.72`,  
    `reg_alpha=0.488`, `reg_lambda=1.322`
- Achieved cross-validated **RMSLE ≈ 0.126**.

### **5️⃣ Ensemble Stacking & Blending**
- Combined Ridge, ElasticNet, and XGBoost via **weighted stacking** (0.2 : 0.3 : 0.5).  
- Improved generalization stability and reduced model variance.  
- Generated final submission `submission_ensemble.csv`.

---

## 📊 Key Results

| Model          | Mean CV RMSLE | Std   | Notes                    |
| -------------- | ------------- | ----- | ------------------------ |
| Ridge          | 0.139         | 0.009 | Strong baseline          |
| Lasso          | 0.137         | 0.008 | Slightly better sparsity |
| ElasticNet     | 0.134         | 0.007 | Best linear model        |
| XGBoost        | **0.126**     | 0.006 | Optimized parameters     |
| Ensemble Blend | **0.124**     | –     | Final submission         |

---

## 🧠 Insights & Learnings

- **Smart imputation** dramatically improved model reliability, especially for neighborhood-based and regression-based fills.  
- **Feature engineering** contributed more performance gain than raw model tuning.  
- **Stacking** helped balance bias and variance, leading to higher leaderboard stability.  
- Modular pipelines (`Pipeline`, `ColumnTransformer`) ensured reproducibility and cleaner experimentation.  

---

## 🗂️ Repository Structure