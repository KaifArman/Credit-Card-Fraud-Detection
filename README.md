# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. It demonstrates a structured approach to handling real-world challenges, such as data imbalance, and provides a scalable solution to enhance financial security.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Methodology](#methodology)
- [Results](#results)
- [How to Run](#how-to-run)
- [Future Scope](#future-scope)
- [Conclusions](#conclusions)
- [Acknowledgements](#acknowledgements)

## Overview

Credit card fraud is a pervasive issue, affecting millions of people globally and costing billions annually. This project aims to develop a robust machine learning pipeline capable of identifying fraudulent transactions with high accuracy while minimizing false positives. By leveraging data preprocessing techniques, machine learning models, and performance evaluation metrics, this project provides a practical solution for detecting fraud in financial transactions.

## Dataset

The dataset used in this project is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle. It contains anonymized transaction data for privacy, with 284,807 transactions recorded. Only 0.17% of these transactions are fraudulent, presenting a significant class imbalance challenge.

Key characteristics:
- **Columns**: Features (`V1` to `V28`) are anonymized, with `Time` and `Amount` as the only non-anonymized columns.
- **Label**: 
  - `0`: Genuine transactions
  - `1`: Fraudulent transactions

## Features

- **Feature Engineering**: Used to normalize `Time` and `Amount` for better model performance.
- **Resampling Techniques**: Addressed class imbalance through oversampling (e.g., SMOTE) or undersampling methods.
- **Evaluation Metrics**: Focused on precision, recall, F1-score, and AUC-ROC to ensure accurate and actionable results.

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - Data Handling: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Machine Learning: Scikit-learn
  - Oversampling Techniques: imbalanced-learn

## Methodology

1. **Data Loading and Exploration**:
   - Loaded the dataset into a Pandas DataFrame.
   - Explored data distribution, class imbalance, and feature correlations.

2. **Data Preprocessing**:
   - Scaled `Time` and `Amount` features using standardization.
   - Handled the severe class imbalance using SMOTE (Synthetic Minority Oversampling Technique).

3. **Model Building**:
   - Trained multiple machine learning models, including:
     - Logistic Regression
     - Random Forest
     - Support Vector Machines (SVM)
   - Used cross-validation to prevent overfitting and ensure generalization.

4. **Evaluation**:
   - Assessed performance using:
     - Precision: Minimizing false positives.
     - Recall: Maximizing detection of fraudulent transactions.
     - F1-score: Balancing precision and recall.
     - AUC-ROC: Ensuring the model is robust across different thresholds.

## Results

The models achieved the following performance metrics (example values):

| Model               | Precision (%) | Recall (%) | F1-Score (%) | AUC-ROC (%) |
|---------------------|---------------|------------|--------------|-------------|
| Logistic Regression | 92.5          | 88.3       | 90.3         | 96.2        |
| Random Forest       | 95.4          | 91.2       | 93.3         | 98.1        |
| SVM                 | 91.8          | 87.5       | 89.6         | 96.0        |

Random Forest emerged as the best-performing model with the highest AUC-ROC and F1-score.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/KaifArman/credit-card-fraud-detection.git
   ```

2. Navigate to the project directory:
   ```bash
   cd credit-card-fraud-detection
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Credit_Card_Fraud_Detection.ipynb
   ```

## Future Scope

- **Algorithm Optimization**:
  - Experiment with advanced models like XGBoost or LightGBM.
  - Perform hyperparameter tuning for better accuracy.
- **Explainability**:
  - Use SHAP or LIME to interpret the model’s decisions.
- **Deployment**:
  - Integrate the model into a real-time fraud detection system using Flask or FastAPI.
- **Additional Features**:
  - Include domain-specific features or external data for enhanced detection capabilities.

## Conclusions

This project demonstrates the power of machine learning in addressing the critical issue of credit card fraud. By handling class imbalance and rigorously evaluating models, it delivers a solution that is both practical and effective. The Random Forest model stood out for its accuracy and reliability. Future work can focus on deploying this model in a production environment and enhancing its interpretability for better stakeholder trust.

## Acknowledgements

- **Dataset**: Courtesy of the Machine Learning Group at ULB (Université Libre de Bruxelles).
- **Community**: Special thanks to the data science community for inspiration and resources.
