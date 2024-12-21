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

The Logistic Regression model has been successfully trained and evaluated on the provided dataset. The accuracy results for both the training and testing data are as follows:

**Accuracy on Training Data:** 94.79%
**Accuracy on Test Data:** 94.42%

These results indicate that the model performs well in predicting the target variable, with a relatively small difference between the training and test accuracy, suggesting good generalization.

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

There are several directions in which this project can be extended:

**Hyperparameter Tuning:** Further improve the model's performance by experimenting with different solvers (e.g., liblinear, saga) and adjusting hyperparameters like C, max_iter, and others using techniques such as GridSearchCV or RandomizedSearchCV.

**Data Preprocessing:** Scaling the data using techniques like Min-Max scaling or Standardization might improve the performance, especially if the features have varying scales.

**Feature Engineering:** Investigate the addition of new features or perform feature selection to improve model accuracy.

**Use of Other Algorithms:** Try other machine learning models like Random Forests, Support Vector Machines (SVM), or XGBoost to compare and potentially enhance predictive performance.

**Model Deployment:** Deploy the model as a web application or integrate it into real-time systems for dynamic predictions.

## Conclusions

The Logistic Regression model has demonstrated robust performance with an accuracy score of around 94% on both the training and test datasets. This indicates that the model is able to generalize well to unseen data, suggesting that Logistic Regression is an appropriate algorithm for this task.

However, there is always room for improvement. Experimenting with different preprocessing techniques, hyperparameter tuning, and other machine learning algorithms could further enhance the model's predictive power.

## Acknowledgements

- **Dataset**: Courtesy of the Machine Learning Group at ULB (Université Libre de Bruxelles).
- **Community**: Special thanks to the data science community for inspiration and resources.
