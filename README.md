# Credit Card Fraud Detection using XGBoost

## Project Overview

This project implements a machine learning model for detecting fraudulent credit card transactions using XGBoost.

The dataset is highly imbalanced, with fraudulent transactions representing only a very small percentage of the total transactions. To handle this class imbalance, the XGBoost `scale_pos_weight` parameter was used.

## Dataset

The project uses the Credit Card Fraud Detection dataset from the Machine Learning Group of ULB.

Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains:

- 284,807 transactions
- 30 input features
- 492 fraudulent transactions
- Target variable: `Class`

Where:

- `0` = Normal transaction
- `1` = Fraudulent transaction

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Google Colab

## Machine Learning Approach

The following workflow was used:

1. Load the dataset
2. Perform exploratory data analysis
3. Check missing values
4. Check duplicate transactions
5. Analyze class imbalance
6. Separate features and target
7. Perform stratified train-test split
8. Calculate `scale_pos_weight`
9. Train an XGBoost classifier
10. Generate predictions
11. Evaluate the model
12. Analyze the confusion matrix
13. Plot ROC Curve
14. Plot Precision-Recall Curve
15. Analyze feature importance

## Model

XGBoost Classifier was used with class imbalance handling through:

`scale_pos_weight`

The model was trained using:

- 300 estimators
- Maximum depth: 6
- Learning rate: 0.1
- Subsample: 0.8
- Column sampling: 0.8
- Random state: 42

## Results

| Metric | Score |
|---|---:|
| Accuracy | 0.9995 |
| Precision | 0.8817 |
| Recall | 0.8367 |
| F1 Score | 0.8586 |
| ROC-AUC | 0.9795 |
| PR-AUC | 0.8826 |

### Confusion Matrix

- True Negatives: 56,853
- False Positives: 11
- False Negatives: 16
- True Positives: 82

The model detected 82 fraudulent transactions out of 98 fraudulent transactions in the test set.

## Why Accuracy Alone Is Not Enough

The dataset is highly imbalanced because fraudulent transactions are much fewer than normal transactions.

Therefore, accuracy alone does not provide a complete picture of fraud detection performance.

Precision, Recall, F1 Score, ROC-AUC and PR-AUC were also used for evaluation.

## False Positives and False Negatives

A False Positive occurs when a legitimate transaction is classified as fraudulent.

A False Negative occurs when a fraudulent transaction is classified as normal.

In fraud detection, both types of errors have practical consequences. False negatives can allow fraudulent transactions to pass through, while false positives can cause legitimate transactions to be flagged for additional verification.

## Project Files

- `Credit_Card_Fraud_Detection_XGBoost.ipynb` — Complete Google Colab notebook
- `README.md` — Project documentation

## Note

The original `creditcard.csv` dataset is not included in this repository because of its large file size. The dataset can be downloaded from Kaggle using the link provided above and uploaded to Google Colab before running the notebook.

## Author

Avinash Verma
