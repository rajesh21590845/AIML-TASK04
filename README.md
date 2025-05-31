# Task 4: Logistic Regression Binary Classification

## 📌 Objective
Build a binary classifier using **Logistic Regression** to classify data from the **Breast Cancer Wisconsin Dataset**.

## 🧰 Tools Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn (optional)

## 🗂 Dataset
[Breast Cancer Wisconsin Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

## 🧪 Steps Performed

1. **Data Loading and Exploration**  
   Loaded dataset, checked missing values, and performed exploratory data analysis.

2. **Preprocessing**  
   - Dropped non-informative columns.
   - Encoded target variable (`M` → 1, `B` → 0).
   - Scaled features using `StandardScaler`.

3. **Train/Test Split**  
   - Used `train_test_split` for an 80/20 train-test split.

4. **Model Training**  
   - Trained a `LogisticRegression` model using scikit-learn.

5. **Evaluation Metrics**  
   - Confusion Matrix
   - Precision, Recall, Accuracy
   - ROC Curve and AUC Score
   - Threshold tuning for classification

6. **Sigmoid Function Visualization**  
   - Plotted the sigmoid function used in logistic regression.

## 📊 Evaluation Results
- **Accuracy**: ~XX%
- **Precision**: ~XX%
- **Recall**: ~XX%
- **AUC Score**: ~XX

## ❓ Interview Questions Answered
- Difference between logistic and linear regression
- Sigmoid function purpose
- Precision vs Recall
- ROC-AUC curve explanation
- What is the confusion matrix?
- Handling class imbalance
- Choosing the classification threshold
- Logistic regression in multi-class problems

## 📎 Repository Structure
