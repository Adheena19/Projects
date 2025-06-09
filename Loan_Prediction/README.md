# 💸 Loan Approval Prediction using Machine Learning

This project is built on the Loan Prediction Practice Problem by **Analytics Vidhya**, which aims to automate the loan approval process based on customer profile inputs.

---

## 📌 Problem Statement

A bank wants to automate the loan eligibility process based on customer details like income, credit history, education, marital status, etc. This is a **binary classification** task (Loan approved: `Y` or `N`).

---

## 🧾 Dataset Overview

**Source**: [Analytics Vidhya Loan Prediction Dataset](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/)

**Columns:**
- Categorical: Gender, Married, Dependents, Education, Self_Employed, Property_Area, Loan_Status
- Numerical: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History

### Sample Stats:
| Feature             | Mean     | Std Dev  | Min | Max     |
|---------------------|----------|----------|-----|---------|
| ApplicantIncome     | 5403.46  | 6109.04  | 150 | 81000   |
| CoapplicantIncome   | 1621.24  | 2926.24  | 0   | 41667   |
| LoanAmount          | 146.41   | 85.59    | 9   | 700     |
| Loan_Amount_Term    | 342      | 65.12    | 12  | 480     |
| Credit_History      | 0.84     | 0.36     | 0   | 1       |

---

## 🛠️ Preprocessing

- Imputation: Missing values filled using median/mode
- Feature Engineering:
  - `Total_Income = ApplicantIncome + CoapplicantIncome`
  - `Log_Total_Income = log(Total_Income)`
  - `EMI = LoanAmount / Loan_Amount_Term`
- Encoding: Categorical variables encoded via Label Encoding & OneHotEncoding

---

## 🤖 Models Used & Accuracy

| Model               | Accuracy Score |
|--------------------|----------------|
| Logistic Regression| **79.4%**       |
| Random Forest       | 75.6%          |
| Decision Tree       | 69.7%          |

Model evaluation used:
- `accuracy_score`

---

## 📊 Correlation Matrix

- `Credit_History` had the highest correlation to `Loan_Status` (~0.56)

---

## 🔮 Final Output

- Predictions saved to `submitted.csv` in the correct format: `Loan_ID` and `Loan_Status`

---

## 🚀 How to Run This Project

1. Clone the repo and navigate to this project:
   ```bash
   git clone https://github.com/Adheena19/Projects.git
   cd Projects/loan_prediction
   ```
2. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Open and run the notebook:
   ```bash
   jupyter notebook loanPredictionAV.ipynb
   ```

---

## ✅ Future Improvements

- Add confusion matrix & ROC-AUC visualization
- Hyperparameter tuning (GridSearchCV or RandomizedSearchCV)
- Model export & Streamlit deployment

---

## 📬 Contact

Feel free to connect with me via [GitHub](https://github.com/Adheena19) 
