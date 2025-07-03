# predict-loan-status
A machine learning project to predict loan approval status using classification models like Logistic Regression and Random Forest. Includes robust data preprocessing, evaluation metrics, and hyperparameter tuning.

## Files Included

- `train_ctrUa4K.csv` — Training dataset
- `test_lAUu6dG.csv` — Test dataset
- `Bank_loan_status.py` — Main script with model training, evaluation, and prediction
- `loan_predictions.csv` — Output predictions for the test set (generated after running the script)

## Project Overview

This project aims to predict whether a loan should be approved (`Y`) or not (`N`) using a variety of applicant features such as income, credit history, loan amount, etc.

## Data Description

1) **Loan_ID** - Unique loan identifier  
2) **Gender** - Gender of the applicant (Male/Female)  
3) **Married** - Marital status (Yes/No)  
4) **Dependents** - Number of dependents (0,1,2,3+)  
5) **Education** - Education level (Graduate/Not Graduate)  
6) **Self_Employed** - Whether self-employed (Yes/No)  
7) **ApplicantIncome** - Income of the applicant  
8) **CoapplicantIncome** - Income of the co-applicant  
9) **LoanAmount** - Loan amount in thousands  
10) **Loan_Amount_Term** - Term of loan in months  
11) **Credit_History** - Credit history meets guidelines (1 = yes, 0 = no)  
12) **Property_Area** - Area type of property (Urban/Semiurban/Rural)  
13) **Loan_Status** - Loan approval status (Y = approved, N = not approved)  

### Models Used:
- Logistic Regression
- Random Forest (with hyperparameter tuning via GridSearchCV)

##  Preprocessing Steps

- Missing values handled using median/mode
- Categorical variables encoded
- Unnecessary features dropped
- Features scaled for logistic regression

##  Evaluation Metrics

The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score
- Confusion Matrix & ROC Curve Visuals

##  How to Run

1. **Python 3.7+** installed.
2. Install dependencies (preferably in a virtual environment):

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
