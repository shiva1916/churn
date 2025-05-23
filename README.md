Customer Churn Prediction in Telecom

This project focuses on building a machine learning model to predict customer churn in the telecom industry. Accurately predicting churn enables telecom companies to take proactive steps in retaining customers and reducing revenue loss.

## Problem Statement

Customer churn is a critical metric for telecom companies. Retaining existing customers is often more cost-effective than acquiring new ones. This project aims to:
	•	Analyze customer data
	•	Identify key churn indicators
	•	Build a classification model to predict whether a customer will churn

## Dataset

The dataset contains customer information such as demographics, service details, account tenure, and usage behavior. The target variable is Churn (Yes/No).

Dataset features include:
	•	Demographics: gender, SeniorCitizen, Partner, Dependents
	•	Services: PhoneService, InternetService, OnlineSecurity, etc.
	•	Account information: tenure, MonthlyCharges, TotalCharges, PaymentMethod, etc.

## No missing values were found after data cleaning.

## Technologies Used
	•	Python
	•	Pandas, NumPy
	•	Seaborn, Matplotlib
	•	Scikit-learn

## Exploratory Data Analysis (EDA)

EDA was performed to understand:
	•	Distribution of churn vs non-churn
	•	Relationship between features and churn
	•	Categorical vs numerical feature insights

Seaborn and matplotlib were used for visualizations.

## Feature Engineering
	•	Converted TotalCharges from object to numeric
	•	Encoded categorical features using Label Encoding
	•	Scaled numerical features using StandardScaler

## Model Building

We tested the following models:
	•	Logistic Regression
	•	Random Forest Classifier
	•	Gradient Boosting Classifier

The data was split into 80% training and 20% testing.

Best Model: Random Forest Classifier

## Accuracy: 80.38%
## Evaluation Metrics

Used:
	•	Accuracy Score
	•	Confusion Matrix
	•	Classification Report (Precision, Recall, F1-Score)

The model performs well on majority class (no churn) and reasonably on minority class (churn).

## Conclusion
	•	The Random Forest model achieved ~80% accuracy.
	•	The model can help identify potential churners and improve customer retention strategies.


