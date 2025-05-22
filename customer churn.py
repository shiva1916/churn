#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd

# Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preview the data
df.head()


# In[6]:


# Basic info: datatypes, non-null counts
df.info()


# In[8]:


# Summary stats for numerical features
df.describe()


# In[10]:


# Check for missing values
df.isnull().sum()


# In[12]:


# Count of churned vs not churned customers
df['Churn'].value_counts()


# In[14]:


# Percentage of churn
df['Churn'].value_counts(normalize=True) * 100


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

# Bar plot of churn distribution
sns.countplot(x='Churn', data=df)
plt.title('Customer Churn Distribution')
plt.show()


# In[18]:


# Convert TotalCharges to numeric (some entries might be empty strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check if there are any NaNs introduced
print(df['TotalCharges'].isnull().sum())

# Drop rows with missing TotalCharges
df = df[df['TotalCharges'].notnull()]


# In[20]:


df.drop('customerID', axis=1, inplace=True)


# In[22]:


# Convert 'Yes'/'No' to 1/0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Convert 'Yes'/'No' in other columns
yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in yes_no_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Handle 'gender'
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# One-hot encode the rest of the categorical columns
df = pd.get_dummies(df, drop_first=True)


# In[24]:


from sklearn.model_selection import train_test_split

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize and fit the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)


# In[28]:


# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[30]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[34]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize and train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred_rf = rf.predict(X_test)

# Evaluate
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))


# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importances
importances = rf.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))  # top 15
plt.title('Top 15 Feature Importances - Random Forest')
plt.tight_layout()
plt.show()

