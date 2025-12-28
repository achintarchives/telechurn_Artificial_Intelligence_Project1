# telechurn_Artificial_Intelligence_Project1
import pandas as pd

df = pd.read_csv('D:\\Telecom_customer_churn.csv')
df.head()
df.shape
df.info()
df['Churn'] = (df['Customer Status'] == 'Churned').astype(int)
df['Churn'].value_counts()
df['Churn'].value_counts(normalize=True)
df.shape
df.isna().sum().sort_values(ascending=False)
df['Customer Status'].value_counts()
df[['Tenure in Months', 'Monthly Charge', 'Total Revenue']].describe()

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

df['Churn'].value_counts(normalize=True)

sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Churn by Contract Type")
plt.xticks(rotation=15)
plt.show()

sns.boxplot(x='Churn', y='Tenure in Months', data=df)
plt.title("Tenure vs Churn")
plt.show()

sns.boxplot(x='Churn', y='Monthly Charge', data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

sns.boxplot(x='Churn', y='Total Revenue', data=df)
plt.title("Total Revenue vs Churn")
plt.show()

sns.countplot(x='Internet Service', hue='Churn', data=df)
plt.title("Churn by Internet Service")
plt.show()

sns.boxplot(x='Churn', y='Age', data=df)
plt.title("Age vs Churn")
plt.show()

sns.countplot(x='Married', hue='Churn', data=df)
plt.title("Marital Status vs Churn")
plt.show()



drop_cols = [
    'Customer ID',
    'City',
    'Zip Code',
    'Latitude',
    'Longitude',
    'Churn Category',
    'Churn Reason',
    'Customer Status'
]

df = df.drop(columns=drop_cols)
df.isna().sum().sort_values(ascending=False)


num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())

for col in cat_cols:
    df[col] = df[col].fillna('Unknown')


df['HasDependents'] = (df['Number of Dependents'] > 0).astype(int)
df['ChargesPerTenure'] = df['Total Charges'] / (df['Tenure in Months'] + 1)
df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)

df.head()
df.shape
df['Churn'].value_counts(normalize=True)

X = df.drop('Churn', axis=1)
y = df['Churn']

X.shape, y.shape

X_encoded = pd.get_dummies(X, drop_first=True)

X_encoded.shape

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

X_scaled.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

y_train.value_counts(normalize=True), y_test.value_counts(normalize=True)

df.columns

print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nChurn distribution:\n", df['Churn'].value_counts(normalize=True))


# LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# 1. Initialize model
lr = LogisticRegression(max_iter=1000)

# 2. Train model
lr.fit(X_train, y_train)

# 3. Predictions
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]

# 4. Evaluation
print("AUC-ROC Score:")
print(roc_auc_score(y_test, y_proba))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# 1. Initialize model
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# 2. Train model
rf.fit(X_train, y_train)

# 3. Predictions
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# 4. Evaluation
print("Random Forest AUC-ROC:")
print(roc_auc_score(y_test, y_proba_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))


from sklearn.metrics import roc_auc_score

lr_auc = roc_auc_score(y_test, y_proba)
rf_auc = roc_auc_score(y_test, y_proba_rf)

print("Logistic Regression AUC:", lr_auc)
print("Random Forest AUC:", rf_auc)

import pandas as pd
import numpy as np

feature_importance = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

feature_importance.head(10)


import seaborn as sns
import matplotlib.pyplot as plt

top_features = feature_importance.head(10)

sns.barplot(x='Importance', y='Feature', data=top_features)
plt.title("Top 10 Features Driving Churn")
plt.show()



