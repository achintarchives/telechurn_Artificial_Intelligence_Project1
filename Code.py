
# CONNECTTEL CHURN PREDICTION 



# 1. Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

import lightgbm as lgb
import shap

plt.style.use("seaborn-v0_8")


# 2. Data Loading

DATA_PATH = r"D:\Telecom_customer_churn.csv"

import os
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"CSV file not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)



# 3. Target Creation

df["Churn"] = (df["Customer Status"] == "Churned").astype(int)


# 4. Drop Leakage / ID Columns

df = df.drop(columns=[
    "Customer ID",
    "City",
    "Zip Code",
    "Latitude",
    "Longitude",
    "Churn Category",
    "Churn Reason",
    "Customer Status"
])


# 5. Missing Values

num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for col in cat_cols:
    df[col] = df[col].fillna("Unknown")


# 6. Feature Engineering

df["HasDependents"] = (df["Number of Dependents"] > 0).astype(int)
df["ChargesPerTenure"] = df["Total Charges"] / (df["Tenure in Months"] + 1)
df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)


# 7. EDA (minimal but valid)

sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn by Contract Type")
plt.xticks(rotation=15)
plt.show()

sns.boxplot(x="Churn", y="Tenure in Months", data=df)
plt.title("Tenure vs Churn")
plt.show()

sns.boxplot(x="Churn", y="Monthly Charge", data=df)
plt.title("Monthly Charge vs Churn")
plt.show()


# 8. Encoding & Scaling

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_encoded = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)


# 9. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 10. Logistic Regression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_proba_lr = lr.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, y_proba_lr)

print("\nLogistic Regression AUC:", lr_auc)
print(classification_report(y_test, lr.predict(X_test)))
print(confusion_matrix(y_test, lr.predict(X_test)))


# 11. Random Forest

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_proba_rf = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, y_proba_rf)

print("\nRandom Forest AUC:", rf_auc)
print(classification_report(y_test, rf.predict(X_test)))
print(confusion_matrix(y_test, rf.predict(X_test)))


# 12. K-Fold Cross Validation

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_auc = cross_val_score(
    LogisticRegression(max_iter=1000),
    X_scaled,
    y,
    cv=cv,
    scoring="roc_auc"
)

print("\nCV AUC Scores:", cv_auc)
print("Mean CV AUC:", cv_auc.mean())


# 13. GridSearchCV (RF)

param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_rf = grid.best_estimator_

print("\nBest RF Params:", grid.best_params_)
print("Best RF CV AUC:", grid.best_score_)


# 14. LightGBM

lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)

lgb_model.fit(X_train, y_train)

y_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
lgb_auc = roc_auc_score(y_test, y_proba_lgb)

print("\nLightGBM AUC:", lgb_auc)


# 15. ROC Curves

for probs, label in [
    (y_proba_lr, "Logistic Regression"),
    (y_proba_rf, "Random Forest"),
    (y_proba_lgb, "LightGBM")
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=label)

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()


# 16. SHAP Interpretation

explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(
    shap_values,
    X_test,
    feature_names=X_encoded.columns
)


# 17. Model Comparison

model_comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "LightGBM"],
    "AUC-ROC": [lr_auc, rf_auc, lgb_auc]
})

print("\nModel Comparison:")
print(model_comparison)
