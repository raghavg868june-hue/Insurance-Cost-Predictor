# ============================================================
#  Medical Insurance Charges Prediction
#  Machine Learning Project | Linear Regression & Beyond
#  Run: python medical_insurance_prediction.py
#  Requires: medical.csv in the same folder as this script
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
matplotlib.rcParams["font.size"] = 13
matplotlib.rcParams["figure.figsize"] = (10, 6)
matplotlib.rcParams["figure.dpi"] = 100

print("✅ All libraries imported successfully!")


# ──────────────────────────────────────────────────────────────
# SECTION 2 — Load Dataset (local CSV)
# ──────────────────────────────────────────────────────────────

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path   = os.path.join(script_dir, "medical.csv")

# Each row in this CSV is wrapped in double-quotes, e.g.:
#   "age,sex,bmi,children,smoker,region,charges"
#   "19,female,27.9,0,yes,southwest,16884.924"
# quotechar+QUOTE_ALL tells pandas to strip the outer quotes,
# then parse the comma-separated values inside normally.
import csv as _csv
df = pd.read_csv(csv_path, quotechar='"', quoting=_csv.QUOTE_ALL)
print(f"✅ Dataset loaded from: {csv_path}")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head())


# ──────────────────────────────────────────────────────────────
# SECTION 3 — Exploratory Data Analysis (EDA)
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("DATASET INFO")
print("=" * 50)
df.info()

print("\n" + "=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)
print(df.describe().round(2))

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
missing = df.isnull().sum()
print(missing)
print(f"\nTotal missing values: {missing.sum()}")

duplicates = df.duplicated().sum()
print(f"\nDuplicate rows found: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"✅ Duplicates removed. New shape: {df.shape}")
else:
    print("✅ No duplicates found.")

for col in ["sex", "smoker", "region"]:
    print(f"\n--- {col.upper()} ---")
    print(df[col].value_counts())


# ──────────────────────────────────────────────────────────────
# SECTION 4 — Data Cleaning
# ──────────────────────────────────────────────────────────────

print("\nData Types:")
print(df.dtypes)

print("\nUnique values in categorical columns:")
for col in ["sex", "smoker", "region"]:
    print(f"  {col}: {df[col].unique()}")


def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    print(f"  '{column}': {len(outliers)} outliers (lower={lower:.2f}, upper={upper:.2f})")
    return outliers


print("\nOutlier Detection (IQR Method):")
detect_outliers_iqr(df, "charges")
detect_outliers_iqr(df, "bmi")
detect_outliers_iqr(df, "age")
print("\nℹ️  Keeping outliers — they represent real extreme medical cases.")


# ──────────────────────────────────────────────────────────────
# SECTION 5 — Feature Engineering
# ──────────────────────────────────────────────────────────────

df_encoded = df.copy()
df_encoded["smoker_code"] = df_encoded["smoker"].map({"no": 0, "yes": 1})
df_encoded["sex_code"]    = df_encoded["sex"].map({"female": 0, "male": 1})
df_encoded = pd.get_dummies(df_encoded, columns=["region"], drop_first=True)
df_encoded.drop(columns=["sex", "smoker"], inplace=True)

print("\n✅ Feature engineering complete!")
print(f"   New shape: {df_encoded.shape}")
print("Columns after encoding:", list(df_encoded.columns))


# ──────────────────────────────────────────────────────────────
# SECTION 6 — Data Visualization  (plots saved as PNG files)
# ──────────────────────────────────────────────────────────────

# Plot 1: Distribution of Target Variable
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["charges"], bins=50, color="steelblue", edgecolor="white")
axes[0].set_title("Distribution of Medical Charges", fontweight="bold")
axes[0].set_xlabel("Charges (USD)")
axes[0].set_ylabel("Frequency")
axes[1].hist(np.log1p(df["charges"]), bins=50, color="darkorange", edgecolor="white")
axes[1].set_title("Log-Transformed Distribution of Charges", fontweight="bold")
axes[1].set_xlabel("Log(Charges)")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("plot_charges_distribution.png", dpi=100, bbox_inches="tight")
plt.show()
print("📊 Plot 1 saved: plot_charges_distribution.png")

# Plot 2: Correlation Heatmap
plt.figure(figsize=(11, 7))
corr_matrix = df_encoded.select_dtypes(include=np.number).corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5, annot_kws={"size": 11})
plt.title("Correlation Heatmap", fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("plot_correlation_heatmap.png", dpi=100, bbox_inches="tight")
plt.show()
print("📊 Plot 2 saved: plot_correlation_heatmap.png")

# Plot 3: Charges by Smoking Status
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="smoker", y="charges", palette=["#2ecc71", "#e74c3c"])
plt.title("Medical Charges: Smoker vs Non-Smoker", fontweight="bold")
plt.xlabel("Smoker")
plt.ylabel("Charges (USD)")
plt.tight_layout()
plt.savefig("plot_smoker_charges.png", dpi=100, bbox_inches="tight")
plt.show()
print("📊 Plot 3 saved: plot_smoker_charges.png")

# Plot 4: Age vs Charges
plt.figure(figsize=(10, 6))
for status, color, label in [("yes", "#e74c3c", "Smoker"), ("no", "#2ecc71", "Non-Smoker")]:
    subset = df[df["smoker"] == status]
    plt.scatter(subset["age"], subset["charges"], color=color, label=label, alpha=0.6, s=20)
plt.title("Age vs Medical Charges (by Smoking Status)", fontweight="bold")
plt.xlabel("Age")
plt.ylabel("Charges (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("plot_age_vs_charges.png", dpi=100, bbox_inches="tight")
plt.show()
print("📊 Plot 4 saved: plot_age_vs_charges.png")

# Plot 5: BMI vs Charges
plt.figure(figsize=(10, 6))
for status, color, label in [("yes", "#e74c3c", "Smoker"), ("no", "#2ecc71", "Non-Smoker")]:
    subset = df[df["smoker"] == status]
    plt.scatter(subset["bmi"], subset["charges"], color=color, label=label, alpha=0.6, s=20)
plt.title("BMI vs Medical Charges (by Smoking Status)", fontweight="bold")
plt.xlabel("BMI")
plt.ylabel("Charges (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("plot_bmi_vs_charges.png", dpi=100, bbox_inches="tight")
plt.show()
print("📊 Plot 5 saved: plot_bmi_vs_charges.png")

# Plot 6: Distribution of Numerical Features
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, color in zip(axes, ["age", "bmi", "children"], ["#3498db", "#9b59b6", "#e67e22"]):
    ax.hist(df[col], bins=30, color=color, edgecolor="white")
    ax.set_title(f"Distribution of {col.title()}", fontweight="bold")
    ax.set_xlabel(col.title())
    ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("plot_numerical_distributions.png", dpi=100, bbox_inches="tight")
plt.show()
print("📊 Plot 6 saved: plot_numerical_distributions.png")

# Plot 7: Charges by Region
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="region", y="charges",
            order=["northeast", "northwest", "southeast", "southwest"],
            palette="Set2")
plt.title("Medical Charges by Region", fontweight="bold")
plt.xlabel("Region")
plt.ylabel("Charges (USD)")
plt.tight_layout()
plt.savefig("plot_region_charges.png", dpi=100, bbox_inches="tight")
plt.show()
print("📊 Plot 7 saved: plot_region_charges.png")


# ──────────────────────────────────────────────────────────────
# SECTION 7 — Train-Test Split & Scaling
# ──────────────────────────────────────────────────────────────

feature_cols = ["age", "bmi", "children", "smoker_code", "sex_code",
                "region_northwest", "region_southeast", "region_southwest"]

X = df_encoded[feature_cols]
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")

scaler = StandardScaler()
numerical_cols = ["age", "bmi", "children"]
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols]  = scaler.transform(X_test[numerical_cols])
print("✅ Feature scaling applied.")


# ──────────────────────────────────────────────────────────────
# SECTION 8 — Model Training
# ──────────────────────────────────────────────────────────────

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    r2   = r2_score(y_te, y_pred)
    mae  = mean_absolute_error(y_te, y_pred)
    mse  = mean_squared_error(y_te, y_pred)
    rmse = np.sqrt(mse)
    print(f"\n{'='*45}\n  {name}\n{'='*45}")
    print(f"  R² Score  : {r2:.4f}  ({r2*100:.2f}%)")
    print(f"  MAE       : ${mae:,.2f}")
    print(f"  MSE       : ${mse:,.2f}")
    print(f"  RMSE      : ${rmse:,.2f}")
    return {"Model": name, "R2": round(r2, 4), "MAE": round(mae, 2),
            "MSE": round(mse, 2), "RMSE": round(rmse, 2), "Predictions": y_pred}


lr_model   = LinearRegression()
lr_results = evaluate_model("Linear Regression",
                            lr_model, X_train_scaled, y_train, X_test_scaled, y_test)

dt_model   = DecisionTreeRegressor(max_depth=6, random_state=42)
dt_results = evaluate_model("Decision Tree Regressor",
                            dt_model, X_train, y_train, X_test, y_test)

rf_model   = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
rf_results = evaluate_model("Random Forest Regressor",
                            rf_model, X_train, y_train, X_test, y_test)


# ──────────────────────────────────────────────────────────────
# SECTION 9 — Model Evaluation Plots
# ──────────────────────────────────────────────────────────────

results_df = pd.DataFrame([
    {k: v for k, v in r.items() if k != "Predictions"}
    for r in [lr_results, dt_results, rf_results]
]).sort_values("R2", ascending=False).reset_index(drop=True)
results_df.index += 1
print("\nMODEL COMPARISON TABLE\n" + "=" * 60)
print(results_df.to_string())
print(f"\n🏆 Best Model: {results_df.iloc[0]['Model']}  (R² = {results_df.iloc[0]['R2']})")

# Bar chart comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
models    = results_df["Model"].str.replace(" Regressor", "", regex=False)
colors    = ["#e74c3c", "#f39c12", "#27ae60"]
bars1 = axes[0].barh(models, results_df["R2"], color=colors, edgecolor="white")
axes[0].set_xlim(0, 1.05)
axes[0].set_title("R² Score (higher = better)", fontweight="bold")
axes[0].set_xlabel("R² Score")
for bar, val in zip(bars1, results_df["R2"]):
    axes[0].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontweight="bold")
bars2 = axes[1].barh(models, results_df["RMSE"], color=colors, edgecolor="white")
axes[1].set_title("RMSE (lower = better)", fontweight="bold")
axes[1].set_xlabel("RMSE (USD)")
for bar, val in zip(bars2, results_df["RMSE"]):
    axes[1].text(val + 50, bar.get_y() + bar.get_height() / 2,
                 f"${val:,.0f}", va="center", fontweight="bold")
plt.suptitle("Model Comparison", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plot_model_comparison.png", dpi=100, bbox_inches="tight")
plt.show()

# Actual vs Predicted
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
model_data = [
    ("Linear Regression", lr_results["Predictions"], "#3498db"),
    ("Decision Tree",     dt_results["Predictions"], "#e67e22"),
    ("Random Forest",     rf_results["Predictions"], "#27ae60"),
]
for ax, (name, preds, color) in zip(axes, model_data):
    ax.scatter(y_test, preds, alpha=0.5, color=color, s=15)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "k--", linewidth=1.5, label="Perfect Prediction")
    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("Actual Charges ($)")
    ax.set_ylabel("Predicted Charges ($)")
    ax.legend(fontsize=9)
plt.suptitle("Actual vs Predicted Charges", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plot_actual_vs_predicted.png", dpi=100, bbox_inches="tight")
plt.show()

# Residual Plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, preds, color) in zip(axes, model_data):
    residuals = y_test - preds
    ax.scatter(preds, residuals, alpha=0.5, color=color, s=15)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5)
    ax.set_title(f"Residuals — {name}", fontweight="bold")
    ax.set_xlabel("Predicted Charges ($)")
    ax.set_ylabel("Residuals ($)")
plt.suptitle("Residual Plots", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plot_residuals.png", dpi=100, bbox_inches="tight")
plt.show()


# ──────────────────────────────────────────────────────────────
# SECTION 10 — Model Improvement (Interaction Feature)
# ──────────────────────────────────────────────────────────────

X_improved = X.copy()
X_improved["bmi_smoker"] = X_improved["bmi"] * X_improved["smoker_code"]
print(f"\n✅ Added interaction feature: bmi_smoker — new feature count: {X_improved.shape[1]}")

X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(
    X_improved, y, test_size=0.2, random_state=42
)
rf_improved    = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
rf_imp_results = evaluate_model(
    "Random Forest (+ Interaction Feature)",
    rf_improved, X_train_imp, y_train_imp, X_test_imp, y_test_imp
)

# Feature Importance
importances  = rf_improved.feature_importances_
feat_df = pd.DataFrame({"Feature": X_improved.columns, "Importance": importances})
feat_df = feat_df.sort_values("Importance", ascending=True)
plt.figure(figsize=(9, 6))
bars = plt.barh(feat_df["Feature"], feat_df["Importance"],
                color=plt.cm.RdYlGn(feat_df["Importance"] / feat_df["Importance"].max()))
plt.title("Feature Importance — Random Forest", fontweight="bold")
plt.xlabel("Importance Score")
for bar, val in zip(bars, feat_df["Importance"]):
    plt.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("plot_feature_importance.png", dpi=100, bbox_inches="tight")
plt.show()
print("📊 Feature importance plot saved.")


# ──────────────────────────────────────────────────────────────
# SECTION 11 — Final Results
# ──────────────────────────────────────────────────────────────

all_results = pd.DataFrame([
    {k: v for k, v in r.items() if k != "Predictions"}
    for r in [lr_results, dt_results, rf_results, rf_imp_results]
]).sort_values("R2", ascending=False).reset_index(drop=True)
all_results.index += 1

print("\n" + "=" * 65)
print("        FINAL MODEL RESULTS SUMMARY")
print("=" * 65)
print(all_results.to_string())

best = all_results.iloc[0]
print(f"\n🏆 BEST MODEL  : {best['Model']}")
print(f"   R² Score    : {best['R2']} ({best['R2']*100:.2f}% variance explained)")
print(f"   MAE         : ${best['MAE']:,.2f}")
print(f"   RMSE        : ${best['RMSE']:,.2f}")

# Final best model plots
best_preds = rf_improved.predict(X_test_imp)
fig, axes  = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test_imp, best_preds, alpha=0.5, color="#27ae60", s=20)
lims = [min(y_test_imp.min(), best_preds.min()), max(y_test_imp.max(), best_preds.max())]
axes[0].plot(lims, lims, "k--", linewidth=2, label="Perfect Prediction")
axes[0].set_title("Best Model: Actual vs Predicted", fontweight="bold")
axes[0].set_xlabel("Actual Charges ($)")
axes[0].set_ylabel("Predicted Charges ($)")
axes[0].legend()

residuals = y_test_imp - best_preds
axes[1].hist(residuals, bins=40, color="#27ae60", edgecolor="white")
axes[1].axvline(x=0, color="black", linestyle="--", linewidth=2)
axes[1].set_title("Residual Distribution", fontweight="bold")
axes[1].set_xlabel("Residual (Actual − Predicted)")
axes[1].set_ylabel("Frequency")

plt.suptitle("Random Forest (Improved) — Final Evaluation",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plot_final_best_model.png", dpi=100, bbox_inches="tight")
plt.show()

print("\n✅ All done! Check your folder for the saved PNG plots.")