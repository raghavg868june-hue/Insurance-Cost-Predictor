# 🏥 Medical Insurance Cost Predictor

> Predicting individual medical insurance charges using demographic and lifestyle factors — with smoking status as the dominant cost driver.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 📌 Overview

This project builds and evaluates machine learning models to predict how much an individual is likely to spend on medical insurance. Using a dataset of 1,338 records, the analysis reveals that **smoking status combined with BMI is by far the strongest predictor of medical costs** — smokers pay dramatically more than non-smokers across all age groups and regions.

Three models were trained and compared:
- Linear Regression
- Decision Tree
- **Random Forest** *(best performer)*

---

## 📊 Key Findings

| Insight | Detail |
|---|---|
| 🚬 Smokers vs Non-Smokers | Smokers' median charges are ~**4× higher** than non-smokers |
| 🏆 Most Important Feature | `bmi_smoker` interaction term (importance score: **0.761**) |
| 📈 Age effect | Charges rise steadily with age, but smoking creates a parallel, far more expensive tier |
| 🌎 Region | Southeast shows slightly higher charges; all regions have similar medians |
| 👨‍👩‍👧 Children | Most insured have 0–2 children; marginal effect on cost |

---

## 🗂️ Dataset Features

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Age of the insured individual |
| `bmi` | Numerical | Body Mass Index |
| `children` | Numerical | Number of dependents covered |
| `smoker` | Categorical | Whether the individual smokes (`yes` / `no`) |
| `sex` | Categorical | Gender (`male` / `female`) |
| `region` | Categorical | US region (`northeast`, `northwest`, `southeast`, `southwest`) |
| `charges` | **Target** | Individual medical costs billed by insurance (USD) |

---

## 🔬 Exploratory Data Analysis

### Distribution of Key Variables
- **Age**: Slightly over-represented at 18–19 (minimum age), otherwise roughly uniform up to 64
- **BMI**: Approximately normally distributed, centered around 30 (overweight threshold)
- **Children**: Right-skewed — majority have 0 dependents
- **Charges**: Heavily right-skewed with a bimodal log-distribution, indicating two cost populations (smokers vs non-smokers)

### Smoking Status Impact
The single most visually striking finding: smokers' charges occupy an almost entirely separate distribution from non-smokers. Even low-BMI smokers tend to out-spend high-BMI non-smokers.

---

## ⚙️ Feature Engineering

A key interaction feature was created:

```python
df['bmi_smoker'] = df['bmi'] * df['smoker_code']
```

This captures the **compounding effect** of high BMI and smoking together — the most expensive combination. This engineered feature accounted for **76.1% of predictive importance** in the Random Forest model.

---

## 🤖 Model Performance

| Model | R² Score | RMSE (USD) |
|---|---|---|
| Linear Regression | 0.807 | $5,956 |
| Decision Tree | 0.870 | $4,890 |
| **Random Forest** | **0.891** | **$4,469** |

The improved Random Forest (with hyperparameter tuning) was selected as the final model.

### Residual Analysis
- **Linear Regression**: Systematic patterns in residuals — struggles with the non-linear smoker/BMI interaction
- **Decision Tree**: Better spread but prone to overfitting
- **Random Forest**: Most balanced residuals, centered near zero with tightest spread

---

## 🏗️ Project Structure

```
insurance-cost-predictor/
│
├── data/
│   └── insurance.csv               # Raw dataset
│
├── notebooks/
│   └── insurance_analysis.ipynb    # Full EDA + modelling notebook
│
├── plots/
│   ├── plot_numerical_distributions.png
│   ├── plot_charges_distribution.png
│   ├── plot_smoker_charges.png
│   ├── plot_region_charges.png
│   ├── plot_age_vs_charges.png
│   ├── plot_bmi_vs_charges.png
│   ├── plot_correlation_heatmap.png
│   ├── plot_model_comparison.png
│   ├── plot_actual_vs_predicted.png
│   ├── plot_residuals.png
│   ├── plot_feature_importance.png
│   └── plot_final_best_model.png
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

### Run the Notebook

```bash
jupyter notebook notebooks/insurance_analysis.ipynb
```

---

## 🧠 Modelling Pipeline

```python
# 1. Load & explore data
# 2. Encode categorical variables (sex, smoker, region)
# 3. Engineer interaction feature: bmi_smoker = bmi × smoker_code
# 4. Log-transform charges to handle skewness (explored)
# 5. Train/test split (80/20)
# 6. Train Linear Regression, Decision Tree, Random Forest
# 7. Evaluate: R², RMSE, residual plots, actual vs predicted
# 8. Tune Random Forest hyperparameters
# 9. Final evaluation on held-out test set
```

---

## 🖼️ Images

### Section 1: Data Overview & Distributions

<div align="center">

<img width="1481" height="381" alt="plot_numerical_distributions" src="https://github.com/user-attachments/assets/6d066c24-2a61-4383-a338-bf3e53a788f2" />

**Figure 1 — Numerical Distributions**
Histograms of Age, BMI, and Children. Age is roughly uniform across adulthood, BMI is normally distributed around 30, and most people have 0 children.

<br/>

<img width="1380" height="480" alt="plot_charges_distribution" src="https://github.com/user-attachments/assets/554bc912-3d7f-43ea-b2c5-86397a9a94cf" />

**Figure 2 — Charges Distribution (Raw vs Log-Transformed)**
Raw charges are heavily right-skewed; log-transformation produces a more bell-shaped distribution, explaining why preprocessing was needed before modelling.

</div>

---

### Section 2: Exploratory Analysis

<div align="center">

<img width="781" height="480" alt="plot_smoker_charges" src="https://github.com/user-attachments/assets/45ec0b42-be75-4a3e-883a-98063e202b27" />

**Figure 3 — Medical Charges: Smoker vs Non-Smoker**
The "aha moment" of the project — smokers pay ~4× more than non-smokers. The most impactful single chart in the entire analysis.

<br/>

<img width="880" height="480" alt="plot_region_charges" src="https://github.com/user-attachments/assets/a1ca094f-a06f-473f-9d7b-419a14b1d814" />

**Figure 4 — Medical Charges by Region**
Box plot across all 4 US regions. All medians are similar; the Southeast is slightly higher. Region is a weak predictor of cost.

<br/>

<img width="980" height="581" alt="plot_age_vs_charges" src="https://github.com/user-attachments/assets/aa97ce01-c4f0-4852-98c7-53cc97bc5f3c" />

**Figure 5 — Age vs Medical Charges (by Smoking Status)**
Two distinct cost bands emerge — smokers form a high-cost band and non-smokers a low-cost band, both rising steadily with age.

<br/>

<img width="980" height="581" alt="plot_bmi_vs_charges" src="https://github.com/user-attachments/assets/4922344c-1461-47d7-a0a9-34cd4bae6b23" />

**Figure 6 — BMI vs Medical Charges (by Smoking Status)**
High BMI combined with smoking drives the highest charges. This chart directly justifies engineering the `bmi_smoker` interaction feature.

<br/>

<img width="1004" height="681" alt="plot_correlation_heatmap" src="https://github.com/user-attachments/assets/a9e8dc97-2076-495e-a7d6-f279a3ec067d" />

**Figure 7 — Correlation Heatmap**
`smoker_code` → `charges` is the strongest pairwise relationship at **0.79**. Sex and region have near-zero correlation with cost.

</div>

---

### Section 3: Modelling & Evaluation

<div align="center">

<img width="1281" height="510" alt="plot_model_comparison" src="https://github.com/user-attachments/assets/556c9d36-4c2a-46ee-8dbd-79b83385f4b7" />

**Figure 8 — Model Comparison (R² & RMSE)**
Random Forest wins with R²=0.891 and RMSE=$4,469, outperforming both Linear Regression and Decision Tree.

<br/>

<img width="1580" height="510" alt="plot_actual_vs_predicted" src="https://github.com/user-attachments/assets/19f6a5fe-3350-451c-bbea-c91872781a9e" />

**Figure 9 — Actual vs Predicted Charges (All Models)**
Random Forest predictions cluster closest to the perfect-prediction diagonal across all charge ranges.

<br/>

<img width="1580" height="510" alt="plot_residuals" src="https://github.com/user-attachments/assets/bc2d72e6-d393-4e11-a4c0-0d8c68306375" />

**Figure 10 — Residual Plots (All Models)**
Linear Regression shows a systematic funnel-shaped bias. Random Forest residuals are most randomly scattered around zero — indicating the best fit.

<br/>

<img width="880" height="581" alt="plot_feature_importance" src="https://github.com/user-attachments/assets/159fabdf-9d45-4fa7-86ec-77136df691bc" />

**Figure 11 — Feature Importance (Random Forest)**
`bmi_smoker` dominates at **0.761**, followed by `age` at 0.131. This validates the feature engineering decision as the single biggest model improvement.

<br/>

<img width="1380" height="510" alt="plot_final_best_model" src="https://github.com/user-attachments/assets/18f9b9d6-e36c-471b-acb6-8f187b74110a" />

**Figure 12 — Final Model Evaluation (Tuned Random Forest)**
The closing chart — tight actual vs predicted fit and a near-zero centered residual distribution confirm the final model's reliability.

</div>

---

## 💡 Conclusions

1. **Smoking is the #1 cost driver** — not age, not BMI alone, but the interaction of BMI and smoking together
2. **Random Forest outperforms** simpler models due to its ability to capture non-linear relationships and interactions
3. **Region has minimal impact** — all four US regions show similar median charges
4. **Feature engineering matters** — the `bmi_smoker` interaction term was the single biggest model improvement

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

Dataset sourced from the [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) on Kaggle, originally from the book *Machine Learning with R* by Brett Lantz.
