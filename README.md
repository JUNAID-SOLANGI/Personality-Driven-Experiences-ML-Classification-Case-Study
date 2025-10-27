# 🧠 Personality-Driven Experiences: ML Classification Case Study

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-Classification-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

> Building intelligent personality inference through behavioral data to power hyper-personalized user experiences

## 📋 Project Overview

**ConnectSphere**, a social-wellness platform, aims to revolutionize user engagement through personality-driven personalization. This project develops a robust machine learning classifier that accurately infers whether users are **Introverts** or **Extroverts** based solely on their in-app behavioral patterns—no explicit quizzes required.

### Business Impact
- 🎯 Personalized activity recommendations
- 💡 Tailored engagement nudges
- 📈 Improved user retention
- 💰 Enhanced premium conversion rates

## 🎓 Research Context

**Researcher:** Junaid  
**Role:** Research Scholar - Data Science

## 📊 Dataset Description

The dataset captures daily behavioral metrics for each user:

| Feature | Type | Description |
|---------|------|-------------|
| `Time_spent_Alone` | Numerical | Hours per day spent in solitary activities |
| `Comfort_Level_with_new_situations` | Categorical | Comfort with new features (Yes/No) |
| `Weekly_Social_Events_Attended` | Numerical | Number of social events per week |
| `Days_Going_Outdoors` | Numerical | Days going outdoors per week |
| `Energy_After_Socializing` | Categorical | Post-socialization energy (Drained: Yes/No) |
| `Size_of_Close-Friend_Circle` | Numerical | Size of close friend network |
| `Weekly_Social-Media_Posts` | Numerical | Social media posts per week |
| **Personality** | **Target** | **Introvert / Extrovert** |

**Dataset Size:** 2,900 user records (1,491 Extroverts, 1,409 Introverts)

## 🔍 Methodology

### 1. Data Preprocessing
- **Missing Value Handling:** Mode imputation for categorical, median for numerical features
- **Class Balance:** Relatively balanced distribution (no resampling needed)
- **Duplicate Removal:** Eliminated redundant records
- **Outlier Analysis:** Visual inspection via boxplots (no significant outliers detected)

### 2. Statistical Analysis
- **Normality Testing:** Shapiro-Wilk test (all features non-normal, p < 0.05)
- **Variance Equality:** Levene's test (unequal variances across groups)
- **Non-Parametric Tests:**
  - Mann-Whitney U Test: All numerical features significantly different between groups (p < 1e-250)
  - Chi-Squared Test: Strong associations for categorical features (p ≈ 0.0)

### 3. Feature Engineering
Created advanced features to capture complex behavioral patterns:

- **Transformations:**
  - `Post_frequency_log`: Log transformation for skewness handling
  - `Time_Spent_Alone_Squared`: Polynomial feature for non-linearity
  - `Post_Frequency_Cubed`: Higher-order polynomial term

- **Interaction Terms:**
  - `GoingOut_Friends_Interaction`: Outdoor activity × Friend circle size
  - `Social_energy_gap`: Friend circle size - Time spent alone
  - `Activity_level`: Social events + Days going outdoors

**Feature Selection:** Removed `Social_Alone_Interaction` due to low correlation (0.027)

### 4. Model Development

Evaluated four classification algorithms:

| Model | Type |
|-------|------|
| Logistic Regression | Linear probabilistic classifier |
| K-Nearest Neighbors | Instance-based learner |
| Decision Tree | Rule-based classifier |
| Random Forest | Ensemble method |

## 📈 Results

### Baseline Performance (Pre-Tuning)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 91.87% | 90.09% | 91.74% | 90.91% | 0.9186 |
| KNN | 91.06% | 88.50% | 91.74% | 90.09% | 0.9113 |
| Decision Tree | 84.96% | 83.33% | 82.57% | 82.95% | 0.8472 |
| Random Forest | 89.00% | 89.00% | 89.00% | 89.00% | N/A |

### Optimized Performance (Post-Tuning with GridSearchCV)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** 🏆 | **92.28%** | **89.47%** | **93.58%** | **91.48%** | **0.9493** |
| Logistic Regression | 92.68% | 90.27% | 93.58% | 91.89% | 0.9187 |
| KNN | 92.68% | 90.27% | 93.58% | 91.89% | 0.9360 |
| Decision Tree | 92.68% | 90.27% | 93.58% | 91.89% | 0.9444 |

### 🎯 Key Findings

**Best Model:** Random Forest achieved the highest AUC-ROC (0.9493), indicating superior discrimination capability between personality types.

**Performance Gains:**
- Decision Tree: +7.72% accuracy improvement (largest gain)
- Random Forest: +3.28% accuracy, +4.58% recall improvement
- All models achieved >92% accuracy after tuning

## 🚀 Business Applications

1. **Activity Suggestion Engine**
   - Low-intensity activities for introverts
   - Group events for extroverts

2. **Smart Engagement Nudges**
   - Outdoor activity prompts for low-activity introverts
   - Social event recommendations for extroverts

3. **Premium Upsell Opportunities**
   - Personality-aligned feature recommendations
   - Customized subscription benefits

## 🛠️ Technical Stack

- **Language:** Python 3.8+
- **ML Libraries:** scikit-learn, pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Statistical Analysis:** scipy
- **Model Optimization:** GridSearchCV

## 📁 Project Structure

```
personality-classification/
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned and engineered features
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory data analysis
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── model_training.py
├── results/
│   ├── figures/             # Plots and visualizations
│   └── models/              # Saved model artifacts
├── docs/
│   └── classification_personality.pdf
├── requirements.txt
└── README.md
```

## 🔬 Key Insights

1. **All behavioral features significantly correlate with personality type** (Mann-Whitney U test, p < 1e-250)
2. **Categorical features show perfect dependency** with personality (Chi-squared, p ≈ 0.0)
3. **Feature engineering boosted model performance** by capturing non-linear patterns
4. **Ensemble methods outperformed** individual classifiers post-tuning
5. **High recall (93.58%)** ensures most introverts/extroverts are correctly identified

## 🎓 Statistical Rigor

- Verified non-normal distributions (Shapiro-Wilk test)
- Confirmed unequal variances (Levene's test)
- Applied appropriate non-parametric tests
- Checked multicollinearity (VIF analysis)
- Used robust imputation strategies (median/mode)

## 📝 License

This project is part of an academic case study for educational purposes.

## 👤 Author

**Junaid**  
Research Scholar - Data Science

---

<p align="center">
  <i>Transforming behavioral insights into personalized experiences 🎯</i>
</p>
