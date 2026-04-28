# 🏦 Credit Risk Application Scorecard

> A production-grade **Probability of Default (PD) Application Scorecard** built on the Home Credit dataset — featuring WoE binning, Logistic Regression modeling, DVC-tracked pipelines, and an interactive Streamlit loan decision system.

a points-based system that takes a borrower's financial and demographic information and returns a credit score, a loan decision, and a 
risk level

---

## 📌 Problem Statement

Financial institutions and banks receives a thousands of loan application every day
to take a decision such as approve/reject loans they need a reliable, interpretable way to assess a borrower's creditworthiness at the time of loan application quickly and consistently and in a way that can be explained to regulators,analyst, and applicants.

This project addresses that problem and builds an **Credit Risk Application Scorecard** — the industry-standard tool used by banks to estimate the **Probability of Default (PD)** for each applicant and translate it into a credit score that drives loan approval or rejection decisions.



## Introduction

I built a Credit Risk Application Scorecard using Logistic Regression, trained 
on 6 merged Home Credit datasets — starting from 557 total features aggregated and created from application, bureau, and behavioural data 
it is build using industry-standard scorecard methodology: WoE binning, IV filtering, 
correlation feature selction, monotonic bin refinement, VIF, and PSI stability check. after this final feature remained  26 

The scorecard works through a simple lookup rule system — each feature value 
falls into a bin, and that bin returns a pre-assigned score point. All bin 
scores are summed with an intercept to produce the final credit score.

The output returns:
- **Credit Score** — total points from all feature bins
- **Loan Decision** — Approve, Review, or Reject based on score band
- **Risk Level** — how risky the applicant is at their score

---

## 🏗️ Project Architecture

```
Raw Data (Home Credit)
        ↓
  Data Ingestion & Validation
        ↓
  Data Transformation & EDA
        ↓
  Feature Engineering (WoE Binning)
        ↓
  Bin Refinement & Selection (IV + Correlation Filter)
        ↓
  Logistic Regression Model Training
        ↓
  Scorecard Generation (Points Scaling)
        ↓
  Model Evaluation (AUC · Gini · KS)
        ↓
  Streamlit Loan Decision App
```

All pipeline stages are tracked and reproducible via **DVC**.

---

# Project Workflow
**The feature engineering pipeline follows industry-standard scorecard methodology:**
starting from 557 features down to 26 — each step driven by a specific statistical reason.

---
### 1. Feature Creation
Aggregated and engineered 557 features from 6 merged Home Credit datasets at SK_ID_CURR level —covering application, bureau, and behavioural data and other dataset
### 2. Univariate Feature Screening
Removed features with high missing rates, high zero concentration, high combination  of special value (-99999, -88888, -77777), and quasi-constant distributions 
dropping any feature where the worst ratio exceeded 95% threshold. 
### 3. Pre-Binning
Each feature binned into 20 bins with a 5% minimum bin size, also the transformed those bins into  WoE transformation and calculated IV  for all bins Features
### 4. IV Filter → 285 features
Removed features with IV < 0.02 as they carry no predictive power.
### 5. Correlation Filter → 123 features
Grouped correlated features and kept only the highest IV feature per group.
### 6. Bin Refinement → 104 features
Manually refined bins to 5–7 per feature with forced monotonicity.
Non-monotonic features were removed  monotonicity is a the requirement in credit scorecards.
### 7. Correlation Filter → 104 features
Applied 0.7 correlation threshold to remove redundant features.
### 8. VIF Filter → 95 features
Removed multicollinear features using Variance Inflation Factor.
### 9. PSI Stability Check → 95 features
Tested feature stability using Population Stability Index.
Test data used as OOT proxy due to data constraints.
### 10. Stepwise Logistic Regression → 67 features
Used stepwise selection to remove statistically insignificant features.
kept the feature with higher iv value 
### 11. Correlation Filter → 58 features
Applied stricter 0.6 threshold for a cleaner feature set.
### 12. Ensemble Feature Selection → 24 features
Applied RFE, Lasso, Stability Selection, Permutation Importance, and SHAP —
then used a voting-based approach keeping features selected by majority of methods.
### 13. Manual Feature Selection → 26 final features
Added back 2 feature from credit balance data that was lost during automated
selection — so the signal of those data feature added back

## 📉 The Trade-Off

Reducing from 67 to 26 features came with a minor drop in metrics —
but the model became simpler, less noisy, and more interpretable.

| Metric | 67 Features | 26 Features | Difference |
|--------|-------------|-------------|------------|
| AUC (Train) | 0.7730 | 0.7669 | -0.0061 |
| AUC (Test) | 0.7723 | 0.7671 | -0.0052 |
| Gini (Train) | 0.5470 | 0.5339 | -0.0131 |
| Gini (Test) | 0.5440 | 0.5343 | -0.0097 |
| KS (Train) | 0.4090 | 0.4034 | -0.0056 |
| KS (Test) | 0.4100 | 0.4094 | -0.0006 |


## 🆚 Traditional PD Model vs Application Scorecard

A traditional PD model outputs a raw probability like 0.23 — precise, 
but meaningless to a loan officer the interpretability problem will need to face.

A scorecard transforms that same probability into an integer score — 645 points 
— where every feature contributes some score points,that are explainable . A rejected 
applicant can be told exactly why therer application is rejected.

This project implements both — **Logistic Regression PD**, 
and scaled into scorecard points using **PDO (Points to Double Odds) scaling**.

Scorecards are:
✅ Actionable — each feature contributes a visible score point<br>
✅ Fully interpretable by credit risk analysts<br>

---

## 🔄 DVC Pipeline Stages

```yaml
stages:
  - 01_data_ingestion
  - 02_data_validation
  - 03_data_transformation
  - 04_feature_quality_filter
  - 05_woe_binning_and_selection
  - 06_bin_refinement_and_selection
  - 07_model_training
  - 08_model_evaluation
  - 09_scorecard_generation
```