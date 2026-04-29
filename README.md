# 🏦 Credit Risk Application Scorecard

> A **Probability of Default (PD) Application Scorecard System** built on the Home Credit dataset — assessing borrower credit worthiness through statistical modeling and delivering an automated loan decision system.

---

## 📌 Problem Statement
Financial institutions and banks receive thousands of loan application every day
to take a decision such as approve/reject loans they need a reliable, interpretable way to assess a borrower's creditworthiness at the time of loan application quickly and consistently and in a way that can be explained to regulators, analyst, and applicants.

## 🎯 Business Objective

The goal of this project is to develop an interpretable **Application Credit Scorecard**
that estimates the **Probability of Default (PD)** for loan applicants and enables:

- Faster and consistent loan approval decisions
- Reduction in credit losses from high-risk borrowers
- Regulatory-compliant model explanations
- Risk-based customer segmentation

The model helps lenders maximize profitability by balancing:

- **Loss from defaulted loans**
- **Profit from approved good loans**
---
## Introduction
I built a **Credit Risk Application Scorecard** using Logistic Regression
the industry-standard tool used by banks to estimate the **Probability of Default (PD)** for each applicant and translate it into a credit score that drives loan approval or rejection decisions.

### How it Works
takes a borrower's financial and demographic information and returns a credit score
The scorecard works through a simple lookup rule system — each feature value 
falls into a bin, and that bin returns a pre-assigned score point. All bin 
scores are summed with an intercept to produce the final credit score.

The output returns:
- **Credit Score** — total points from all feature bins
- **Loan Decision** — Approve, Review, or Reject based on score band
- **Risk Level** — how risky the applicant is at their score

---
## 📂 Dataset

This project uses the **Home Credit Default Risk** dataset — a real-world credit risk dataset published on Kaggle, containing detailed financial, demographic, and behavioural information on loan applicants.

The model is trained on **6 merged tables**:
| Table | Description |
|---|---|
| `application_train` | Core applicant data — demographics, income, loan details |
| `bureau` | Applicant's past loans from other financial institutions |
| `bureau_balance` | Monthly balance history of bureau loans |
| `previous_application` | Prior loan applications made at Home Credit |
| `installments_payments` | Repayment history on previous loans |
| `credit_card_balance` | Monthly credit card balance snapshots |

Starting from **557 aggregated features** across these tables, the pipeline reduces to a final **30 features** through a rigorous feature selection process.

> 📎 Dataset source: [Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)


# Project Workflow

starting from **557 total features** aggregated and created from application, bureau, and behavioural data 
reduced to **30 final features**  *industry-standard scorecard methodology* through 8 statistical elimination stages.

![Feature Selection Pipeline](image\feature_pipeline.jpg)
*Figure 1: End-to-end feature selection process — 557 raw features reduced to 30 final features across 8 elimination stages.*

A **Logistic Regression** model was trained on the final 30 features — chosen 
for its interpretability and regulatory compliance — and calibrated to produce 
reliable **Probability of Default (PD)** estimates.


The PD estimates are translated into a **Credit Score** — a simple number 
that summarises a borrower's risk, where every feature bin contributes 
a score point to the final score.

🧮 Scorecard Scaling<br>
Credit Score = Intercept Points + Σ Bin Scores

where<br>
PDO: 20<br>
Base Score: 600<br>
Base Odds: 50:1<br>
Factor          = PDO / ln(2)        → 20 / ln(2)<br>
Offset          = Base Score + Factor × ln(Base Odds)   → 600 + Factor × ln(50)<br>

Intercept Points = Offset − Factor × model_intercept<br>
Bin Score = −(WoE × Coefficient) × Factor<br>

All pipeline stages are tracked via **DVC** and fully reproducible.

📊 Scorecard Validation<br>
The scorecard is validated across **10 equal-width score bands** — each band 
reports the bad rate, average PD, and cumulative bad capture rate, confirming 
the score consistently separates good and bad borrowers as it moves up the range.


> The bottom 2 score bands capture **51%** of all bad loans.

Setting Cut-offs<br>
Loan decision cutoffs are set using an **Expected Value Framework** 
— balancing the cost of a bad loan against the profit of a good one 
to determine the optimal **Approve, Review, or Reject** thresholds.


Streamlit Application<br>
A **Streamlit application** was built to accept a single borrower's details 
and return the credit score, PD estimate, and loan decision in real time. 

---
## 📈 Model Performance

| Metric | Train | Test |
|---|---|---|
| AUC | 0.7659 | 0.7663 |
| Gini | 0.5318 | 0.5327 |
| KS | 0.3981 | 0.4032 |
| Brier Score (Calibrated) | 0.0676 | 0.0676 |

---

## 📉 The Trade-Off

Reducing from 67 to 30 features came with a minor drop in metrics —
but the model became simpler, less noisy, and more interpretable.

| Metric | 67 Features | 30 Features | Difference |
|--------|-------------|-------------|------------|
| AUC (Train) | 0.7730 | 0.7659 | -0.0071 |
| AUC (Test) | 0.7723 | 0.7663 | -0.0060 |
| Gini (Train) | 0.5470 | 0.5318 | -0.0152 |
| Gini (Test) | 0.5440 | 0.5327 | -0.0113 |
| KS (Train) | 0.4090 | 0.3981 | -0.0109 |
| KS (Test) | 0.4100 | 0.4032 | -0.0068 |

---
## 🏦 Loan Decision Rules

Loan decisions are based on the final credit score:

| Credit Score | Decision | Risk Level |
|---|---|---|
| 748 and above | ✅ Approve | 🟢 Very Low Risk |
| 733 – 747 | ✅ Approve | 🟢 Low Risk |
| 725 – 732 | ✅ Approve | 🟠 Medium Risk |
| 705 – 724 | ⚠️ Manual Review | 🔴 High Risk |
| Below 705 | ❌ Reject | 🔴 Very High Risk |

> ⚠️ Scores below 706 fall in the top 2 risk deciles — capturing **51% of all defaults.**

Cutoffs are derived using an **Expected Value Framework** — balancing the 
cost of approving a bad loan against the profit of approving a good one.


## 🔍 Model Explainability

The scorecard is fully interpretable due to its structure:

- All variables are transformed using Weight of Evidence (WoE)
- Each bin represents a monotonic risk relationship
- Logistic regression coefficients map directly to score contributions
- Final score is a sum of independent feature contributions
- Business interpretability for credit officers