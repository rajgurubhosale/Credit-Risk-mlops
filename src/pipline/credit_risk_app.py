import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from src.entity.artifact_entity import ScorecardArtifact

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="wide")

st.title("💳 Credit Risk Predictor")
st.markdown("Fill in the applicant details below. Derived features are calculated automatically.")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DN_MISSING_PLACEHOLDER = -99999.0
D_MISSING_PLACEHOLDER  = -88888.0
N_MISSING_PLACEHOLDER  = -77777.0

def safe_ratio(numerator, denominator):
    n_missing = (numerator is None) or np.isnan(numerator)
    d_missing = (denominator is None) or np.isnan(denominator)
    if n_missing and d_missing:
        return DN_MISSING_PLACEHOLDER
    if d_missing:
        return D_MISSING_PLACEHOLDER
    if n_missing:
        return N_MISSING_PLACEHOLDER
    if denominator == 0:
        return np.nan
    return numerator / denominator

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – External Sources
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🔢 External Credit Sources")
st.info("These scores come from external credit bureaus")
col1, col2, col3 = st.columns(3)

with col1:
    ext1 = st.number_input(
        "EXT_SOURCE_1",
        min_value=0.0, max_value=1.0, value=0.5, step=0.001,
        format="%.4f",
        help="External score 1 · range: 0 – 1"
    )

with col2:
    ext2 = st.number_input(
        "EXT_SOURCE_2",
        min_value=0.0, max_value=1.0, value=0.5, step=0.001,
        format="%.4f",
        help="External score 2 · range: 0 – 1"
    )

with col3:
    ext3 = st.number_input(
        "EXT_SOURCE_3",
        min_value=0.0, max_value=1.0, value=0.5, step=0.001,
        format="%.4f",
        help="External score 3 · range: 0 – 1"
    )

# Derived: EXT_SOURCE_WEIGHTED
weights             = np.array([1.9405, 2.4851, 2.7281])
ext_vals            = np.array([ext1, ext2, ext3])
eff_denom           = weights.sum()
ext_source_weighted = float((ext_vals * weights).sum() / eff_denom)

col1, = st.columns(1)

col1.metric(
    "EXT_SOURCE_WEIGHTED",
    f"{ext_source_weighted:.6f}"
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – Loan Amounts
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("💰 Loan Amounts")
col1, col2, col3 = st.columns(3)
#Credit amount of the loan,
# Loan annuity,

with col1:
    amt_credit = st.number_input(
        "AMT_CREDIT (loan amount)",
        min_value=0.0, max_value=10_000_000.0, value=500_000.0, step=1000.0,
        format="%.2f",
        help = "credit amount of loan application"
    )

with col2:
    amt_annuity = st.number_input(
        "AMT_ANNUITY (monthly payment)",
        min_value=0.0, max_value=500_000.0, value=25_000.0, step=500.0,
        format="%.2f",
        help = "monthly emi payment"
        
    )

with col3:
    amt_goods_price = st.number_input(
        "AMT_GOODS_PRICE",
        min_value=40_500.0, max_value=4_050_000.0, value=450_000.0, step=1000.0,
        format="%.2f",
        help="Price of the product or asset being purchased using the loan."
    )

# Derived ratios
annuity_credit_ratio      = safe_ratio(amt_annuity, amt_credit)
goods_credit_ratio        = safe_ratio(amt_goods_price, amt_credit)

col1, col2 = st.columns(2)
col1.metric("ANNUITY_CREDIT_RATIO",      f"{annuity_credit_ratio:.6f}",help='Ratio of EMI to total loan amount.')

col2.metric("GOODS_CREDIT_RATIO",        f"{goods_credit_ratio:.6f}",
            help="Ratio of goods value to loan amount.")


st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – Personal Information  ← NEW
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("👤 Personal Information")
col1, col2, col3 = st.columns(3)

with col1:
    years_age = st.number_input(
        "YEARS_AGE",
        min_value=18.0, max_value=100.0, value=35.0, step=1.0,
        format="%.0f",
        help="Applicant age in years"
    )

with col2:
    name_family_status = st.selectbox(
        "NAME_FAMILY_STATUS",
        options=["Married", "Single / not married", "Civil marriage",
                 "Separated", "Widow", "Unknown"],
        index=0,
        help="Marital status of applicant"
    )

with col3:
    flag_document_3 = st.selectbox(
        "FLAG_DOCUMENT_3",
        options=[1, 0],
        index=0,
        format_func=lambda x: "Provided (1)" if x == 1 else "Not Provided (0)",
        help="Whether document 3 was provided"
    )

col4, col5, col6 = st.columns(3)

with col4:
    region_rating = st.selectbox(
        "REGION_RATING_CLIENT_W_CITY",
        options=[1, 2, 3],
        index=1,
        help="City-level credit risk rating; higher values (3) indicate regions with historically higher default risk."
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – Employment & DPD
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🏢 Employment & Payment History")
col1, col2, col3 = st.columns(3)

with col1:
    years_employed = st.number_input(
        "YEARS_EMPLOYED",
        min_value=0.0, max_value=70.0, value=5.0, step=1.0,
        format="%.1f",
        help="Years the applicant has been employed"
    )

with col2:
    name_income_type = st.selectbox(
        "NAME_INCOME_TYPE",
        options=["Working", "Commercial associate", "State servant",
                 "Pensioner", "RARE",],
        index=0,
        help="Type of income source"
    )

with col3:
    ip_worst_dpd = st.number_input(
        "IP_WORST_DPD_720D (days past due)",
        min_value=0.0, max_value=720.0, value=0.0, step=1.0,
        format="%.0f",
        help="Worst days-past-due in last 720 days"
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – Bureau Features
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🏦 Bureau Credit Features")
col1, col2, col3 = st.columns(3)

with col1:
    b_debt_input   = st.number_input("AMT_CREDIT_SUM_DEBT",
                                     min_value=0.0, max_value=50_000_000.0,
                                     value=100_000.0, step=1000.0, format="%.2f",
                                     help='Remaining unpaid amount from those approved loans.')
    
    b_credit_input = st.number_input("AMT_CREDIT_SUM",
                                     min_value=0.0, max_value=50_000_000.0,
                                     value=500_000.0, step=1000.0, format="%.2f",
                                     help='Total amount originally approved across all bureau loans.')
    
    b_debt_to_credit = safe_ratio(b_debt_input, b_credit_input)
    st.metric("B_DEBT_TO_CREDIT_RATIO", f"{b_debt_to_credit:.6f}",help='debt/credit Percentage of previously approved credit that is still unpaid.')

with col2:
    b_num_active = st.number_input(
        "B_NUM_ACTIVE_CREDIT_720D",
        min_value=0, max_value=50, value=0, step=1,
        help="Active credits in last 720 days · max ≈ 50"
    )

with col3:
    b_credit_duration_min = st.number_input(
        "B_CREDIT_DURATION_MIN (days)",
        min_value=-3650.0, max_value=3650.0, value=180.0, step=10.0,
        format="%.0f",
        help="Min of (DAYS_CREDIT_ENDDATE − DAYS_CREDIT) What is the shortest credit term this customer has taken in the past where shorter durations indicate slightly lower risk and missing or only long-duration loans indicate higher risk"
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – Installment Payment Features
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📋 Installment Payment Features")
col1, col2= st.columns(2)

with col1:
    ip_ratio_late = st.number_input(
        "IP_RATIO_LATE_PAYMENTS_2160D",
        min_value=0.0, max_value=1.0, value=0.0, step=0.01,
        format="%.4f",
        help="Ratio of late payments over 2160 days higher values mean more late payments and higher credit risk."
    )

with col2:
    ip_num_completed = st.number_input(
        "IP_NUM_COMPLETED_LOANS",
        min_value=0.0, max_value=100.0, value=2.0, step=1.0,
        format="%.0f",
        help="Number of completed loans"
    )


st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – Credit Card / ATM Features
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("💳 Credit Card & ATM Features")
col1, col2 = st.columns(2)

with col1:
    cb_atm_freq = st.number_input(
        "CB_AVG_ATM_WITHDRAWAL_FREQ_6M",
        min_value=0.0, max_value=35.0, value=1.0, step=0.1,
        format="%.2f",
        help="Avg ATM withdrawal count per month over last 6 months · data max ≈ 35"
    )
with col2:
    cb_wt_credit_util = st.number_input(
        "CB_WT_CREDIT_UTIL_TREND_3M_12M",
        min_value=-10.0, max_value=10.0, value=0.0, step=0.01,
        format="%.4f",
        help=" credit utilisation trend Change in how much credit the customer is using recently; higher values mean the customer is using more credit and risk is increasing."
    )



st.divider()

# SECTION 8 – Previous Application Features

st.subheader("📂 Previous Application Features")
col1, col2 = st.columns(2)

with col1:
    pa_ratio_approved = st.number_input(
        "PA_RATIO_APPROVED_LOANS",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        format="%.4f",
        help="Ratio of past approved loan applications; higher values indicate better credit history and lower risk."
    )

with col2:
    pa_avg_annuity_pos = st.number_input(
        "PA_AVG_AMT_ANNUITY_POS",
        min_value=0.0, max_value=500_000.0, value=10_000.0, step=500.0,
        format="%.2f",
        help="Average annuity(EMI) amount on POS previous applications"
    )

col1,col2 = st.columns(2)

pa_risk_1080 = st.number_input(
    "PA_AVG_RISK_WEIGHT_1080D",
    0.0, 4.0, 2.0, 0.1,
    format="%.2f",
    help="PA_AVG_RISK_WEIGHT_1080D: Average risk level of past loan applications in the last ~3 years; higher values indicate riskier past borrowing."
)

st.markdown("**PA Credit-to-Application Ratios (by portfolio type)**")
col1, col2 = st.columns(2)
pa_ratio_cash = col1.number_input(
    "PA_RATIO_CREDIT_APPLICATION_Cash",
    min_value=-2.0, max_value=2.0, value=0.0, step=0.01, format="%.4f",
    help='for CASH LOANS Credit given vs applied amount (bank approval behavior).'
)
pa_ratio_pos = col2.number_input(
    "PA_RATIO_CREDIT_APPLICATION_POS",
    min_value=-2.0, max_value=2.0, value=0.0, step=0.01, format="%.4f",
    help='for POS LOANS Credit given vs applied amount (bank approval behavior).'
    
)

pa_ratio_credit_annuity_pos = st.number_input(
    "PA_RATIO_AMT_CREDIT_TO_ANNUITY_POS",
    min_value=0.0, max_value=100.0, value=10.0, step=0.1, format="%.4f",
    help="whether a customer tends to take heavy POS loans relative to their repayment strength,so it means higher the value higher the risk is RATIO:i.e SUM(AMT_CREDIT) / SUM(AMT_ANNUITY) for POS previous applications"
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 – Categorical Features
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🏷️ Categorical Features")
col1, col2, col3, col4 = st.columns(4)

with col1:
    occupation_group = st.selectbox(
        "OCCUPATION_GROUP",
        options=["LOW_SKILL", "SKILLED_PRO", "MANAGERS", "SERVICE", "MISSING"],
        index=1
    )

with col2:
    org_group = st.selectbox(
        "ORG_GROUP",
        options=["ORG_PRIVATE", "ORG_STABLE", "ORG_OTHER", "ORG_UNSTABLE"],
        index=0
    )

with col3:
    education_type = st.selectbox(
        "NAME_EDUCATION_TYPE",
        options=["Higher education", "Secondary / secondary special",
                 "Incomplete higher", "Lower secondary",
                 "Academic degree", "MISSING"],
        index=0
    )

with col4:
    gender = st.selectbox(
        "CODE_GENDER",
        options=["M", "F", "Missing"],
        index=0
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_scorecard_artifacts():
    scorecard_artifact    = ScorecardArtifact()
    categorical_rules     = pd.read_csv(scorecard_artifact.scorecard_categorical_rules)
    numerical_rules       = pd.read_csv(scorecard_artifact.scorecard_numerical_rules)
    categorical_lookup    = joblib.load(scorecard_artifact.scorecard_categorical_lookup)
    numerical_lookup      = joblib.load(scorecard_artifact.scorecard_numerical_lookup)
    scorecard_table       = pd.read_csv(scorecard_artifact.final_scorecard_table_path)
    return {
        "categorical_rules":  categorical_rules,
        "numerical_rules":    numerical_rules,
        "categorical_lookup": categorical_lookup,
        "numerical_lookup":   numerical_lookup,
        "scorecard_table":    scorecard_table,
    }

artifacts          = load_scorecard_artifacts()
categorical_lookup = artifacts["categorical_lookup"]
numerical_lookup   = artifacts["numerical_lookup"]

# ─────────────────────────────────────────────────────────────────────────────
# SCORING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def single_get_numerical_column_score(numerical_lookup, feature, value):
    feature_lookup = numerical_lookup[feature]
    if value in feature_lookup['special']:
        return feature_lookup['special'][value]
    if not feature_lookup['interval'].empty:
        try:
            feature_low  = feature_lookup['interval'].index[0].left
            feature_high = feature_lookup['interval'].index[-1].right
            if value < feature_low:
                return feature_lookup['interval'].iloc[0]
            elif value > feature_high:
                return feature_lookup['interval'].iloc[-1]
            else:
                return feature_lookup['interval'][value]
        except KeyError:
            pass
    if not feature_lookup['discrete'].empty:
        try:
            if value < feature_lookup['discrete'].index.min():
                return feature_lookup['discrete'].iloc[0]
            elif value > feature_lookup['discrete'].index.max():
                return feature_lookup['discrete'].iloc[-1]
            else:
                return feature_lookup['discrete'][value]
        except KeyError:
            pass
    return np.nan


def single_get_cat_score(categorical_lookup, feature, value):
    feature_lookup = categorical_lookup[feature]
    return feature_lookup.get(value, feature_lookup.get('RARE', np.nan))


def single_score_applicant(numerical_lookup, categorical_lookup, user_info: dict):
    total_score = 0
    breakdown   = {}

    for feature in numerical_lookup.keys():
        if feature in user_info:
            value = user_info[feature]
            if pd.isna(value):
                value = -99999.0
            score = single_get_numerical_column_score(numerical_lookup, feature, value)
            if not pd.isna(score):
                total_score        += score
                breakdown[feature]  = score

    for feature in categorical_lookup.keys():
        if feature in user_info:
            value = user_info[feature]
            if pd.isna(value):
                value = 'MISSING'
            score = single_get_cat_score(categorical_lookup, feature, value)
            if not pd.isna(score):
                total_score        += score
                breakdown[feature]  = score

    total_score = 719.8443041917163 + total_score

    return {
        'total_score': round(total_score, 4),
        'breakdown':   breakdown,
    }

# PREDICT BUTTON

    


predict_btn = st.button(
    "🚀 Run Prediction",
    type="primary",
    use_container_width=True,
    key="run_prediction_main"
)

if predict_btn:

    features = {
        # External sources
        "EXT_SOURCE_1":                       ext1,
        "EXT_SOURCE_3":                       ext3,
        "EXT_SOURCE_WEIGHTED":                ext_source_weighted,

        # Loan amounts
        "AMT_CREDIT":                         amt_credit,
        "AMT_GOODS_PRICE":                    amt_goods_price,
        "ANNUITY_CREDIT_RATIO":               annuity_credit_ratio,
        "GOODS_CREDIT_RATIO":                 goods_credit_ratio,

        # Personal  
        "YEARS_AGE":                          years_age,
        "NAME_FAMILY_STATUS":                 name_family_status,
        "FLAG_DOCUMENT_3":                    flag_document_3,

        # Employment
        "YEARS_EMPLOYED":                     years_employed,
        "NAME_INCOME_TYPE":                   name_income_type,
        "IP_WORST_DPD_720D":                  ip_worst_dpd,

        # Bureau
        "B_DEBT_TO_CREDIT_RATIO":             b_debt_to_credit,
        "B_NUM_ACTIVE_CREDIT_720D":           b_num_active,
        "B_CREDIT_DURATION_MIN":              b_credit_duration_min,
        # Installment
        "IP_RATIO_LATE_PAYMENTS_2160D":       ip_ratio_late,
        "IP_NUM_COMPLETED_LOANS":             ip_num_completed,
        "CB_WT_CREDIT_UTIL_TREND_3M_12M":     cb_wt_credit_util,

        # ATM / card
        "CB_AVG_ATM_WITHDRAWAL_FREQ_6M":      cb_atm_freq,
        "REGION_RATING_CLIENT_W_CITY":        region_rating,

        # Previous applications
        "PA_RATIO_APPROVED_LOANS":            pa_ratio_approved,
        "PA_AVG_AMT_ANNUITY_POS":             pa_avg_annuity_pos,
        "PA_AVG_RISK_WEIGHT_1080D":           pa_risk_1080,
        "PA_RATIO_CREDIT_APPLICATION_Cash":   pa_ratio_cash,
        "PA_RATIO_CREDIT_APPLICATION_POS":    pa_ratio_pos,

        # Categorical
        "OCCUPATION_GROUP":                   occupation_group,
        "ORG_GROUP":                          org_group,
        "NAME_EDUCATION_TYPE":                education_type,
        "CODE_GENDER":                        gender,
    }

    result = single_score_applicant(numerical_lookup, categorical_lookup, features)
    score  = result['total_score']

    st.divider()
    st.subheader("📊 Prediction Result")



    col_score, col_decision, col_risk = st.columns(3)

    with col_score:
        st.metric(label="💳 Credit Score", value=score)
        
    with col_decision:
        if score >= 725:
            st.success("✅ APPROVE")
        elif score >= 705:
            st.warning("⚠️ MANUAL REVIEW")
        else:
            st.error("❌ REJECT")

    with col_risk:
        if score >= 748:
            st.success("🟢 Very Low Risk")
        elif score >= 733:
            st.success("🟢 Low Risk")
        elif score >= 725:
            st.warning("🟠 Medium Risk")
        elif score >= 705:
            st.warning("🔴 High Risk")
        else:
            st.error("🔴 Very High Risk")
    if score < 706:
        st.warning("⚠️ Score falls in top 2 risk deciles — captures 51% of all defaults.")

    st.divider()

    with st.expander("🔍 Feature Score Breakdown", expanded=False):
        
        breakdown_df = pd.DataFrame(
            list(result["breakdown"].items()),
            columns=["Feature", "Score"]
        )

        # Sort DESC (positive on top, negative bottom)
        breakdown_df = breakdown_df.sort_values(by="Score", ascending=False)

        # Style function
        def highlight_score(val):
            if val < 0:
                return "color: red; font-weight: bold;"
            else:
                return "color: green; font-weight: bold;"

        styled_df = breakdown_df.style.applymap(highlight_score, subset=["Score"])

        st.dataframe(styled_df, use_container_width=True)
        
    with st.expander("🔍 User Input Breakdown", expanded=False):
        st.json(features)