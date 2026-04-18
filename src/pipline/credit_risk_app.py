import streamlit as st
import numpy as np
from src.entity.artifact_entity import ScorecardArtifact

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="wide")

st.title("💳 Credit Risk Predictor")
st.markdown("Fill in the applicant details below. Derived features are calculated automatically.")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (same sentinel values you use in your pipeline)
# ─────────────────────────────────────────────────────────────────────────────
DN_MISSING_PLACEHOLDER = -99999.0   # both numerator & denominator missing
D_MISSING_PLACEHOLDER  = -88888.0   # denominator only missing
N_MISSING_PLACEHOLDER  = -77777.0   # numerator only missing

def safe_ratio(numerator, denominator):
    """Mirrors RatioFeatureMixin._create_ratio_feature for scalar inputs."""
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
col1, col2, col3 = st.columns(3)

with col1:
    ext1 = st.number_input(
        "EXT_SOURCE_1",
        min_value=0.0, max_value=1.0, value=0.5, step=0.001,
        format="%.4f",
        help="External score 1 · range: 0.0146 – 0.9627"
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
        help="External score 3 · range: 0.0005 – 0.896"
    )

# Derived: EXT_SOURCE_WEIGHTED
weights   = np.array([1.9405, 2.4851, 2.7281])
ext_vals  = np.array([ext1, ext2, ext3])
eff_denom = weights.sum()                          # all provided → all weights active
ext_source_weighted = float((ext_vals * weights).sum() / eff_denom)

st.info(f"📐 **EXT_SOURCE_WEIGHTED** (calculated) = `{ext_source_weighted:.6f}`")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – Loan Amounts
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("💰 Loan Amounts")
col1, col2, col3 = st.columns(3)

with col1:
    amt_credit = st.number_input(
        "AMT_CREDIT (loan amount)",
        min_value=0.0, max_value=10_000_000.0, value=500_000.0, step=1000.0,
        format="%.2f"
    )

with col2:
    amt_annuity = st.number_input(
        "AMT_ANNUITY (monthly payment)",
        min_value=0.0, max_value=500_000.0, value=25_000.0, step=500.0,
        format="%.2f"
    )

with col3:
    amt_goods_price = st.number_input(
        "AMT_GOODS_PRICE",
        min_value=40_500.0, max_value=4_050_000.0, value=450_000.0, step=1000.0,
        format="%.2f",
        help="range: 40 500 – 4 050 000"
    )

# Derived ratios
annuity_credit_ratio  = safe_ratio(amt_annuity, amt_credit)
goods_credit_ratio    = safe_ratio(amt_goods_price, amt_credit)
credit_ext_product    = ext_source_weighted * amt_credit

col1, col2, col3 = st.columns(3)
col1.metric("ANNUITY_CREDIT_RATIO", f"{annuity_credit_ratio:.6f}")
col2.metric("GOODS_CREDIT_RATIO",   f"{goods_credit_ratio:.6f}")
col3.metric("CREDIT_EXT_SOURCE_PRODUCT", f"{credit_ext_product:,.2f}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – Employment & DPD
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🏢 Employment & Payment History")
col1, col2 = st.columns(2)

with col1:
    years_employed = st.number_input(
        "YEARS_EMPLOYED",
        min_value=0.0, max_value=70.0, value=5.0, step=0.5,
        format="%.1f",
        help="Years the applicant has been employed"
    )

with col2:
    ip_worst_dpd = st.number_input(
        "IP_WORST_DPD_720D (days past due)",
        min_value=0.0, max_value=1000.0, value=0.0, step=1.0,
        format="%.0f",
        help="Worst days-past-due in last 720 days · data max ≈ 705"
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – Bureau Features
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🏦 Bureau Credit Features")
col1, col2, col3 = st.columns(3)

with col1:
    b_debt_input  = st.number_input("Total Debt (for B_DEBT_TO_CREDIT_RATIO)",
                                    min_value=0.0, max_value=50_000_000.0,
                                    value=100_000.0, step=1000.0, format="%.2f")
    b_credit_input = st.number_input("Total Credit (for B_DEBT_TO_CREDIT_RATIO)",
                                     min_value=0.0, max_value=50_000_000.0,
                                     value=500_000.0, step=1000.0, format="%.2f")
    b_debt_to_credit = safe_ratio(b_debt_input, b_credit_input)
    st.metric("B_DEBT_TO_CREDIT_RATIO", f"{b_debt_to_credit:.6f}")

with col2:
    b_num_active = st.number_input(
        "B_NUM_ACTIVE_CREDIT_720D",
        min_value=0, max_value=50, value=2, step=1,
        help="Active credits in last 720 days · max ≈ 50"
    )

with col3:
    b_credit_duration_min = st.number_input(
        "B_CREDIT_DURATION_MIN (days)",
        min_value=-3650.0, max_value=3650.0, value=180.0, step=10.0,
        format="%.0f",
        help="Min of (DAYS_CREDIT_ENDDATE − DAYS_CREDIT) across bureau records"
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – Installment Payment Features
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📋 Installment Payment Features")
col1, col2, col3 = st.columns(3)

with col1:
    ip_ratio_late = st.number_input(
        "IP_RATIO_LATE_PAYMENTS_2160D",
        min_value=0.0, max_value=1.0, value=0.0, step=0.01,
        format="%.4f",
        help="Ratio of late payments over 2160 days · max = 1.0"
    )

with col2:
    ip_num_completed = st.number_input(
        "IP_NUM_COMPLETED_LOANS",
        min_value=0.0, max_value=100.0, value=0.0, step=1.0,
        format="%.0f",
        help="Number of completed loans (can be NaN → enter 0 if unknown)"
    )

with col3:
    cb_wt_credit_util = st.number_input(
        "CB_WT_CREDIT_UTIL_TREND_3M_12M",
        min_value=-10.0, max_value=10.0, value=0.0, step=0.01,
        format="%.4f",
        help="Weighted credit utilisation trend · data max ≈ 6.45"
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – Credit Card / ATM Features
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
    region_rating = st.selectbox(
        "REGION_RATING_CLIENT_W_CITY",
        options=[1, 2, 3],
        index=1,
        help="City-level region rating (1 = best, 3 = worst)"
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – Previous Application Features
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📂 Previous Application Features")

col1, col2 = st.columns(2)
with col1:
    pa_ratio_approved = st.number_input(
        "PA_RATIO_APPROVED_LOANS",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        format="%.4f",
        help="Fraction of previous applications that were approved"
    )

with col2:
    pa_avg_annuity_pos = st.number_input(
        "PA_AVG_AMT_ANNUITY_POS",
        min_value=0.0, max_value=500_000.0, value=10_000.0, step=500.0,
        format="%.2f",
        help="Average annuity amount on POS previous applications"
    )

st.markdown("**PA_AVG_RISK_WEIGHT (by time window)**")
col4 = st.columns(1)[0]

pa_risk_1080 = col4.number_input(
    "PA_AVG_RISK_WEIGHT_1080D",
    0.0, 4.0, 2.0, 0.1,
    format="%.2f"
)
st.markdown("**PA Credit-to-Application Ratios (by portfolio type)**")
col1, col2 = st.columns(2)
pa_ratio_cash = col1.number_input(
    "PA_RATIO_CREDIT_APPLICATION_Cash",
    min_value=-5.0, max_value=5.0, value=0.0, step=0.01, format="%.4f"
)
pa_ratio_pos  = col2.number_input(
    "PA_RATIO_CREDIT_APPLICATION_POS",
    min_value=-5.0, max_value=5.0, value=0.0, step=0.01, format="%.4f"
)

pa_ratio_credit_annuity_pos = st.number_input(
    "PA_RATIO_AMT_CREDIT_TO_ANNUITY_POS",
    min_value=0.0, max_value=100.0, value=10.0, step=0.1, format="%.4f",
    help="SUM(AMT_CREDIT) / SUM(AMT_ANNUITY) for POS previous applications"
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 – Categorical Features
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
        options=["Higher education", "Secondary", "Lower secondary", "MISSING"],
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
# PREDICT BUTTON – assemble feature dict
# ─────────────────────────────────────────────────────────────────────────────
predict_btn = st.button(
    "🚀 Run Prediction",
    type="primary",
    use_container_width=True,
    key="run_prediction_main"
)


import pandas as pd
import streamlit as st
from pathlib import Path
import joblib

@st.cache_data
def load_scorecard_artifacts():
    
    scorecard_artifact = ScorecardArtifact()
    
    categorical_rules = pd.read_csv(scorecard_artifact.scorecard_categorical_rules)
    numerical_rules    = pd.read_csv(scorecard_artifact.scorecard_numerical_rules)

    categorical_lookup = joblib.load(scorecard_artifact.scorecard_categorical_lookup)
    numerical_lookup   = joblib.load(scorecard_artifact.scorecard_numerical_lookup)

    # scorecard table (handle multiple formats)
    scorecard_table = pd.read_csv(scorecard_artifact.final_scorecard_table_path)
    

    return {
        "categorical_rules": categorical_rules,
        "numerical_rules": numerical_rules,
        "categorical_lookup": categorical_lookup,
        "numerical_lookup": numerical_lookup,
        "scorecard_table": scorecard_table
    }


artifacts = load_scorecard_artifacts()

categorical_rules = artifacts["categorical_rules"]
numerical_rules    = artifacts["numerical_rules"]
categorical_lookup = artifacts["categorical_lookup"]
numerical_lookup   = artifacts["numerical_lookup"]
scorecard_table    = artifacts["scorecard_table"]
def single_get_numerical_column_score(numerical_lookup,feature, value): 
        
    feature_lookup = numerical_lookup[feature]
    
    # Check special values first (-99999, -77777 etc.)
    if value in feature_lookup['special']:
        return feature_lookup['special'][value]
    
    # Check interval bins
    if not feature_lookup['interval'].empty:
        
        try:
            feature_low = feature_lookup['interval'].index[0].left
            feature_high = feature_lookup['interval'].index[-1].right
            
            if value < feature_low:
                return feature_lookup['interval'].iloc[0]
            elif value > feature_high:                
                return feature_lookup['interval'].iloc[-1]
            else:
                return feature_lookup['interval'][value]
        except KeyError:
            
            pass
    
    # Check discrete bins (1, 2, 3 etc.)
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
 
  
def single_get_cat_score(categorical_lookup,feature, value):
    feature_lookup = categorical_lookup[feature]
    
    # direct lookup, fallback to RARE if unseen category
    return feature_lookup.get(value, feature_lookup.get('RARE', np.nan))
 
def single_score_applicant(numerical_lookup,categorical_lookup,user_info: dict):
 
 
        total_score = 0
        breakdown   = {}
        
        # 1. Numerical features
        for feature in numerical_lookup.keys():
            if feature in user_info:
                value = user_info[feature]
                
                # if value is NaN → treat as special -99999
                if pd.isna(value):
                    value = -99999.0
                
                score = single_get_numerical_column_score(numerical_lookup,feature, value)
                
                if not pd.isna(score):       
                    total_score        += score
                    breakdown[feature]  = score
        
        # 2. Categorical features
        for feature in categorical_lookup.keys():
            if feature in user_info:
                value = user_info[feature]
                
                # if value is NaN → treat as MISSING
                if pd.isna(value):
                    value = 'MISSING'
                
                score = single_get_cat_score(categorical_lookup,feature, value)
                
                if not pd.isna(score):        
                    total_score        += score
                    breakdown[feature]  = score
                        # intercept points calculted earlier
        total_score = 719.8443041917163 + total_score
                
        return {
            'total_score': round(total_score, 4),
            'breakdown':   breakdown
        }
    
if predict_btn:

    features = {
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_3": ext3,
        "EXT_SOURCE_WEIGHTED": ext_source_weighted,
        "CREDIT_EXT_SOURCE_PRODUCT": credit_ext_product,
        "AMT_GOODS_PRICE": amt_goods_price,
        "ANNUITY_CREDIT_RATIO": annuity_credit_ratio,
        "GOODS_CREDIT_RATIO": goods_credit_ratio,
        "YEARS_EMPLOYED": years_employed,
        "IP_WORST_DPD_720D": ip_worst_dpd,
        "B_DEBT_TO_CREDIT_RATIO": b_debt_to_credit,
        "B_NUM_ACTIVE_CREDIT_720D": b_num_active,
        "B_CREDIT_DURATION_MIN": b_credit_duration_min,
        "IP_RATIO_LATE_PAYMENTS_2160D": ip_ratio_late,
        "IP_NUM_COMPLETED_LOANS": ip_num_completed,
        "CB_WT_CREDIT_UTIL_TREND_3M_12M": cb_wt_credit_util,
        "REGION_RATING_CLIENT_W_CITY": region_rating,
        "PA_RATIO_APPROVED_LOANS": pa_ratio_approved,
        "PA_AVG_AMT_ANNUITY_POS": pa_avg_annuity_pos,
        "PA_AVG_RISK_WEIGHT_1080D": pa_risk_1080,
        "PA_RATIO_CREDIT_APPLICATION_Cash": pa_ratio_cash,
        "PA_RATIO_CREDIT_APPLICATION_POS": pa_ratio_pos,
        "OCCUPATION_GROUP": occupation_group,
        "ORG_GROUP": org_group,
        "NAME_EDUCATION_TYPE": education_type,
        "CODE_GENDER": gender,
    }

    result = single_score_applicant(numerical_lookup, categorical_lookup, features)
    score  = result['total_score']

    st.divider()
    st.subheader("📊 Prediction Result")

    # ── Score display ──────────────────────────────────────────────────────
    col_score, col_decision, col_risk = st.columns(3)

    with col_score:
        st.metric(label="💳 Credit Score", value=score)

    # ── Decision ───────────────────────────────────────────────────────────
    with col_decision:
        if score >= 740:
            st.success("✅ APPROVE")
        elif score >= 706:
            st.warning("⚠️ MANUAL REVIEW")
        else:
            st.error("❌ REJECT")

    # ── Risk Band ──────────────────────────────────────────────────────────
    with col_risk:
        if score >= 757:
            st.success(" Low Risk " )
        elif score >= 733:
            st.warning("Medium Risk ")
        elif score >= 706:
            st.warning("High Risk ")
        else:
            st.error(" Very High Risk  ")

    # ── Capture rate warning ───────────────────────────────────────────────
    if score < 706:
        st.warning("⚠️ Score falls in top 2 risk deciles — captures 51% of all defaults.")

    st.divider()

    # ── Breakdown ──────────────────────────────────────────────────────────
    with st.expander("🔍 Feature Score Breakdown", expanded=False):
        st.json(result["breakdown"])
        
    with st.expander("🔍 User input   Breakdown", expanded=False):
        st.json(features)
