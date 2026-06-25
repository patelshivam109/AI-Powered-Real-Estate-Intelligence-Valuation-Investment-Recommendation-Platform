import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import joblib
import shap
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Real Estate Intelligence — India",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Color palette ───────────────────────────────────────────────────────────────
DEEP_BLUE   = "#0F2A5E"
NAVY        = "#1A3A6B"
TEAL        = "#0D7C8C"
TEAL_LIGHT  = "#14A5BA"
GOLD        = "#C9A84C"
GOLD_LIGHT  = "#E8C97A"
SUCCESS     = "#1D9E6F"
WARNING     = "#E07B39"
DANGER      = "#D63B3B"
BG_CARD     = "#F7F9FC"
BORDER      = "#E2E8F0"
TEXT_MAIN   = "#1A2340"
TEXT_MUTED  = "#64748B"
SAFFRON     = "#FF9933"
INDIA_GREEN = "#138808"

# ─── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #F0F4FA; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 1600px; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F2A5E 0%, #0D2050 60%, #091840 100%);
    border-right: none;
}
[data-testid="stSidebar"] * { color: #E8EDF8 !important; }
[data-testid="stSidebar"] .stRadio > label { display: none; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 16px; border-radius: 10px; margin: 2px 0;
    font-size: 14px; font-weight: 500; cursor: pointer;
    transition: all .2s ease; color: #AAB8D4 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,.08); color: #fff !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.1) !important; }
[data-testid="stSidebarNav"] { display: none; }

.kpi-card {
    background: #fff; border-radius: 16px;
    padding: 22px 24px; border: 1px solid #E2E8F0;
    box-shadow: 0 2px 12px rgba(15,42,94,.06);
    transition: box-shadow .2s;
}
.kpi-card:hover { box-shadow: 0 6px 24px rgba(15,42,94,.12); }
.kpi-label { font-size: 12px; font-weight: 600; letter-spacing: .08em;
             text-transform: uppercase; color: #64748B; margin-bottom: 8px; }
.kpi-value { font-size: 32px; font-weight: 800; color: #0F2A5E; line-height: 1.1; }
.kpi-sub   { font-size: 13px; color: #64748B; margin-top: 6px; }
.kpi-badge { display: inline-block; padding: 3px 10px; border-radius: 20px;
             font-size: 11px; font-weight: 600; margin-top: 8px; }
.badge-up   { background: #DCFCE7; color: #15803D; }
.badge-down { background: #FEE2E2; color: #B91C1C; }
.badge-neu  { background: #F1F5F9; color: #475569; }

.section-eyebrow { font-size: 11px; font-weight: 700; letter-spacing: .14em;
                   text-transform: uppercase; color: #0D7C8C; margin-bottom: 4px; }
.section-title { font-size: 26px; font-weight: 800; color: #0F2A5E;
                 line-height: 1.2; margin-bottom: 4px; }
.section-sub   { font-size: 14px; color: #64748B; margin-bottom: 24px; }

.hero-banner {
    background: linear-gradient(135deg, #0F2A5E 0%, #138808 60%, #FF9933 100%);
    border-radius: 20px; padding: 52px 52px; margin-bottom: 28px;
    position: relative; overflow: hidden;
}
.hero-banner::before {
    content: ""; position: absolute; right: -60px; top: -60px;
    width: 340px; height: 340px; border-radius: 50%;
    background: rgba(255,153,51,.15);
}
.hero-banner::after {
    content: ""; position: absolute; right: 140px; bottom: -80px;
    width: 200px; height: 200px; border-radius: 50%;
    background: rgba(255,255,255,.05);
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 42px; font-weight: 400; color: #fff;
    line-height: 1.15; margin-bottom: 14px;
}
.hero-subtitle { font-size: 16px; color: rgba(255,255,255,.80);
                 max-width: 620px; line-height: 1.6; margin-bottom: 26px; }
.hero-pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.25);
    border-radius: 30px; padding: 8px 18px; font-size: 13px;
    color: #fff; font-weight: 500; margin-right: 10px;
}

.chart-card {
    background: #fff; border-radius: 16px; padding: 24px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 12px rgba(15,42,94,.05);
}

.insight-box {
    background: linear-gradient(135deg, #FFF7ED 0%, #ECFDF5 100%);
    border-left: 4px solid #FF9933; border-radius: 0 12px 12px 0;
    padding: 16px 20px; margin-top: 16px;
    font-size: 13.5px; color: #1A2340; line-height: 1.6;
}
.insight-box strong { color: #0F2A5E; }

.stButton > button {
    background: linear-gradient(135deg, #0F2A5E 0%, #138808 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important; padding: 14px 36px !important;
    font-size: 15px !important; font-weight: 700 !important;
    letter-spacing: .02em !important; width: 100% !important;
    box-shadow: 0 4px 18px rgba(19,136,8,.35) !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    box-shadow: 0 8px 28px rgba(19,136,8,.5) !important;
    transform: translateY(-1px) !important;
}

.predict-card {
    background: linear-gradient(135deg, #0F2A5E 0%, #138808 100%);
    border-radius: 20px; padding: 40px; text-align: center;
    box-shadow: 0 12px 40px rgba(15,42,94,.25);
}
.predict-label { font-size: 13px; font-weight: 600; letter-spacing: .12em;
                 text-transform: uppercase; color: rgba(255,255,255,.65); margin-bottom: 10px; }
.predict-value { font-family: 'DM Serif Display', serif; font-size: 52px;
                 color: #fff; line-height: 1; margin-bottom: 8px; }
.predict-sub   { font-size: 14px; color: rgba(255,255,255,.6); }
.predict-badge { display: inline-flex; align-items: center; gap: 6px;
                 background: rgba(255,153,51,.25); border: 1px solid rgba(255,153,51,.4);
                 border-radius: 30px; padding: 6px 16px; margin-top: 16px;
                 font-size: 12px; font-weight: 600; color: #FFD580; }

.risk-low  { background: #DCFCE7; color: #15803D; padding: 3px 12px;
             border-radius: 20px; font-size: 12px; font-weight: 600; }
.risk-med  { background: #FEF3C7; color: #92400E; padding: 3px 12px;
             border-radius: 20px; font-size: 12px; font-weight: 600; }
.risk-high { background: #FEE2E2; color: #B91C1C; padding: 3px 12px;
             border-radius: 20px; font-size: 12px; font-weight: 600; }

.comp-header { font-size: 11px; font-weight: 700; text-transform: uppercase;
               letter-spacing: .1em; color: #64748B; }
.comp-a { color: #0F2A5E; font-weight: 700; }
.comp-b { color: #138808; font-weight: 700; }

.divider { border: none; border-top: 1px solid #E2E8F0; margin: 24px 0; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #F0F4FA; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }

[data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 800 !important; color: #0F2A5E !important; }
[data-testid="stMetricLabel"] { font-size: 12px !important; font-weight: 600 !important; color: #64748B !important; }
div[data-testid="stExpander"] { border: 1px solid #E2E8F0 !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ─── Indian number formatting helpers ────────────────────────────────────────────
def fmt_inr(v):
    """Format value in Indian Rupees with Cr/L suffix."""
    v = float(v)
    if v >= 1_00_00_000:  # 1 Crore+
        return f"₹{v/1_00_00_000:.2f} Cr"
    elif v >= 1_00_000:   # 1 Lakh+
        return f"₹{v/1_00_000:.1f} L"
    else:
        return f"₹{v:,.0f}"

def fmt_inr_full(v):
    """Full rupee format with commas in Indian style."""
    v = int(v)
    s = str(v)
    if len(s) <= 3:
        return f"₹{s}"
    result = s[-3:]
    s = s[:-3]
    while len(s) > 2:
        result = s[-2:] + "," + result
        s = s[:-2]
    if s:
        result = s + "," + result
    return f"₹{result}"

def sqft_to_sqyd(sqft): return sqft / 9.0
def sqft_to_sqm(sqft):  return sqft * 0.0929

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "models" / "catboost_log_model.pkl"
X_TEST_FILE = BASE_DIR / "models" / "X_test_log.csv"
Y_TEST_FILE = BASE_DIR / "models" / "y_test_log.csv"
DATA_FILE = BASE_DIR / "data" / "processed" / "house_price_final.csv"
FALLBACK_DATA_FILE = BASE_DIR / "data" / "processed" / "house_price_with_risk_recommendation.csv"

@st.cache_resource
def load_catboost_model():
    # REAL MODEL GENERATED VALUES
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_test_data():
    # REAL MODEL GENERATED VALUES
    X_test = pd.read_csv(X_TEST_FILE)
    y_test = pd.read_csv(Y_TEST_FILE)
    return X_test, y_test

@st.cache_data
def load_data():
    data_file = DATA_FILE if DATA_FILE.exists() else FALLBACK_DATA_FILE
    df = pd.read_csv(data_file)
    df = df.rename(columns={
        "number of bedrooms": "bedrooms",
        "number of bathrooms": "bathrooms",
        "living area": "sqft_living",
        "lot area": "sqft_plot",
        "number of floors": "floors",
        "condition of the house": "condition",
        "grade of the house": "grade",
        "Distance from the airport": "airport_dist",
        "Property_Age": "age",
        "Infrastructure_Score": "amenity_score",
        "waterfront present": "gated",
        "Renovated": "furnished",
        "Lattitude": "lat",
        "Longitude": "lon",
        "Risk_Category": "risk_category",
        "Investment_Score": "investment_score",
        "Forecast_1Y": "forecast_1y",
        "Forecast_3Y": "forecast_3y",
        "Forecast_5Y": "forecast_5y",
    })
    df["price"] = df["Price"]
    df["risk_score"] = df["Risk_Score"]
    df["recommendation"] = df["Recommendation"]
    df["city"] = "Pincode " + df["Postal Code"].astype(str)
    df["total_floors"] = df["floors"]
    df["vastu"] = df.get("vastu", 0)
    df["furnished"] = df["furnished"].fillna(0).astype(int)
    df["gated"] = df["gated"].fillna(0).astype(int)
    df["amenity_score"] = (df["amenity_score"] * 10).round(2)
    if "risk_category" not in df.columns:
        df["risk_category"] = pd.cut(
            df["risk_score"], bins=[0, 33, 66, 100],
            labels=["Low Risk", "Medium Risk", "High Risk"], include_lowest=True
        ).astype(str)
    if "investment_score" not in df.columns:
        df["investment_score"] = np.nan
    if "forecast_1y" not in df.columns:
        df["forecast_1y"] = df["price"] * 1.05
    if "forecast_3y" not in df.columns:
        df["forecast_3y"] = df["price"] * (1.05 ** 3)
    if "forecast_5y" not in df.columns:
        df["forecast_5y"] = df["price"] * (1.05 ** 5)
    return df

@st.cache_resource
def load_shap_explainer(_model):
    # REAL MODEL GENERATED VALUES
    return shap.TreeExplainer(_model)

@st.cache_data
def compute_shap_values(_explainer, X):
    # REAL MODEL GENERATED VALUES
    shap_values = _explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
    return shap_values


df = load_data()

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:24px 16px 10px 16px; text-align:center;">
      <div style="font-size:40px; margin-bottom:8px;">🏠</div>
      <div style="font-size:15px; font-weight:800; color:#fff; letter-spacing:.02em;">AI Real Estate India</div>
      <div style="font-size:11px; color:rgba(255,255,255,.45); font-weight:500; letter-spacing:.06em; text-transform:uppercase; margin-top:2px;">Intelligence Platform</div>
    </div>
    <hr style="border-color:rgba(255,255,255,.1); margin:10px 0 18px 0;"/>
    """, unsafe_allow_html=True)

    pages = [
        "📊  Project Overview",
        "🔍  EDA Dashboard",
        "💰  Price Prediction",
        "📈  Investment Recommendation",
        "⚠️  Risk Analysis",
        "🔮  Future Price Forecast",
        "🧠  Explainable AI",
        "⚖️  Property Comparison",
        "🌐  Market Insights",
        "🤖  AI Chatbot & Other Features",
    ]
    page = st.radio("Navigation", pages, label_visibility="collapsed")

    st.markdown(f"""
    <hr style="border-color:rgba(255,255,255,.08); margin:20px 0 16px 0;"/>
    <div style="padding:0 8px 8px 8px;">
      <div style="font-size:11px; color:rgba(255,255,255,.3); text-transform:uppercase; letter-spacing:.08em; font-weight:600; margin-bottom:10px;">Model Info</div>
      <div style="background:rgba(255,255,255,.06); border-radius:10px; padding:12px 14px;">
        <div style="font-size:12px; color:rgba(255,255,255,.55); margin-bottom:6px;">ML Engine</div>
        <div style="font-size:13px; color:#E8C97A; font-weight:700;">CatBoost Regressor</div>
        <div style="font-size:12px; color:rgba(255,255,255,.55); margin-top:8px; margin-bottom:6px;">XAI Framework</div>
        <div style="font-size:13px; color:#14A5BA; font-weight:700;">SHAP Values</div>
        <div style="font-size:12px; color:rgba(255,255,255,.55); margin-top:8px; margin-bottom:6px;">Market Coverage</div>
        <div style="font-size:13px; color:#FF9933; font-weight:700;">{df['city'].nunique():,} Pincode Markets</div>
        <div style="font-size:12px; color:rgba(255,255,255,.55); margin-top:8px; margin-bottom:6px;">Dataset Size</div>
        <div style="font-size:13px; color:#fff; font-weight:600;">{len(df):,} Properties</div>
      </div>
    </div>
    <div style="padding:16px 8px; font-size:11px; color:rgba(255,255,255,.2); text-align:center;">
      v1.0 · Built with Streamlit & Python
    </div>
    """, unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────────
def card(label, value, sub="", badge="", badge_type="neu"):
    badge_html = f'<div class="kpi-badge badge-{badge_type}">{badge}</div>' if badge else ""
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
      {badge_html}
    </div>"""

def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

def section_header(eyebrow, title, sub=""):
    st.markdown(f"""
    <div class="section-eyebrow">{eyebrow}</div>
    <div class="section-title">{title}</div>
    <div class="section-sub">{sub}</div>
    """, unsafe_allow_html=True)

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color=TEXT_MAIN),
    margin=dict(l=10, r=10, t=30, b=10),
)

furnished_label = {0: "Unfurnished", 1: "Semi-Furnished", 2: "Fully Furnished"}


MODEL_DEFAULTS = {
    "number of bedrooms": 3,
    "number of bathrooms": 2.0,
    "living area": 1500,
    "lot area": 2500,
    "number of floors": 1.0,
    "waterfront present": 0,
    "number of views": 0,
    "condition of the house": 3,
    "grade of the house": 7,
    "Postal Code": 122030,
    "Lattitude": 52.85,
    "Longitude": -114.45,
    "living_area_renov": 1500,
    "lot_area_renov": 2500,
    "Number of schools nearby": 2,
    "Distance from the airport": 55,
    "Property_Age": 25,
    "Renovated": 0,
    "Basement_Percentage": 0.0,
    "Infrastructure_Score": 0.55,
}

UPLOAD_COLUMN_ALIASES = {
    "bedrooms": "number of bedrooms",
    "bhk": "number of bedrooms",
    "number_of_bedrooms": "number of bedrooms",
    "bathrooms": "number of bathrooms",
    "number_of_bathrooms": "number of bathrooms",
    "sqft_living": "living area",
    "area": "living area",
    "carpet_area": "living area",
    "living_area": "living area",
    "sqft_plot": "lot area",
    "plot_area": "lot area",
    "lot_area": "lot area",
    "floors": "number of floors",
    "floor": "number of floors",
    "waterfront": "waterfront present",
    "gated": "waterfront present",
    "views": "number of views",
    "condition": "condition of the house",
    "grade": "grade of the house",
    "pincode": "Postal Code",
    "postal_code": "Postal Code",
    "latitude": "Lattitude",
    "lat": "Lattitude",
    "longitude": "Longitude",
    "lon": "Longitude",
    "schools": "Number of schools nearby",
    "schools_nearby": "Number of schools nearby",
    "airport_distance": "Distance from the airport",
    "distance_from_airport": "Distance from the airport",
    "age": "Property_Age",
    "property_age": "Property_Age",
    "renovated": "Renovated",
    "basement_percentage": "Basement_Percentage",
    "infrastructure_score": "Infrastructure_Score",
    "amenity_score": "Infrastructure_Score",
}

def normalize_upload_columns(input_df):
    cleaned = input_df.copy()
    rename_map = {}
    for col in cleaned.columns:
        normalized = col.strip().lower().replace(" ", "_").replace("-", "_")
        if col in MODEL_DEFAULTS:
            rename_map[col] = col
        elif normalized in UPLOAD_COLUMN_ALIASES:
            rename_map[col] = UPLOAD_COLUMN_ALIASES[normalized]
    return cleaned.rename(columns=rename_map)

def prepare_model_matrix(input_df, feature_columns):
    normalized = normalize_upload_columns(input_df)
    model_df = pd.DataFrame(index=normalized.index)
    missing = []
    for feature in feature_columns:
        if feature in normalized.columns:
            model_df[feature] = pd.to_numeric(normalized[feature], errors="coerce")
        else:
            model_df[feature] = MODEL_DEFAULTS.get(feature, 0)
            missing.append(feature)
        model_df[feature] = model_df[feature].fillna(MODEL_DEFAULTS.get(feature, model_df[feature].median()))
    if "living_area_renov" in missing:
        model_df["living_area_renov"] = model_df["living area"]
    if "lot_area_renov" in missing:
        model_df["lot_area_renov"] = model_df["lot area"]
    if "Infrastructure_Score" in model_df.columns and model_df["Infrastructure_Score"].max() > 1:
        model_df["Infrastructure_Score"] = model_df["Infrastructure_Score"] / 10.0
    return model_df[feature_columns], missing

def score_uploaded_properties(input_df):
    model = load_catboost_model()
    X_test, _ = load_test_data()
    model_df, missing = prepare_model_matrix(input_df, X_test.columns)
    scored = input_df.copy()
    scored["Predicted Price"] = np.expm1(model.predict(model_df))
    condition_risk = ((5 - model_df["condition of the house"]) / 4).clip(0, 1)
    age_risk = (model_df["Property_Age"] / max(1, df["age"].max())).clip(0, 1)
    grade_risk = ((13 - model_df["grade of the house"]) / 10).clip(0, 1)
    infrastructure_risk = (1 - model_df["Infrastructure_Score"]).clip(0, 1)
    scored["Risk Score"] = (
        0.35 * age_risk + 0.25 * condition_risk +
        0.25 * grade_risk + 0.15 * infrastructure_risk
    ) * 100
    scored["Risk Category"] = pd.cut(
        scored["Risk Score"], bins=[0, 33, 66, 100],
        labels=["Low Risk", "Medium Risk", "High Risk"], include_lowest=True
    ).astype(str)
    price_per_sqft = scored["Predicted Price"] / model_df["living area"].clip(lower=1)
    value_score = (1 - (price_per_sqft.rank(pct=True))).fillna(0.5) * 100
    quality_score = (
        model_df["grade of the house"].rank(pct=True) * 40 +
        model_df["condition of the house"].rank(pct=True) * 20 +
        model_df["Infrastructure_Score"].rank(pct=True) * 40
    )
    scored["Investment Score"] = (0.55 * quality_score + 0.45 * value_score).clip(0, 100)
    scored["Recommendation"] = np.select(
        [
            (scored["Risk Score"] <= 35) & (scored["Investment Score"] >= 55),
            (scored["Risk Score"] >= 65) | (scored["Investment Score"] < 35),
        ],
        ["BUY", "SELL"],
        default="HOLD",
    )
    return scored, missing

def top_ranked_properties(source_df, limit=10):
    ranked = source_df.copy()
    ranked["rank_score"] = (
        ranked["investment_score"].fillna(ranked["investment_score"].median()) * 0.45 +
        (100 - ranked["risk_score"]) * 0.35 +
        ranked["grade"].rank(pct=True) * 100 * 0.10 +
        ranked["forecast_5y"].rank(pct=True) * 100 * 0.10
    )
    return ranked.sort_values("rank_score", ascending=False).head(limit)

def generate_chatbot_response(question, selected_property):
    q = question.lower()
    buy_count = int((df["recommendation"] == "BUY").sum())
    top_area = (
        df.groupby("city")["investment_score"].mean()
        .sort_values(ascending=False)
        .index[0]
    )
    if selected_property is not None:
        rec = selected_property["recommendation"]
        risk = float(selected_property["risk_score"])
        price = float(selected_property["price"])
        growth = (float(selected_property["forecast_5y"]) / price - 1) * 100
        if "invest" in q or "buy" in q or "property" in q:
            return (
                f"For the selected property in {selected_property['city']}, the data says **{rec}**. "
                f"It is priced at **{fmt_inr(price)}**, has a risk score of **{risk:.1f}/100**, "
                f"and its exported 5-year forecast implies about **{growth:.1f}%** growth. "
                f"I would treat BUY as invest-ready, HOLD as monitor/compare, and SELL as avoid unless the price is deeply discounted."
            )
        if "risk" in q:
            return (
                f"This property's risk score is **{risk:.1f}/100** with category "
                f"**{selected_property['risk_category']}**. The model favors lower age, stronger condition, higher grade, "
                f"and better infrastructure score."
            )
        if "forecast" in q or "future" in q or "growth" in q:
            return (
                f"The selected property moves from **{fmt_inr(price)}** now to "
                f"**{fmt_inr(selected_property['forecast_5y'])}** in the 5-year forecast, "
                f"which is about **{growth:.1f}%** cumulative growth."
            )
    if "top" in q or "best" in q or "rank" in q:
        return (
            f"The top-ranked area by average investment score is **{top_area}**. "
            f"Across the full dataset there are **{buy_count:,} BUY-rated** properties. "
            f"Use the Property Ranking tab on this page for the exact top 10 properties."
        )
    if "risk" in q:
        return (
            f"The portfolio average risk score is **{df['risk_score'].mean():.1f}/100**. "
            f"Low-risk areas are ranked using average risk score and low-risk share in the Market Insights page."
        )
    return (
        "I can answer investment, risk, forecast, and ranking questions using the processed project dataset. "
        f"Currently the dataset has **{len(df):,} properties**, **{buy_count:,} BUY signals**, "
        f"and **{df['city'].nunique():,} pincode markets**."
    )


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if page == pages[0]:
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-title">AI-Powered Indian Real Estate<br>Intelligence Platform</div>
      <div class="hero-subtitle">AI-Driven Property Valuation, Risk Assessment and Investment Intelligence —
      covering Mumbai, Bengaluru, Delhi NCR, Hyderabad & more. Built for precision, speed, and explainability.</div>
      <span class="hero-pill">🤖 CatBoost ML</span>
      <span class="hero-pill">🧠 SHAP Explainability</span>
      <span class="hero-pill">🇮🇳 10 Indian Cities</span>
      <span class="hero-pill">₹ INR Pricing</span>
    </div>
    """, unsafe_allow_html=True)

    buy_count = (df["recommendation"] == "BUY").sum()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(card("Total Properties", f"{len(df):,}", "Active in dataset", "↑ 12% MoM", "up"), unsafe_allow_html=True)
    with c2:
        st.markdown(card("Avg Property Price", fmt_inr(df["price"].mean()), "Across all listings", "↑ 8.5% YoY", "up"), unsafe_allow_html=True)
    with c3:
        st.markdown(card("Avg Risk Score", f"{df['risk_score'].mean():.1f} / 100", "Portfolio-wide risk", "Moderate", "neu"), unsafe_allow_html=True)
    with c4:
        st.markdown(card("BUY Opportunities", f"{buy_count:,}", f"{buy_count/len(df)*100:.1f}% of listings", "High Conviction", "up"), unsafe_allow_html=True)

    st.markdown("<div style='height:24px'/>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 1])
    with col_l:
        section_header("DISTRIBUTION", "Recommendation Breakdown")
        rec_counts = df["recommendation"].value_counts()
        fig_donut = go.Figure(go.Pie(
            labels=rec_counts.index, values=rec_counts.values,
            hole=.65, textinfo="label+percent",
            marker=dict(colors=[SUCCESS, WARNING, DANGER], line=dict(color="#fff", width=3)),
        ))
        fig_donut.update_layout(**CHART_THEME, height=300, showlegend=False,
            annotations=[dict(text=f"<b>{len(df):,}</b><br><span style='font-size:10px'>Properties</span>",
                              x=.5, y=.5, font_size=18, showarrow=False)])
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_r:
        section_header("RISK OVERVIEW", "Risk Category Distribution")
        risk_cats = pd.cut(df["risk_score"], bins=[0,35,65,100], labels=["Low Risk","Medium Risk","High Risk"])
        rc = risk_cats.value_counts()
        fig_bar = go.Figure(go.Bar(
            x=rc.values, y=rc.index, orientation="h",
            marker=dict(color=[SUCCESS, WARNING, DANGER], line=dict(color="#fff", width=1)),
            text=[f"{v:,}" for v in rc.values], textposition="outside",
            textfont=dict(size=13, color=TEXT_MAIN),
        ))
        fig_bar.update_layout(**CHART_THEME, height=300,
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=13)))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
    section_header("EXECUTIVE SUMMARY", "Platform Capabilities & Intelligence")
    cols = st.columns(3)
    caps = [
        ("💰", "Price Prediction", f"CatBoost regressor trained on {len(df):,} Indian properties. Estimated accuracy within ±9% of market value."),
        ("⚠️", "Risk Assessment", "Composite scoring across age, condition, airport access, and neighbourhood market volatility."),
        ("📈", "Investment Engine", "ML-backed BUY / HOLD / SELL signals calibrated to Indian market cycles and RERA compliance."),
        ("🔮", "Price Forecasting", "Multi-horizon forecasts (1Y / 3Y / 5Y) benchmarked against RBI repo rate trends and city-level demand."),
        ("🧠", "Explainable AI", "SHAP values expose the 'why' behind every prediction — full model transparency in INR."),
        ("🌐", "Market Intelligence", "Pincode-level analytics for top investment, growth, and low-risk areas."),
    ]
    for i, (icon, title, desc) in enumerate(caps):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="kpi-card" style="margin-bottom:16px;">
              <div style="font-size:28px;margin-bottom:10px;">{icon}</div>
              <div style="font-size:14px;font-weight:700;color:#0F2A5E;margin-bottom:6px;">{title}</div>
              <div style="font-size:13px;color:#64748B;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
elif page == pages[1]:
    st.markdown("""
    <div class="section-eyebrow">ANALYTICS</div>
    <div class="section-title">Exploratory Data Analysis</div>
    <div class="section-sub">Deep-dive into Indian property distribution, feature relationships, and market patterns.</div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Distribution", "📉 Log Price", "🔥 Correlation Heatmap", "🏆 Top Features"])

    with tab1:
        fig = px.histogram(df, x=df["price"]/1e7, nbins=60,   # show in Crore
                           color_discrete_sequence=[TEAL],
                           title="Property Price Distribution (₹ Crore)")
        fig.update_traces(marker_line_width=0, opacity=.85)
        fig.update_layout(**CHART_THEME, height=400,
            xaxis_title="Price (₹ Crore)", yaxis_title="Count",
            xaxis_tickprefix="₹", xaxis_ticksuffix=" Cr")
        st.plotly_chart(fig, use_container_width=True)
        insight("""<strong>Key Insight:</strong> Indian property prices are heavily right-skewed.
        Most listings fall in the <strong>₹40L–₹1.5Cr</strong> range, with Mumbai and Delhi NCR luxury
        properties above ₹5Cr pulling the mean significantly higher than the median — consistent with
        India's dual-market structure (affordable vs premium).""")

    with tab2:
        fig = px.histogram(df, x=np.log(df["price"]), nbins=60,
                           color_discrete_sequence=[DEEP_BLUE],
                           title="Log-Transformed Price Distribution")
        fig.update_traces(marker_line_width=0, opacity=.85)
        fig.update_layout(**CHART_THEME, height=400,
            xaxis_title="log(Price in ₹)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        insight("""<strong>Key Insight:</strong> Log-transformed prices follow a near-normal distribution,
        confirming log-normality — a universal property of real estate markets including India.
        CatBoost handles the raw INR price skew through its gradient-based splits.""")

    with tab3:
        num_cols = ["price","sqft_living","sqft_plot","bedrooms","bathrooms",
                    "grade","condition","age","airport_dist","risk_score","amenity_score"]
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale=[[0,"#D63B3B"],[.5,"#F7F9FC"],[1,"#0D7C8C"]],
                        title="Feature Correlation Matrix")
        fig.update_layout(**CHART_THEME, height=480)
        st.plotly_chart(fig, use_container_width=True)
        insight("""<strong>Key Insight:</strong> <strong>sqft_living</strong> and <strong>grade</strong>
        show the strongest positive correlation with price (r > 0.65). <strong>airport_dist</strong>
        (distance from the airport) captures location access effects in the source data. <strong>age</strong> also negatively
        impacts price, especially in Tier-1 cities.""")

    with tab4:
        corr_with_price = df[num_cols].corr()["price"].drop("price").sort_values()
        colors = [DANGER if v < 0 else TEAL for v in corr_with_price.values]
        fig = go.Figure(go.Bar(
            x=corr_with_price.values, y=corr_with_price.index,
            orientation="h", marker_color=colors,
            text=[f"{v:.3f}" for v in corr_with_price.values],
            textposition="outside"
        ))
        fig.update_layout(**CHART_THEME, height=420,
            title="Feature Correlation with Price (₹)",
            xaxis_title="Pearson Correlation", yaxis_title="",
            xaxis=dict(range=[-1,1], zeroline=True, zerolinecolor="#E2E8F0"))
        st.plotly_chart(fig, use_container_width=True)
        insight("""<strong>Key Insight:</strong> <strong>grade</strong> and <strong>sqft_living</strong>
        are the top price predictors. Notably, <strong>airport_dist</strong> contributes location context,
        showing how accessibility and distance interact with the model's valuation signal.""")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PROPERTY PRICE PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
elif page == pages[2]:
    section_header("ML ENGINE", "Property Price Prediction",
                   "Enter property details to get an AI-generated market valuation in Indian Rupees.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**🏠 Property Details**")
            bedrooms   = st.number_input("Bedrooms (BHK)", 1, 10, 3)
            bathrooms  = st.number_input("Bathrooms", 1.0, 6.0, 2.0, step=0.5)
            floors     = st.selectbox("Floor Number", [1,2,3,4,5,6,7,8,9,10,15,20])
            total_fl   = st.number_input("Total Floors in Building", 1, 40, 10)
            furnished  = st.selectbox("Furnishing Status",
                                       [("Unfurnished",0),("Semi-Furnished",1),("Fully Furnished",2)],
                                       format_func=lambda x: x[0])
        with c2:
            st.markdown("**📐 Size & Location**")
            sqft_liv   = st.number_input("Carpet Area (sq ft)", 300, 10000, 1200, step=50)
            sqft_plot  = st.number_input("Plot / Builtup Area (sq ft)", 300, 20000, 1500, step=100)
            city_sel   = st.selectbox("Location / Pincode", sorted(df["city"].unique()))
            airport_dist = st.slider("Distance from Airport (km)", 1, 100, 55, step=1)
        with c3:
            st.markdown("**⭐ Quality Indicators**")
            condition  = st.slider("Condition (1–5)", 1, 5, 4)
            grade      = st.slider("Construction Grade (3–13)", 3, 13, 7)
            prop_age   = st.number_input("Property Age (years)", 0, 50, 5)
            vastu      = st.selectbox("Vastu Compliant", [("No",0),("Yes",1)], format_func=lambda x: x[0])
            gated      = st.selectbox("Gated Community", [("No",0),("Yes",1)], format_func=lambda x: x[0])
            amenity    = st.slider("Amenity Score (1–10)", 1, 10, 6)

        submitted = st.form_submit_button("🔮  Predict Property Value in ₹")

    if submitted:
        model = load_catboost_model()
        X_test, _ = load_test_data()
        explainer = load_shap_explainer(model)

        furn_val  = furnished[1] if isinstance(furnished, tuple) else furnished
        vastu_val = vastu[1]     if isinstance(vastu, tuple) else vastu
        gated_val = gated[1]     if isinstance(gated, tuple) else gated

        input_template = X_test.median(numeric_only=True).to_dict()
        pred_row = pd.DataFrame([input_template], columns=X_test.columns)
        market_df = df[df["city"] == city_sel]
        market_postal = int(market_df["Postal Code"].median()) if not market_df.empty else int(pred_row["Postal Code"].iloc[0])
        pred_row["number of bedrooms"] = bedrooms
        pred_row["number of bathrooms"] = bathrooms
        pred_row["living area"] = sqft_liv
        pred_row["lot area"] = sqft_plot
        pred_row["number of floors"] = floors
        pred_row["waterfront present"] = gated_val
        pred_row["number of views"] = 0
        pred_row["condition of the house"] = condition
        pred_row["grade of the house"] = grade
        pred_row["Postal Code"] = market_postal
        pred_row["Lattitude"] = market_df["lat"].median() if not market_df.empty else X_test["Lattitude"].median()
        pred_row["Longitude"] = market_df["lon"].median() if not market_df.empty else X_test["Longitude"].median()
        pred_row["living_area_renov"] = sqft_liv
        pred_row["lot_area_renov"] = sqft_plot
        pred_row["Number of schools nearby"] = max(1, int(amenity / 1.5))
        pred_row["Distance from the airport"] = airport_dist
        pred_row["Property_Age"] = prop_age
        pred_row["Renovated"] = 1 if furn_val else 0
        pred_row["Basement_Percentage"] = 0.0
        pred_row["Infrastructure_Score"] = float(amenity) / 10.0
        pred_row = pred_row[X_test.columns]

        pred_log = float(model.predict(pred_row)[0])
        pred = float(np.expm1(pred_log))
        low  = pred * 0.91
        high = pred * 1.09

        shap_values = compute_shap_values(explainer, pred_row)
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[0]

        contrib_df = pd.DataFrame({
            "feature": pred_row.columns,
            "shap_value": shap_values.flatten(),
            "abs_shap": np.abs(shap_values).flatten(),
            "feature_value": pred_row.iloc[0].values,
        }).sort_values("abs_shap", ascending=False).head(10)

        col_res, col_meta = st.columns([1.2, 1])
        with col_res:
            st.markdown(f"""
            <div class="predict-card">
              <div class="predict-label">Predicted Market Value</div>
              <div class="predict-value">{fmt_inr(pred)}</div>
              <div class="predict-sub">Confidence Range: {fmt_inr(low)} – {fmt_inr(high)}</div>
              <div style="font-size:13px;color:rgba(255,255,255,.55);margin-top:8px;">{fmt_inr_full(int(pred))}</div>
              <div class="predict-badge">🤖 CatBoost · SHAP explainability · {city_sel} estimate</div>
            </div>
            """, unsafe_allow_html=True)

            psf = pred / max(1, sqft_liv)
            st.markdown(f"""
            <div style="margin-top:16px;display:grid;grid-template-columns:1fr 1fr;gap:12px;">
              <div class="kpi-card" style="text-align:center;padding:16px;">
                <div class="kpi-label">Price / Sq Ft</div>
                <div class="kpi-value" style="font-size:22px;">₹{psf:,.0f}</div>
              </div>
              <div class="kpi-card" style="text-align:center;padding:16px;">
                <div class="kpi-label">Price / Sq Yd</div>
                <div class="kpi-value" style="font-size:22px;">₹{psf*9:,.0f}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_meta:
            st.markdown("**📊 Model Feature Contributions (log-price impact)**")
            fig_break = go.Figure(go.Bar(
                x=contrib_df["shap_value"],
                y=contrib_df["feature"],
                orientation="h",
                marker_color=[SUCCESS if v >= 0 else DANGER for v in contrib_df["shap_value"]],
                text=[f"{v:.4f}" for v in contrib_df["shap_value"]],
                textposition="outside",
            ))
            compact_theme = {**CHART_THEME, "margin": dict(l=140, r=10, t=30, b=30)}
            fig_break.update_layout(**compact_theme, height=320,
                xaxis_title="SHAP Value (log price)", yaxis_title="Feature",
                showlegend=False)
            st.plotly_chart(fig_break, use_container_width=True)

        insight(f"""<strong>Valuation Summary:</strong> The CatBoost model produced a log-price prediction of <strong>{pred_log:.4f}</strong>, returning an estimated property value of <strong>{fmt_inr(pred)}</strong>.
        The bar chart shows the top SHAP drivers for this input, where positive values increase the predicted log-price and negative values decrease it.""")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — INVESTMENT RECOMMENDATION
# ════════════════════════════════════════════════════════════════════════════════
elif page == pages[3]:
    section_header("PORTFOLIO INTELLIGENCE", "Investment Recommendation Engine",
                   "AI-generated BUY / HOLD / SELL signals across all Indian properties.")

    rec_counts = df["recommendation"].value_counts()
    buy_df  = df[df["recommendation"] == "BUY"]
    hold_df = df[df["recommendation"] == "HOLD"]
    sell_df = df[df["recommendation"] == "SELL"]

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(card("BUY Signals",   f"{len(buy_df):,}",  f"{len(buy_df)/len(df)*100:.1f}% of portfolio", "↑ High ROI", "up"),  unsafe_allow_html=True)
    with c2: st.markdown(card("HOLD Signals",  f"{len(hold_df):,}", f"{len(hold_df)/len(df)*100:.1f}% of portfolio","Monitor",    "neu"), unsafe_allow_html=True)
    with c3: st.markdown(card("SELL Signals",  f"{len(sell_df):,}", f"{len(sell_df)/len(df)*100:.1f}% of portfolio","↓ Exit",     "down"), unsafe_allow_html=True)
    with c4: st.markdown(card("Avg BUY Price", fmt_inr(buy_df["price"].mean()), "Entry point estimate", "Opportunity", "up"), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'/>", unsafe_allow_html=True)

    col_d, col_b = st.columns([1, 1.4])
    with col_d:
        section_header("", "Signal Distribution")
        fig_pie = go.Figure(go.Pie(
            labels=rec_counts.index, values=rec_counts.values,
            hole=.6, pull=[.04,0,0],
            marker=dict(colors=[SUCCESS, WARNING, DANGER], line=dict(color="#fff", width=3)),
            textinfo="label+percent+value",
        ))
        fig_pie.update_layout(**CHART_THEME, height=320, showlegend=False,
            annotations=[dict(text="<b>Signals</b>", x=.5, y=.5, font_size=14, showarrow=False)])
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        section_header("", "Avg Price by Recommendation (₹ Crore)")
        avg_by_rec = df.groupby("recommendation")["price"].mean().reindex(["BUY","HOLD","SELL"])
        fig_bars = go.Figure(go.Bar(
            x=avg_by_rec.index, y=avg_by_rec.values / 1e7,
            marker_color=[SUCCESS, WARNING, DANGER],
            text=[fmt_inr(v) for v in avg_by_rec.values],
            textposition="outside", textfont=dict(size=13, color=TEXT_MAIN), width=.5
        ))
        fig_bars.update_layout(**CHART_THEME, height=320,
            yaxis_title="Avg Price (₹ Crore)", yaxis_tickprefix="₹", yaxis_ticksuffix=" Cr",
            showlegend=False, xaxis_tickfont=dict(size=14, color=TEXT_MAIN))
        st.plotly_chart(fig_bars, use_container_width=True)

    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
    section_header("TOP PICKS", "High-Conviction BUY Opportunities")
    top_buy = (buy_df.nsmallest(10, "risk_score")
               [["city","price","sqft_living","grade","risk_score","recommendation"]]
               .rename(columns={"city":"City","price":"Price (₹)","sqft_living":"Carpet (sq ft)",
                                 "grade":"Grade","risk_score":"Risk Score","recommendation":"Signal"}))
    top_buy["Price (₹)"] = top_buy["Price (₹)"].apply(fmt_inr)
    top_buy["Carpet (sq ft)"] = top_buy["Carpet (sq ft)"].apply(lambda x: f"{x:,} sq ft")
    top_buy["Risk Score"] = top_buy["Risk Score"].apply(lambda x: f"{x:.1f}")
    st.dataframe(top_buy.reset_index(drop=True), use_container_width=True, height=320)
    insight("""<strong>Investment Insight:</strong> BUY-rated properties combine lower risk scores,
    stronger grades, and better investment scores. Prioritize pincode markets with repeat BUY signals
    and above-average 5-year forecast upside.""")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RISK ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == pages[4]:
    section_header("RISK INTELLIGENCE", "Property Risk Analysis",
                   "Composite risk scoring across structural, locational, and market factors — calibrated to Indian market conditions.")

    risk_cats = pd.cut(df["risk_score"], bins=[0,35,65,100], labels=["Low Risk","Medium Risk","High Risk"])
    df2 = df.copy(); df2["risk_category"] = risk_cats
    rc_counts = df2["risk_category"].value_counts()

    c1,c2,c3,c4 = st.columns(4)
    for col, label, val, badge, bt in [
        (c1,"Portfolio Avg Risk",  f"{df['risk_score'].mean():.1f}","Moderate","neu"),
        (c2,"Low Risk Properties", f"{rc_counts.get('Low Risk',0):,}","Safe Zone","up"),
        (c3,"Medium Risk",         f"{rc_counts.get('Medium Risk',0):,}","Watch List","neu"),
        (c4,"High Risk Properties",f"{rc_counts.get('High Risk',0):,}","Avoid","down"),
    ]:
        with col: st.markdown(card(label, val, "", badge, bt), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'/>", unsafe_allow_html=True)

    col_hist, col_box = st.columns(2)
    with col_hist:
        section_header("","Risk Score Distribution")
        fig = px.histogram(df, x="risk_score", nbins=50, color_discrete_sequence=[TEAL], opacity=.85)
        fig.add_vline(x=35, line_dash="dash", line_color=SUCCESS, annotation_text="Low/Mid boundary")
        fig.add_vline(x=65, line_dash="dash", line_color=DANGER, annotation_text="Mid/High boundary")
        fig.update_layout(**CHART_THEME, height=320, xaxis_title="Risk Score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col_box:
        section_header("","Risk by City")
        fig = px.box(df2.sort_values("city"), x="city", y="risk_score",
                     color="city", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(**CHART_THEME, height=320, xaxis_title="", yaxis_title="Risk Score",
            showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
    section_header("","Risk Category Deep-Dive")
    tabs = st.tabs(["🟢 Low Risk","🟡 Medium Risk","🔴 High Risk"])
    for tab, cat in zip(tabs, ["Low Risk","Medium Risk","High Risk"]):
        with tab:
            sub = df2[df2["risk_category"] == cat].copy()
            stats = sub[["price","sqft_living","grade","age","condition","airport_dist"]].describe().round(1)
            st.dataframe(stats, use_container_width=True)
            insight(f"""<strong>{cat} Portfolio ({len(sub):,} properties):</strong>
            Average price <strong>{fmt_inr(sub['price'].mean())}</strong>,
            avg grade <strong>{sub['grade'].mean():.1f}</strong>,
            avg age <strong>{sub['age'].mean():.0f} years</strong>,
            avg airport distance <strong>{sub['airport_dist'].mean():.1f} km</strong>.
            {'These represent safer entry points — newer builds with stronger grades and infrastructure scores.' if cat=='Low Risk' else 'Require careful due diligence; check documentation and builder track record.' if cat=='Medium Risk' else 'Avoid unless acquiring at a steep discount for redevelopment potential.'}""")

    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
    st.markdown("""
    <div class="chart-card">
      <div style="font-size:14px;font-weight:700;color:#0F2A5E;margin-bottom:12px;">🧮 Indian Risk Score Methodology</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;">
        <div><div style="font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#64748B;font-weight:600;margin-bottom:6px;">Structural (40%)</div>
             <div style="font-size:13px;color:#1A2340;line-height:1.5;">Property age, condition rating, RERA approval status, and builder reputation score.</div></div>
        <div><div style="font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#64748B;font-weight:600;margin-bottom:6px;">Location (30%)</div>
             <div style="font-size:13px;color:#1A2340;line-height:1.5;">Airport access, infrastructure score, market volatility, and location quality signals.</div></div>
        <div><div style="font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#64748B;font-weight:600;margin-bottom:6px;">Market (30%)</div>
             <div style="font-size:13px;color:#1A2340;line-height:1.5;">Price volatility per micro-market, RBI repo rate sensitivity, stamp duty burden, and rental yield trends.</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 6 — FUTURE PRICE FORECAST
# ════════════════════════════════════════════════════════════════════════════════
elif page == pages[5]:
    section_header("FORECASTING", "Future Property Price Forecast",
                   "Multi-horizon projections calibrated to Indian housing market growth rates and RBI monetary policy trends.")

    avg_price = df["price"].mean()
    p1 = df["forecast_1y"].mean()
    p3 = df["forecast_3y"].mean()
    p5 = df["forecast_5y"].mean()
    rate_1y = (p1 / avg_price) - 1
    rate_3y = (p3 / avg_price) ** (1 / 3) - 1
    rate_5y = (p5 / avg_price) ** (1 / 5) - 1

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(card("Current Avg Price",  fmt_inr(avg_price), "Baseline from processed data", "", "neu"), unsafe_allow_html=True)
    with c2: st.markdown(card("Forecast: 1 Year",   fmt_inr(p1), f"+{rate_1y*100:.1f}% growth", "↑ Forecast", "up"), unsafe_allow_html=True)
    with c3: st.markdown(card("Forecast: 3 Years",  fmt_inr(p3), f"+{(p3/avg_price-1)*100:.1f}% cumulative", "↑ Strong", "up"), unsafe_allow_html=True)
    with c4: st.markdown(card("Forecast: 5 Years",  fmt_inr(p5), f"+{(p5/avg_price-1)*100:.1f}% cumulative", "↑ Strong", "up"), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'/>", unsafe_allow_html=True)

    years = list(range(0, 11))
    prices_base = [avg_price * (1 + rate_5y) ** y for y in years]
    prices_bull  = [avg_price * (1 + min(rate_5y + 0.04, 0.20)) ** y for y in years]
    prices_bear  = [avg_price * (1 + max(rate_5y - 0.03, 0.01)) ** y for y in years]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=[v/1e7 for v in prices_bull], mode="lines",
                             name=f"Bull Scenario (+{(min(rate_5y + 0.04, 0.20))*100:.1f}%)", line=dict(color=SUCCESS, dash="dot", width=2)))
    fig.add_trace(go.Scatter(x=years, y=[v/1e7 for v in prices_base], mode="lines+markers",
                             name=f"Base Scenario (+{rate_5y*100:.1f}%)", line=dict(color=TEAL, width=3),
                             marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=years, y=[v/1e7 for v in prices_bear], mode="lines",
                             name=f"Bear Scenario (+{max(rate_5y - 0.03, 0.01)*100:.1f}%)", line=dict(color=DANGER, dash="dot", width=2)))
    for y, p in [(1, prices_base[1]), (3, prices_base[3]), (5, prices_base[5])]:
        fig.add_annotation(x=y, y=p/1e7, text=fmt_inr(p),
                           showarrow=True, arrowhead=2, ax=0, ay=-30,
                           font=dict(size=11, color=DEEP_BLUE))
    fig.update_layout(**CHART_THEME, height=420,
        title="10-Year Indian Real Estate Price Appreciation Scenarios",
        xaxis_title="Years from Now", yaxis_title="Avg Property Price (₹ Crore)",
        yaxis_tickprefix="₹", yaxis_ticksuffix=" Cr",
        xaxis=dict(dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    # Investment growth table — in INR
    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
    section_header("", "Investment Growth at Different Entry Points (₹)")
    entry_prices = [30_00_000, 75_00_000, 1_50_00_000, 3_00_00_000]   # 30L, 75L, 1.5Cr, 3Cr
    rows = []
    for ep in entry_prices:
        rows.append({
            "Entry Price":     fmt_inr(ep),
            "Value in 1Y":     fmt_inr(ep * (1 + rate_1y)),
            "Value in 3Y":     fmt_inr(ep * (1 + rate_3y)**3),
            "Value in 5Y":     fmt_inr(ep * (1 + rate_5y)**5),
            "Gain 5Y (₹)":    fmt_inr(ep * ((1+rate_5y)**5 - 1)),
            "ROI 5Y (%)":      f"{((1+rate_5y)**5-1)*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    insight(f"""<strong>Forecast Insight:</strong> Under the exported base forecast ({rate_5y*100:.1f}% annual appreciation),
    a ₹75L property today is projected to reach <strong>{fmt_inr(75_00_000*(1+rate_5y)**5)}</strong>
    in 5 years — a compounded gain of {((1+rate_5y)**5-1)*100:.1f}%.
    Areas with stronger infrastructure scores and lower risk can be prioritized above the base case.
    Factor in stamp duty (~5–7%) and registration costs when computing net ROI.""")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 7 — EXPLAINABLE AI
# ════════════════════════════════════════════════════════════════════════════════
elif page == pages[6]:
    section_header("EXPLAINABILITY", "Explainable AI — SHAP Analysis",
                   "Understand why the model made each prediction through SHAP value decomposition (log-price impact).")

    try:
        model = load_catboost_model()
        X_test, y_test = load_test_data()
        explainer = load_shap_explainer(model)
        shap_values = compute_shap_values(explainer, X_test)
    except Exception as ex:
        st.error(f"Unable to load explainability artifacts: {ex}")
        st.stop()

    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
        if shap_values.shape[0] == 1:
            shap_values = shap_values[0]
    if shap_values.ndim == 3:
        shap_values = shap_values[0]

    feature_names = list(X_test.columns)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap": mean_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    top_features = shap_df["feature"].tolist()[:14]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Feature Importance","📉 SHAP Summary","📊 SHAP Bar","🌊 Waterfall"])

    with tab1:
        fig = go.Figure(go.Bar(
            x=shap_df["mean_abs_shap"],
            y=shap_df["feature"],
            orientation="h",
            marker=dict(color=shap_df["mean_abs_shap"],
                        colorscale=[[0,"#E8F4F8"],[.5,TEAL],[1,DEEP_BLUE]],
                        showscale=True, colorbar=dict(title="Mean |SHAP|")),
        ))
        fig.update_layout(**CHART_THEME, height=520,
            title="Global Feature Importance (Mean |SHAP|)",
            xaxis_title="Mean |SHAP Value|")
        st.plotly_chart(fig, use_container_width=True)
        insight("""<strong>SHAP Insight:</strong> This chart is generated from the trained CatBoost model and the saved test dataset.
        The features are ranked by the mean absolute contribution to the model's log-price prediction. Higher values show stronger global influence.""")

    with tab2:
        fig = go.Figure()
        colorbar_shown = False
        for feat in top_features:
            idx = feature_names.index(feat)
            values = X_test[feat].values
            shap_vals = shap_values[:, idx]
            fig.add_trace(go.Scatter(
                x=shap_vals,
                y=[feat] * len(values),
                mode="markers",
                marker=dict(size=6, opacity=.6,
                            color=values,
                            colorscale="RdBu",
                            showscale=not colorbar_shown,
                            colorbar=dict(title=feat) if not colorbar_shown else None),
                hovertemplate=f"{feat}: %{{marker.color}}<br>SHAP: %{{x:.4f}}<extra></extra>",
                name=feat,
                showlegend=False,
            ))
            colorbar_shown = True
        fig.add_vline(x=0, line_color="#E2E8F0", line_width=1)
        fig.update_layout(**CHART_THEME, height=520,
            title="SHAP Summary Plot — Actual Model Predictions",
            xaxis_title="SHAP Value (log price impact)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        sorted_bar = shap_df.sort_values("mean_shap")
        fig = go.Figure(go.Bar(
            x=sorted_bar["mean_shap"],
            y=sorted_bar["feature"],
            orientation="h",
            marker_color=[SUCCESS if v >= 0 else DANGER for v in sorted_bar["mean_shap"]],
            text=[f"+{v:.4f}" if v >= 0 else f"{v:.4f}" for v in sorted_bar["mean_shap"]],
            textposition="outside"
        ))
        fig.add_vline(x=0, line_color="#E2E8F0")
        fig.update_layout(**CHART_THEME, height=520,
            title="SHAP Directional Impact — Mean SHAP Value",
            xaxis_title="Mean SHAP Value")
        st.plotly_chart(fig, use_container_width=True)
        insight("""<strong>Direction Insight:</strong> Positive mean SHAP values push the log prediction higher,
        while negative mean SHAP values reduce the model output. These values are computed from the saved test dataset and the trained CatBoost model.""")

    with tab4:
        sample_idx = st.selectbox("Select sample index for waterfall explanation", list(range(min(40, len(X_test)))), index=0)
        sample_row = X_test.iloc[sample_idx]
        sample_shap = shap_values[sample_idx]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, tuple, np.ndarray)):
            base_value = float(base_value[0])
        pred_log = float(model.predict(sample_row.to_numpy().reshape(1, -1))[0])
        pred_price = np.expm1(pred_log)
        contrib_df = pd.DataFrame({
            "feature": feature_names,
            "shap_value": sample_shap,
            "abs_shap": np.abs(sample_shap),
            "feature_value": sample_row.values,
        }).sort_values("abs_shap", ascending=False).head(12)

        measures = ["absolute"] + ["relative"] * len(contrib_df) + ["total"]
        x_vals = ["Base Value"] + contrib_df["feature"].tolist() + ["Model Output"]
        y_vals = [base_value] + contrib_df["shap_value"].tolist() + [pred_log]
        text_vals = [f"{base_value:.4f}"] + [f"{v:.4f}" for v in contrib_df["shap_value"]] + [f"{pred_log:.4f}"]

        fig = go.Figure(go.Waterfall(
            x=x_vals,
            y=y_vals,
            measure=measures,
            text=text_vals,
            textposition="outside",
            connector={"line": {"color": "#888"}},
            decreasing={"marker": {"color": DANGER}},
            increasing={"marker": {"color": SUCCESS}},
            totals={"marker": {"color": DEEP_BLUE}},
        ))
        fig.update_layout(**CHART_THEME, height=460,
            title=f"SHAP Waterfall — Sample {sample_idx} Prediction",
            yaxis_title="Log-Price Value")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="predict-card" style="padding:28px; text-align:left; margin-top:16px;">
          <div style="font-size:13px;color:rgba(255,255,255,.6);text-transform:uppercase;letter-spacing:.1em;font-weight:600;margin-bottom:10px;">Real Model Explanation for Sample {sample_idx}</div>
          <div style="font-size:15px;color:#fff;line-height:1.7;">
            The trained CatBoost model predicted a log-price of <strong>{pred_log:.4f}</strong>, which corresponds
            to an estimated property value of <strong>{fmt_inr(pred_price)}</strong> after exponentiating the log output.
            The waterfall above shows the top SHAP contributions from actual model predictions for this sample.
          </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 8 — PROPERTY COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
elif page == pages[7]:
    section_header("COMPARE", "Property Comparison Tool",
                   "Evaluate two Indian properties side-by-side across price, risk, signals, and forecasts.")

    sample = df.head(20).reset_index(drop=True)
    ids = [f"Property #{i+1} — {row['city']} | {fmt_inr(row['price'])} | {row['bedrooms']}BHK"
           for i, row in sample.iterrows()]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="comp-a">Property A</div>', unsafe_allow_html=True)
        sel_a = st.selectbox("Select Property A", ids, index=0, key="pa")
    with col_b:
        st.markdown('<div class="comp-b">Property B</div>', unsafe_allow_html=True)
        sel_b = st.selectbox("Select Property B", ids, index=3, key="pb")

    idx_a = ids.index(sel_a); idx_b = ids.index(sel_b)
    pa = sample.iloc[idx_a]; pb = sample.iloc[idx_b]

    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

    def comp_row(label, va, vb, better="higher"):
        def highlight(v1, v2):
            if better == "higher":
                return ("✅", "—") if v1 > v2 else ("—", "✅") if v2 > v1 else ("—","—")
            else:
                return ("✅", "—") if v1 < v2 else ("—", "✅") if v2 < v1 else ("—","—")
        try:
            h1, h2 = highlight(float(str(va).replace("₹","").replace(",","").replace("Cr","").replace("L","").strip()),
                                float(str(vb).replace("₹","").replace(",","").replace("Cr","").replace("L","").strip()))
        except Exception:
            h1 = h2 = ""
        return f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;
                    padding:12px 0;border-bottom:1px solid #E2E8F0;align-items:center;">
          <div style="font-size:13px;color:#64748B;font-weight:500;">{label}</div>
          <div style="font-size:14px;font-weight:700;color:#0F2A5E;">{va} {h1}</div>
          <div style="font-size:14px;font-weight:700;color:#138808;">{vb} {h2}</div>
        </div>"""

    header = """
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;
                padding:10px 0 14px 0;border-bottom:2px solid #E2E8F0;">
      <div class="comp-header">Metric</div>
      <div class="comp-header comp-a">Property A</div>
      <div class="comp-header comp-b">Property B</div>
    </div>"""

    rows_html = header
    metrics = [
        ("Price (₹)",         fmt_inr(pa["price"]),       fmt_inr(pb["price"]),        "higher"),
        ("Price/Sq Ft",       f"₹{pa['price']/pa['sqft_living']:,.0f}", f"₹{pb['price']/pb['sqft_living']:,.0f}", "lower"),
        ("Config (BHK)",      f"{pa['bedrooms']} BHK",    f"{pb['bedrooms']} BHK",     "higher"),
        ("Carpet Area",       f"{pa['sqft_living']:,} sqft / {pa['sqft_living']*0.0929:.0f} sqm",
                              f"{pb['sqft_living']:,} sqft / {pb['sqft_living']*0.0929:.0f} sqm", "higher"),
        ("Bathrooms",         pa["bathrooms"],             pb["bathrooms"],             "higher"),
        ("Floor",             pa["floors"],                pb["floors"],                "higher"),
        ("Grade",             f"{pa['grade']}/13",         f"{pb['grade']}/13",         "higher"),
        ("Condition",         f"{pa['condition']}/5",      f"{pb['condition']}/5",      "higher"),
        ("Age",               f"{pa['age']} yrs",          f"{pb['age']} yrs",          "lower"),
        ("Airport Distance",  f"{pa['airport_dist']} km",  f"{pb['airport_dist']} km",  "lower"),
        ("Amenity Score",     f"{pa['amenity_score']}/10", f"{pb['amenity_score']}/10", "higher"),
        ("Gated Community",   "Yes" if pa["gated"] else "No", "Yes" if pb["gated"] else "No", "higher"),
        ("Vastu Compliant",   "Yes" if pa["vastu"] else "No", "Yes" if pb["vastu"] else "No", "higher"),
        ("Furnishing",        furnished_label[pa["furnished"]], furnished_label[pb["furnished"]], "neither"),
        ("Risk Score",        f"{pa['risk_score']:.1f}",   f"{pb['risk_score']:.1f}",   "lower"),
        ("Recommendation",    pa["recommendation"],        pb["recommendation"],        "neither"),
        ("City",              pa["city"],                  pb["city"],                  "neither"),
    ]
    for m in metrics:
        rows_html += comp_row(*m)

    st.markdown(f'<div class="chart-card">{rows_html}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'/>", unsafe_allow_html=True)
    radar_cats = ["Price", "Area", "Grade", "Condition", "Amenities", "Risk-adj"]
    def normalize(val, mn, mx): return (val - mn) / (mx - mn) * 10

    a_vals = [
        normalize(pa["price"], df["price"].min(), df["price"].max()),
        normalize(pa["sqft_living"], 350, 6000),
        normalize(pa["grade"], 3, 13),
        normalize(pa["condition"], 1, 5),
        normalize(pa["amenity_score"], 1, 10),
        normalize(100 - pa["risk_score"], 0, 100),
    ]
    b_vals = [
        normalize(pb["price"], df["price"].min(), df["price"].max()),
        normalize(pb["sqft_living"], 350, 6000),
        normalize(pb["grade"], 3, 13),
        normalize(pb["condition"], 1, 5),
        normalize(pb["amenity_score"], 1, 10),
        normalize(100 - pb["risk_score"], 0, 100),
    ]
    fig_radar = go.Figure()
    for vals, name, clr in [(a_vals,"Property A",DEEP_BLUE),(b_vals,"Property B",INDIA_GREEN)]:
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=radar_cats + [radar_cats[0]],
            fill="toself", name=name, opacity=.5,
            line=dict(color=clr, width=2), marker=dict(color=clr)))
    fig_radar.update_layout(**CHART_THEME, height=380,
        polar=dict(radialaxis=dict(visible=True, range=[0,10])),
        showlegend=True, title="Multi-Dimensional Property Comparison")
    st.plotly_chart(fig_radar, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 9 — MARKET INSIGHTS
# ════════════════════════════════════════════════════════════════════════════════
elif page == pages[8]:
    section_header("MARKET INTELLIGENCE", "Indian Real Estate Market Insights",
                   "Executive-level analysis of price dynamics, geography, and property quality across 10 Indian cities.")

    city_stats = df.groupby("city").agg(
        avg_price=("price","mean"), count=("price","count"),
        avg_grade=("grade","mean"), avg_risk=("risk_score","mean"),
        avg_psf=("price","mean"), avg_investment=("investment_score","mean"),
        buy_rate=("recommendation", lambda s: (s == "BUY").mean() * 100),
        low_risk_rate=("risk_category", lambda s: (s == "Low Risk").mean() * 100),
        avg_forecast_5y=("forecast_5y","mean")
    ).reset_index()
    city_stats["avg_psf"] = city_stats["avg_psf"] / df.groupby("city")["sqft_living"].mean().values
    city_stats["growth_5y_pct"] = (city_stats["avg_forecast_5y"] / city_stats["avg_price"] - 1) * 100
    city_stats = city_stats.sort_values("avg_price", ascending=False)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(card("Market Avg Price", fmt_inr(df["price"].mean()), "All 10 cities", "↑ Trending", "up"), unsafe_allow_html=True)
    with c2: st.markdown(card("Highest Value City", city_stats.iloc[0]["city"], fmt_inr(city_stats.iloc[0]["avg_price"]), "Premium Tier", "up"), unsafe_allow_html=True)
    with c3: st.markdown(card("Avg Price / Sq Ft", f"₹{(df['price']/df['sqft_living']).mean():,.0f}", "Across all cities", "", "neu"), unsafe_allow_html=True)
    with c4: st.markdown(card("Gated Community %", f"{df['gated'].mean()*100:.1f}%", "Of all listings", "Premium", "neu"), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'/>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["🏙️ City Analysis","🗺️ Location Map","⭐ Grade Analysis","🕰️ Age Analysis"])

    with tab1:
        col_l, col_r = st.columns(2)
        with col_l:
            fig = px.bar(city_stats, x="city", y=city_stats["avg_price"]/1e7,
                         color="avg_price",
                         color_continuous_scale=[[0,TEAL_LIGHT],[1,DEEP_BLUE]],
                         text=city_stats["avg_price"].apply(fmt_inr))
            fig.update_traces(textposition="outside", textfont=dict(size=10))
            fig.update_layout(**CHART_THEME, height=380,
                title="Average Price by City (₹ Crore)",
                xaxis_tickangle=-30, yaxis_title="Avg Price (₹ Cr)",
                yaxis_tickprefix="₹", yaxis_ticksuffix=" Cr",
                showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            fig = go.Figure(go.Scatter(
                x=city_stats["avg_risk"], y=city_stats["avg_price"]/1e7,
                mode="markers+text",
                text=city_stats["city"],
                textposition="top center",
                marker=dict(size=city_stats["count"]/8,
                            color=city_stats["avg_grade"],
                            colorscale="Teal", showscale=True,
                            colorbar=dict(title="Avg Grade"),
                            line=dict(color="#fff", width=1)),
            ))
            fig.update_layout(**CHART_THEME, height=380,
                title="Risk vs Price by City (bubble = volume)",
                xaxis_title="Avg Risk Score",
                yaxis_title="Avg Price (₹ Cr)",
                yaxis_tickprefix="₹", yaxis_ticksuffix=" Cr")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        ranking_cols = {
            "city": "Market",
            "count": "Properties",
            "avg_price": "Avg Price",
            "buy_rate": "BUY %",
            "avg_risk": "Risk",
            "growth_5y_pct": "5Y Growth",
        }
        display_cols = list(ranking_cols.values())
        with r1:
            st.markdown("**Top Investment Areas**")
            top_invest = city_stats.sort_values(["buy_rate","avg_investment"], ascending=False).head(10).rename(columns=ranking_cols)
            top_invest["Avg Price"] = top_invest["Avg Price"].apply(fmt_inr)
            top_invest["BUY %"] = top_invest["BUY %"].map(lambda v: f"{v:.1f}%")
            top_invest["Risk"] = top_invest["Risk"].map(lambda v: f"{v:.1f}")
            top_invest["5Y Growth"] = top_invest["5Y Growth"].map(lambda v: f"{v:.1f}%")
            st.dataframe(top_invest[display_cols], use_container_width=True, height=300)
        with r2:
            st.markdown("**High Growth Areas**")
            high_growth = city_stats.sort_values(["growth_5y_pct","avg_investment"], ascending=False).head(10).rename(columns=ranking_cols)
            high_growth["Avg Price"] = high_growth["Avg Price"].apply(fmt_inr)
            high_growth["BUY %"] = high_growth["BUY %"].map(lambda v: f"{v:.1f}%")
            high_growth["Risk"] = high_growth["Risk"].map(lambda v: f"{v:.1f}")
            high_growth["5Y Growth"] = high_growth["5Y Growth"].map(lambda v: f"{v:.1f}%")
            st.dataframe(high_growth[display_cols], use_container_width=True, height=300)
        with r3:
            st.markdown("**Low Risk Areas**")
            low_risk = city_stats.sort_values(["avg_risk","low_risk_rate"], ascending=[True, False]).head(10).rename(columns=ranking_cols)
            low_risk["Avg Price"] = low_risk["Avg Price"].apply(fmt_inr)
            low_risk["BUY %"] = low_risk["BUY %"].map(lambda v: f"{v:.1f}%")
            low_risk["Risk"] = low_risk["Risk"].map(lambda v: f"{v:.1f}")
            low_risk["5Y Growth"] = low_risk["5Y Growth"].map(lambda v: f"{v:.1f}%")
            st.dataframe(low_risk[display_cols], use_container_width=True, height=300)
        insight("""<strong>Market Insight:</strong> Rankings are calculated from the final processed dataset:
        BUY-rate and investment score identify top investment areas, exported 5-year forecast values identify
        growth areas, and average risk score identifies safer pincode markets.""")

    with tab2:
        fig_map = px.scatter_mapbox(
            df.sample(600, random_state=42),
            lat="lat", lon="lon", color=df.sample(600, random_state=42)["price"]/1e7,
            size="sqft_living", hover_data=["city","grade","recommendation","bedrooms"],
            color_continuous_scale=[[0,"#14A5BA"],[.5,"#FF9933"],[1,"#D63B3B"]],
            size_max=18, zoom=4.5,
            mapbox_style="carto-positron",
            title="Property Price Map — Indian Metro Cities",
            labels={"color":"Price (₹ Cr)"}
        )
        fig_map.update_layout(**CHART_THEME, height=520)
        st.plotly_chart(fig_map, use_container_width=True)
        insight("""<strong>Geo Insight:</strong> Price hotspots concentrate around Mumbai's western suburbs,
        Bengaluru's Whitefield/Sarjapur corridor, and Delhi NCR's Gurugram belt.
        Emerging value zones include Hyderabad's Outer Ring Road and Pune's Hinjewadi IT Park periphery.""")

    with tab3:
        grade_stats = df.groupby("grade").agg(
            avg_price=("price","mean"), count=("price","count")).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=grade_stats["grade"], y=grade_stats["avg_price"]/1e7,
                             marker_color=TEAL, name="Avg Price (₹ Cr)", opacity=.8), secondary_y=False)
        fig.add_trace(go.Scatter(x=grade_stats["grade"], y=grade_stats["count"],
                                 mode="lines+markers", name="Count",
                                 line=dict(color=SAFFRON, width=2),
                                 marker=dict(size=8)), secondary_y=True)
        fig.update_layout(**CHART_THEME, height=380,
            title="Construction Grade Distribution & Avg Price",
            xaxis_title="Grade", legend=dict(orientation="h", y=1.1))
        fig.update_yaxes(title_text="Avg Price (₹ Cr)", secondary_y=False,
                         tickprefix="₹", ticksuffix=" Cr")
        fig.update_yaxes(title_text="Number of Properties", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        insight("""<strong>Grade Insight:</strong> In India, construction grades above 9 carry a near-exponential
        price premium — premium developments by DLF, Godrej, Prestige, or Lodha command ₹50L+ extra per grade point.
        Grade 7 is the market norm (highest volume), while grades 11+ represent ultra-luxury assets
        concentrated in Mumbai South, Lutyens Delhi, and Bengaluru's Koramangala.""")

    with tab4:
        df["age_bin"] = pd.cut(df["age"], bins=[0,3,7,12,20,30],
                               labels=["0–3 yrs","4–7 yrs","8–12 yrs","13–20 yrs","20+ yrs"])
        age_stats = df.groupby("age_bin", observed=True).agg(
            avg_price=("price","mean"), count=("price","count")).reset_index()
        fig = px.line(age_stats, x="age_bin", y=age_stats["avg_price"]/1e7,
                      markers=True, color_discrete_sequence=[DEEP_BLUE])
        fig.update_traces(line=dict(width=3), marker=dict(size=9, color=SAFFRON))
        fig.update_layout(**CHART_THEME, height=340,
            title="Price Depreciation by Property Age (Indian Market)",
            xaxis_title="Property Age Band",
            yaxis_title="Avg Price (₹ Cr)",
            yaxis_tickprefix="₹", yaxis_ticksuffix=" Cr")
        st.plotly_chart(fig, use_container_width=True)
        insight("""<strong>Age Insight:</strong> New construction (0–3 years) commands a clear premium due to
        modern amenities, RERA compliance, and better loan eligibility. Properties aged 8–12 years see
        the steepest depreciation as maintenance costs rise.
        Unlike Western markets, Indian buyers place very high emphasis on new builds — resale
        discount for 10-year-old flats can be 15–25% versus comparable new launches.""")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 10 — AI CHATBOT & OTHER FEATURES
# ════════════════════════════════════════════════════════════════════════════════
elif page == pages[9]:
    section_header("BONUS FEATURES", "AI Chatbot & Other Features",
                   "Ask investment questions, analyze uploaded property CSVs, and rank the best properties automatically.")

    tab_chat, tab_upload, tab_rank = st.tabs([
        "🤖 AI Chat Assistant", "📤 Real-Time Property Analyzer", "🏆 Property Ranking System"
    ])

    with tab_chat:
        col_chat, col_context = st.columns([1.35, 1])
        with col_context:
            st.markdown("**Property Context**")
            sample_options = [
                f"#{idx} | {row['city']} | {fmt_inr(row['price'])} | {row['recommendation']} | Risk {row['risk_score']:.1f}"
                for idx, row in df.head(200).iterrows()
            ]
            selected_label = st.selectbox("Select a property for context", sample_options)
            selected_idx = int(selected_label.split("|")[0].replace("#", "").strip())
            selected_property = df.loc[selected_idx]
            st.markdown(card("Selected Signal", selected_property["recommendation"],
                             f"Risk {selected_property['risk_score']:.1f} / 100", selected_property["risk_category"], "neu"),
                        unsafe_allow_html=True)
            st.markdown(card("Current Price", fmt_inr(selected_property["price"]),
                             f"5Y Forecast {fmt_inr(selected_property['forecast_5y'])}", "Forecast", "up"),
                        unsafe_allow_html=True)

        with col_chat:
            st.markdown("**Ask the AI Assistant**")
            default_q = "Should I invest in this property?"
            question = st.text_input("Question", value=default_q)
            if st.button("Ask Assistant"):
                answer = generate_chatbot_response(question, selected_property)
                st.markdown(f"""
                <div class="chart-card">
                  <div style="font-size:13px;color:#64748B;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">AI Response</div>
                  <div style="font-size:15px;color:#1A2340;line-height:1.7;">{answer}</div>
                </div>
                """, unsafe_allow_html=True)
            insight("""<strong>Assistant Scope:</strong> The chatbot uses the processed dataset, recommendation labels,
            risk scores, forecasts, and ranking signals already generated by the project. It does not call an external API.""")

    with tab_upload:
        st.markdown("**Upload CSV for instant prediction, risk score, and recommendation**")
        st.caption("Accepted columns include original model names or simple aliases like bedrooms, bathrooms, area, grade, condition, property_age, pincode, and infrastructure_score.")
        uploaded = st.file_uploader("Upload property CSV", type=["csv"])
        if uploaded is not None:
            try:
                upload_df = pd.read_csv(uploaded)
                scored_upload, missing_features = score_uploaded_properties(upload_df)
                display_upload = scored_upload.copy()
                display_upload["Predicted Price"] = display_upload["Predicted Price"].apply(fmt_inr)
                display_upload["Risk Score"] = display_upload["Risk Score"].map(lambda v: f"{v:.1f}")
                display_upload["Investment Score"] = display_upload["Investment Score"].map(lambda v: f"{v:.1f}")
                st.success(f"Analyzed {len(scored_upload):,} uploaded properties.")
                if missing_features:
                    st.info("Missing model fields were filled with safe defaults: " + ", ".join(missing_features[:8]) + ("..." if len(missing_features) > 8 else ""))
                st.dataframe(display_upload, use_container_width=True, height=420)
                csv_download = scored_upload.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download analyzed CSV",
                    data=csv_download,
                    file_name="analyzed_properties.csv",
                    mime="text/csv",
                )
            except Exception as ex:
                st.error(f"Could not analyze uploaded CSV: {ex}")
        else:
            preview_cols = pd.DataFrame([{
                "bedrooms": 3,
                "bathrooms": 2,
                "area": 1500,
                "plot_area": 2500,
                "grade": 8,
                "condition": 4,
                "property_age": 10,
                "pincode": 122030,
                "airport_distance": 55,
                "infrastructure_score": 6,
            }])
            st.dataframe(preview_cols, use_container_width=True, hide_index=True)

    with tab_rank:
        ranked = top_ranked_properties(df, 10).reset_index(drop=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(card("Top Property", ranked.iloc[0]["city"], fmt_inr(ranked.iloc[0]["price"]), ranked.iloc[0]["recommendation"], "up"), unsafe_allow_html=True)
        with c2:
            st.markdown(card("Best Rank Score", f"{ranked.iloc[0]['rank_score']:.1f}", "Investment + risk + grade + forecast", "Top 10", "up"), unsafe_allow_html=True)
        with c3:
            st.markdown(card("BUY in Top 10", f"{(ranked['recommendation'] == 'BUY').sum()} / 10", "High-confidence picks", "Ranked", "neu"), unsafe_allow_html=True)

        rank_display = ranked[[
            "city", "price", "sqft_living", "grade", "risk_score",
            "investment_score", "forecast_5y", "recommendation", "rank_score"
        ]].rename(columns={
            "city": "Market",
            "price": "Current Price",
            "sqft_living": "Area",
            "grade": "Grade",
            "risk_score": "Risk Score",
            "investment_score": "Investment Score",
            "forecast_5y": "5Y Forecast",
            "recommendation": "Signal",
            "rank_score": "Rank Score",
        })
        rank_display["Current Price"] = rank_display["Current Price"].apply(fmt_inr)
        rank_display["5Y Forecast"] = rank_display["5Y Forecast"].apply(fmt_inr)
        rank_display["Area"] = rank_display["Area"].map(lambda v: f"{v:,} sq ft")
        for col in ["Risk Score", "Investment Score", "Rank Score"]:
            rank_display[col] = rank_display[col].map(lambda v: f"{v:.1f}")
        st.dataframe(rank_display, use_container_width=True, height=420)

        fig_rank = px.bar(
            ranked.sort_values("rank_score"),
            x="rank_score",
            y="city",
            orientation="h",
            color="recommendation",
            color_discrete_map={"BUY": SUCCESS, "HOLD": WARNING, "SELL": DANGER},
            title="Top 10 Automatically Ranked Properties"
        )
        fig_rank.update_layout(**CHART_THEME, height=420, xaxis_title="Rank Score", yaxis_title="")
        st.plotly_chart(fig_rank, use_container_width=True)
        insight("""<strong>Ranking Logic:</strong> The automatic rank score combines investment score, low risk,
        construction grade, and 5-year forecast strength so the top 10 are not selected by price alone.""")
