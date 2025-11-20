# ================================================================
# CUSTOMER ANALYTICS STREAMLIT APP
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import base64
import io
from fpdf import FPDF
from datetime import datetime

# ---------------------------------------------------------
# 1. Set Page Config (must be FIRST)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    layout="wide",
    page_icon="📊"
)

# ---------------------------------------------------------
# GLOBAL CSS (hover, fade-in, metric card class)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Fade-in for main content */
    .main {
        animation: fadein 0.6s;
    }
    @keyframes fadein {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    /* Metric card hover + base */
    .metric-card {
        border-radius: 12px;
        padding: 14px;
        color: white;
        box-shadow: 0 4px 10px rgba(0,0,0,0.12);
        transition: transform 0.18s ease-in-out, box-shadow 0.18s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 10px 22px rgba(0,0,0,0.18);
    }
    /* Small tweaks for dark theme */
    .stApp {
        background-color: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Helpers: load icon
# ---------------------------------------------------------
def load_icon(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

icon = load_icon("assets/icon.png")

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
header_html = f"""
    <div style="
        background: linear-gradient(to right, #0B1F3F, #006D7F, #00AFC4, #00CFEA);
        padding: 18px 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 14px;
    ">
        {f'<img src="data:image/png;base64,{icon}" style="width:55px; height:55px;">' if icon else ''}
        <div>
            <h1 style="color:white; margin:0; font-size:32px;">Customer Analytics Dashboard</h1>
            <p style="color:white; margin-top:6px; font-size:14px;">
                Powered by Python • RFM • K-Means • Cohorts • Transaction Insights
            </p>
        </div>
    </div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA DIRECTORY & CONFIG
# ---------------------------------------------------------
DATA_DIR = Path("data")

# ---------------------------------------------------------
# CACHING: Load CSVS and heavy computations
# ---------------------------------------------------------
@st.cache_data
def load_csv_cached(path):
    return pd.read_csv(path)

@st.cache_data
def compute_rfm(df, date_col):
    # df must be filtered_tx with proper date_col converted
    # frequency: unique invoiceno if exists else count
    freq_agg = ("invoiceno", "nunique") if "invoiceno" in df.columns else ("customerid", "count")
    recency_ser = df.groupby("customerid")[date_col].max().reset_index().rename(columns={date_col: "last_date"})
    max_date = df[date_col].max()
    recency_ser["recency"] = recency_ser["last_date"].apply(lambda d: (max_date - d).days)
    freq_mon = (
        df.groupby("customerid")
        .agg(frequency=(freq_agg[0], freq_agg[1]) if freq_agg else ("customerid", "count"))
    )
    # monetary: sum(quantity * unit_price) if unit_price present
    if "unit_price" in df.columns and "quantity" in df.columns:
        monetary = df.groupby("customerid").apply(lambda x: (x["unit_price"] * x["quantity"]).sum()).rename("monetary")
    else:
        monetary = df.groupby("customerid").size().rename("monetary")
    # combine
    r = recency_ser.set_index("customerid").join(freq_mon).join(monetary)
    r = r.reset_index().rename(columns={"index": "customerid"})
    # ensure columns exist
    if "frequency" not in r.columns:
        r = r.rename(columns={r.columns[1]: "frequency"})
    return r.reset_index(drop=True)

# ---------------------------------------------------------
# VALIDATION FUNCTION
# ---------------------------------------------------------
def validate_columns(df, required_cols, name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"❌ {name} is missing columns: {', '.join(missing)}")
        return False
    return True

# ---------------------------------------------------------
# METRIC CARD (uses CSS class)
# ---------------------------------------------------------
def metric_card(title, value, subtitle=None):
    subtitle_html = f'<div style="font-size:12px; opacity:0.9; margin-top:6px;">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
        <div class="metric-card" style="
            background: linear-gradient(to right, #0B1F3F, #006D7F, #00AFC4, #00CFEA);
        ">
            <div style="font-size:14px; opacity:0.9;">{title}</div>
            <div style="font-size:26px; font-weight:700; margin-top:6px;">{value}</div>
            {subtitle_html}
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# THEME TOGGLE & SESSION STATE INIT
# ---------------------------------------------------------
st.sidebar.header("Settings")
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)

if theme == "Dark":
    st.markdown(
        """
        <style>
            .stApp { background-color: #0B0F16; color: #E6EEF2; }
            .css-1d391kg, .css-1v3fvcr { color: #E6EEF2; }
        </style>
        """,
        unsafe_allow_html=True
    )

# session state defaults
if "selected_country" not in st.session_state:
    st.session_state.selected_country = "All"
if "date_range" not in st.session_state:
    st.session_state.date_range = None
if "filters_applied" not in st.session_state:
    st.session_state.filters_applied = False

# ---------------------------------------------------------
# LOAD DATA (cached) with spinner & validation
# ---------------------------------------------------------
with st.spinner("Loading datasets..."):
    try:
        final_df = load_csv_cached(DATA_DIR / "customer_segments_final.csv")
    except Exception:
        final_df = None
    try:
        tx_df = load_csv_cached(DATA_DIR / "transaction_history.csv")
    except Exception:
        tx_df = None

if final_df is None or tx_df is None:
    st.error("Missing required files in /data folder. Expected: customer_segments_final.csv and transaction_history.csv")
    st.stop()

# Normalize column names
final_df.columns = final_df.columns.str.lower().str.strip()
tx_df.columns = tx_df.columns.str.lower().str.strip()

# Expected cols
cust_col = "customerid"
seg_col = "km_segment"
country_col = "country" if "country" in tx_df.columns else None
date_col = "invoice_date" if "invoice_date" in tx_df.columns else None

# validate basic columns
if not validate_columns(final_df, [cust_col, seg_col], "Customer Segments"):
    st.stop()
tx_required = [cust_col]
if date_col:
    tx_required.append(date_col)
if "unit_price" in tx_df.columns:
    tx_required += ["unit_price", "quantity"]
if not validate_columns(tx_df, tx_required, "Transactions"):
    st.stop()

# convert date column
if date_col:
    tx_df[date_col] = pd.to_datetime(tx_df[date_col], errors="coerce")
    if tx_df[date_col].isna().all():
        st.warning("invoice_date column could not be parsed — date filters and cohorts will be disabled.")
        date_col = None

# ---------------------------------------------------------
# SIDEBAR FILTERS (with session state)
# ---------------------------------------------------------
st.sidebar.header("Filters")

# Country filter
country_list = ["All"]
if country_col:
    country_list += sorted(tx_df[country_col].dropna().unique().tolist())

st.session_state.selected_country = st.sidebar.selectbox("Country", country_list, index=country_list.index(st.session_state.selected_country) if st.session_state.selected_country in country_list else 0)

# Date filter
if date_col:
    min_date = tx_df[date_col].min().date()
    max_date = tx_df[date_col].max().date()
    dr = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    st.session_state.date_range = dr
else:
    st.session_state.date_range = None

# Apply filters button (gives visual feedback)
if st.sidebar.button("Apply Filters"):
    st.session_state.filters_applied = True
    st.success("Filters applied successfully!", icon="✅")

# Page navigation
page = st.sidebar.radio(
    "Page",
    ["Overview", "Segments", "Cohorts", "CLTV & Actions", "Customer Lookup", "Export"]
)

# ---------------------------------------------------------
# APPLY FILTERS & REBUILD RFM (cached compute)
# ---------------------------------------------------------
filtered_tx = tx_df.copy()

if st.session_state.selected_country != "All" and country_col:
    filtered_tx = filtered_tx[filtered_tx[country_col] == st.session_state.selected_country]

if st.session_state.date_range and date_col:
    start, end = pd.to_datetime(st.session_state.date_range[0]), pd.to_datetime(st.session_state.date_range[1])
    filtered_tx = filtered_tx[(filtered_tx[date_col] >= start) & (filtered_tx[date_col] <= end)]

if filtered_tx.empty:
    st.warning("No data for selected filters.")
    st.stop()

with st.spinner("Computing RFM metrics..."):
    rfm_df = compute_rfm(filtered_tx, date_col if date_col else tx_df.columns[0])

# Merge with final_df segments
filtered_final = rfm_df.merge(
    final_df[[cust_col, seg_col]],
    on=cust_col,
    how="left"
).dropna(subset=[seg_col])

# ensure recency/frequency/monetary column names exist in merged df
recency_c = "recency"
freq_c = "frequency"
mon_c = "monetary"
for c in [recency_c, freq_c, mon_c]:
    if c not in filtered_final.columns:
        filtered_final[c] = np.nan

# ---------------------------------------------------------------
# PAGE: OVERVIEW
# ---------------------------------------------------------------
if page == "Overview":
    st.title("Customer Analytics Overview")

    total_customers = int(filtered_final[cust_col].nunique())
    total_revenue = float(filtered_final[mon_c].sum(skipna=True)) if filtered_final[mon_c].notna().any() else 0.0
    total_segments = int(filtered_final[seg_col].nunique())

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Customers", f"{total_customers:,}", subtitle="Unique active customers")
    with c2:
        metric_card("Revenue", f"₹ {total_revenue:,.0f}", subtitle="Filtered period")
    with c3:
        metric_card("Active Segments", f"{total_segments}", subtitle="Defined clusters")

    st.markdown("### Revenue by Segment")
    seg_rev = (
        filtered_final.groupby(seg_col)
        .agg(customers=(cust_col, "nunique"), monetary=(mon_c, "sum"))
        .reset_index()
    )

    fig_bar = px.bar(
        seg_rev,
        x=seg_col,
        y="monetary",
        text="customers",
        hover_data={seg_col: True, "monetary": ":,.2f", "customers": True},
        title="Segment Monetary Contribution"
    )
    fig_bar.update_layout(margin=dict(t=40, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### Top 10 Customers")
    st.dataframe(filtered_final.sort_values(mon_c, ascending=False).head(10))

    # One-click CSV export for overview summary
    def get_segment_summary(df):
        return df.groupby(seg_col).agg(
            customers=(cust_col, "nunique"),
            avg_recency=(recency_c, "mean"),
            avg_frequency=(freq_c, "mean"),
            avg_monetary=(mon_c, "mean")
        ).reset_index()

    summary_df = get_segment_summary(filtered_final)
    csv_bytes = summary_df.to_csv(index=False).encode()
    st.download_button("Download Segment Summary CSV", csv_bytes, "segment_summary.csv", "text/csv")

   # PDF download (fixed)
def generate_pdf_bytes(df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Customer Segment Summary", ln=True)
    pdf.ln(4)

    for _, row in df.iterrows():
        text = " | ".join(
            f"{k}: {round(v,2) if isinstance(v,(float,int)) else v}"
            for k, v in row.to_dict().items()
        )
        pdf.multi_cell(0, 8, text)
        pdf.ln(1)

    # Correct: return PDF bytes directly
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    return pdf_bytes


pdf_bytes = generate_pdf_bytes(summary_df)

st.download_button(
    "Download Segment Summary PDF",
    pdf_bytes,
    "segment_summary.pdf",
    "application/pdf"
)


# ---------------------------------------------------------------
# PAGE: SEGMENTS
# ---------------------------------------------------------------
     elif page == "Segments":
    st.title("Segment Explorer")

    segments = ["All"] + sorted(filtered_final[seg_col].unique().tolist())
    sel_segment = st.selectbox("Select Segment", segments)

    seg_df = filtered_final if sel_segment == "All" else filtered_final[filtered_final[seg_col] == sel_segment]

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Customers", f"{seg_df[cust_col].nunique():,}")
    with c2: metric_card("Median Recency", f"{seg_df[recency_c].median():.0f}")
    with c3: metric_card("Median Frequency", f"{seg_df[freq_c].median():.0f}")
    with c4: metric_card("Median Monetary", f"{seg_df[mon_c].median():.0f}")

    st.markdown("### Distributions")
    st.plotly_chart(px.histogram(seg_df, x=recency_c, nbins=30, title="Recency Distribution"), use_container_width=True)
    st.plotly_chart(px.histogram(seg_df, x=freq_c, nbins=30, title="Frequency Distribution"), use_container_width=True)
    st.plotly_chart(px.histogram(seg_df, x=mon_c, nbins=30, title="Monetary Distribution"), use_container_width=True)

    st.markdown("### Segment Data")
    st.dataframe(seg_df)

# ---------------------------------------------------------------
# PAGE: CUSTOMER LOOKUP
# ---------------------------------------------------------------
elif page == "Customer Lookup":
    st.title("Customer Lookup")
    cid = st.text_input("Enter Customer ID")

    if cid:
        try:
            cid_val = int(cid)
            row = filtered_final[filtered_final[cust_col] == cid_val]
            if row.empty:
                st.warning("Customer not found.")
            else:
                st.subheader("Customer Summary")
                st.dataframe(row)

                c1, c2, c3, c4 = st.columns(4)
                with c1: metric_card("Recency", f"{int(row[recency_c].values[0])}")
                with c2: metric_card("Frequency", f"{int(row[freq_c].values[0])}")
                with c3: metric_card("Monetary", f"{row[mon_c].values[0]:.2f}")
                with c4: metric_card("Segment", f"{row[seg_col].values[0]}")
        except ValueError:
            st.error("Customer ID must be numeric.")

# ---------------------------------------------------------------
# PAGE: COHORTS
# ---------------------------------------------------------------
elif page == "Cohorts":
    st.title("Cohort Retention Analysis")

    if date_col is None:
        st.error("Missing invoice_date column.")
        st.stop()

    dfc = filtered_tx.copy()
    dfc["invoice_month"] = dfc[date_col].dt.to_period("M").dt.to_timestamp()
    dfc["cohort_month"] = dfc.groupby("customerid")["invoice_month"].transform("min")
    cohort = dfc.groupby(["cohort_month", "invoice_month"]).agg(customers=("customerid", "nunique")).reset_index()
    cohort["period"] = (
        (cohort["invoice_month"].dt.year - cohort["cohort_month"].dt.year) * 12 +
        (cohort["invoice_month"].dt.month - cohort["cohort_month"].dt.month)
    )
    pivot = cohort.pivot_table(index="cohort_month", columns="period", values="customers").fillna(0)
    retention = pivot.div(pivot.iloc[:, 0], axis=0)

    st.subheader("Retention Heatmap")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(retention, cmap="YlGnBu", ax=ax, annot=False)
    st.pyplot(fig)

    st.subheader("Retention Table")
    st.dataframe(retention)

# ---------------------------------------------------------------
# PAGE: CLTV & ACTIONS
# ---------------------------------------------------------------
elif page == "CLTV & Actions":
    st.title("Segment Playbook & Actions")

    seg_summary = (
        filtered_final.groupby(seg_col)
        .agg(
            customers=(cust_col, "nunique"),
            median_recency=(recency_c, "median"),
            median_frequency=(freq_c, "median"),
            median_monetary=(mon_c, "median")
        )
        .reset_index()
    )
    st.dataframe(seg_summary)

    st.markdown("### Recommended Actions")
    actions = {
        "Elite": "Exclusive perks, priority access, premium bundles.",
        "Loyal High-Value": "Referral rewards, tier upgrades, loyalty bonuses.",
        "Mass Regulars": "Cross-sell nudges, free shipping thresholds.",
        "Dormant": "Win-back emails, targeted discounts, reactivation flows."
    }

    for seg in filtered_final[seg_col].unique():
        st.markdown(f"**{seg}** — {actions.get(seg, 'No action defined.')}")

# ---------------------------------------------------------------
# PAGE: EXPORT
# ---------------------------------------------------------------
elif page == "Export":
    st.title("Export Data")

    csv_customers = filtered_final.to_csv(index=False).encode()
    st.download_button("Download Filtered Customers (CSV)", csv_customers, "export_filtered_customers.csv", "text/csv")

    csv_tx = filtered_tx.to_csv(index=False).encode()
    st.download_button("Download Filtered Transactions (CSV)", csv_tx, "export_filtered_transactions.csv", "text/csv")

    summary = filtered_final.groupby(seg_col).agg(
        customers=(cust_col, "nunique"),
        monetary=(mon_c, "sum")
    ).reset_index()
    st.download_button("Download Segmentation Summary (CSV)", summary.to_csv(index=False).encode(), "export_segment_summary.csv", "text/csv")

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown(
    """
    <div style="
        width:100%;
        margin-top:30px;
        padding:14px;
        border-radius:10px;
        background: linear-gradient(to right, #0B1F3F, #006D7F, #00AFC4, #00CFEA);
        text-align:center;
        color:white;
    ">
        <div style="margin-bottom:8px;">
            <a href="https://www.linkedin.com/in/vthenge" target="_blank" style="margin-right:20px;">
                <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="28" style="vertical-align:middle;"/>
            </a>
            <a href="https://github.com/Vikrantthenge" target="_blank">
                <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="28" style="vertical-align:middle; filter: invert(1);"/>
            </a>
        </div>
        <div style="font-size:12px;">Built by <strong>Vikrant Thenge</strong> • Customer Analytics Dashboard</div>
    </div>
    """,
    unsafe_allow_html=True
)
