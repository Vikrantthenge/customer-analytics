# ================================================================
# CUSTOMER ANALYTICS STREAMLIT APP - FINAL POLISHED VERSION
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import base64

# ---------------------------------------------------------
# 1. Set Page Config (must be FIRST)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    layout="wide",
    page_icon="📊"
)

# ---------------------------------------------------------
# HEADER WITH ICON (TOP)
# ---------------------------------------------------------

def load_icon(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

icon = load_icon("assets/icon.png")

st.markdown(f"""
    <div style="
        background: linear-gradient(to right, #0B1F3F, #006D7F, #00AFC4, #00CFEA);
        padding: 25px 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 18px;
    ">
        <img src="data:image/png;base64,{icon}" 
             style="width:55px; height:55px;">
        <div>
            <h1 style="color:white; margin:0; font-size:40px;">
                Customer Analytics Dashboard
            </h1>
            <p style="color:white; margin-top:5px; font-size:18px;">
                Powered by Python • PostgreSQL • RFM Segmentation • K-Means Clustering • Cohort Analysis • Transaction Insights
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA DIRECTORY
# ---------------------------------------------------------
DATA_DIR = Path("data")

# ---------------------------------------------------------
# METRIC CARD (Gradient Style)
# ---------------------------------------------------------
def metric_card(title, value, subtitle=None):
    st.markdown(f"""
        <div style="
            background: linear-gradient(to right, #0B1F3F, #006D7F, #00AFC4, #00CFEA);
            padding: 18px;
            border-radius: 12px;
            color: white;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            margin-bottom: 15px;
        ">
            <h4 style="margin:0; font-size:20px;">{title}</h4>
            <h2 style="margin:5px 0 0; font-size:32px; font-weight:bold;">{value}</h2>
            {'<p style="margin:4px 0 0; font-size:14px;">' + subtitle + '</p>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# 1. LOAD FILES
# ---------------------------------------------------------------

def load_csv(path):
    try:
        return pd.read_csv(path)
    except:
        return None

final_df = load_csv(DATA_DIR / "customer_segments_final.csv")
tx_df = load_csv(DATA_DIR / "transaction_history.csv")

if final_df is None or tx_df is None:
    st.error("Missing required files in /data folder.")
    st.stop()

# Normalize column names
final_df.columns = final_df.columns.str.lower().str.strip()
tx_df.columns = tx_df.columns.str.lower().str.strip()

# Expected columns
cust_col = "customerid"
recency_c = "recency"
freq_c = "frequency"
mon_c = "monetary"
seg_col = "km_segment"

country_col = "country" if "country" in tx_df.columns else None
date_col = "invoice_date" if "invoice_date" in tx_df.columns else None

if date_col:
    tx_df[date_col] = pd.to_datetime(tx_df[date_col], errors="coerce")

# ---------------------------------------------------------------
# 2. SIDEBAR FILTERS
# ---------------------------------------------------------------

st.sidebar.header("Filters")

# Country filter
country_list = ["All"]
if country_col:
    country_list += sorted(tx_df[country_col].dropna().unique().tolist())

selected_country = st.sidebar.selectbox("Country", country_list)

# Date filter
if date_col:
    min_date = tx_df[date_col].min().date()
    max_date = tx_df[date_col].max().date()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
else:
    date_range = None

# Page navigation
page = st.sidebar.radio(
    "Page",
    ["Overview", "Segments", "Cohorts", "CLTV & Actions", "Customer Lookup", "Export"]
)

# ---------------------------------------------------------------
# 3. APPLY FILTERS TO TRANSACTIONS & REBUILD RFM
# ---------------------------------------------------------------

filtered_tx = tx_df.copy()

if selected_country != "All" and country_col:
    filtered_tx = filtered_tx[filtered_tx[country_col] == selected_country]

if date_range and date_col:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_tx = filtered_tx[(filtered_tx[date_col] >= start) & (filtered_tx[date_col] <= end)]

if filtered_tx.empty:
    st.warning("No data for selected filters.")
    st.stop()

# Rebuild RFM-like metrics
filtered_customers = (
    filtered_tx.groupby("customerid")
    .agg(
        recency=("invoice_date", lambda x: (filtered_tx[date_col].max() - x.max()).days),
        frequency=("invoiceno", "nunique") if "invoiceno" in filtered_tx.columns else ("customerid", "count"),
        monetary=("unit_price", lambda x: (filtered_tx.loc[x.index, "quantity"] * x).sum())
    )
    .reset_index()
)

filtered_final = filtered_customers.merge(
    final_df[[cust_col, seg_col]],
    on="customerid",
    how="left"
).dropna(subset=[seg_col])

# ---------------------------------------------------------------
# PAGE 1: OVERVIEW
# ---------------------------------------------------------------

if page == "Overview":
    st.title("Customer Analytics Overview")

    total_customers = filtered_final[cust_col].nunique()
    total_revenue = filtered_final[mon_c].sum()
    total_segments = filtered_final[seg_col].nunique()

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Customers", f"{total_customers:,}")
    with c2: metric_card("Revenue", f"{total_revenue:,.0f}")
    with c3: metric_card("Active Segments", total_segments)

    st.markdown("### Revenue by Segment")
    seg_rev = (
        filtered_final.groupby(seg_col)
        .agg(customers=(cust_col, "nunique"), monetary=(mon_c, "sum"))
        .reset_index()
    )
    st.plotly_chart(
        px.bar(seg_rev, x=seg_col, y="monetary", text="customers", title="Segment Monetary Contribution"),
        use_container_width=True
    )

    st.markdown("### Top 10 Customers")
    st.dataframe(filtered_final.sort_values(mon_c, ascending=False).head(10))

# ---------------------------------------------------------------
# PAGE 2: SEGMENTS
# ---------------------------------------------------------------

elif page == "Segments":
    st.title("Segment Explorer")

    segments = ["All"] + sorted(filtered_final[seg_col].unique().tolist())
    sel_segment = st.selectbox("Select Segment", segments)

    seg_df = filtered_final if sel_segment == "All" else filtered_final[filtered_final[seg_col] == sel_segment]

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Customers", seg_df[cust_col].nunique())
    with c2: metric_card("Median Recency", seg_df[recency_c].median())
    with c3: metric_card("Median Frequency", seg_df[freq_c].median())
    with c4: metric_card("Median Monetary", seg_df[mon_c].median())

    st.markdown("### Distributions")
    st.plotly_chart(px.histogram(seg_df, x=recency_c), use_container_width=True)
    st.plotly_chart(px.histogram(seg_df, x=freq_c), use_container_width=True)
    st.plotly_chart(px.histogram(seg_df, x=mon_c), use_container_width=True)

    st.markdown("### Segment Data")
    st.dataframe(seg_df)

# ---------------------------------------------------------------
# PAGE 3: CUSTOMER LOOKUP
# ---------------------------------------------------------------

elif page == "Customer Lookup":
    st.title("Customer Lookup")
    cid = st.text_input("Enter Customer ID")

    if cid:
        try:
            cid = int(cid)
        except:
            st.error("Customer ID must be numeric.")
            st.stop()

        row = filtered_final[filtered_final[cust_col] == cid]

        if row.empty:
            st.warning("Customer not found.")
        else:
            st.subheader("Customer Summary")
            st.dataframe(row)

            c1, c2, c3, c4 = st.columns(4)
            with c1: metric_card("Recency", row[recency_c].values[0])
            with c2: metric_card("Frequency", row[freq_c].values[0])
            with c3: metric_card("Monetary", row[mon_c].values[0])
            with c4: metric_card("Segment", row[seg_col].values[0])

# ---------------------------------------------------------------
# PAGE 4: COHORTS
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
    sns.heatmap(retention, cmap="YlGnBu", annot=False)
    st.pyplot(fig)

    st.subheader("Retention Table")
    st.dataframe(retention)

# ---------------------------------------------------------------
# PAGE 5: CLTV & ACTIONS
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
# PAGE 6: EXPORT
# ---------------------------------------------------------------

elif page == "Export":
    st.title("Export Data")

    if st.button("Export Filtered Customers"):
        out = DATA_DIR / "export_filtered_customers.csv"
        filtered_final.to_csv(out, index=False)
        st.success(f"Saved: {out}")

    if st.button("Export Filtered Transactions"):
        out = DATA_DIR / "export_filtered_transactions.csv"
        filtered_tx.to_csv(out, index=False)
        st.success(f"Saved: {out}")

    if st.button("Export Segmentation Summary"):
        summary = filtered_final.groupby(seg_col).agg(
            customers=(cust_col, "nunique"),
            monetary=(mon_c, "sum")
        ).reset_index()
        out = DATA_DIR / "export_segment_summary.csv"
        summary.to_csv(out, index=False)
        st.success(f"Saved: {out}")


# ---------------------------------------------------------------

st.markdown(
    """
    <div style="
        width:100%;
        margin-top:40px;
        padding:20px;
        border-radius:12px;
        background: linear-gradient(to right, #0B1F3F, #006D7F, #00AFC4, #00CFEA);
        text-align:center;
        color:white;
    ">

        <!-- Social Links with Emojis -->
        <div style="margin-bottom: 8px;">
            <a href="https://www.linkedin.com/in/vthenge" 
               target="_blank" 
               style="margin-right: 25px; text-decoration: none; font-size: 22px; color:white;">
               🔗 LinkedIn
            </a>

            <a href="https://github.com/Vikrantthenge" 
               target="_blank" 
               style="text-decoration: none; font-size: 22px; color:white;">
               💻 GitHub
            </a>
        </div>

        <!-- Footer Text -->
        <div style="font-size: 14px; margin-top: 10px;">
            Built by <strong>Vikrant Thenge</strong> • Customer Analytics Dashboard
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

