# ================================================================
# CUSTOMER ANALYTICS STREAMLIT APP 
# ================================================================
# Single-file Streamlit app:
# - Pages: Overview, Segments, Cohorts, CLTV & Actions, Customer Lookup, Export
# - Features: caching, spinners, validation, session state, theme toggle
# - UI polish: CSS (hover, fade-in), metric cards, expanders, tooltips
# - Analytics: robust RFM calculation, monthly trends, segment scatterplot
# - Export: CSV + PDF 
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import base64
import io
from datetime import datetime
from fpdf import FPDF

# ---------------------------------------------------------
# 0. Page config (must be first)
# ---------------------------------------------------------
st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide", page_icon="📊")

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
    /* Metric card base + hover */
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
    /* small tweaks */
    .st-download-link > a { color: white; }
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
# Config: data directory
# ---------------------------------------------------------
DATA_DIR = Path("data")

# ---------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------
@st.cache_data
def load_csv_cached(path):
    return pd.read_csv(path)

@st.cache_data
def compute_rfm_cached(tx, date_col, has_invoiceno, has_unit_price_qty):
    # tx: filtered transactions DataFrame
    # returns rfm df with columns: customerid, recency, frequency, monetary
    if tx.empty:
        return pd.DataFrame(columns=["customerid", "recency", "frequency", "monetary"])

    # frequency
    if has_invoiceno:
        freq = tx.groupby("customerid")["invoiceno"].nunique().rename("frequency")
    else:
        freq = tx.groupby("customerid").size().rename("frequency")

    # monetary
    if has_unit_price_qty:
        monetary = (tx.groupby("customerid").apply(lambda x: (x["unit_price"] * x["quantity"]).sum()).rename("monetary"))
    else:
        monetary = tx.groupby("customerid").size().rename("monetary")

    # recency
    if date_col:
        max_date = tx[date_col].max()
        last_purchase = tx.groupby("customerid")[date_col].max().rename("last_purchase")
        recency = (max_date - last_purchase).dt.days.rename("recency")
    else:
        # if no date, fill with NaN
        recency = pd.Series(np.nan, index=freq.index, name="recency")

    rfm = pd.concat([recency, freq, monetary], axis=1).reset_index()
    rfm = rfm.rename(columns={"index":"customerid"}) if "customerid" not in rfm.columns else rfm
    # ensure proper columns
    rfm = rfm.rename(columns={rfm.columns[0]: "customerid"}) if rfm.columns[0] != "customerid" else rfm
    return rfm[["customerid", "recency", "frequency", "monetary"]]

# ---------------------------------------------------------
# Validation helper
# ---------------------------------------------------------
def validate_columns(df, required_cols, name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"❌ {name} missing columns: {', '.join(missing)}")
        return False
    return True

# ---------------------------------------------------------
# Metric card renderer
# ---------------------------------------------------------
def metric_card(title, value, subtitle=None):
    subtitle_html = f'<div style="font-size:12px; opacity:0.9; margin-top:6px;">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
        <div class="metric-card" style="
            background: linear-gradient(to right, #0B1F3F, #006D7F, #00AFC4, #00CFEA);
        ">
            <div style="font-size:14px; opacity:0.95;">{title}</div>
            <div style="font-size:26px; font-weight:700; margin-top:6px;">{value}</div>
            {subtitle_html}
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: theme, session state init
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

# session defaults
if "selected_country" not in st.session_state:
    st.session_state.selected_country = "All"
if "date_range" not in st.session_state:
    st.session_state.date_range = None
if "filters_applied" not in st.session_state:
    st.session_state.filters_applied = False

# ---------------------------------------------------------
# Load datasets with spinner + validation
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
    st.error("Missing datasets in /data: expected customer_segments_final.csv and transaction_history.csv")
    st.stop()

# normalize columns
final_df.columns = final_df.columns.str.lower().str.strip()
tx_df.columns = tx_df.columns.str.lower().str.strip()

# expected columns - keep flexible
cust_col = "customerid"
seg_col = "km_segment"
country_col = "country" if "country" in tx_df.columns else None
date_col = "invoice_date" if "invoice_date" in tx_df.columns else None

# Basic validation
if not validate_columns(final_df, [cust_col, seg_col], "Customer Segments"):
    st.stop()

tx_required = [cust_col]
if date_col:
    tx_required.append(date_col)
# unit_price & quantity optional - monetary calc will use them if present
has_invoiceno = "invoiceno" in tx_df.columns
has_unit_price_qty = ("unit_price" in tx_df.columns) and ("quantity" in tx_df.columns)

if not validate_columns(tx_df, tx_required, "Transactions"):
    st.stop()

# parse dates safely
if date_col:
    tx_df[date_col] = pd.to_datetime(tx_df[date_col], errors="coerce")
    if tx_df[date_col].isna().all():
        st.warning("invoice_date could not be parsed. Date-based features will be disabled.")
        date_col = None

# ---------------------------------------------------------
# Sidebar: Filters
# ---------------------------------------------------------
st.sidebar.header("Filters")

# Country selector
country_list = ["All"]
if country_col:
    country_list += sorted(tx_df[country_col].dropna().unique().tolist())
st.session_state.selected_country = st.sidebar.selectbox("Country", country_list, index=country_list.index(st.session_state.selected_country) if st.session_state.selected_country in country_list else 0)

# Date range
if date_col:
    min_date = tx_df[date_col].min().date()
    max_date = tx_df[date_col].max().date()
    dr = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    st.session_state.date_range = dr
else:
    st.session_state.date_range = None

# Apply filters button
if st.sidebar.button("Apply Filters"):
    st.session_state.filters_applied = True
    st.success("Filters applied", icon="✅")

# Page nav
page = st.sidebar.radio("Page", ["Overview", "Segments", "Cohorts", "CLTV & Actions", "Customer Lookup", "Export"])

# ---------------------------------------------------------
# Apply filters to transactions
# ---------------------------------------------------------
filtered_tx = tx_df.copy()

if country_col and st.session_state.selected_country != "All":
    filtered_tx = filtered_tx[filtered_tx[country_col] == st.session_state.selected_country]

if st.session_state.date_range and date_col:
    start, end = pd.to_datetime(st.session_state.date_range[0]), pd.to_datetime(st.session_state.date_range[1])
    filtered_tx = filtered_tx[(filtered_tx[date_col] >= start) & (filtered_tx[date_col] <= end)]

if filtered_tx.empty:
    st.warning("No data after applying filters.")
    st.stop()

# ---------------------------------------------------------
# Compute RFM (cached)
# ---------------------------------------------------------
with st.spinner("Computing RFM metrics..."):
    rfm_df = compute_rfm_cached(filtered_tx, date_col, has_invoiceno, has_unit_price_qty)

# Merge with segments
filtered_final = rfm_df.merge(final_df[[cust_col, seg_col]], on=cust_col, how="left").dropna(subset=[seg_col])

# Ensure RFM col names exist
for col in ["recency", "frequency", "monetary"]:
    if col not in filtered_final.columns:
        filtered_final[col] = np.nan

# ---------------------------------------------------------
# Utility: segment summary and exporters (used in Overview)
# ---------------------------------------------------------
def get_segment_summary(df):
    return df.groupby(seg_col).agg(
        customers=(cust_col, "nunique"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean")
    ).reset_index().sort_values("avg_monetary", ascending=False)

def generate_pdf_bytes(df, title="Customer Segment Summary"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, title, ln=True)
    pdf.ln(4)
    # header row
    headers = list(df.columns)
    header_line = " | ".join(headers)
    pdf.set_font("Arial", style="B", size=11)
    pdf.multi_cell(0, 7, header_line)
    pdf.set_font("Arial", size=11)
    pdf.ln(2)
    for _, row in df.iterrows():
        values = [str(round(v,2)) if isinstance(v, (float, int, np.floating, np.integer)) else str(v) for v in row.tolist()]
        line = " | ".join(values)
        pdf.multi_cell(0, 7, line)
    # return bytes
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    return pdf_bytes

# ---------------------------------------------------------
# PAGE: OVERVIEW
# ---------------------------------------------------------
if page == "Overview":
    st.title("Customer Analytics Overview")

    total_customers = int(filtered_final[cust_col].nunique())
    total_revenue = float(filtered_final["monetary"].sum(skipna=True)) if filtered_final["monetary"].notna().any() else 0.0
    total_segments = int(filtered_final[seg_col].nunique())

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Customers", f"{total_customers:,}", subtitle="Unique active customers")
    with c2:
        metric_card("Revenue", f"₹ {total_revenue:,.0f}", subtitle="Filtered period revenue")
    with c3:
        metric_card("Active Segments", f"{total_segments}", subtitle="Segment count")

    st.markdown("### Revenue by Segment")
    seg_rev = filtered_final.groupby(seg_col).agg(customers=(cust_col,"nunique"), monetary=("monetary","sum")).reset_index()
    if seg_rev.empty:
        st.info("No segment revenue to display.")
    else:
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

    st.markdown("### Monthly Trends")
    # Monthly revenue and new customers if date exists
    if date_col:
        monthly = filtered_tx.copy()
        monthly["month"] = monthly[date_col].dt.to_period("M").dt.to_timestamp()
        monthly_rev = monthly.groupby("month").apply(lambda x: (x["unit_price"] * x["quantity"]).sum() if has_unit_price_qty else x.shape[0]).rename("revenue").reset_index()
        monthly_customers = monthly.groupby("month")["customerid"].nunique().rename("active_customers").reset_index()
        mdf = monthly_rev.merge(monthly_customers, on="month", how="left")
        if not mdf.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mdf["month"], y=mdf["revenue"], mode="lines+markers", name="Revenue"))
            fig.add_trace(go.Bar(x=mdf["month"], y=mdf["active_customers"], name="Active Customers", yaxis="y2", opacity=0.6))
            # second y-axis
            fig.update_layout(
                yaxis=dict(title="Revenue"),
                yaxis2=dict(title="Active Customers", overlaying="y", side="right"),
                legend=dict(orientation="h"),
                title="Monthly Revenue & Active Customers",
                margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No monthly data available.")
    else:
        st.info("Invoice date not available — monthly trends disabled.")

    st.markdown("### Top 10 Customers")
    st.dataframe(filtered_final.sort_values("monetary", ascending=False).head(10))

    # Segment summary + exporters
    summary_df = get_segment_summary(filtered_final)
    csv_bytes = summary_df.to_csv(index=False).encode()
    st.download_button("Download Segment Summary CSV", csv_bytes, "segment_summary.csv", "text/csv")
    pdf_bytes = generate_pdf_bytes(summary_df)
    st.download_button("Download Segment Summary PDF", pdf_bytes, "segment_summary.pdf", "application/pdf")

    with st.expander("Insights (static)"):
        st.write("- Elite segments often contribute disproportionate revenue.")
        st.write("- Consider win-back campaigns for Dormant segments.")
        st.write("- Frequency and monetary value are positively correlated for top segments.")

# ---------------------------------------------------------
# PAGE: SEGMENTS
# ---------------------------------------------------------
elif page == "Segments":
    st.title("Segment Explorer")

    segments = ["All"] + sorted(filtered_final[seg_col].unique().tolist()) if not filtered_final.empty else ["All"]
    sel_segment = st.selectbox("Select Segment", segments, index=0)

    seg_df = filtered_final if sel_segment == "All" else filtered_final[filtered_final[seg_col] == sel_segment]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Customers", f"{seg_df[cust_col].nunique():,}")
    with c2:
        metric_card("Median Recency", f"{seg_df['recency'].median():.0f}" if seg_df['recency'].notna().any() else "N/A")
    with c3:
        metric_card("Median Frequency", f"{seg_df['frequency'].median():.0f}" if seg_df['frequency'].notna().any() else "N/A")
    with c4:
        metric_card("Median Monetary", f"{seg_df['monetary'].median():.0f}" if seg_df['monetary'].notna().any() else "N/A")

    st.markdown("### Distributions")
    st.plotly_chart(px.histogram(seg_df, x="recency", nbins=30, title="Recency Distribution"), use_container_width=True)
    st.plotly_chart(px.histogram(seg_df, x="frequency", nbins=30, title="Frequency Distribution"), use_container_width=True)
    st.plotly_chart(px.histogram(seg_df, x="monetary", nbins=30, title="Monetary Distribution"), use_container_width=True)

    st.markdown("### Segment Data")
    st.dataframe(seg_df)

    # Cluster scatter: Recency vs Monetary colored by segment
    st.markdown("### Segment Scatter (Recency vs Monetary)")
    if seg_df.empty:
        st.info("No segment data for scatter plot.")
    else:
        # choose axes safe
        if seg_df["recency"].notna().any() and seg_df["monetary"].notna().any():
            fig = px.scatter(
                seg_df,
                x="recency",
                y="monetary",
                color=seg_col,
                hover_data=[cust_col, "frequency"],
                title="Recency vs Monetary by Segment"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Recency or Monetary not available for scatter plot.")

# ---------------------------------------------------------
# PAGE: CUSTOMER LOOKUP
# ---------------------------------------------------------
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
                with c1:
                    metric_card("Recency", f"{int(row['recency'].values[0])}" if not pd.isna(row['recency'].values[0]) else "N/A")
                with c2:
                    metric_card("Frequency", f"{int(row['frequency'].values[0])}" if not pd.isna(row['frequency'].values[0]) else "N/A")
                with c3:
                    metric_card("Monetary", f"{row['monetary'].values[0]:.2f}" if not pd.isna(row['monetary'].values[0]) else "N/A")
                with c4:
                    metric_card("Segment", f"{row[seg_col].values[0]}")
        except ValueError:
            st.error("Customer ID must be numeric.")

# ---------------------------------------------------------
# PAGE: COHORTS
# ---------------------------------------------------------
elif page == "Cohorts":
    st.title("Cohort Retention Analysis")

    # ---- Check invoice_date column ----
    if "invoice_date" not in filtered_tx.columns:
        st.error("invoice_date column not found. Cohort analysis cannot run.")
        st.stop()

    dfc = filtered_tx.copy()

    # Convert date safely
    dfc["invoice_date"] = pd.to_datetime(dfc["invoice_date"], errors="coerce")

    if dfc["invoice_date"].isna().all():
        st.error("invoice_date has no valid entries. Cohort analysis cannot run.")
        st.stop()

    # ---- Build cohort ----
    dfc["invoice_month"] = dfc["invoice_date"].dt.to_period("M").dt.to_timestamp()

    dfc["cohort_month"] = (
        dfc.groupby("customerid")["invoice_month"].transform("min")
    )

    cohort = (
        dfc.groupby(["cohort_month", "invoice_month"])
        .agg(customers=("customerid", "nunique"))
        .reset_index()
    )

    cohort["period"] = (
        (cohort["invoice_month"].dt.year - cohort["cohort_month"].dt.year) * 12 +
        (cohort["invoice_month"].dt.month - cohort["cohort_month"].dt.month)
    )

    pivot = cohort.pivot_table(
        index="cohort_month",
        columns="period",
        values="customers"
    ).fillna(0)

    retention = pivot.div(pivot.iloc[:, 0], axis=0)

    # ---- Visuals ----
    st.subheader("Retention Heatmap")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(retention, cmap="YlGnBu", annot=True, fmt=".0%", ax=ax)
    st.pyplot(fig)

    st.subheader("Retention Table")
    st.dataframe(retention)


# Pivot into a retention matrix
pivot = cohort.pivot_table(
    index="cohort_month",
    columns="period",
    values="customers"
).fillna(0)

# Retention percentages
retention = pivot.div(pivot.iloc[:, 0], axis=0)
