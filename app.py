# ================================================================
# CUSTOMER ANALYTICS STREAMLIT APP - FINAL (with ML, Forecast, Insights)
# ================================================================
# Features:
# - Pages: Overview, Segments, Cohorts, CLTV & Actions, Customer Lookup, Export
# - Additions: CLTV model (RandomForest/Linear), Monthly revenue forecast (LinearRegression), Insights panel
# - Safe guards and fallbacks so UI doesn't crash if libraries/data missing
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
from datetime import timedelta

# ML imports with defensive fallback
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ---------------------------------------------------------
# 1. Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide", page_icon="📊")

# ---------------------------------------------------------
# Header + icon
# ---------------------------------------------------------
def load_icon(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

icon = load_icon("assets/icon.png")

st.markdown(f"""
    <div style="
        background: linear-gradient(to right, #0B1F3F, #006D7F, #00AFC4, #00CFEA);
        padding: 18px 20px;
        border-radius: 10px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
    ">
        {f'<img src="data:image/png;base64,{icon}" style="width:55px; height:55px;">' if icon else ''}
        <div>
            <h1 style="color:white; margin:0; font-size:30px;">Customer Analytics Dashboard</h1>
            <p style="color:white; margin-top:6px; font-size:13px;">
                Powered by Python • RFM • CLTV Model • Revenue Forecast • Cohorts • Transaction Insights
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Metric card helper
# ---------------------------------------------------------
def metric_card(title, value, subtitle=None):
    subtitle_html = f'<div style="font-size:12px; opacity:0.9; margin-top:6px;">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
        <div style="
            background: linear-gradient(to right, #0B1F3F, #006D7F, #00AFC4, #00CFEA);
            padding: 14px;
            border-radius: 10px;
            color: white;
            box-shadow: 0 6px 18px rgba(0,0,0,0.12);
            margin-bottom: 12px;
        ">
            <div style="font-size:13px; opacity:0.95;">{title}</div>
            <div style="font-size:22px; font-weight:700; margin-top:6px;">{value}</div>
            {subtitle_html}
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Data directory & loaders
# ---------------------------------------------------------
DATA_DIR = Path("data")

def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

# Allow optional upload from sidebar
st.sidebar.markdown("### Optional: upload CSVs")
uploaded_segments = st.sidebar.file_uploader("Upload customer_segments_final.csv", type=["csv"])
uploaded_tx = st.sidebar.file_uploader("Upload transaction_history.csv", type=["csv"])

if uploaded_segments:
    final_df = pd.read_csv(uploaded_segments)
else:
    final_df = load_csv(DATA_DIR / "customer_segments_final.csv")

if uploaded_tx:
    tx_df = pd.read_csv(uploaded_tx)
else:
    tx_df = load_csv(DATA_DIR / "transaction_history.csv")

if final_df is None or tx_df is None:
    st.error("Missing required files: customer_segments_final.csv and transaction_history.csv (upload optional).")
    st.stop()

# Normalize columns
final_df.columns = final_df.columns.str.lower().str.strip()
tx_df.columns = tx_df.columns.str.lower().str.strip()

cust_col = "customerid"
seg_col = "km_segment"
recency_c = "recency"
freq_c = "frequency"
mon_c = "monetary"

country_col = "country" if "country" in tx_df.columns else None
date_col = "invoice_date" if "invoice_date" in tx_df.columns else None

if date_col:
    tx_df[date_col] = pd.to_datetime(tx_df[date_col], errors="coerce")

# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
st.sidebar.header("Filters")
country_list = ["All"]
if country_col:
    country_list += sorted(tx_df[country_col].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Country", country_list)

if date_col:
    min_date = tx_df[date_col].min().date()
    max_date = tx_df[date_col].max().date()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
else:
    date_range = None

page = st.sidebar.radio("Page", ["Overview", "Segments", "Cohorts", "CLTV & Actions", "Customer Lookup", "Export"])

# ---------------------------------------------------------
# Apply filters and rebuild RFM (defensive)
# ---------------------------------------------------------
filtered_tx = tx_df.copy()
if selected_country != "All" and country_col:
    filtered_tx = filtered_tx[filtered_tx[country_col] == selected_country]

if date_range and date_col:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_tx = filtered_tx[(filtered_tx[date_col] >= start) & (filtered_tx[date_col] <= end)]

if filtered_tx.empty:
    st.warning("No data for selected filters.")
    st.stop()

# safe revenue column if unit_price & quantity present
if "unit_price" in filtered_tx.columns and "quantity" in filtered_tx.columns:
    filtered_tx["revenue"] = filtered_tx["unit_price"] * filtered_tx["quantity"]
else:
    # fallback: count transactions as revenue=1 (useful for demos)
    filtered_tx["revenue"] = 1.0

# Rebuild RFM-like metrics
def build_rfm(tx):
    # recency = days since last purchase (relative to max date in tx)
    max_date = tx[date_col] if date_col and date_col in tx.columns else None
    if date_col and date_col in tx.columns:
        max_date = tx[date_col].max()
        recency = tx.groupby("customerid")[date_col].max().pipe(lambda s: (max_date - s).dt.days.rename("recency"))
    else:
        recency = pd.Series(np.nan, name="recency")

    if "invoiceno" in tx.columns:
        frequency = tx.groupby("customerid")["invoiceno"].nunique().rename("frequency")
    else:
        frequency = tx.groupby("customerid").size().rename("frequency")

    monetary = tx.groupby("customerid")["revenue"].sum().rename("monetary")

    rfm = pd.concat([recency, frequency, monetary], axis=1).reset_index()
    rfm = rfm.rename(columns={rfm.columns[0]: "customerid"}) if rfm.columns[0] != "customerid" else rfm
    return rfm[["customerid", "recency", "frequency", "monetary"]]

filtered_final = build_rfm(filtered_tx).merge(final_df[[cust_col, seg_col]], on="customerid", how="left").dropna(subset=[seg_col])

# ensure numeric columns exist
for c in ["recency", "frequency", "monetary"]:
    if c not in filtered_final.columns:
        filtered_final[c] = np.nan

# helper: get segment summary
def get_segment_summary(df):
    return df.groupby(seg_col).agg(
        customers=(cust_col, "nunique"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        total_monetary=("monetary", "sum")
    ).reset_index().sort_values("total_monetary", ascending=False)

# ---------------------------------------------------------
# ML: CLTV Model (train if possible) - defensive
# - target: customer revenue in last 90 days (relative to max date)
# - features: recency, frequency, monetary (RFM)
# ---------------------------------------------------------
CLTV_MODEL = None
cltv_ready = False
cltv_note = ""

if SKLEARN_AVAILABLE:
    try:
        # compute target: revenue in last 90 days
        if date_col and date_col in filtered_tx.columns:
            max_d = filtered_tx[date_col].max()
            cutoff = max_d - pd.Timedelta(days=90)
            future_window = filtered_tx[filtered_tx[date_col] > cutoff].groupby("customerid")["revenue"].sum().rename("future_90d")
            # join to rfm
            cltv_df = filtered_final.merge(future_window.reset_index(), on="customerid", how="left").fillna(0)
            features = cltv_df[["recency", "frequency", "monetary"]].fillna(0)
            target = cltv_df["future_90d"].fillna(0)
            # only train if some positive target exists and enough rows
            if len(cltv_df) >= 20 and target.sum() > 0:
                X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
                # try RandomForest first
                try:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                except Exception:
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                CLTV_MODEL = model
                cltv_ready = True
                cltv_note = f"Trained model on {len(X_train)} samples."
            else:
                cltv_note = "Not enough data or zero future revenue to train CLTV model."
        else:
            cltv_note = "invoice_date missing; cannot compute CLTV target."
    except Exception as e:
        cltv_note = "CLTV model training failed: " + str(e)
else:
    cltv_note = "sklearn not available in environment; CLTV model disabled."

# ---------------------------------------------------------
# Forecast: simple linear regression on monthly revenue
# ---------------------------------------------------------
forecast_df = None
forecast_note = ""
forecast_horizon = 3  # months

try:
    if date_col and date_col in filtered_tx.columns:
        monthly = filtered_tx.copy()
        monthly["month"] = monthly[date_col].dt.to_period("M").dt.to_timestamp()
        monthly_rev = monthly.groupby("month")["revenue"].sum().reset_index().sort_values("month")
        if len(monthly_rev) >= 6:
            # numeric index
            monthly_rev["month_idx"] = np.arange(len(monthly_rev))
            X = monthly_rev[["month_idx"]].values
            y = monthly_rev["revenue"].values
            if SKLEARN_AVAILABLE:
                lr = LinearRegression()
                lr.fit(X, y)
                last_idx = monthly_rev["month_idx"].iloc[-1]
                future_idx = np.arange(last_idx + 1, last_idx + 1 + forecast_horizon)
                preds = lr.predict(future_idx.reshape(-1, 1))
                future_months = pd.date_range(monthly_rev["month"].iloc[-1] + pd.offsets.MonthBegin(1), periods=forecast_horizon, freq="MS")
                forecast_df = pd.DataFrame({"month": future_months, "predicted_revenue": preds})
                forecast_note = f"Linear trend forecast for next {forecast_horizon} months."
            else:
                forecast_note = "sklearn not available; forecast disabled."
        else:
            forecast_note = "Not enough monthly points (need >=6) to produce a reliable forecast."
    else:
        forecast_note = "invoice_date missing; cannot produce forecast."
except Exception as e:
    forecast_note = "Forecast failed: " + str(e)
    forecast_df = None

# ---------------------------------------------------------------
# Pages (Overview, Segments, Cohorts, CLTV & Actions, Customer Lookup, Export)
# ---------------------------------------------------------------

# -----------------------
# Overview
# -----------------------
if page == "Overview":
    st.title("Customer Analytics Overview")

    total_customers = int(filtered_final[cust_col].nunique())
    total_revenue = float(filtered_final["monetary"].sum(skipna=True)) if filtered_final["monetary"].notna().any() else 0.0
    total_segments = int(filtered_final[seg_col].nunique()) if seg_col in filtered_final.columns else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Customers", f"{total_customers:,}", subtitle="Unique active customers")
    with c2:
        metric_card("Revenue", f"₹ {total_revenue:,.0f}", subtitle="Filtered period revenue")
    with c3:
        metric_card("Active Segments", f"{total_segments}", subtitle="Segment count")

    st.markdown("### Revenue by Segment")
    seg_rev = get_segment_summary(filtered_final)[[seg_col, "total_monetary", "customers"]].rename(columns={"total_monetary": "monetary"})
    if seg_rev.empty:
        st.info("No segment revenue to display.")
    else:
        fig_bar = px.bar(seg_rev, x=seg_col, y="monetary", text="customers", hover_data={"monetary": ":,.2f"}, title="Segment Monetary Contribution")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### Top 10 Customers")
    st.dataframe(filtered_final.sort_values("monetary", ascending=False).head(10))

    # Monthly Trends (Time-Series)
    st.markdown("### Monthly Revenue & Active Users")
    if date_col and date_col in filtered_tx.columns:
        ts = filtered_tx.copy()
        ts["month"] = ts[date_col].dt.to_period("M").dt.to_timestamp()
        monthly = ts.groupby("month").agg(revenue=("revenue", "sum"), users=("customerid", "nunique")).reset_index()
        if monthly.empty:
            st.info("No monthly data.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["revenue"], mode="lines+markers", name="Revenue"))
            fig.add_trace(go.Bar(x=monthly["month"], y=monthly["users"], name="Active Users", opacity=0.6, yaxis="y2"))
            fig.update_layout(
                yaxis=dict(title="Revenue"),
                yaxis2=dict(title="Active Users", overlaying="y", side="right"),
                title="Monthly Revenue & Active Users",
                legend=dict(orientation="h"),
                margin=dict(t=40)
            )
            st.plotly_chart(fig, use_container_width=True)

        # show forecast if available
        if forecast_df is not None:
            st.markdown("### Forecast (next 3 months)")
            figf = px.line(forecast_df, x="month", y="predicted_revenue", title="Predicted Revenue (Linear Trend)")
            st.plotly_chart(figf, use_container_width=True)
        else:
            st.info(forecast_note)
    else:
        st.info("Invoice date not available — monthly trends disabled.")

    # Insights panel
    st.markdown("### Quick Insights")
    with st.expander("Auto-generated insights"):
        # Top segment
        try:
            top_seg = seg_rev.sort_values("monetary", ascending=False).iloc[0][seg_col]
            st.write(f"- **Top segment by revenue:** {top_seg}")
        except Exception:
            st.write("- Top segment: n/a")

        # Segment trend drop - compute month-on-month revenue per segment and flag largest drop
        try:
            if date_col and date_col in filtered_tx.columns:
                seg_month = filtered_tx.copy()
                seg_month["month"] = seg_month[date_col].dt.to_period("M").dt.to_timestamp()
                seg_month_rev = seg_month.groupby([seg_col, "month"])["revenue"].sum().reset_index()
                # compute last two months change
                last_month = seg_month_rev["month"].max()
                prev_month = (last_month - pd.DateOffset(months=1)).to_period("M").to_timestamp()
                last_rev = seg_month_rev[seg_month_rev["month"] == last_month].set_index(seg_col)["revenue"]
                prev_rev = seg_month_rev[seg_month_rev["month"] == prev_month].set_index(seg_col)["revenue"]
                change = (last_rev - prev_rev).dropna()
                if not change.empty:
                    worst_seg = change.idxmin()
                    st.write(f"- **Segment with largest MoM drop:** {worst_seg} ({change.min():.0f})")
                else:
                    st.write("- Segment trend: insufficient data")
            else:
                st.write("- Segment trend: invoice_date missing")
        except Exception:
            st.write("- Segment trend: error computing")

        # Highest retention cohort (if cohorts available)
        try:
            if date_col and date_col in filtered_tx.columns:
                dfc = filtered_tx.copy()
                dfc["invoice_month"] = dfc[date_col].dt.to_period("M").dt.to_timestamp()
                dfc["cohort_month"] = dfc.groupby("customerid")["invoice_month"].transform("min")
                cohort_table = dfc.groupby(["cohort_month", "invoice_month"]).agg(customers=("customerid", "nunique")).reset_index()
                cohort_table["period"] = (
                    (cohort_table["invoice_month"].dt.year - cohort_table["cohort_month"].dt.year) * 12
                    + (cohort_table["invoice_month"].dt.month - cohort_table["cohort_month"].dt.month)
                )
                pivot = cohort_table.pivot_table(index="cohort_month", columns="period", values="customers").fillna(0)
                retention = pivot.div(pivot.iloc[:,0], axis=0)
                # highest retention in period=1 (month after cohort)
                if retention.shape[1] > 1:
                    best_cohort = retention.iloc[:,1].idxmax()
                    best_val = retention.iloc[:,1].max()
                    st.write(f"- **Highest 1-month retention cohort:** {best_cohort.date()} (~{best_val:.0%})")
                else:
                    st.write("- Retention: not enough periods")
            else:
                st.write("- Retention: invoice_date missing")
        except Exception:
            st.write("- Retention: error computing")

# -----------------------
# Segments
# -----------------------
elif page == "Segments":
    st.title("Segment Explorer")

    segments = ["All"] + sorted(filtered_final[seg_col].unique().tolist())
    sel_segment = st.selectbox("Select Segment", segments)

    seg_df = filtered_final if sel_segment == "All" else filtered_final[filtered_final[seg_col] == sel_segment]

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Customers", f"{seg_df[cust_col].nunique():,}")
    with c2: metric_card("Median Recency", f"{seg_df['recency'].median():.0f}" if seg_df['recency'].notna().any() else "N/A")
    with c3: metric_card("Median Frequency", f"{seg_df['frequency'].median():.0f}" if seg_df['frequency'].notna().any() else "N/A")
    with c4: metric_card("Median Monetary", f"{seg_df['monetary'].median():.0f}" if seg_df['monetary'].notna().any() else "N/A")

    st.markdown("### Distributions")
    st.plotly_chart(px.histogram(seg_df, x="recency", nbins=30, title="Recency Distribution"), use_container_width=True)
    st.plotly_chart(px.histogram(seg_df, x="frequency", nbins=30, title="Frequency Distribution"), use_container_width=True)
    st.plotly_chart(px.histogram(seg_df, x="monetary", nbins=30, title="Monetary Distribution"), use_container_width=True)

    st.markdown("### Segment Data")
    st.dataframe(seg_df)

    st.markdown("### Recency vs Monetary (Cluster Visualization)")
    if seg_df[recency_c].notna().any() and seg_df[mon_c].notna().any():
        fig = px.scatter(seg_df, x=recency_c, y=mon_c, color=seg_col, hover_data=[cust_col, freq_c], title="Recency vs Monetary by Segment")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for scatter plot.")

# -----------------------
# Customer Lookup
# -----------------------
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
                with c1: metric_card("Recency", f"{int(row['recency'].values[0])}" if not pd.isna(row['recency'].values[0]) else "N/A")
                with c2: metric_card("Frequency", f"{int(row['frequency'].values[0])}" if not pd.isna(row['frequency'].values[0]) else "N/A")
                with c3: metric_card("Monetary", f"{row['monetary'].values[0]:.2f}" if not pd.isna(row['monetary'].values[0]) else "N/A")
                with c4: metric_card("Segment", f"{row[seg_col].values[0]}")
                # If CLTV model ready, show predicted CLTV for this customer
                if cltv_ready and CLTV_MODEL is not None:
                    cust_feat = row[["recency", "frequency", "monetary"]].fillna(0)
                    try:
                        pred = CLTV_MODEL.predict(cust_feat)[0]
                        st.write(f"**Predicted next-90-days revenue (CLTV proxy):** ₹ {pred:,.2f}")
                    except Exception:
                        st.write("CLTV prediction failed for this customer.")
                else:
                    st.write(f"CLTV model: {cltv_note}")

        except ValueError:
            st.error("Customer ID must be numeric.")

# -----------------------
# Cohorts
# -----------------------
elif page == "Cohorts":
    st.title("Cohort Retention Analysis")

    if date_col is None or date_col not in filtered_tx.columns:
        st.error("Missing invoice_date column; cohorts disabled.")
    else:
        dfc = filtered_tx.copy()
        dfc["invoice_month"] = dfc[date_col].dt.to_period("M").dt.to_timestamp()
        dfc["cohort_month"] = dfc.groupby("customerid")["invoice_month"].transform("min")

        cohort = dfc.groupby(["cohort_month", "invoice_month"]).agg(customers=("customerid", "nunique")).reset_index()
        cohort["period"] = (
            (cohort["invoice_month"].dt.year - cohort["cohort_month"].dt.year) * 12 +
            (cohort["invoice_month"].dt.month - cohort["cohort_month"].dt.month)
        )

        pivot = cohort.pivot_table(index="cohort_month", columns="period", values="customers").fillna(0)
        if pivot.empty:
            st.warning("Not enough data to build cohort matrix.")
        else:
            retention = pivot.div(pivot.iloc[:, 0], axis=0)
            st.subheader("Retention Heatmap")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(retention, cmap="YlGnBu", annot=True, fmt=".0%", ax=ax)
            st.pyplot(fig)
            st.subheader("Retention Table")
            st.dataframe(retention)

# -----------------------
# CLTV & Actions (page)
# -----------------------
elif page == "CLTV & Actions":
    st.title("CLTV & Action Playbook")

    st.markdown("### Segment Summary (RFM)")
    seg_summary = get_segment_summary(filtered_final)
    if seg_summary.empty:
        st.info("No segment summary available.")
    else:
        st.dataframe(seg_summary)

    st.markdown("### CLTV Model Status")
    st.write(cltv_note)
    if cltv_ready and CLTV_MODEL is not None:
        st.success("CLTV model is ready. You can preview predictions below.")
        # show sample predictions
        try:
            sample = filtered_final.sample(min(10, len(filtered_final))).copy()
            sample_feats = sample[["recency", "frequency", "monetary"]].fillna(0)
            sample["predicted_90d_revenue"] = CLTV_MODEL.predict(sample_feats)
            st.dataframe(sample[[cust_col, "recency", "frequency", "monetary", "predicted_90d_revenue"]])
            # CSV download of predictions
            csv_pred = sample.to_csv(index=False).encode()
            st.download_button("Download sample CLTV predictions (CSV)", csv_pred, "cltv_sample.csv", "text/csv")
        except Exception as e:
            st.write("Failed to generate CLTV preview:", e)
    else:
        st.info("CLTV model not trained. " + cltv_note)

    st.markdown("### Recommended Actions (by segment)")
    default_actions = {
        "Elite": "Exclusive perks, priority access, premium bundles.",
        "Loyal High-Value": "Referral rewards, tier upgrades, loyalty bonuses.",
        "Mass Regulars": "Cross-sell nudges, free shipping thresholds.",
        "Dormant": "Win-back emails, targeted discounts, reactivation flows."
    }
    for seg in seg_summary[seg_col].unique() if not seg_summary.empty else []:
        st.markdown(f"**{seg}** — {default_actions.get(seg, 'Define targeted campaign and test offers.')}")

# -----------------------
# Export
# -----------------------
elif page == "Export":
    st.title("Export Data & Artifacts")
    st.markdown("Download filtered datasets and summaries")

    csv_customers = filtered_final.to_csv(index=False).encode()
    st.download_button("Download Filtered Customers (CSV)", csv_customers, "export_filtered_customers.csv", "text/csv")

    csv_tx = filtered_tx.to_csv(index=False).encode()
    st.download_button("Download Filtered Transactions (CSV)", csv_tx, "export_filtered_transactions.csv", "text/csv")

    summary = get_segment_summary(filtered_final)
    st.download_button("Download Segmentation Summary (CSV)", summary.to_csv(index=False).encode(), "export_segment_summary.csv", "text/csv")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div style="
    width:100%;
    margin-top:30px;
    padding:12px;
    border-radius:10px;
    background: linear-gradient(to right, #0B1F3F, #006D7F, #00AFC4, #00CFEA);
    text-align:center;
    color:white;
">
    <a href="https://www.linkedin.com/in/vthenge" target="_blank" style="margin-right:16px;">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="24" style="vertical-align:middle;"/>
    </a>
    <a href="https://github.com/Vikrantthenge" target="_blank">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="24" style="vertical-align:middle; filter: invert(1);"/>
    </a>
    <div style="font-size:12px; margin-top:8px;">Built by <strong>Vikrant Thenge</strong> • Customer Analytics Dashboard</div>
</div>
""", unsafe_allow_html=True)
