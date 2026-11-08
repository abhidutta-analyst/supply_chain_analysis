"""
Supply Chain Delivery Performance Dashboard
============================================
Interactive Streamlit dashboard built on top of the Supply_Chain_Analysis.ipynb
workflow and the DataCo Supply Chain dataset (180,519 raw rows -> 172,765
cleaned order-item records, Jan 2015 - Feb 2018).

Page structure (sidebar navigation):
    1. Overview         - headline KPIs, delay distribution, profitability mix
    2. Delivery Performance - delay rates over time and across dimensions
    3. Root Cause Analysis  - shipping-mode miscalibration, region drill-down,
                              time-of-order seasonality
    4. Prediction Model     - metrics, confusion matrix, feature importance,
                              interactive late/on-time prediction form
    5. Data Explorer        - filterable, searchable table with CSV download

Preprocessing, feature engineering, encoding, train/test split, SMOTE and the
Random Forest classifier replicate the notebook exactly (random_state=42,
80/20 stratified split). The trained pipeline is persisted with joblib so the
model trains only on the very first launch.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Page config & shared styling constants
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Supply Chain Delivery Performance",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette - validated categorical slots (light mode), status colors, and ink
# tokens from the modern professional dashboard design system.
CAT = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#db2777",
       "#0891b2", "#4f46e5", "#dc2626"]
PRIMARY = CAT[0]          # Royal blue - single-series bars / lines
SECONDARY = CAT[1]        # Violet
STATUS_GOOD = "#059669"   # Emerald
STATUS_CRITICAL = "#dc2626" # Vibrant red
INK = "#0f172a"           # Slate 900
INK_2 = "#475569"         # Slate 600
INK_MUTED = "#64748b"     # Slate 500
GRID = "#e2e8f0"          # Slate 200

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

# Raw CSV column names (kept verbatim - they contain spaces/parentheses)
COL_TYPE = "Type"
COL_REAL = "Days for shipping (real)"
COL_SCHED = "Days for shipment (scheduled)"
COL_SALES = "Sales per customer"
COL_DELIVERY_STATUS = "Delivery Status"
COL_LATE_RISK = "Late_delivery_risk"
COL_CATEGORY = "Category Name"
COL_SEGMENT = "Customer Segment"
COL_DEPARTMENT = "Department Name"
COL_REGION = "Order Region"
COL_STATUS = "Order Status"
COL_ORDER_DATE = "order date (DateOrders)"
COL_SHIP_DATE = "shipping date (DateOrders)"
COL_SHIPPING_MODE = "Shipping Mode"
COL_PROFIT = "Order Profit Per Order"
COL_PRODUCT = "Product Name"
COL_PRICE = "Product Price"

# Engineered columns
COL_PROC = "Order Processing Time"
COL_DELAY = "Delay"
COL_IS_DELAYED = "Is_Delayed"
COL_MONTH = "order_month"
COL_DAY = "order_day"
COL_HOUR = "order_hour"
COL_PROFIT_FLAG = "Profitability Flag"

# Model feature list - identical to the notebook (cell 28)
MODEL_FEATURES = [COL_TYPE, COL_SCHED, COL_CATEGORY, COL_SEGMENT,
                  COL_DEPARTMENT, COL_REGION, COL_SHIPPING_MODE,
                  COL_MONTH, COL_HOUR]

MODEL_PATH = Path(__file__).parent / "rf_late_delivery_model.joblib"
DATA_PATH = Path(__file__).parent / "DataCoSupplyChainDataset.csv"


# ---------------------------------------------------------------------------
# Data loading & preprocessing (mirrors the notebook exactly)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading and preparing dataset…")
def load_data() -> pd.DataFrame:
    """Load the raw CSV and apply the notebook's cleaning + feature engineering."""
    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    # Notebook cleaning: drop identifier/PII/redundant columns
    columns_to_drop = [
        "Product Description", "Product Image", "Customer Email",
        "Customer Password", "Customer Fname", "Customer Lname",
        "Customer Street", "Customer Zipcode", "Order Zipcode",
        "Longitude", "Latitude", "Order Item Cardprod Id", "Order Item Id",
        "Order Item Discount", "Order Item Discount Rate",
        "Order Item Product Price", "Order Item Quantity", "Order Item Total",
        "Category Id", "Department Id", "Order Id", "Order Customer Id",
        "Customer Id", "Product Card Id", "Product Category Id",
        "Benefit per order", "Product Status", "Customer City", "Order City",
        "Order Country", "Order State", "Customer State", "Market",
    ]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    # Cancelled shipments never enter the delivery pipeline
    df = df[df[COL_DELIVERY_STATUS] != "Shipping canceled"]

    # Date parsing
    for c in [COL_ORDER_DATE, COL_SHIP_DATE]:
        df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=False)

    # Feature engineering (notebook cell 10 & 12)
    df[COL_PROC] = (df[COL_SHIP_DATE] - df[COL_ORDER_DATE]).dt.days
    df[COL_DELAY] = df[COL_PROC] - df[COL_SCHED]
    df[COL_IS_DELAYED] = df[COL_DELAY] > 0
    df[COL_MONTH] = df[COL_ORDER_DATE].dt.month
    df[COL_DAY] = df[COL_ORDER_DATE].dt.day_name()
    df[COL_HOUR] = df[COL_ORDER_DATE].dt.hour
    df[COL_PROFIT_FLAG] = np.where(df[COL_PROFIT] > 0, "Profit",
                                   np.where(df[COL_PROFIT] < 0, "Loss",
                                            "Break-even"))
    return df


# ---------------------------------------------------------------------------
# Model training / loading (mirrors the notebook exactly)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training model (first launch only)…")
def get_model_and_test_data():
    """Reproduce the notebook's ML pipeline, persisting the model with joblib.

    Steps, in the notebook's order:
      1. Select the 9 model features + target (Late_delivery_risk).
      2. Frequency-encode the 6 categorical features (drop raw strings).
      3. 80/20 stratified train/test split, random_state=42.
      4. SMOTE oversampling on the training set only (random_state=42).
      5. RandomForestClassifier(random_state=42) fit on balanced data.

    Returns model, X_test, y_test, and the per-column frequency maps so the
    interactive prediction form can encode raw user input identically.
    """
    df = load_data()

    # 1. Features & target
    X_raw = df[MODEL_FEATURES].copy()
    y = df[COL_LATE_RISK]

    # 2. Frequency encoding - construct X in the exact order of MODEL_FEATURES
    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    freq_maps: dict[str, pd.Series] = {}
    encoded_cols = {}
    for col in MODEL_FEATURES:
        if col in cat_cols:
            freq = X_raw[col].value_counts(normalize=True)
            encoded_cols[f"{col}_freq"] = X_raw[col].map(freq)
            freq_maps[col] = freq
        else:
            encoded_cols[col] = X_raw[col]
    X = pd.DataFrame(encoded_cols)

    # 3. Train/test split - identical seed & stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Notebook drops NaN training rows before SMOTE (test set is untouched)
    non_nan_indices = X_train.dropna().index
    X_train_clean = X_train.loc[non_nan_indices]
    y_train_clean = y_train.loc[non_nan_indices]

    # 4. SMOTE on training data only
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_clean, y_train_clean)

    # 5. Random Forest - reuse the saved model when present and features match
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        if hasattr(model, "feature_names_in_") and list(model.feature_names_in_) != list(X.columns):
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train_bal, y_train_bal)
            joblib.dump(model, MODEL_PATH)
    else:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_bal, y_train_bal)
        joblib.dump(model, MODEL_PATH)

    # Feature importance straight from the fitted forest
    importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance")

    return model, X_test, y_test, importances, freq_maps


def encode_input(raw: dict, freq_maps: dict[str, pd.Series]) -> pd.DataFrame:
    """Frequency-encode one raw observation dict exactly like training."""
    row = {}
    for col in MODEL_FEATURES:
        if col in freq_maps:                      # categorical -> frequency
            value = raw[col]
            row[f"{col}_freq"] = float(freq_maps[col].get(value, 0.0))
        else:                                     # already numeric
            row[col] = raw[col]
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Shared chart helpers (thin marks, hairline grids, one axis per chart)
# ---------------------------------------------------------------------------

def style_axes(fig: go.Figure, y_title: str | None = None,
               x_title: str | None = None,
               showlegend: bool | None = None) -> go.Figure:
    """Apply the shared recessive chrome: hairline grid, muted ink, no clutter."""
    layout_kwargs = dict(
        font=dict(color=INK_2, size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        hoverlabel=dict(font_size=12),
    )
    # Legend presence: always for >= 2 series (set explicitly); hidden by
    # default for single-series charts.
    if showlegend is not None:
        layout_kwargs["showlegend"] = showlegend
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(
        gridcolor=GRID, gridwidth=1, zeroline=False,
        linecolor=GRID, ticks="outside", tickcolor=GRID,
        tickfont=dict(color=INK_MUTED, size=11),
        title_font=dict(color=INK_2, size=12),
    )
    fig.update_yaxes(
        gridcolor=GRID, gridwidth=1, zeroline=False,
        linecolor=GRID, ticks="outside", tickcolor=GRID,
        tickfont=dict(color=INK_MUTED, size=11),
        title_font=dict(color=INK_2, size=12),
    )
    if x_title:
        fig.update_xaxes(title_text=x_title)
    if y_title:
        fig.update_yaxes(title_text=y_title)
    return fig


def fmt_money(v: float) -> str:
    """Notebook's compact money format (e.g. 7.5M $)."""
    if abs(v) >= 1e6:
        return f"{v / 1e6:.1f}M $"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.1f}K $"
    return f"{v:.0f} $"


def kpi_card(label: str, value: str, sub: str | None = None,
             color: str | None = None) -> None:
    """Lightweight stat tile rendered as scoped HTML inside an st.container."""
    sub_html = (f'<div style="font-size:12px;color:{INK_MUTED};margin-top:4px">'
                f"{sub}</div>") if sub else ""
    value_style = f"color:{color};" if color else f"color:{INK};"
    st.markdown(
        f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;
                    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.04);
                    border-radius:12px;padding:18px 22px;height:100%">
          <div style="font-size:11px;font-weight:600;color:{INK_2};text-transform:uppercase;
                      letter-spacing:0.8px">{label}</div>
          <div style="font-size:28px;font-weight:700;{value_style}
                      line-height:1.2;margin-top:6px">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar: navigation + global filters (one row, scoping every page)
# ---------------------------------------------------------------------------

df = load_data()

st.sidebar.header("Supply Chain Delivery Performance")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Delivery Performance", "Root Cause Analysis",
     "Prediction Model", "Data Explorer"],
    label_visibility="collapsed",
)
st.sidebar.divider()

# Global filters - every page renders from this same slice.
st.sidebar.subheader("Filters")

date_range = st.sidebar.date_input(
    "Order date range",
    value=(df[COL_ORDER_DATE].min().date(), df[COL_ORDER_DATE].max().date()),
    min_value=df[COL_ORDER_DATE].min().date(),
    max_value=df[COL_ORDER_DATE].max().date(),
)

region_opts = ["All"] + sorted(df[COL_REGION].dropna().unique().tolist())
f_region = st.sidebar.selectbox("Order Region", region_opts)

mode_opts = ["All"] + sorted(df[COL_SHIPPING_MODE].dropna().unique().tolist())
f_mode = st.sidebar.selectbox("Shipping Mode", mode_opts)

segment_opts = ["All"] + sorted(df[COL_SEGMENT].dropna().unique().tolist())
f_segment = st.sidebar.selectbox("Customer Segment", segment_opts)

status_opts = ["All"] + sorted(df[COL_STATUS].dropna().unique().tolist())
f_status = st.sidebar.selectbox("Order Status", status_opts)

f_late_only = st.sidebar.checkbox("Late deliveries only", value=False)


def _as_date_pair(value):
    """Normalize st.date_input output (None, one date, or a 1/2-date tuple)."""
    if value is None:
        return None, None
    if isinstance(value, (list, tuple)):
        if not value:
            return None, None
        start = value[0]
        end = value[-1] if len(value) > 1 else None
        return start, end
    return value, value  # a single selected date


@st.cache_data(show_spinner=False)
def apply_filters(_df: pd.DataFrame, start, end, region, mode, segment,
                  status, late_only) -> pd.DataFrame:
    out = _df
    if start is not None:
        out = out[out[COL_ORDER_DATE].dt.date >= start]
    if end is not None:
        out = out[out[COL_ORDER_DATE].dt.date <= end]
    if region != "All":
        out = out[out[COL_REGION] == region]
    if mode != "All":
        out = out[out[COL_SHIPPING_MODE] == mode]
    if segment != "All":
        out = out[out[COL_SEGMENT] == segment]
    if status != "All":
        out = out[out[COL_STATUS] == status]
    if late_only:
        out = out[out[COL_IS_DELAYED]]
    return out


_start, _end = _as_date_pair(date_range)
fdf = apply_filters(df, _start, _end, f_region, f_mode,
                    f_segment, f_status, f_late_only)

if fdf.empty:
    st.warning("No orders match the current filters. Adjust the sidebar "
               "filters to see data.")
    st.stop()


# ===========================================================================
# PAGE 1 - OVERVIEW
# ===========================================================================

if page == "Overview":
    st.title("📦 Overview")
    st.caption(f"Order-item records from {fdf[COL_ORDER_DATE].min():%b %Y} to "
               f"{fdf[COL_ORDER_DATE].max():%b %Y} · "
               f"{len(fdf):,} rows after filters")

    # ---- KPI row (notebook cell 14 definitions) ----
    total = len(fdf)
    late = int(fdf[COL_IS_DELAYED].sum())
    late_pct = late / total * 100
    on_time_pct = 100 - late_pct
    avg_delay_late = fdf.loc[fdf[COL_IS_DELAYED], COL_DELAY].mean()
    p90_delay = fdf.loc[fdf[COL_IS_DELAYED], COL_DELAY].quantile(0.90)
    total_profit = fdf.loc[fdf[COL_PROFIT] > 0, COL_PROFIT].sum()
    loss_delays = fdf.loc[fdf[COL_IS_DELAYED], COL_PROFIT].sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total deliveries", f"{total:,}")
    with c2:
        kpi_card("On-time", f"{on_time_pct:.1f}%",
                 sub=f"{total - late:,} orders on time or early",
                 color=STATUS_GOOD)
    with c3:
        kpi_card("Late", f"{late_pct:.1f}%",
                 sub=f"{late:,} orders shipped late", color=STATUS_CRITICAL)
    with c4:
        kpi_card("Avg delay (late orders)",
                 f"{avg_delay_late:.2f} days" if pd.notna(avg_delay_late) else "—",
                 sub=f"90% of delays ≤ {p90_delay:.0f} days")

    st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        kpi_card("Total profit (profitable orders)", fmt_money(total_profit))
    with c6:
        kpi_card("Profit on late orders", fmt_money(loss_delays))
    with c7:
        kpi_card("Loss-making orders",
                 f"{(fdf[COL_PROFIT_FLAG] == 'Loss').mean() * 100:.1f}%",
                 sub=f"{(fdf[COL_PROFIT_FLAG] == 'Loss').sum():,} orders below cost")
    with c8:
        kpi_card("Median sales per customer",
                 f"${fdf[COL_SALES].median():,.2f}")

    st.divider()

    # ---- Delay distribution & profitability mix ----
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Delay distribution")
        st.caption("Days late (positive) or early (negative) vs the promised "
                   "schedule. 31% of orders are exactly 1 day late.")

        delay_dist = (fdf[COL_DELAY].value_counts(normalize=True)
                      .sort_index() * 100).reset_index()
        delay_dist.columns = ["Delay (days)", "Pct"]

        # Ordered categories (early -> late) take a one-hue ordinal ramp;
        # the "1 day late" peak is emphasized in the danger color.
        peak = delay_dist["Pct"].idxmax()
        colors = [STATUS_CRITICAL if i == peak else PRIMARY
                  for i in range(len(delay_dist))]
        fig = go.Figure(go.Bar(
            x=delay_dist["Delay (days)"].astype(str),
            y=delay_dist["Pct"],
            marker_color=colors,
            width=0.62,
            text=[f"{v:.1f}%" for v in delay_dist["Pct"]],
            textposition="outside",
            textfont=dict(color=INK_2, size=11),
            cliponaxis=False,
            hovertemplate="Delay: %{x} days<br>Share: %{y:.1f}%<extra></extra>",
        ))
        fig = style_axes(fig, y_title="% of orders", x_title="Delay (days)")
        fig.update_yaxes(range=[0, delay_dist["Pct"].max() * 1.18])
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Profitability mix")
        st.caption("Share of orders by profit flag — 18.7% of orders lose "
                   "money outright, a bigger leak than lateness itself.")

        flag_counts = fdf[COL_PROFIT_FLAG].value_counts()

        # Status colors: profit = good, loss = critical, break-even = muted.
        flag_colors = {"Profit": STATUS_GOOD, "Loss": STATUS_CRITICAL,
                       "Break-even": INK_MUTED}

        fig = go.Figure(go.Bar(
            orientation="h",
            y=list(flag_counts.index),
            x=flag_counts.values,
            marker_color=[flag_colors.get(f, PRIMARY) for f in flag_counts.index],
            width=0.55,
            text=[f"{v:,} ({v / flag_counts.sum() * 100:.1f}%)"
                  for v in flag_counts.values],
            textposition="outside",
            textfont=dict(color=INK_2, size=11),
            cliponaxis=False,
            hovertemplate="%{y}: %{x:,} orders<extra></extra>",
        ))
        fig = style_axes(fig, x_title="Orders")
        fig.update_xaxes(range=[0, flag_counts.max() * 1.35])
        st.plotly_chart(fig, use_container_width=True)

    # ---- Monthly trend ----
    st.subheader("Late-delivery rate over time")
    st.caption("Share of orders shipping late, by month of order date.")

    monthly = (fdf.groupby(fdf[COL_ORDER_DATE].dt.to_period("M"))[COL_IS_DELAYED]
               .mean().mul(100).reset_index())
    monthly.columns = ["Month", "Late %"]
    monthly["Label"] = monthly["Month"].dt.strftime("%b %Y")
    monthly_x = np.arange(len(monthly))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_x, y=monthly["Late %"], mode="lines",
        line=dict(color=PRIMARY, width=2), fill="tozeroy",
        fillcolor="rgba(42,120,214,0.10)",
        hovertemplate="%{customdata}<br>Late: %{y:.1f}%<extra></extra>",
        customdata=monthly["Label"],
    ))
    # Selective direct label: only the peak month (never every point)
    peak_i = monthly["Late %"].idxmax()
    fig.add_trace(go.Scatter(
        x=[monthly_x[peak_i]], y=[monthly["Late %"].iloc[peak_i]],
        mode="markers+text",
        marker=dict(size=9, color=PRIMARY,
                    line=dict(color="#fcfcfb", width=2)),
        text=[f"{monthly['Late %'].iloc[peak_i]:.1f}%"],
        textposition="top center", textfont=dict(color=INK_2, size=11),
        hoverinfo="skip",
    ))
    fig = style_axes(fig, y_title="Late deliveries (%)")
    fig.update_xaxes(tickvals=monthly_x,
                     ticktext=monthly["Label"], tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 2 - DELIVERY PERFORMANCE
# ===========================================================================

elif page == "Delivery Performance":
    st.title("🚚 Delivery Performance")
    st.caption("Delay rates across dimensions, all rendered from the same "
               "filtered slice as the sidebar.")

    dim_opts = {"Order Region": COL_REGION, "Shipping Mode": COL_SHIPPING_MODE,
                "Customer Segment": COL_SEGMENT, "Order Status": COL_STATUS,
                "Payment Type": COL_TYPE, "Department": COL_DEPARTMENT,
                "Product Category": COL_CATEGORY}
    dim_label = st.selectbox("Break down by dimension", list(dim_opts.keys()))
    dim_col = dim_opts[dim_label]

    top_n = st.slider("Show top N categories", 5, 25, 10)

    # ---- Delay % by selected dimension ----
    st.subheader(f"Delay % by {dim_label.lower()}")
    st.caption("Share of orders in each category that shipped late. "
               "The dashed reference line marks the overall average.")

    by_dim = (fdf.groupby(dim_col)[COL_IS_DELAYED].agg(["mean", "count"])
              .reset_index())
    by_dim.columns = [dim_col, "delay_pct", "orders"]
    by_dim["delay_pct"] *= 100
    by_dim = by_dim.sort_values("delay_pct", ascending=False).head(top_n)

    overall = fdf[COL_IS_DELAYED].mean() * 100

    fig = go.Figure(go.Bar(
        orientation="h",
        y=by_dim[dim_col][::-1],          # largest at top
        x=by_dim["delay_pct"][::-1],
        marker_color=PRIMARY,             # one series -> one hue
        width=0.6,
        text=[f"{v:.1f}%" for v in by_dim["delay_pct"][::-1]],
        textposition="outside",
        textfont=dict(color=INK_2, size=11),
        cliponaxis=False,
        customdata=by_dim["orders"][::-1],
        hovertemplate=("%{y}<br>Late: %{x:.1f}%<br>%{customdata:,} orders"
                       "<extra></extra>"),
    ))
    fig.add_vline(x=overall, line_width=1, line_dash="dash",
                  line_color=INK_MUTED)
    fig.add_annotation(x=overall, y=1.05, yref="paper",
                       text=f"Overall {overall:.1f}%", showarrow=False,
                       font=dict(color=INK_MUTED, size=11))
    fig = style_axes(fig, x_title="Late deliveries (%)")
    fig.update_xaxes(range=[0, max(by_dim["delay_pct"].max() * 1.2, overall * 1.1)])
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---- Shipping mode: scheduled vs actual (the headline finding) ----
    st.subheader("Shipping mode: scheduled vs actual shipping days")
    st.caption("The premium tiers promise less time than they deliver. "
               "First Class is scheduled for 1 day and always takes 2.")

    mode_stats = (fdf.groupby(COL_SHIPPING_MODE)
                  .agg(scheduled=(COL_SCHED, "mean"),
                       actual=(COL_REAL, "mean"),
                       orders=(COL_REAL, "count"))
                  .reset_index())
    mode_order = ["Same Day", "First Class", "Second Class", "Standard Class"]
    mode_stats["ord"] = mode_stats[COL_SHIPPING_MODE].map(
        {m: i for i, m in enumerate(mode_order)})
    mode_stats = mode_stats.sort_values("ord")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Scheduled days", x=mode_stats[COL_SHIPPING_MODE],
        y=mode_stats["scheduled"], marker_color=CAT[0], width=0.32,
        offsetgroup="sched",
        hovertemplate="%{x}<br>Scheduled: %{y:.2f} days<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Actual days", x=mode_stats[COL_SHIPPING_MODE],
        y=mode_stats["actual"], marker_color=CAT[1], width=0.32,
        offsetgroup="actual",
        hovertemplate="%{x}<br>Actual: %{y:.2f} days<extra></extra>",
    ))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.12,
                                                   title=""))
    fig = style_axes(fig, y_title="Average shipping days")
    st.plotly_chart(fig, use_container_width=True)

    # ---- Delivery status mix ----
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Delivery status mix")
        st.caption("Dataset flag combining early, on-time and late outcomes.")
        ds = fdf[COL_DELIVERY_STATUS].value_counts()
        ds_colors = {"Late delivery": STATUS_CRITICAL,
                     "Shipping on time": STATUS_GOOD,
                     "Advance shipping": PRIMARY}
        fig = go.Figure(go.Bar(
            orientation="h", y=list(ds.index)[::-1], x=list(ds.values)[::-1],
            marker_color=[ds_colors.get(k, PRIMARY) for k in list(ds.index)[::-1]],
            width=0.55,
            text=[f"{v:,} ({v / ds.sum() * 100:.1f}%)"
                  for v in list(ds.values)[::-1]],
            textposition="outside", textfont=dict(color=INK_2, size=11),
            cliponaxis=False,
            hovertemplate="%{y}: %{x:,} orders<extra></extra>",
        ))
        fig = style_axes(fig, x_title="Orders")
        fig.update_xaxes(range=[0, ds.max() * 1.32])
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Shipping volume by mode")
        st.caption("Order count per shipping mode — context for the delay "
                   "rates above.")
        sv = fdf[COL_SHIPPING_MODE].value_counts()
        fig = go.Figure(go.Bar(
            orientation="h", y=list(sv.index)[::-1], x=list(sv.values)[::-1],
            marker_color=PRIMARY, width=0.55,
            text=[f"{v:,}" for v in list(sv.values)[::-1]],
            textposition="outside", textfont=dict(color=INK_2, size=11),
            cliponaxis=False,
            hovertemplate="%{y}: %{x:,} orders<extra></extra>",
        ))
        fig = style_axes(fig, x_title="Orders")
        fig.update_xaxes(range=[0, sv.max() * 1.25])
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 3 - ROOT CAUSE ANALYSIS
# ===========================================================================

elif page == "Root Cause Analysis":
    st.title("🔍 Root Cause Analysis")
    st.caption("Why deliveries run late: schedule miscalibration dominates; "
               "region and season matter far less.")

    # ---- Finding callouts from the notebook/README ----
    c1, c2, c3 = st.columns(3)
    c1.info("**Schedule miscalibration** — First Class is scheduled for 1 day "
            "and takes 2, 100% of the time. Second Class runs ~80% late. "
            "Re-baselining these promises fixes most recorded lateness.")
    c2.info("**Structural, not behavioral** — delay rates sit in a narrow "
            "53–55% band across segment, payment type, day and hour. "
            "Global scheduling beats micro-targeting.")
    c3.info("**Margin holds for now** — mean profit is ~$20–23 at every delay "
            "level. Lateness is a reliability problem; the margin leak is the "
            "18.7% of orders sold below cost.")

    st.divider()

    # ---- Notebook cell 23: top drivers within a region ----
    st.subheader("Top drivers of late delivery by region")
    st.caption("Every factor level inside the chosen region, ranked by its "
               "late rate (notebook's drill-down, extended to any region).")

    region_pick = st.selectbox(
        "Region", sorted(fdf[COL_REGION].dropna().unique().tolist()),
        index=(sorted(fdf[COL_REGION].dropna().unique().tolist()).index("Central Africa")
               if "Central Africa" in fdf[COL_REGION].unique() else 0))

    drivers = [COL_SHIPPING_MODE, COL_SEGMENT, COL_DEPARTMENT, COL_TYPE,
               COL_STATUS]
    frames = []
    df_region = fdf[fdf[COL_REGION] == region_pick]
    for factor in drivers:
        temp = (df_region.groupby(factor)[COL_IS_DELAYED]
                .agg(total_orders="count", late_orders="sum")
                .reset_index())
        temp["delay_pct"] = temp["late_orders"] / temp["total_orders"] * 100
        temp["Factor"] = factor + " : " + temp[factor].astype(str)
        frames.append(temp[["Factor", "delay_pct", "total_orders"]])
    driver_df = (pd.concat(frames)
                 .sort_values("delay_pct", ascending=False).head(10))

    fig = go.Figure(go.Bar(
        orientation="h", y=driver_df["Factor"][::-1],
        x=driver_df["delay_pct"][::-1],
        marker_color=PRIMARY, width=0.6,
        text=[f"{v:.1f}%" for v in driver_df["delay_pct"][::-1]],
        textposition="outside", textfont=dict(color=INK_2, size=11),
        cliponaxis=False,
        customdata=driver_df["total_orders"][::-1],
        hovertemplate=("%{y}<br>Late: %{x:.1f}%<br>%{customdata:,} orders"
                       "<extra></extra>"),
    ))
    fig = style_axes(fig, x_title="Late deliveries (%)")
    fig.update_xaxes(range=[0, driver_df["delay_pct"].max() * 1.2])
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---- Profit vs delay (notebook cell 16) - two charts, one axis each ----
    st.subheader("Profit vs delay duration")
    st.caption("Left: mean profit per order barely moves with delay length. "
               "Right: total profit concentration by delay bucket.")

    profit_metrics = (fdf.groupby(COL_DELAY)[COL_PROFIT]
                      .agg(mean_profit="mean", total_profit="sum",
                           order_count="count").reset_index())

    pm_l, pm_r = st.columns(2)
    with pm_l:
        fig = go.Figure(go.Bar(
            x=profit_metrics[COL_DELAY].astype(str),
            y=profit_metrics["mean_profit"],
            marker_color=PRIMARY, width=0.6,
            text=[f"${v:.1f}" for v in profit_metrics["mean_profit"]],
            textposition="outside", textfont=dict(color=INK_2, size=11),
            cliponaxis=False,
            hovertemplate="Delay %{x} days<br>Mean profit: $%{y:.2f}"
                          "<extra></extra>",
        ))
        fig = style_axes(fig, y_title="Mean profit ($)",
                         x_title="Delay (days)")
        fig.update_yaxes(range=[0, profit_metrics["mean_profit"].max() * 1.25])
        st.plotly_chart(fig, use_container_width=True, key="pm_mean")
    with pm_r:
        fig = go.Figure(go.Bar(
            x=profit_metrics[COL_DELAY].astype(str),
            y=profit_metrics["total_profit"],
            marker_color=SECONDARY, width=0.6,
            text=[fmt_money(v) for v in profit_metrics["total_profit"]],
            textposition="outside", textfont=dict(color=INK_2, size=11),
            cliponaxis=False,
            hovertemplate="Delay %{x} days<br>Total profit: %{customdata}"
                          "<extra></extra>",
            customdata=[fmt_money(v) for v in profit_metrics["total_profit"]],
        ))
        fig = style_axes(fig, y_title="Total profit ($)", x_title="Delay (days)")
        fig.update_yaxes(range=[0, profit_metrics["total_profit"].max() * 1.25])
        st.plotly_chart(fig, use_container_width=True, key="pm_total")

    st.divider()

    # ---- Time-based seasonality (notebook cells 24-25) ----
    st.subheader("When orders are placed vs when they run late")
    st.caption("Delay % by month, day of week and hour of order — the spread "
               "is narrow (roughly 52–57%), confirming lateness is structural.")

    tb_l, tb_m, tb_r = st.columns(3)

    with tb_l:
        by_month = (fdf.groupby(COL_MONTH)[COL_IS_DELAYED].mean().mul(100)
                    .reset_index())
        fig = go.Figure(go.Scatter(
            x=by_month[COL_MONTH], y=by_month[COL_IS_DELAYED],
            mode="lines+markers", line=dict(color=PRIMARY, width=2),
            marker=dict(size=8, color=PRIMARY,
                        line=dict(color="#fcfcfb", width=2)),
            hovertemplate="%{x}<br>Late: %{y:.1f}%<extra></extra>",
        ))
        fig = style_axes(fig, y_title="Late %", x_title="Month")
        fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=MONTH_LABELS)
        st.plotly_chart(fig, use_container_width=True, key="tb_month")

    with tb_m:
        by_day = (fdf.groupby(COL_DAY)[COL_IS_DELAYED].mean().mul(100)
                  .reset_index())
        by_day["ord"] = by_day[COL_DAY].map({d: i for i, d in enumerate(DAY_ORDER)})
        by_day = by_day.sort_values("ord")
        fig = go.Figure(go.Bar(
            x=by_day[COL_DAY], y=by_day[COL_IS_DELAYED],
            marker_color=PRIMARY, width=0.6,
            hovertemplate="%{x}<br>Late: %{y:.1f}%<extra></extra>",
        ))
        fig = style_axes(fig, y_title="Late %", x_title="Day of week")
        fig.update_xaxes(tickangle=-45)
        fig.update_yaxes(range=[0, by_day[COL_IS_DELAYED].max() * 1.2])
        st.plotly_chart(fig, use_container_width=True, key="tb_day")

    with tb_r:
        by_hour = (fdf.groupby(COL_HOUR)[COL_IS_DELAYED].mean().mul(100)
                   .reset_index())
        fig = go.Figure(go.Scatter(
            x=by_hour[COL_HOUR], y=by_hour[COL_IS_DELAYED],
            mode="lines+markers", line=dict(color=PRIMARY, width=2),
            marker=dict(size=8, color=PRIMARY,
                        line=dict(color="#fcfcfb", width=2)),
            hovertemplate="%{x}:00<br>Late: %{y:.1f}%<extra></extra>",
        ))
        fig = style_axes(fig, y_title="Late %", x_title="Hour of day")
        fig.update_xaxes(tickvals=list(range(0, 24, 3)))
        st.plotly_chart(fig, use_container_width=True, key="tb_hour")


# ===========================================================================
# PAGE 4 - PREDICTION MODEL
# ===========================================================================

elif page == "Prediction Model":
    st.title("🤖 Prediction Model")
    st.caption("Random Forest classifier predicting the dataset's "
               "`Late_delivery_risk` flag from attributes known at dispatch "
               "— no leakage from actual shipping times.")

    (model, X_test, y_test, importances,
     freq_maps) = get_model_and_test_data()

    # ------------------------------------------------------------------
    # Model card
    # ------------------------------------------------------------------
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Model card")
        st.markdown(
            f"""
| | |
|---|---|
| **Algorithm** | RandomForestClassifier |
| **Target** | `Late_delivery_risk` (1 = late) |
| **Features** | 9 (6 categorical, frequency-encoded) |
| **Train rows** | {len(X_test) * 4:,} (80%) |
| **Test rows** | {len(X_test):,} (20%) |
| **Balancing** | SMOTE (train only) |
| **Random seed** | 42 (split, SMOTE, forest) |
| **Persisted** | `{MODEL_PATH.name}` via joblib |
            """
        )
        with st.expander("Feature list"):
            st.markdown(
                "\n".join(f"- `{f}`" for f in MODEL_FEATURES))

    with right:
        st.subheader("Performance on held-out test set")
        st.caption("Displayed values must match the notebook's output; any "
                   "deviation is flagged below the cards.")

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc:.2f}")
        m2.metric("Precision (late)", f"{prec:.2f}")
        m3.metric("Recall (late)", f"{rec:.2f}")
        m4.metric("F1 (late)", f"{f1:.2f}")

        # Flag any drift from the notebook's reported numbers instead of
        # silently displaying different values.
        expected = {"Accuracy": (acc, 0.74), "Precision": (prec, 0.79),
                    "Recall": (rec, 0.75)}
        drift = {k: (v[0], v[1]) for k, v in expected.items()
                 if abs(v[0] - v[1]) > 0.005}
        if drift:
            st.warning(
                "⚠️ **Discrepancy vs notebook** — " +
                ", ".join(f"{k}: dashboard {got:.4f} vs notebook {exp:.2f}"
                          for k, (got, exp) in drift.items()) +
                ". Likely a library-version difference; numbers are shown as "
                "computed here, not overridden.")

    st.divider()

    # ------------------------------------------------------------------
    # Confusion matrix + classification report
    # ------------------------------------------------------------------
    cm_l, cm_r = st.columns(2)

    with cm_l:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_test, y_pred)
        labels = ["On time (0)", "Late (1)"]
        z = cm[::-1]  # row 1 (late) on top for readability
        fig = go.Figure(go.Heatmap(
            z=z, x=labels, y=labels[::-1], colorscale=[[0, "#f5f5f3"],
                                                       [1, PRIMARY]],
            showscale=False, xgap=2, ygap=2,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}"
                          "<br>Count: %{z:,}<extra></extra>",
        ))
        for i in range(2):
            for j in range(2):
                val = z[i][j]
                fig.add_annotation(x=labels[j], y=labels[::-1][i],
                                   text=f"{val:,}", showarrow=False,
                                   font=dict(color="white" if val > cm.max() / 2
                                             else INK, size=16))
        fig = style_axes(fig, y_title="Actual", x_title="Predicted")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True, key="cm")

    with cm_r:
        st.subheader("Classification report")
        report = classification_report(y_test, y_pred, output_dict=True)
        rep_df = (pd.DataFrame(report).T
                  .drop(index=["accuracy"])
                  .rename(index={"0": "On time (0)", "1": "Late (1)",
                                 "macro avg": "Macro avg",
                                 "weighted avg": "Weighted avg"}))
        rep_df["support"] = rep_df["support"].astype(int)
        st.dataframe(
            rep_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}",
                                 "f1-score": "{:.2f}", "support": "{:,}"})
            .background_gradient(subset=["f1-score"], cmap="Blues", vmin=0, vmax=1),
            use_container_width=True,
        )
        st.caption("Baseline: always guessing \"late\" scores 0.57 accuracy "
                   "— the model adds ~17 points.")

    st.divider()

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    st.subheader("Feature importance")
    st.caption("Impurity-based importance from the fitted forest. Scheduled "
               "shipping days and shipping mode dominate — the schedule "
               "itself is the signal, echoing the root-cause finding.")

    pretty = {f"{COL_TYPE}_freq": "Payment type", f"{COL_SCHED}": "Scheduled days",
              f"{COL_CATEGORY}_freq": "Product category",
              f"{COL_SEGMENT}_freq": "Customer segment",
              f"{COL_DEPARTMENT}_freq": "Department",
              f"{COL_REGION}_freq": "Order region",
              f"{COL_SHIPPING_MODE}_freq": "Shipping mode",
              f"{COL_MONTH}": "Order month", f"{COL_HOUR}": "Order hour"}
    imp = importances.copy()
    imp["Label"] = imp["Feature"].map(pretty).fillna(imp["Feature"])

    fig = go.Figure(go.Bar(
        orientation="h", y=imp["Label"], x=imp["Importance"],
        marker_color=PRIMARY, width=0.6,
        text=[f"{v:.3f}" for v in imp["Importance"]],
        textposition="outside", textfont=dict(color=INK_2, size=11),
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig = style_axes(fig, x_title="Importance")
    fig.update_xaxes(range=[0, imp["Importance"].max() * 1.18])
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig, use_container_width=True, key="fi")

    st.divider()

    # ------------------------------------------------------------------
    # Interactive prediction form
    # ------------------------------------------------------------------
    st.subheader("Try it: predict a delivery")
    st.caption("Enter order attributes known at dispatch. Inputs are "
               "frequency-encoded exactly like training, then scored by the "
               "persisted model.")

    with st.form("prediction_form", clear_on_submit=False):
        pf1, pf2, pf3 = st.columns(3)
        with pf1:
            in_type = st.selectbox("Payment type",
                                   sorted(freq_maps[COL_TYPE].index))
            in_sched = st.selectbox("Scheduled shipping days",
                                    sorted(df[COL_SCHED].unique()))
            in_category = st.selectbox("Product category",
                                       sorted(freq_maps[COL_CATEGORY].index))
        with pf2:
            in_segment = st.selectbox("Customer segment",
                                      sorted(freq_maps[COL_SEGMENT].index))
            in_department = st.selectbox("Department",
                                         sorted(freq_maps[COL_DEPARTMENT].index))
            in_region = st.selectbox("Order region",
                                     sorted(freq_maps[COL_REGION].index))
        with pf3:
            in_mode = st.selectbox("Shipping mode",
                                   ["Same Day", "First Class", "Second Class",
                                    "Standard Class"])
            in_month = st.select_slider("Order month", options=list(range(1, 13)),
                                        format_func=lambda m: MONTH_LABELS[m - 1])
            in_hour = st.slider("Order hour", 0, 23, 12)

        submitted = st.form_submit_button("Predict delivery outcome",
                                          type="primary", use_container_width=True)

    if submitted:
        raw = {COL_TYPE: in_type, COL_SCHED: in_sched, COL_CATEGORY: in_category,
               COL_SEGMENT: in_segment, COL_DEPARTMENT: in_department,
               COL_REGION: in_region, COL_SHIPPING_MODE: in_mode,
               COL_MONTH: in_month, COL_HOUR: in_hour}
        X_in = encode_input(raw, freq_maps)
        pred = model.predict(X_in)[0]
        proba = model.predict_proba(X_in)[0]

        if pred == 1:
            st.error(f"### 🔴 Late delivery predicted — {proba[1] * 100:.1f}% risk")
            st.markdown(
                "This order profile matches historical patterns of late "
                "shipment. Consider a faster fulfilment path or an honest "
                "re-promise at checkout.")
        else:
            st.success(f"### 🟢 On time predicted — {proba[0] * 100:.1f}% "
                       "confidence")
        with st.expander("Prediction details"):
            st.write("**Model input (after frequency encoding)**")
            st.dataframe(X_in.T.rename(columns={0: "value"}),
                         use_container_width=True)
            st.write(f"**Class probabilities** — on time: "
                     f"**{proba[0]:.3f}** · late: **{proba[1]:.3f}**")


# ===========================================================================
# PAGE 5 - DATA EXPLORER
# ===========================================================================

else:  # Data Explorer
    st.title("🗂️ Data Explorer")
    st.caption(f"{len(fdf):,} cleaned order-item rows (notebook cleaning "
               "applied). Filter, search, sort — then download the slice.")

    # Column visibility
    all_cols = list(fdf.columns)
    default_show = [COL_ORDER_DATE, COL_TYPE, COL_SEGMENT, COL_CATEGORY,
                    COL_DEPARTMENT, COL_REGION, COL_STATUS, COL_SHIPPING_MODE,
                    COL_SCHED, COL_REAL, COL_PROC, COL_DELAY, COL_IS_DELAYED,
                    COL_PROFIT, COL_PROFIT_FLAG, COL_LATE_RISK]
    with st.expander("Columns shown"):
        chosen = st.multiselect("Pick columns", all_cols, default=default_show)
    show_cols = chosen if chosen else all_cols

    # Quick search across string columns
    search = st.text_input("Search (matches any text column)",
                           placeholder="e.g. Standard Class, Central Africa…")
    view = fdf
    if search:
        mask = False
        for c in show_cols:
            if view[c].dtype == object:
                mask = mask | view[c].astype(str).str.contains(
                    search, case=False, na=False)
        view = view[mask]

    # Per-column filters
    with st.expander("Advanced filters"):
        av1, av2, av3 = st.columns(3)
        with av1:
            if COL_CATEGORY in show_cols:
                cats = st.multiselect("Category",
                                      sorted(view[COL_CATEGORY].dropna().unique()))
                if cats:
                    view = view[view[COL_CATEGORY].isin(cats)]
        with av2:
            if COL_SHIPPING_MODE in show_cols:
                modes = st.multiselect("Shipping mode",
                                       sorted(view[COL_SHIPPING_MODE].dropna().unique()))
                if modes:
                    view = view[view[COL_SHIPPING_MODE].isin(modes)]
        with av3:
            delays = st.multiselect(
                "Delay (days)",
                sorted(view[COL_DELAY].dropna().unique().tolist(),
                       key=lambda x: (x is None, x)))
            if delays:
                view = view[view[COL_DELAY].isin(delays)]

    st.write(f"**{len(view):,} rows** match")
    st.dataframe(view[show_cols], use_container_width=True, height=460,
                 hide_index=True)

    # Download the current slice
    @st.cache_data(show_spinner=False)
    def to_csv(df_slice: pd.DataFrame) -> bytes:
        return df_slice.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download current view as CSV",
        data=to_csv(view[show_cols]),
        file_name="supply_chain_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Dataset & cleaning notes"):
        st.markdown(
            """
**Raw file**: 180,519 rows × 53 columns → **cleaned**: 172,765 rows × 20 columns
(Jan 2015 – Feb 2018, one row per order item).

Cleaning applied (identical to the notebook):
- Dropped identifier, PII (names/emails/passwords) and redundant columns —
  `Product Description` is empty everywhere, `Order Zipcode` missing in 86% of rows
- Removed `Shipping canceled` orders (never enter the delivery pipeline)
- Parsed both date columns to datetime

Engineered features: **Order Processing Time**, **Delay**
(processing − scheduled), **Is_Delayed** (delay > 0), **order_month / day / hour**,
and **Profitability Flag** (Profit / Loss / Break-even).
            """)
