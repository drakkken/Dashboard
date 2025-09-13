# app.py (fixed)
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Marketing Intelligence — Clean Dashboard", layout="wide")


# Helpers: robust file / col handling

def find_file_by_keyword(keywords):
    files = os.listdir(".")
    for kw in keywords:
        for f in files:
            if kw.lower() in f.lower() and f.lower().endswith(".csv"):
                return f
    return None

def clean_numeric(s):
    """
    Clean numeric values robustly. Accepts Series or DataFrame (takes first numeric column).
    Returns a numeric Series (float), NaNs where conversion fails.
    """
    # If a DataFrame slipped in, try to pick a sensible column (prefer numeric)
    if isinstance(s, pd.DataFrame):
        # pick first numeric column, otherwise first column
        num_cols = s.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            s = s[num_cols[0]]
        else:
            s = s.iloc[:, 0]

    # now ensure series
    s = pd.Series(s)  # converts scalar/array-like safely to Series
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    # remove non-numeric characters and convert
    cleaned = s.astype(str).str.replace(r"[^\d.\-eE]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")

def safe_sum(col):
    """
    Return scalar sum for a Series or DataFrame column-like object.
    Works if a DataFrame was accidentally passed too.
    """
    if isinstance(col, pd.DataFrame):
        # sum numeric columns then sum across them
        num = col.select_dtypes(include=[np.number])
        if not num.empty:
            return float(num.sum().sum())
        # fallback: try first column
        col = col.iloc[:, 0]
    # ensure numeric
    s = pd.to_numeric(col, errors="coerce").fillna(0)
    return float(s.sum())

def safe_int(col):
    """Return int(safe_sum(col)) with safe fallback to 0."""
    val = safe_sum(col)
    try:
        return int(round(val))
    except Exception:
        return 0


# Normalizers

def normalize_marketing_columns(df):
    rename = {}
    for c in df.columns:
        lc = c.strip().lower().replace("_", " ")
        if lc == "date":
            rename[c] = "date"
        elif "tactic" in lc:
            rename[c] = "tactic"
        elif "state" in lc:
            rename[c] = "state"
        elif "campaign" in lc:
            rename[c] = "campaign"
        elif "impress" in lc:
            rename[c] = "impressions"
        elif lc in ("impression","impression count"):
            rename[c] = "impressions"
        elif "click" in lc:
            rename[c] = "clicks"
        elif "spend" in lc:
            rename[c] = "spend"
        elif "attributed" in lc and "revenue" in lc:
            rename[c] = "attributed_revenue"
        elif "revenue" in lc and "attributed" not in lc:
            rename[c] = "attributed_revenue"
    df = df.rename(columns=rename)
    # ensure columns exist
    for needed in ["date","impressions","clicks","spend","attributed_revenue","tactic","state","campaign"]:
        if needed not in df.columns:
            df[needed] = np.nan
    # clean numeric safely
    df["impressions"] = clean_numeric(df["impressions"]).fillna(0).astype(int)
    df["clicks"] = clean_numeric(df["clicks"]).fillna(0).astype(int)
    df["spend"] = clean_numeric(df["spend"]).fillna(0.0)
    df["attributed_revenue"] = clean_numeric(df["attributed_revenue"]).fillna(0.0)
    # parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df[["date","tactic","state","campaign","impressions","clicks","spend","attributed_revenue"]]

def normalize_business_columns(df):
    rename = {}
    for c in df.columns:
        lc = c.strip().lower().replace("_"," ")
        # careful precedence with boolean ops
        if lc == "date":
            rename[c] = "date"
        elif ("order" in lc or "orders" in lc) and "new" not in lc:
            rename[c] = "orders"
        elif "new order" in lc:
            rename[c] = "new_orders"
        elif "new customer" in lc or "new customers" in lc:
            rename[c] = "new_customers"
        elif "total revenue" in lc or "total revenue" == lc:
            rename[c] = "total_revenue"
        elif "gross" in lc and "profit" in lc:
            rename[c] = "gross_profit"
        elif "cogs" in lc:
            rename[c] = "COGS"
        elif lc in ("# of orders","orders"):
            rename[c] = "orders"
    df = df.rename(columns=rename)
    # ensure numeric cols exist and are cleaned
    for col in ["orders","new_orders","new_customers","total_revenue","gross_profit","COGS"]:
        if col in df.columns:
            df[col] = clean_numeric(df[col]).fillna(0.0)
        else:
            df[col] = 0.0
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # ensure orders are integer-like
    df["orders"] = pd.to_numeric(df["orders"], errors="coerce").fillna(0).astype(int)
    return df[["date","orders","new_orders","new_customers","total_revenue","gross_profit","COGS"]]

# Load files (robust)

fb_file = find_file_by_keyword(["facebook", "fb"])
gg_file = find_file_by_keyword(["google"])
tt_file = find_file_by_keyword(["tiktok", "tik tok", "ticktok"])
biz_file = find_file_by_keyword(["business", "business.csv", "businessdata"])

# fallback to exact names
if not fb_file and os.path.exists("facebook.csv"): fb_file = "facebook.csv"
if not gg_file and os.path.exists("google.csv"): gg_file = "google.csv"
if not tt_file and os.path.exists("tiktok.csv"): tt_file = "tiktok.csv"
if not biz_file and os.path.exists("business.csv"): biz_file = "business.csv"

missing = []
if not fb_file: missing.append("Facebook CSV")
if not gg_file: missing.append("Google CSV")
if not tt_file: missing.append("TikTok CSV")
if not biz_file: missing.append("Business CSV")

if missing:
    st.error("Missing files: " + ", ".join(missing))
    st.info("Place your CSVs in this folder. Expected names include: facebook.csv, google.csv, tiktok.csv, business.csv")
    st.stop()

# read files
try:
    fb_raw = pd.read_csv(fb_file)
    gg_raw = pd.read_csv(gg_file)
    tt_raw = pd.read_csv(tt_file)
    biz_raw = pd.read_csv(biz_file)
except Exception as e:
    st.error(f"Error reading CSVs: {e}")
    st.stop()

# normalize
fb = normalize_marketing_columns(fb_raw)
gg = normalize_marketing_columns(gg_raw)
tt = normalize_marketing_columns(tt_raw)

fb["channel"] = "Facebook"
gg["channel"] = "Google"
tt["channel"] = "TikTok"

marketing = pd.concat([fb, gg, tt], ignore_index=True)
marketing = marketing.dropna(subset=["date"]).sort_values("date")

biz = normalize_business_columns(biz_raw)
biz = biz.dropna(subset=["date"]).sort_values("date")


# Derived metrics

def add_metrics(df):
    df = df.copy()
    df["CTR"] = np.where(df["impressions"]>0, df["clicks"] / df["impressions"], np.nan)
    df["CPC"] = np.where(df["clicks"]>0, df["spend"] / df["clicks"], np.nan)
    df["CPM"] = np.where(df["impressions"]>0, df["spend"] / df["impressions"] * 1000, np.nan)
    df["ROAS"] = np.where(df["spend"]>0, df["attributed_revenue"] / df["spend"], np.nan)
    return df

marketing = add_metrics(marketing)

# Aggregations
daily_channel = marketing.groupby(["date","channel"], as_index=False).agg({
    "impressions":"sum","clicks":"sum","spend":"sum","attributed_revenue":"sum"
})
daily_channel = add_metrics(daily_channel)

daily_total = daily_channel.groupby("date", as_index=False).agg({
    "impressions":"sum","clicks":"sum","spend":"sum","attributed_revenue":"sum"
})
daily_total = add_metrics(daily_total)

campaign_agg = marketing.groupby(["channel","campaign"], as_index=False).agg({
    "impressions":"sum","clicks":"sum","spend":"sum","attributed_revenue":"sum"
})
campaign_agg = add_metrics(campaign_agg)

state_agg = marketing.groupby("state", as_index=False).agg({
    "impressions":"sum","clicks":"sum","spend":"sum","attributed_revenue":"sum"
})
state_agg = add_metrics(state_agg)

# merge daily marketing with business by date
combined = pd.merge(biz, daily_total, on="date", how="left").fillna(0)
combined = combined.sort_values("date")
combined["spend"] = combined["spend"].astype(float)
combined["attributed_revenue"] = combined["attributed_revenue"].astype(float)

# filteres 

st.sidebar.header("Filters")
min_date = marketing["date"].min().date()
max_date = marketing["date"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

channels = ["Facebook","Google","TikTok"]
selected_channels = st.sidebar.multiselect("Channels", channels, default=channels)

tactic_options = marketing["tactic"].dropna().unique().tolist()
selected_tactics = st.sidebar.multiselect("Tactic (optional)", tactic_options)

state_options = marketing["state"].dropna().unique().tolist()
selected_states = st.sidebar.multiselect("State (optional)", state_options)

# apply filters
start_dt = pd.to_datetime(date_range[0])
end_dt = pd.to_datetime(date_range[1])
mask = (marketing["date"] >= start_dt) & (marketing["date"] <= end_dt) & (marketing["channel"].isin(selected_channels))
if selected_tactics:
    mask &= marketing["tactic"].isin(selected_tactics)
if selected_states:
    mask &= marketing["state"].isin(selected_states)
filtered = marketing[mask].copy()
if filtered.empty:
    st.warning("No marketing rows after applying filters — try widening the date range / channels / tactic / state.")

# filtered aggregations
daily_filtered = filtered.groupby("date", as_index=False).agg({
    "impressions":"sum","clicks":"sum","spend":"sum","attributed_revenue":"sum"
})
daily_filtered = add_metrics(daily_filtered)
channel_filtered = filtered.groupby("channel", as_index=False).agg({
    "impressions":"sum","clicks":"sum","spend":"sum","attributed_revenue":"sum"
})
channel_filtered = add_metrics(channel_filtered)
campaign_filtered = filtered.groupby(["channel","campaign"], as_index=False).agg({
    "impressions":"sum","clicks":"sum","spend":"sum","attributed_revenue":"sum"
})
campaign_filtered = add_metrics(campaign_filtered)
state_filtered = filtered.groupby("state", as_index=False).agg({
    "impressions":"sum","clicks":"sum","spend":"sum","attributed_revenue":"sum"
})
state_filtered = add_metrics(state_filtered)

# biz filtered & combined
biz_filtered = biz[(biz["date"]>=start_dt) & (biz["date"]<=end_dt)].sort_values("date")
combined_filtered = pd.merge(biz_filtered, daily_filtered, on="date", how="left").fillna(0)


# Top KPI row

st.title("Marketing Intelligence — Simple & Beautiful (Light)")
st.markdown("Clean, actionable dashboard that connects marketing activity to business outcomes. Use the sidebar to filter.")


total_spend = safe_sum(daily_filtered["spend"])
total_attr = safe_sum(daily_filtered["attributed_revenue"])
total_impr = safe_sum(daily_filtered["impressions"])
total_clicks = safe_sum(daily_filtered["clicks"])
orders = safe_int(biz_filtered["orders"]) if not biz_filtered.empty else 0
total_rev = safe_sum(biz_filtered["total_revenue"]) if not biz_filtered.empty else 0.0
gross_profit = safe_sum(biz_filtered["gross_profit"]) if not biz_filtered.empty else 0.0

avg_ctr = (total_clicks / total_impr) if total_impr > 0 else np.nan
avg_cpc = (total_spend / total_clicks) if total_clicks > 0 else np.nan
avg_roas = (total_attr / total_spend) if total_spend > 0 else np.nan
spend_per_order = (total_spend / orders) if orders > 0 else np.nan

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Spend (₹)", f"₹{total_spend:,.0f}")
k2.metric("Attributed Revenue (₹)", f"₹{total_attr:,.0f}")
k3.metric("Orders", f"{orders:,}")
k4.metric("Business Revenue (₹)", f"₹{total_rev:,.0f}")
k5.metric("Avg ROAS", f"{avg_roas:.2f}x" if not pd.isna(avg_roas) else "—")
k6.metric("Avg CTR", f"{avg_ctr:.2%}" if not pd.isna(avg_ctr) else "—")

k7,k8,k9 = st.columns(3)
k7.metric("Avg CPC (₹)", f"₹{avg_cpc:.2f}" if not pd.isna(avg_cpc) else "—")
k8.metric("Spend per Order (₹)", f"₹{spend_per_order:,.0f}" if not pd.isna(spend_per_order) else "—")
k9.metric("Gross Profit (₹)", f"₹{gross_profit:,.0f}")

st.markdown("---")


# Time series: spend / attr rev & orders

st.subheader("Trends — Spend, Attributed Revenue & Orders")
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=daily_filtered['date'], y=daily_filtered['spend'],
                         mode='lines+markers', name='Marketing Spend', line=dict(width=2)), secondary_y=False)
fig.add_trace(go.Scatter(x=daily_filtered['date'], y=daily_filtered['attributed_revenue'],
                         mode='lines+markers', name='Attributed Revenue', line=dict(dash='dash')), secondary_y=False)
if not biz_filtered.empty:
    fig.add_trace(go.Bar(x=biz_filtered['date'], y=biz_filtered['orders'], name='Orders', opacity=0.6), secondary_y=True)
fig.update_layout(template="plotly_white", height=420, legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0))
fig.update_yaxes(title_text="Spend / Attributed Rev (₹)", secondary_y=False)
fig.update_yaxes(title_text="Orders", secondary_y=True)
st.plotly_chart(fig, use_container_width=True)


# Channel overview: pies + grouped bar + ROAS ranking

st.subheader("Channel performance")

col1, col2 = st.columns([1,1])
with col1:
    if channel_filtered["spend"].sum() > 0:
        fig_pie_spend = px.pie(channel_filtered, names='channel', values='spend', hole=0.45,
                               title="Spend share by Channel", template="plotly_white",
                               color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie_spend.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie_spend, use_container_width=True)
    else:
        st.info("No spend data in the selected filters for channel pie chart.")
with col2:
    if channel_filtered["attributed_revenue"].sum() > 0:
        fig_pie_rev = px.pie(channel_filtered, names='channel', values='attributed_revenue', hole=0.45,
                             title="Attributed revenue share by Channel", template="plotly_white",
                             color_discrete_sequence=px.colors.qualitative.Pastel2)
        fig_pie_rev.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie_rev, use_container_width=True)
    else:
        st.info("No attributed revenue data in the selected filters for revenue pie chart.")

ch = channel_filtered.sort_values("spend", ascending=False)
if not ch.empty:
    fig_bar = go.Figure(data=[
        go.Bar(name='Attributed Revenue', x=ch['channel'], y=ch['attributed_revenue'], marker_color=px.colors.qualitative.Pastel[1]),
        go.Bar(name='Spend', x=ch['channel'], y=ch['spend'], marker_color=px.colors.qualitative.Pastel[0])
    ])
    fig_bar.update_layout(barmode='group', title="Attributed Revenue vs Spend by Channel", template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)

if not ch.empty:
    ch_roas = ch[['channel','ROAS']].sort_values('ROAS', ascending=False)
    fig_roas = px.bar(ch_roas, x='ROAS', y='channel', orientation='h', title="ROAS (higher is better)", template="plotly_white")
    st.plotly_chart(fig_roas, use_container_width=True)

st.markdown("---")


st.subheader("State-level summary (top states by spend)")
if not state_filtered.empty:
    state_top = state_filtered.sort_values("spend", ascending=False).head(12)
    fig_state = px.bar(state_top, x='spend', y='state', orientation='h', title="Top states by Spend", template="plotly_white",
                       text=state_top['attributed_revenue'].map(lambda x: f"₹{x:,.0f}"))
    fig_state.update_yaxes(categoryorder='total ascending')
    st.plotly_chart(fig_state, use_container_width=True)
else:
    st.info("No state-level data for current filters.")

st.markdown("---")

#making a downloadable 
st.subheader("Top campaigns (filtered) — actionable list")
if not campaign_filtered.empty:
    camp = campaign_filtered.copy().sort_values("attributed_revenue", ascending=False)
    camp_display = camp[['channel','campaign','spend','attributed_revenue','CTR','CPC','ROAS']].head(50).fillna(0)
    camp_display['spend'] = camp_display['spend'].map(lambda x: f"₹{x:,.0f}")
    camp_display['attributed_revenue'] = camp_display['attributed_revenue'].map(lambda x: f"₹{x:,.0f}")
    camp_display['CTR'] = camp_display['CTR'].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
    camp_display['CPC'] = camp_display['CPC'].map(lambda x: f"₹{x:.2f}" if pd.notna(x) else "—")
    camp_display['ROAS'] = camp_display['ROAS'].map(lambda x: f"{x:.2f}x" if pd.notna(x) else "—")
    st.dataframe(camp_display, use_container_width=True)
    csv = camp.to_csv(index=False)
    st.download_button("Download campaign data (filtered)", csv, file_name="campaigns_filtered.csv", mime="text/csv")
else:
    st.info("No campaigns found for current filters.")

st.markdown("---")

# -------------------------
# Quick diagnostics / correlations
# -------------------------
st.subheader("Quick diagnostics")
if not combined_filtered.empty and combined_filtered['spend'].std() and combined_filtered['orders'].std():
    corr_spend_orders = combined_filtered['spend'].corr(combined_filtered['orders'])
    corr_attr_total = combined_filtered['attributed_revenue'].corr(combined_filtered['total_revenue'])
    st.write(f"- Correlation (daily) Spend ↔ Orders: **{corr_spend_orders:.2f}**")
    st.write(f"- Correlation (daily) Attributed Marketing Revenue ↔ Business Revenue: **{corr_attr_total:.2f}**")
else:
    st.info("Not enough combined daily data to compute diagnostics.")

st.markdown("---")
st.caption("Design choices: light theme, pastel palette, focused KPIs and charts for quick decisions. Use sidebar filters to isolate problems/opportunities.")
