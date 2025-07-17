import os
from datetime import datetime
import streamlit as st
import pandas as pd
import re

import os

PARQUET_FILE = "enriched_output.parquet"

st.set_page_config(layout="wide")

# Constants
PARQUET_FILE = "enriched_output.parquet"
USEFUL_COLUMNS = [
    "booking_date_datetime", "Date", "net_premium", "gross_premium",
    "Month", "Grouping", "segment", "Insurer", "State", "Day",
    "net_retention", "retention%", "policy_type", "case_type", "Zone", 
    "Policy Info|odPremium", "Policy Info|tpPremium", "vehicle_type_plan_name"
]

MONTH_ORDER = [
    "April", "May", "June", "July", "August", "September",
    "October", "November", "December", "January", "February", "March"
]
SEGMENT_ORDER = ["Car", "Bike", "CV", "School Bus"]
GROUPING_ORDER = [f"G{i}" for i in range(1, 12)]

# -------------------------------
# Load and Cache Data
# -------------------------------
@st.cache_data(persist="disk")
def load_data(mod_time):
    df = pd.read_parquet(PARQUET_FILE, columns=USEFUL_COLUMNS)

    df["booking_date_datetime"] = pd.to_datetime(df["booking_date_datetime"], errors="coerce")
    df["segment"] = df["segment"].astype(str).str.strip()
    df["Insurer"] = df["Insurer"].astype(str).str.strip()
    df["Grouping"] = df["Grouping"].astype(str).str.strip()
    df["vehicle_type_plan_name"] = df["vehicle_type_plan_name"].astype(str).str.strip()
    df["Month"] = pd.Categorical(df["Month"], categories=MONTH_ORDER, ordered=True)
    df["Grouping"] = pd.Categorical(df["Grouping"], categories=GROUPING_ORDER, ordered=True)
    df["segment"] = pd.Categorical(df["segment"], categories=SEGMENT_ORDER, ordered=True)
    df["net_premium"] = pd.to_numeric(df["net_premium"], errors="coerce").fillna(0)
    df["gross_premium"] = pd.to_numeric(df["gross_premium"], errors="coerce").fillna(0)
    df["net_retention"] = pd.to_numeric(df["net_retention"], errors="coerce").fillna(0)
    df["policy_type"] = df["policy_type"].astype(str).str.strip()
    df["case_type"] = df["case_type"].astype(str).str.strip()
    df["Zone"] = df["Zone"].astype(str).str.strip()
    df["Policy Info|odPremium"] = pd.to_numeric(df["Policy Info|odPremium"], errors="coerce").fillna(0)
    df["Policy Info|tpPremium"] = pd.to_numeric(df["Policy Info|tpPremium"], errors="coerce").fillna(0)
    return df[df["segment"].isin(SEGMENT_ORDER)]

mod_time = os.path.getmtime(PARQUET_FILE)  # Get last modified time
df = load_data(mod_time)  # Main filtered dataframe
# -------------------------------
# Total Caluclation
# -------------------------------
def add_totals_row_col(df):
    if df.empty or df.shape[1] == 0:
        return df
    df.loc["Total"] = df.sum(numeric_only=True)
    df["Row Total"] = df.sum(axis=1, numeric_only=True)
    return df
    
# -------------------------------
# Sidebar Filter Setup
# -------------------------------
if "reset_filters" not in st.session_state:
    st.session_state.reset_filters = False

def reset_sidebar():
    st.session_state.reset_filters = True
    for k in ["selected_months", "selected_groupings", "selected_segments", "selected_insurers", "selected_states"]:
        st.session_state.pop(k, None)

if st.sidebar.button("🔄 Clear Filters", on_click=reset_sidebar):
    st.rerun()

st.sidebar.title("📊 Filter Options")

# Date filter
time_filter = st.sidebar.radio("Select Time Filter", ["LMTD vs MTD", "Custom Day Range", "FTD vs FTD-1", "Week on Week"])
today = datetime.now().date()
selected_day = today.day
selected_week = today.isocalendar()[1]  # ISO week number
prev_week = today.isocalendar().week - 1
current_week = today.isocalendar().week

if time_filter == "Custom Day Range":
    start_day = st.sidebar.number_input("Start Day (1–31)", 1, 31, value=1)
    end_day = st.sidebar.number_input("End Day (1–31)", 1, 31, value=selected_day)
    df_time = df[(df["Day"] >= start_day) & (df["Day"] <= end_day)]
    date_label = f"Custom Range: Day {start_day} to {end_day}"

elif time_filter == "FTD vs FTD-1":
    ftd_day = selected_day - 1
    prev_day = selected_day - 2
    df_time = df[df["Day"].isin([prev_day, ftd_day])]
    date_label = f"FTD vs FTD-1 (Day {prev_day} vs {ftd_day})"

elif time_filter == "Week on Week":
    df["booking_week"] = df["booking_date_datetime"].dt.isocalendar().week
    current_week = today.isocalendar().week
    prev_week = current_week - 1
    df_time = df[df["booking_week"].isin([prev_week, current_week])]
    date_label = f"Week on Week (Week {prev_week} vs Week {current_week})"

else:  # MTD
    df_time = df[df["Day"] <= selected_day]
    date_label = f"LMTD vs MTD (Till Day {selected_day})"

# Month filter logic
available_months = df["Month"].dropna().unique().tolist()
month_options_sorted = [m for m in MONTH_ORDER if m in available_months]

# Default last 2 months (LMTD vs MTD)
current_month = datetime.now().strftime("%B")
if current_month in MONTH_ORDER:
    current_idx = MONTH_ORDER.index(current_month)
    last_month = MONTH_ORDER[current_idx - 1] if current_idx > 0 else None
    default_months = [m for m in [last_month, current_month] if m in month_options_sorted]
else:
    default_months = month_options_sorted[-2:]

def multiselect_with_reset(label, options, key, fallback):
    return st.sidebar.multiselect(label, ["All"] + options, default=st.session_state.get(key, ["All"]), key=key)

month_selection = multiselect_with_reset("📅 Filter by Month(s)", month_options_sorted, "selected_months", default_months)

if "All" in month_selection or not month_selection:
    selected_months = default_months
else:
    selected_months = month_selection

# Other filters
selected_groupings = st.sidebar.multiselect("📌 Filter by Grouping", ["All"] + GROUPING_ORDER, default=["All"])
selected_segments = st.sidebar.multiselect("🔍 Filter by Segment", ["All"] + SEGMENT_ORDER, default=["All"])
selected_insurers = st.sidebar.multiselect("🏢 Filter by Insurance Company", ["All"] + sorted(df["Insurer"].dropna().unique().tolist()), default=["All"])
selected_states = st.sidebar.multiselect("🌍 Filter by State", ["All"] + sorted(df["State"].dropna().unique().tolist()), default=["All"])
selected_zones = st.sidebar.multiselect("🏙️ Filter by Zone", ["All"] + sorted(df["Zone"].dropna().unique().tolist()), default=["All"])



def resolve_filter(selection, full_list):
    return full_list if "All" in selection or not selection else selection

selected_groupings = resolve_filter(selected_groupings, GROUPING_ORDER)
selected_segments = resolve_filter(selected_segments, SEGMENT_ORDER)
selected_insurers = resolve_filter(selected_insurers, df["Insurer"].unique().tolist())
selected_states = resolve_filter(selected_states, df["State"].unique().tolist())
selected_zones = resolve_filter(selected_zones, df["Zone"].unique().tolist())
# Final filtered data
st.session_state.reset_filters = False
df_filtered = df_time[
    (df_time["Grouping"].isin(selected_groupings)) &
    (df_time["segment"].isin(selected_segments)) &
    (df_time["Insurer"].isin(selected_insurers)) &
    (df_time["State"].isin(selected_states))&
    (df_time["Zone"].isin(selected_zones))
].copy()

# Apply Month filter only when applicable
if time_filter in ["LMTD vs MTD", "Custom Day Range"]:
    df_filtered = df_filtered[df_filtered["Month"].isin(selected_months)]
if time_filter == "Week on Week":
    df_filtered["Week"] = df_filtered["booking_date_datetime"].dt.isocalendar().week
elif time_filter == "FTD vs FTD-1":
    df_filtered["Day"] = df_filtered["booking_date_datetime"].dt.day
# -------------------------------
# Formatting Helpers
# -------------------------------
def format_inr_val(x):
    if pd.isnull(x): return "-"
    try: return f"₹{int(round(x)):,}"
    except: return x

def safe_format(val, pct=True):
    try:
        if pd.isnull(val): return "-"
        return f"{val:.2f}%" if pct else f"₹{int(round(val)):,}"
    except:
        return "-"

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Grouping × Segment",
    "📊 Insurer-wise",
    "📈 Insurer × Segment",
    "📉 MoM Change View",
    "📍 VLI Tracking",
    "🏙️ Insurer × Zone",
])

# -------------------------------
# Tab 1: Grouping × Segment
# -------------------------------
with tab1:
    st.markdown("#### 📊 Retention% and Net Premium by Grouping & Segment")
    st.markdown(f"**🔎 View:** {date_label}")

    total_prem = df_filtered["net_premium"].sum()
    total_ret = df_filtered["net_retention"].sum()
    avg_ret = (total_ret / total_prem * 100) if total_prem > 0 else 0

    with st.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("📦 Total Net Premium", format_inr_val(total_prem))
        col2.metric("📈 Total Net Retention", format_inr_val(total_ret))
        col3.metric("💹 Avg Retention%", f"{avg_ret:.2f}%")


    # Set grouping and column index
    if time_filter == "Week on Week":
        group_fields = ["Grouping", "segment", "Week"]
        col_field = ["segment", "Week"]
        col_idx = pd.MultiIndex.from_product([selected_segments, [prev_week, current_week]], names=col_field)
    elif time_filter == "FTD vs FTD-1":
        group_fields = ["Grouping", "segment", "Day"]
        col_field = ["segment", "Day"]
        col_idx = pd.MultiIndex.from_product([selected_segments, [selected_day - 2, selected_day - 1]], names=col_field)
    else:
        group_fields = ["Grouping", "segment", "Month"]
        col_field = ["segment", "Month"]
        col_idx = pd.MultiIndex.from_product([selected_segments, selected_months], names=col_field)

    # Retention%
    df_ret_group = df_filtered.groupby(group_fields, observed=True).agg({
        "net_retention": "sum",
        "net_premium": "sum"
    }).reset_index()
    df_ret_group["Retention%"] = (df_ret_group["net_retention"] / df_ret_group["net_premium"]).where(df_ret_group["net_premium"] > 0, 0) * 100

    pivot_ret = df_ret_group.pivot(index="Grouping", columns=col_field, values="Retention%")
    pivot_ret = pivot_ret.reindex(columns=col_idx).reindex(index=selected_groupings)
    pivot_ret.columns = [' - '.join(map(str, col)).strip() for col in pivot_ret.columns.values]  # flatten
    st.subheader("🔹 Retention%")
    st.dataframe(pivot_ret.map(lambda x: safe_format(x, pct=True)), use_container_width=True)

    # Net Premium
    pivot_prem = pd.pivot_table(
        df_filtered,
        index="Grouping",
        columns=col_field,
        values="net_premium",
        aggfunc="sum",
        fill_value=0,
        observed=True
    ).reindex(columns=col_idx).reindex(index=selected_groupings)

    pivot_prem = add_totals_row_col(pivot_prem)
    pivot_prem.columns = [' - '.join(map(str, col)).strip() for col in pivot_prem.columns.values]  # flatten
    st.subheader("🔹 Net Premium")
    st.dataframe(pivot_prem.map(format_inr_val), use_container_width=True)

    st.subheader("🔹 Net Retention (Grouping × Segment × Month)")
    pivot_retval = pd.pivot_table(
        df_filtered,
        index="Grouping",
        columns=["segment", "Month"],
        values="net_retention",
        aggfunc="sum",
        fill_value=0,
        observed=True
    ).reindex(columns=col_idx).reindex(index=selected_groupings)

    pivot_retval = add_totals_row_col(pivot_retval)
    st.dataframe(pivot_retval.map(format_inr_val), use_container_width=True)


# -------------------------------
# Tab 2: Insurer × Month / Week / Day
# -------------------------------
with tab2:
    st.markdown("#### 📊 Retention%, Net Premium and Net Retention by Insurer × Time")
    st.markdown(f"**🔎 View:** {date_label}")

    # Determine grouping column
    if time_filter == "Week on Week":
        time_col = "Week"
        time_values = [prev_week, selected_week]
    elif time_filter == "FTD vs FTD-1":
        time_col = "Day"
        time_values = [selected_day - 2, selected_day - 1]
    else:
        time_col = "Month"
        time_values = selected_months

    # --------------------
    # Retention% Table
    # --------------------
    st.subheader(f"🔹 Retention% (Insurer × {time_col})")

    df_ret = df_filtered.groupby(["Insurer", time_col], observed=True).agg({
        "net_retention": "sum",
        "net_premium": "sum"
    }).reset_index()
    df_ret["Retention%"] = (df_ret["net_retention"] / df_ret["net_premium"]).where(df_ret["net_premium"] > 0, 0) * 100

    pivot_ret = df_ret.pivot(index="Insurer", columns=time_col, values="Retention%")
    pivot_ret = pivot_ret[[v for v in time_values if v in pivot_ret.columns]]
    pivot_ret.columns = [str(col) for col in pivot_ret.columns]
    st.dataframe(pivot_ret.map(lambda x: safe_format(x, pct=True)), use_container_width=True)

    # --------------------
    # Net Premium Table
    # --------------------
    st.subheader(f"🔹 Net Premium (Insurer × {time_col})")

    df_prem = df_filtered.groupby(["Insurer", time_col], observed=True)["net_premium"].sum().reset_index()
    pivot_prem = df_prem.pivot(index="Insurer", columns=time_col, values="net_premium")
    pivot_prem = pivot_prem[[v for v in time_values if v in pivot_prem.columns]]
    pivot_prem = add_totals_row_col(pivot_prem)
    pivot_prem.columns = [str(col) for col in pivot_prem.columns]
    st.dataframe(pivot_prem.map(format_inr_val), use_container_width=True)

    # --------------------
    # Net Retention Table (NEW)
    # --------------------
    st.subheader(f"🔹 Net Retention (Insurer × {time_col})")

    df_retval = df_filtered.groupby(["Insurer", time_col], observed=True)["net_retention"].sum().reset_index()
    pivot_retval = df_retval.pivot(index="Insurer", columns=time_col, values="net_retention")
    pivot_retval = pivot_retval[[v for v in time_values if v in pivot_retval.columns]]
    pivot_retval = add_totals_row_col(pivot_retval)
    pivot_retval.columns = [str(col) for col in pivot_retval.columns]
    st.dataframe(pivot_retval.map(format_inr_val), use_container_width=True)

# -------------------------------
# Tab 3: Insurer × Segment × Month
# -------------------------------
with tab3:
    st.markdown("#### 📊 Retention% and Net Premium by Insurer × Segment × Time")
    st.markdown(f"**🔎 View:** {date_label}")

    # Determine time column
    if time_filter == "Week on Week":
        time_col = "Week"
        time_values = [prev_week, selected_week]
    elif time_filter == "FTD vs FTD-1":
        time_col = "Day"
        time_values = [selected_day - 2, selected_day - 1]
    else:
        time_col = "Month"
        time_values = selected_months

    col_idx = pd.MultiIndex.from_product([selected_segments, time_values], names=["segment", time_col])

    # --------------------
    # Retention% Table
    # --------------------
    st.subheader(f"🔹 Retention% (Insurer × Segment × {time_col})")

    df_ret = df_filtered.groupby(["Insurer", "segment", time_col], observed=True).agg({
        "net_retention": "sum",
        "net_premium": "sum"
    }).reset_index()
    df_ret["Retention%"] = (df_ret["net_retention"] / df_ret["net_premium"]).where(df_ret["net_premium"] > 0, 0) * 100

    pivot_ret = df_ret.pivot(index="Insurer", columns=["segment", time_col], values="Retention%")
    pivot_ret = pivot_ret.reindex(columns=col_idx)
    pivot_ret.columns = [' - '.join(map(str, col)).strip() for col in pivot_ret.columns]
    st.dataframe(pivot_ret.map(lambda x: safe_format(x, pct=True)), use_container_width=True)

    # --------------------
    # Net Premium Table
    # --------------------
    st.subheader(f"🔹 Net Premium (Insurer × Segment × {time_col})")

    df_prem = df_filtered.groupby(["Insurer", "segment", time_col], observed=True)["net_premium"].sum().reset_index()
    pivot_prem = df_prem.pivot(index="Insurer", columns=["segment", time_col], values="net_premium")
    pivot_prem = pivot_prem.reindex(columns=col_idx)
    pivot_prem = add_totals_row_col(pivot_prem)
    pivot_prem.columns = [' - '.join(map(str, col)).strip() for col in pivot_prem.columns]
    st.dataframe(pivot_prem.map(format_inr_val), use_container_width=True)

    st.subheader("🔹 Net Retention (Insurer × Segment × Month)")
    df_retval = df_filtered.groupby(["Insurer", "segment", "Month"], observed=True)["net_retention"].sum().reset_index()
    pivot_retval = df_retval.pivot(index="Insurer", columns=["segment", "Month"], values="net_retention")
    pivot_retval = pivot_retval.reindex(columns=col_idx)
    pivot_retval = add_totals_row_col(pivot_retval)
    st.dataframe(pivot_retval.map(format_inr_val), use_container_width=True)

# -------------------------------
# Tab 4: MoM Comparison
# -------------------------------
with tab4:
    st.markdown("#### 📉 Change View: Net Premium & Retention%")
    st.markdown(f"**🔎 View:** {date_label}")

    # Select dynamic time column and label
    if time_filter == "Week on Week":
        time_col = "Week"
        time_values = [prev_week, selected_week]
    elif time_filter == "FTD vs FTD-1":
        time_col = "Day"
        time_values = [selected_day - 2, selected_day - 1]
    else:
        time_col = "Month"
        time_values = selected_months[-2:] if len(selected_months) >= 2 else selected_months

    if len(time_values) < 2:
        st.warning("⚠️ Need at least 2 time periods to compare.")
    else:
        t1, t2 = time_values[0], time_values[1]
        df_comp = df_filtered[df_filtered[time_col].isin([t1, t2])].copy()

        # Make sure both time values are present in data
        actual_values = sorted(df_comp[time_col].dropna().unique().tolist())
        if len(actual_values) < 2:
            st.warning("⚠️ Not enough data for selected time comparison.")
        else:
            col_idx = pd.MultiIndex.from_product([selected_segments, [t1, t2]], names=["segment", time_col])

            # Net Premium Pivot
            prem_pivot = pd.pivot_table(
                df_comp,
                index="Grouping",
                columns=["segment", time_col],
                values="net_premium",
                aggfunc="sum",
                fill_value=0,
                observed=True
            ).reindex(columns=col_idx).reindex(index=selected_groupings)

            # Retention% Pivot
            df_comp["retention%"] = (df_comp["net_retention"] / df_comp["net_premium"]).where(df_comp["net_premium"] > 0, 0) * 100
            ret_pivot = pd.pivot_table(
                df_comp,
                index="Grouping",
                columns=["segment", time_col],
                values="retention%",
                aggfunc="mean",
                fill_value=0,
                observed=True
            ).reindex(columns=col_idx).reindex(index=selected_groupings)

            # Delta Tables
            diff_data = {}
            for seg in selected_segments:
                diff_data[(seg, "Premium Δ")] = prem_pivot.get((seg, t2), 0) - prem_pivot.get((seg, t1), 0)
                diff_data[(seg, "Retention% Δ")] = ret_pivot.get((seg, t2), 0) - ret_pivot.get((seg, t1), 0)

            df_diff = pd.DataFrame(diff_data, index=selected_groupings)

            # Display Format
            display_df = pd.DataFrame()
            for col in df_diff.columns:
                is_pct = "Retention" in col[1]
                display_df[col] = df_diff[col].map(lambda x: safe_format(x, pct=is_pct))

            display_df.columns = [f"{seg} - {metric}" for seg, metric in display_df.columns]
            display_df.index.name = "Grouping"

            def highlight_diff(val):
                try:
                    val = float(str(val).replace("₹", "").replace(",", "").replace("%", ""))
                    if val > 0:
                        return "color: green;"
                    elif val < 0:
                        return "color: red;"
                except:
                    return ""
                return ""

            st.dataframe(display_df.style.map(highlight_diff), use_container_width=True)

# -------------------------------
# Tab 5: VLI Tracking — Actuals till last month, Estimate till Mar
# -------------------------------
import os
import re
import pandas as pd
import streamlit as st
from datetime import datetime
from io import StringIO

with tab5:
    st.markdown('#### 📍 VLI Tracking Table (Apr–Mar View, with Estimations and Payouts)')

    @st.cache_data(show_spinner=True)
    def load_motor_mis_data(folder='motor_mis_monthly_data'):
        usecols = ['booking_date_ist', 'company', 'vehicle_type', 'vehicle_sub_type', 'policy_type',
                   'net_premium', 'final_od', 'total_tp_premium', 'sbi_segment', 'rto_4', 'Sales_Channel']
        dfs = []
        for file in sorted(os.listdir(folder)):
            if file.endswith('.csv') and '_' in file:
                try:
                    df = pd.read_csv(os.path.join(folder, file), usecols=usecols)
                    month_str, yy = file.replace('.csv', '').split('_')
                    df['Month'] = month_str[:3].title()
                    df['Year'] = int('20' + yy) if len(yy) == 2 else int(yy)
                    dfs.append(df)
                except:
                    pass
        df = pd.concat(dfs, ignore_index=True)
        df = df.rename(columns={
            'company': 'Insurer',
            'final_od': 'Policy Info|odPremium',
            'total_tp_premium': 'Policy Info|tpPremium'
        })
        df['policy_type'] = df['policy_type'].astype(str).str.lower().str.replace(' ', '_')
        df['vehicle_type'] = df['vehicle_type'].astype(str).str.lower().str.strip()
        df['vehicle_sub_type'] = df['vehicle_sub_type'].astype(str).str.lower().str.strip()
        df['segment'] = df['vehicle_type']
        return df

    df_full = load_motor_mis_data()

    col1, col2, col3 = st.columns([2, 3, 2])
    with col1:
        growth_rate = st.number_input("📈 Growth Rate for Estimation (%)", 0, 100, 10, step=1) / 100
    with col2:
        sales_channel_options = df_full["Sales_Channel"].dropna().unique().tolist()
        selected_sales_channels = st.multiselect("🎯 Filter by Sales Channel", sales_channel_options, default=sales_channel_options)
    with col3:
        download_placeholder = st.empty()

    df_full = df_full[df_full["Sales_Channel"].isin(selected_sales_channels)]

    fiscal_months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar']
    current_month = datetime.now().strftime('%b')
    current_year = datetime.now().year
    current_month_idx = fiscal_months.index(current_month)

    try:
        rules_df = pd.read_csv('mapping_data/vli_tracking.csv')
    except:
        st.stop()

    @st.cache_data
    def load_preferred_less_mapping():
        return pd.read_excel('mapping_data/Preferred - Less Preferred RTO A-B Category list.xlsx')

    @st.cache_data
    def load_rto_tagging_master():
        return pd.read_excel('mapping_data/RTO Category Tagging Master.xlsb', engine='pyxlsb')

    preferred_less_df = load_preferred_less_mapping()
    rto_tagging_df = load_rto_tagging_master()

    def extract_values(val, col_values):
        if pd.isna(val) or str(val).strip().lower() == 'all':
            return col_values.unique().tolist()
        return [v.strip().lower().replace(' ', '_') for v in str(val).split(',')]

    def parse_target(val):
        if pd.isna(val) or val.strip() == '':
            return '', ''
        parts = [x.strip() for x in val.split(',')]
        return (parts[0], parts[1]) if len(parts) == 2 else (parts[0], '')

    def format_inr(val):
        try:
            return f'₹{val:,.0f}'
        except:
            return val

    def compute_payout(total_val, rule_row):
        max_thresh = 0
        payout_percent = 0
        for i in range(1, 5):
            thresh, reward = parse_target(rule_row.get(f'Target_{i}', ''))
            try:
                if thresh:
                    thresh_val = float(thresh) * 1e7
                    if total_val >= thresh_val and thresh_val > max_thresh:
                        max_thresh = thresh_val
                        payout_percent = float(str(reward).replace('%', '').strip()) / 100
            except:
                continue
        return total_val * payout_percent

    fiscal_years = {}
    for year in sorted(df_full['Year'].unique()):
        fy_label = f'FY {year}-{str(year+1)[-2:]}'
        months = [(mon, year) for mon in fiscal_months[:9]] + [(mon, year+1) for mon in fiscal_months[9:]]
        fiscal_years[fy_label] = months

    def normalize_segment(s):
        return re.sub(r'[^a-z]', '', str(s).lower().strip())

    all_rows = []

    for insurer_name, group in rules_df.groupby('Insurer'):
        insurer_name_clean = insurer_name.strip().lower()
        with st.container():
            st.markdown(f"### 🏢 {insurer_name}")
            rows = []
            for _, rule in group.iterrows():
                premium_type = str(rule['premium']).strip().lower()
                premium_col_map = {'net': 'net_premium', 'od': 'Policy Info|odPremium', 'tp': 'Policy Info|tpPremium'}
                premium_col = premium_col_map.get(premium_type)
                if premium_col not in df_full.columns:
                    continue

                insurer_mask = df_full['Insurer'].apply(lambda x: insurer_name_clean in str(x).lower())
                normalized_segment = normalize_segment(rule['segment'])

                if insurer_name_clean == 'sbi' and normalized_segment == 'cvpreferred':
                    pref_rows = preferred_less_df[
                        preferred_less_df['Preferred/Less Preferred'].str.lower().str.strip() == 'preferred'
                    ]
                    merged = pref_rows.merge(
                        rto_tagging_df,
                        left_on=['RTO State', 'RTO Category'],
                        right_on=['RTO State', 'Cat.'],
                        how='inner'
                    )
                    lob_list = pref_rows['LOB'].dropna().str.strip().str.lower().unique().tolist()
                    rto_codes = merged['RTO_CODE'].dropna().astype(str).str.strip().unique().tolist()
                    df_match = df_full[
                        insurer_mask &
                        df_full['sbi_segment'].astype(str).str.strip().str.lower().isin(lob_list) &
                        df_full['rto_4'].astype(str).str.strip().isin(rto_codes)
                    ].copy()

                elif insurer_name_clean == 'sbi' and normalized_segment == 'cvlesspreferred':
                    pref_rows = preferred_less_df[
                        preferred_less_df['Preferred/Less Preferred'].str.lower().str.strip() == 'less preferred'
                    ]
                    merged = pref_rows.merge(
                        rto_tagging_df,
                        on='RTO State',
                        how='inner'
                    )
                    lob_list = pref_rows['LOB'].dropna().str.strip().str.lower().unique().tolist()
                    rto_codes = merged['RTO_CODE'].dropna().astype(str).str.strip().unique().tolist()
                    df_match = df_full[
                        insurer_mask &
                        df_full['sbi_segment'].astype(str).str.strip().str.lower().isin(lob_list) &
                        df_full['rto_4'].astype(str).str.strip().isin(rto_codes)
                    ].copy()

                else:
                    segment_vals = extract_values(rule['segment'], df_full['segment'])
                    policy_vals = extract_values(rule['policy_type'], df_full['policy_type'])
                    if any('school_bus' in seg for seg in segment_vals):
                        segment_mask = (df_full['vehicle_type'] == 'pcv') & (df_full['vehicle_sub_type'] == 'bus')
                    else:
                        segment_mask = df_full['segment'].isin(segment_vals)
                    policy_mask = df_full['policy_type'].isin(policy_vals)
                    df_match = df_full[insurer_mask & segment_mask & policy_mask].copy()

                if df_match.empty:
                    continue

                monthly_actuals = df_match.groupby(['Month', 'Year'], observed=True)[premium_col].sum()

                row = {
                    'Insurer': insurer_name,
                    'Segment': rule['segment'],
                    'Policy Type': rule['policy_type'],
                    'Premium Type': premium_type.upper()
                }
                for i in range(1, 5):
                    thresh, pct = parse_target(rule.get(f'Target_{i}', ''))
                    row[f'Target_{i}_Threshold'] = thresh
                    row[f'Target_{i}_Payout'] = pct

                for fy_label in sorted(fiscal_years.keys(), reverse=True):
                    total_val = 0
                    prev_val = 0
                    for mon, yr in fiscal_years[fy_label]:
                        is_future = (yr > current_year) or (yr == current_year and fiscal_months.index(mon) >= current_month_idx)
                        val = monthly_actuals.get((mon, yr), 0)
                        if not is_future:
                            prev_val = val
                            row[f'{mon}_{str(yr)[-2:]}'] = format_inr(val)
                        else:
                            if prev_val == 0:
                                row[f'{mon}_{str(yr)[-2:]}'] = format_inr(0)
                                continue
                            prev_val *= (1 + growth_rate)
                            row[f'{mon}_{str(yr)[-2:]}'] = format_inr(prev_val)
                            val = prev_val
                        total_val += val
                    row[f'Total {fy_label}'] = format_inr(total_val)
                    row[f'Payout {fy_label}'] = format_inr(compute_payout(total_val, rule))

                rows.append(row)
                all_rows.append(row)

            if rows:
                df_display = pd.DataFrame(rows)
                st.dataframe(df_display, use_container_width=True)

    if all_rows:
        df_all = pd.DataFrame(all_rows)
        df_download = df_all.copy()
        for col in df_download.columns:
            if df_download[col].dtype == object:
                df_download[col] = df_download[col].str.replace("₹", "").str.replace(",", "")
        buf = StringIO()
        df_download.to_csv(buf, index=False)
        download_placeholder.download_button(
            label='⬇️ Download Full VLI Table',
            data=buf.getvalue(),
            file_name='vli_tracking.csv',
            mime='text/csv',
            use_container_width=True
        )
    else:
        st.info('ℹ️ No matching rules/data found.')


# -------------------------------
# Tab 6: ZonexSegment
# -------------------------------
with tab6:
    st.markdown("#### 🏙️ Retention%, Net Premium and Net Retention by Insurer × Zone")
    st.markdown(f"**🔎 View:** {date_label}")

    # Determine time column and values
    if time_filter == "Week on Week":
        time_col = "Week"
        time_values = [prev_week, selected_week]
    elif time_filter == "FTD vs FTD-1":
        time_col = "Day"
        time_values = [selected_day - 2, selected_day - 1]
    else:
        time_col = "Month"
        time_values = selected_months

    # ---------------------------
    # Retention% Table
    # ---------------------------
    st.subheader("🔹 Retention% (Insurer × Zone × Time)")

    df_ret = df_filtered.groupby(["Insurer", "Zone", time_col], observed=True).agg({
        "net_retention": "sum",
        "net_premium": "sum"
    }).reset_index()
    df_ret["Retention%"] = (df_ret["net_retention"] / df_ret["net_premium"]).where(df_ret["net_premium"] > 0, 0) * 100

    pivot_ret = df_ret.pivot(index="Insurer", columns=["Zone", time_col], values="Retention%")
    pivot_ret = pivot_ret.fillna(0)

    # Reorder columns: Sort Zone → Time
    pivot_ret = pivot_ret.reindex(
        columns=sorted(pivot_ret.columns, key=lambda x: (str(x[0]), x[1]))
    )

    st.dataframe(pivot_ret.map(lambda x: safe_format(x, pct=True)), use_container_width=True)

    # ---------------------------
    # Net Premium Table
    # ---------------------------
    st.subheader("🔹 Net Premium (Insurer × Zone × Time)")

    df_prem = df_filtered.groupby(["Insurer", "Zone", time_col], observed=True)["net_premium"].sum().reset_index()
    pivot_prem = df_prem.pivot(index="Insurer", columns=["Zone", time_col], values="net_premium")
    pivot_prem = add_totals_row_col(pivot_prem.fillna(0))
    pivot_prem = pivot_prem.reindex(
        columns=sorted(pivot_prem.columns, key=lambda x: (str(x[0]), x[1]))
    )

    st.dataframe(pivot_prem.map(format_inr_val), use_container_width=True)

    # ---------------------------
    # Net Retention Table
    # ---------------------------
    st.subheader("🔹 Net Retention (Insurer × Zone × Time)")

    df_retval = df_filtered.groupby(["Insurer", "Zone", time_col], observed=True)["net_retention"].sum().reset_index()
    pivot_retval = df_retval.pivot(index="Insurer", columns=["Zone", time_col], values="net_retention")
    pivot_retval = add_totals_row_col(pivot_retval.fillna(0))
    pivot_retval = pivot_retval.reindex(
        columns=sorted(pivot_retval.columns, key=lambda x: (str(x[0]), x[1]))
    )

    st.dataframe(pivot_retval.map(format_inr_val), use_container_width=True)


# -------------------------------
# Export
# -------------------------------
st.sidebar.markdown("### 📥 Export")
st.sidebar.download_button(
    label="Download Filtered CSV",
    data=df_filtered.to_csv(index=False),
    file_name="filtered_data.csv",
    mime="text/csv"
)