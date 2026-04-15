import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import geopandas as gpd
import plotly.express as px
import json
from pathlib import Path

from Data_cleaning_Syria import (
    load_clean_acled,
    load_clean_food,
    build_conflict_panel,
    merge_food
)

# ----------------------------
# GLOBAL PROJECT PATH
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent

# ----------------------------
# PAGE SETUP
# ----------------------------
st.set_page_config(
    page_title="Syria Conflict Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Syria Conflict Dashboard")
st.caption(
    "Track protests, violence, food prices, and district-level regression results across Syrian regions."
)

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    acled_path = BASE_DIR / "data" / "acled_syria.csv"
    food_path = BASE_DIR / "data" / "wfp_food_prices_syr.csv"

    if not acled_path.exists():
        st.error(f"Missing file: {acled_path}")
        st.stop()

    if not food_path.exists():
        st.error(f"Missing file: {food_path}")
        st.stop()

    acled = load_clean_acled(acled_path)
    food = load_clean_food(food_path)
    panel = build_conflict_panel(acled)
    return panel, food


@st.cache_data
def load_map():
    shp_path = BASE_DIR / "shapefiles" / "syr_admin_boundaries" / "syr_admin1.shp"

    if not shp_path.exists():
        st.error(f"Missing shapefile: {shp_path}")
        st.stop()

    gdf = gpd.read_file(shp_path)

    if "admin1Name" in gdf.columns:
        name_col = "admin1Name"
    elif "shapeName" in gdf.columns:
        name_col = "shapeName"
    else:
        name_col = gdf.columns[0]

    gdf["admin1"] = gdf[name_col].replace({
        "Latakia": "Lattakia",
        "Idlib": "Idleb",
        "As-Suwayda": "As-Sweida",
        "Deir ez-Zor": "Deir Ez-Zor",
        "Deir-ez-Zor": "Deir Ez-Zor",
        "Al-Hasakah": "Al Hasakah",
        "Al-Hasakeh": "Al Hasakah",
        "Dar'a": "Dara",
        "Tartus": "Tartous",
        "Damascus Countryside": "Rural Damascus",
        "Rif Dimashq": "Rural Damascus",
        "Qunaitra": "Quneitra"
    })

    return gdf.to_crs(epsg=4326)


df_panel_base, food = load_data()
gdf = load_map()

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.header("Controls")

metric = st.sidebar.selectbox(
    "Select Metric",
    [
        "Protests",
        "Civilian Targeting",
        "Battles",
        "Remote Violence",
        "Riots",
        "Food Prices",
        "Food Price Change",
        "Protests vs Food Prices",
        "Protests vs Food Price Change",
        "Protests vs Food Price Change (Lagged)",
        "Protests vs Civilian Targeting (Lagged)",
        "Total Protests by Repression",
        "Total Protests by Price Change (Binned by Repression)"
    ]
)

food_metrics = {
    "Food Prices",
    "Food Price Change",
    "Protests vs Food Prices",
    "Protests vs Food Price Change",
    "Protests vs Food Price Change (Lagged)",
    "Total Protests by Price Change (Binned by Repression)"
}

food_categories = sorted(food["category"].dropna().unique())
selected_category = "All categories"

if metric in food_metrics:
    selected_category = st.sidebar.selectbox(
        "Select Food Category",
        ["All categories"] + food_categories
    )

regions = ["Overall"] + sorted(df_panel_base["admin1"].dropna().unique())
region = st.sidebar.selectbox("Select Region", regions)

compare = False
region2 = None

if region != "Overall":
    compare = st.sidebar.checkbox("Compare with another region")
    if compare:
        region2 = st.sidebar.selectbox(
            "Select Second Region",
            [r for r in regions if r not in ["Overall", region]]
        )

year_min = int(df_panel_base["year"].min())
year_max = int(df_panel_base["year"].max())

year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max)
)

time_unit = st.sidebar.selectbox("View By", ["Yearly", "Monthly"])

map_zoom_mode = st.sidebar.radio(
    "Map View",
    ["Overall map", "Zoom to selected district"]
)

# ----------------------------
# PREP DATA
# ----------------------------
df_panel = merge_food(df_panel_base, food, selected_category)

df_panel["civilian_targeting_lag1"] = (
    df_panel.groupby("admin1")["civilian_targeting"].shift(1)
)

df_panel["price_change_lag1"] = (
    df_panel.groupby("admin1")["price_change"].shift(1)
)

df_panel["repression_level"] = pd.qcut(
    df_panel["civilian_targeting_lag1"],
    q=3,
    labels=["Low", "Medium", "High"],
    duplicates="drop"
)

df_panel["price_shock_bin"] = pd.cut(
    df_panel["price_change_lag1"],
    bins=[-float("inf"), -0.05, 0.05, float("inf")],
    labels=["Price Drop", "Stable", "Price Increase"]
)

df_filtered = df_panel[
    (df_panel["year"] >= year_range[0]) &
    (df_panel["year"] <= year_range[1])
].copy()

region_data = (
    df_filtered.copy()
    if region == "Overall"
    else df_filtered[df_filtered["admin1"] == region].copy()
)

# ----------------------------
# KPI ROW
# ----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Region", region)
k2.metric("Years", f"{year_range[0]}–{year_range[1]}")
k3.metric("View", time_unit)
k4.metric("Food Category", selected_category if metric in food_metrics else "N/A")

st.divider()

# ----------------------------
# HELPERS
# ----------------------------
def summarize(data):
    if time_unit == "Yearly":
        summary = data.groupby("year").agg({
            "protests": "sum",
            "civilian_targeting": "sum",
            "battles": "sum",
            "remote_violence": "sum",
            "riots": "sum",
            "price": "mean",
            "price_change": "mean",
            "price_change_lag1": "mean",
            "civilian_targeting_lag1": "mean"
        }).reset_index()
        x = summary["year"]
    else:
        summary = data.groupby("month").agg({
            "protests": "sum",
            "civilian_targeting": "sum",
            "battles": "sum",
            "remote_violence": "sum",
            "riots": "sum",
            "price": "mean",
            "price_change": "mean",
            "price_change_lag1": "mean",
            "civilian_targeting_lag1": "mean"
        }).reset_index()
        summary["month_str"] = summary["month"].astype(str)
        x = summary["month_str"]

    return summary, x


def format_axis(ax, x):
    if time_unit == "Monthly":
        x = list(x)
        step = max(1, len(x) // 12)
        ticks = list(range(0, len(x), step))
        labels = [x[i] for i in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45, ha="right")


def build_map_data(df):
    grouped = df.groupby("admin1", as_index=False).agg({
        "protests": "sum",
        "civilian_targeting": "sum",
        "battles": "sum",
        "remote_violence": "sum",
        "riots": "sum",
        "price": "mean",
        "price_change": "mean",
        "price_change_lag1": "mean",
        "civilian_targeting_lag1": "mean"
    })

    mapping = {
        "Protests": "protests",
        "Civilian Targeting": "civilian_targeting",
        "Battles": "battles",
        "Remote Violence": "remote_violence",
        "Riots": "riots",
        "Food Prices": "price",
        "Food Price Change": "price_change",
        "Protests vs Food Prices": "price",
        "Protests vs Food Price Change": "price_change",
        "Protests vs Food Price Change (Lagged)": "price_change_lag1",
        "Protests vs Civilian Targeting (Lagged)": "civilian_targeting_lag1",
        "Total Protests by Repression": "civilian_targeting_lag1",
        "Total Protests by Price Change (Binned by Repression)": "price_change_lag1"
    }

    return grouped, mapping.get(metric, "protests")


def get_map_view(gdf_map):
    if map_zoom_mode == "Overall map" or region == "Overall":
        centroid = gdf_map.geometry.union_all().centroid
        return centroid.y, centroid.x, 5.2

    selected = gdf_map[gdf_map["admin1"] == region]
    centroid = selected.geometry.union_all().centroid
    return centroid.y, centroid.x, 5.9


# ----------------------------
# GRAPH
# ----------------------------
st.subheader(metric)
st.caption("Chart view for selected filters.")

fig, ax = plt.subplots(figsize=(10, 5.5))

summary, x = summarize(region_data)

if metric == "Protests":
    ax.plot(x, summary["protests"], linewidth=2)
elif metric == "Civilian Targeting":
    ax.plot(x, summary["civilian_targeting"], linewidth=2)
elif metric == "Battles":
    ax.plot(x, summary["battles"], linewidth=2)
elif metric == "Remote Violence":
    ax.plot(x, summary["remote_violence"], linewidth=2)
elif metric == "Riots":
    ax.plot(x, summary["riots"], linewidth=2)
elif metric == "Food Prices":
    ax.plot(x, summary["price"], linewidth=2)
elif metric == "Food Price Change":
    ax.plot(x, summary["price_change"], linewidth=2)
elif metric == "Protests vs Food Prices":
    ax.scatter(region_data["price"], region_data["protests"], alpha=0.7)
    ax.set_xlabel("Food Price")
    ax.set_ylabel("Protests")
elif metric == "Protests vs Food Price Change":
    ax.scatter(region_data["price_change"], region_data["protests"], alpha=0.7)
    ax.set_xlabel("Food Price Change")
    ax.set_ylabel("Protests")
elif metric == "Protests vs Food Price Change (Lagged)":
    ax.scatter(region_data["price_change_lag1"], region_data["protests"], alpha=0.7)
    ax.set_xlabel("Lagged Food Price Change")
    ax.set_ylabel("Protests")
elif metric == "Protests vs Civilian Targeting (Lagged)":
    ax.scatter(region_data["civilian_targeting_lag1"], region_data["protests"], alpha=0.7)
    ax.set_xlabel("Lagged Civilian Targeting")
    ax.set_ylabel("Protests")
elif metric == "Total Protests by Repression":
    tmp = region_data.groupby("repression_level", observed=False)["protests"].sum()
    ax.bar(tmp.index.astype(str), tmp.values)
elif metric == "Total Protests by Price Change (Binned by Repression)":
    tmp = (
        region_data.groupby(
            ["price_shock_bin", "repression_level"],
            observed=False
        )["protests"]
        .sum()
        .unstack()
        .fillna(0)
    )
    tmp.plot(kind="bar", ax=ax)

if metric in {
    "Protests",
    "Civilian Targeting",
    "Battles",
    "Remote Violence",
    "Riots",
    "Food Prices",
    "Food Price Change"
}:
    format_axis(ax, x)
    ax.set_xlabel("Time")
    ax.set_ylabel(metric)

ax.set_title(f"{metric} in {region}")
ax.grid(alpha=0.2)
plt.tight_layout()

st.pyplot(fig, use_container_width=True)

st.divider()

# ----------------------------
# MAP
# ----------------------------
st.subheader(f"{metric} Map")
st.caption("Map view across districts.")

map_df, color_col = build_map_data(df_filtered)
choropleth = gdf.merge(map_df, on="admin1", how="left").copy()

geojson = json.loads(choropleth[["admin1", "geometry"]].to_json())
lat, lon, zoom = get_map_view(choropleth)

fig_map = px.choropleth_mapbox(
    choropleth,
    geojson=geojson,
    locations="admin1",
    featureidkey="properties.admin1",
    color=color_col,
    hover_name="admin1",
    color_continuous_scale="OrRd",
    mapbox_style="white-bg",
    center={"lat": lat, "lon": lon},
    zoom=zoom,
    opacity=0.82
)

fig_map.update_layout(
    height=700,
    margin={"r": 0, "t": 0, "l": 0, "b": 0}
)

st.plotly_chart(fig_map, width="stretch")

st.divider()

# ----------------------------
# REGRESSIONS
# ----------------------------
st.subheader("District-Level Regression Results")

@st.cache_data
def run_regressions(df):
    reg_df = df.dropna(subset=[
        "protests",
        "price_change_lag1",
        "civilian_targeting_lag1"
    ]).copy()

    rows = []

    for district in sorted(reg_df["admin1"].dropna().unique()):
        sub = reg_df[reg_df["admin1"] == district].copy()

        if len(sub) < 24 or sub["protests"].nunique() < 2:
            continue

        try:
            model = smf.ols(
                "protests ~ price_change_lag1 + civilian_targeting_lag1 + price_change_lag1:civilian_targeting_lag1",
                data=sub
            ).fit(cov_type="HC1")

            rows.append({
                "admin1": district,
                "n_obs": int(model.nobs),
                "r_squared": round(model.rsquared, 3),
                "coef_interaction": round(
                    model.params.get(
                        "price_change_lag1:civilian_targeting_lag1",
                        float("nan")
                    ),
                    3
                ),
                "p_interaction": round(
                    model.pvalues.get(
                        "price_change_lag1:civilian_targeting_lag1",
                        float("nan")
                    ),
                    3
                )
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


regression_results = run_regressions(df_filtered)

if regression_results.empty:
    st.warning("No regressions available.")
else:
    st.dataframe(regression_results, width="stretch")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    coef_plot = regression_results.sort_values("coef_interaction")

    ax2.barh(
        coef_plot["admin1"],
        coef_plot["coef_interaction"]
    )
    ax2.set_title("Interaction Effect by District")
    ax2.set_xlabel("Coefficient")
    ax2.set_ylabel("District")
    ax2.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
