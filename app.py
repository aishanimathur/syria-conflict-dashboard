import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import geopandas as gpd
import plotly.express as px
import json

from Data_cleaning_Syria import (
    load_clean_acled,
    load_clean_food,
    build_conflict_panel,
    merge_food
)

# ----------------------------
# PAGE SETUP
# ----------------------------
st.set_page_config(
    page_title="Syria Conflict Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Syria Conflict Dashboard")
st.caption("Track protests, violence, food prices, and district-level regression results across Syrian regions.")

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    acled = load_clean_acled("/Users/aishanimathur/Downloads/ACLED Data_2026-03-18.csv")
    food = load_clean_food("/Users/aishanimathur/Downloads/wfp_food_prices_syr (1).csv")
    panel = build_conflict_panel(acled)
    return panel, food

@st.cache_data
def load_map():
    gdf = gpd.read_file("/Users/aishanimathur/Downloads/syr_admin_boundaries.shp/syr_admin1.shp")

    if "admin1Name" in gdf.columns:
        name_col = "admin1Name"
    elif "shapeName" in gdf.columns:
        name_col = "shapeName"
    else:
        name_col = gdf.columns[0]

    gdf["admin1"] = gdf[name_col].replace({
        "Latakia": "Lattakia",
        "Idlib": "Idleb",
        "Ar-Raqqa": "Ar-Raqqa",
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
# SIDEBAR UI
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
# MERGE FOOD
# ----------------------------
df_panel = merge_food(df_panel_base, food, selected_category)

# ----------------------------
# ADD LAGS + REPRESSION LEVELS
# ----------------------------
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

# ----------------------------
# FILTER DATA
# ----------------------------
df_filtered = df_panel[
    (df_panel["year"] >= year_range[0]) &
    (df_panel["year"] <= year_range[1])
].copy()

region_data = df_filtered.copy() if region == "Overall" else df_filtered[df_filtered["admin1"] == region].copy()

# ----------------------------
# KPI ROW
# ----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Region", region)
k2.metric("Years", f"{year_range[0]}–{year_range[1]}")
k3.metric("Time View", time_unit)
k4.metric("Food Category", selected_category if metric in food_metrics else "N/A")

st.divider()

# ----------------------------
# LABEL HELPERS
# ----------------------------
def get_metric_labels(metric):
    if metric == "Protests":
        return "Total protest event count", "Total protest event count by district"
    elif metric == "Civilian Targeting":
        return "Total civilian targeting event count", "Total civilian targeting event count by district"
    elif metric == "Battles":
        return "Total battle event count", "Total battle event count by district"
    elif metric == "Remote Violence":
        return "Total remote violence event count", "Total remote violence event count by district"
    elif metric == "Riots":
        return "Total riot event count", "Total riot event count by district"
    elif metric == "Food Prices":
        return "Average food price", "Average food price by district"
    elif metric == "Food Price Change":
        return "Average food price change", "Average food price change by district"
    elif metric == "Protests vs Food Prices":
        return "Total protest count and average food price", "Average food price by district"
    elif metric == "Protests vs Food Price Change":
        return "Total protest count and average food price change", "Average food price change by district"
    elif metric == "Protests vs Food Price Change (Lagged)":
        return "Total protest count and average lagged food price change", "Average lagged food price change by district"
    elif metric == "Protests vs Civilian Targeting (Lagged)":
        return "Total protest count and average lagged civilian targeting", "Average lagged civilian targeting by district"
    elif metric == "Total Protests by Repression":
        return "Total protest event count", "Average lagged civilian targeting by district"
    elif metric == "Total Protests by Price Change (Binned by Repression)":
        return "Total protest event count", "Average lagged food price change by district"
    else:
        return "Value", "Value by district"

def get_graph_title(metric, region):
    if metric == "Protests":
        return f"Protests over Time in {region}"
    elif metric == "Civilian Targeting":
        return f"Civilian Targeting over Time in {region}"
    elif metric == "Battles":
        return f"Battles over Time in {region}"
    elif metric == "Remote Violence":
        return f"Remote Violence over Time in {region}"
    elif metric == "Riots":
        return f"Riots over Time in {region}"
    elif metric == "Food Prices":
        return f"Food Prices over Time in {region}"
    elif metric == "Food Price Change":
        return f"Food Price Change over Time in {region}"
    elif metric == "Protests vs Food Prices":
        return f"Protests and Food Prices in {region}"
    elif metric == "Protests vs Food Price Change":
        return f"Protests and Food Price Change in {region}"
    elif metric == "Protests vs Food Price Change (Lagged)":
        return f"Do Protests Rise after Food Price Shocks in {region}?"
    elif metric == "Protests vs Civilian Targeting (Lagged)":
        return f"Do Protests Rise after Repression in {region}?"
    elif metric == "Total Protests by Repression":
        return f"Total Protests by Repression in {region}"
    elif metric == "Total Protests by Price Change (Binned by Repression)":
        return f"Total Protests by Price Shock and Repression in {region}"
    return f"{metric} in {region}"

def get_graph_description(metric):
    descriptions = {
        "Protests": "This chart shows total protest events over time in the selected region.",
        "Civilian Targeting": "This chart shows total civilian targeting events over time in the selected region.",
        "Battles": "This chart shows total battle events over time in the selected region.",
        "Remote Violence": "This chart shows total remote violence events over time in the selected region.",
        "Riots": "This chart shows total riot events over time in the selected region.",
        "Food Prices": "This chart shows average food prices over time in the selected region.",
        "Food Price Change": "This chart shows average food price change over time in the selected region.",
        "Protests vs Food Prices": "Each point shows the relationship between food prices and protest counts in the selected region.",
        "Protests vs Food Price Change": "Each point shows the relationship between food price change and protest counts in the selected region.",
        "Protests vs Food Price Change (Lagged)": "Each point shows whether higher food price change in the previous period is associated with more protests later.",
        "Protests vs Civilian Targeting (Lagged)": "Each point shows whether higher civilian targeting in the previous period is associated with more protests later.",
        "Total Protests by Repression": "This chart shows total protests grouped by levels of lagged repression.",
        "Total Protests by Price Change (Binned by Repression)": "This chart shows total protests grouped by food price shock category and repression level."
    }
    return descriptions.get(metric, "This chart shows the selected metric over the chosen period.")

def get_map_title(metric):
    titles = {
        "Protests": "Protest Map",
        "Civilian Targeting": "Civilian Targeting Map",
        "Battles": "Battles Map",
        "Remote Violence": "Remote Violence Map",
        "Riots": "Riots Map",
        "Food Prices": "Food Price Map",
        "Food Price Change": "Food Price Change Map",
        "Protests vs Food Prices": "Food Price Map",
        "Protests vs Food Price Change": "Food Price Change Map",
        "Protests vs Food Price Change (Lagged)": "Lagged Food Price Change Map",
        "Protests vs Civilian Targeting (Lagged)": "Lagged Civilian Targeting Map",
        "Total Protests by Repression": "Lagged Civilian Targeting Map",
        "Total Protests by Price Change (Binned by Repression)": "Lagged Food Price Change Map"
    }
    return titles.get(metric, "Map")

def get_map_explanation(metric):
    explanations = {
        "Protests": "Map shows total protest events by district during the selected years.",
        "Civilian Targeting": "Map shows total civilian targeting events by district during the selected years.",
        "Battles": "Map shows total battle events by district during the selected years.",
        "Remote Violence": "Map shows total remote violence events by district during the selected years.",
        "Riots": "Map shows total riot events by district during the selected years.",
        "Food Prices": "Map shows average food price by district during the selected years.",
        "Food Price Change": "Map shows average food price change by district during the selected years.",
        "Protests vs Food Prices": "Map shows average food price by district during the selected years.",
        "Protests vs Food Price Change": "Map shows average food price change by district during the selected years.",
        "Protests vs Food Price Change (Lagged)": "Map shows average prior-period food price change by district.",
        "Protests vs Civilian Targeting (Lagged)": "Map shows average prior-period civilian targeting by district.",
        "Total Protests by Repression": "Chart groups protests by repression level. Map shows average prior-period civilian targeting by district.",
        "Total Protests by Price Change (Binned by Repression)": "Chart groups protests by food price shock and repression. Map shows average prior-period food price change by district."
    }
    return explanations.get(metric, "Map shows the selected metric by district.")

y_label, map_label = get_metric_labels(metric)
graph_title = get_graph_title(metric, region)
graph_description = get_graph_description(metric)
map_title = get_map_title(metric)
map_description = get_map_explanation(metric)

# ----------------------------
# SUMMARIZE FUNCTION
# ----------------------------
def summarize(data, time_unit):
    if time_unit == "Yearly":
        summary = data.groupby("year").agg({
            "protests": "sum",
            "civilian_targeting": "sum",
            "civilian_targeting_lag1": "sum",
            "battles": "sum",
            "remote_violence": "sum",
            "riots": "sum",
            "price": "last",
            "price_change": "mean",
            "price_change_lag1": "mean"
        }).reset_index()
        x = summary["year"]
    else:
        summary = data.groupby("month").agg({
            "protests": "sum",
            "civilian_targeting": "sum",
            "civilian_targeting_lag1": "sum",
            "battles": "sum",
            "remote_violence": "sum",
            "riots": "sum",
            "price": "mean",
            "price_change": "mean",
            "price_change_lag1": "mean"
        }).reset_index()
        summary["month_str"] = summary["month"].astype(str)
        x = summary["month_str"]

    return summary, x

# ----------------------------
# HELPER FOR CLEAN X LABELS
# ----------------------------
def format_time_axis(ax, x_values, time_unit):
    if time_unit == "Monthly":
        x_values = list(x_values)
        if len(x_values) > 18:
            step = max(1, len(x_values) // 12)
            ticks = list(range(0, len(x_values), step))
            labels = [x_values[i] for i in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=45, ha="right")
        else:
            ax.tick_params(axis="x", rotation=45)
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")

# ----------------------------
# STANDARD LINE PLOT FUNCTION
# ----------------------------
def plot_region(ax, data, name):
    summary, x = summarize(data, time_unit)

    if metric == "Protests":
        ax.plot(x, summary["protests"], linewidth=2, label=f"{name} - Total protests")
    elif metric == "Civilian Targeting":
        ax.plot(x, summary["civilian_targeting"], linewidth=2, label=f"{name} - Total civilian targeting")
    elif metric == "Battles":
        ax.plot(x, summary["battles"], linewidth=2, label=f"{name} - Total battles")
    elif metric == "Remote Violence":
        ax.plot(x, summary["remote_violence"], linewidth=2, label=f"{name} - Total remote violence")
    elif metric == "Riots":
        ax.plot(x, summary["riots"], linewidth=2, label=f"{name} - Total riots")
    elif metric == "Food Prices":
        ax.plot(x, summary["price"], linewidth=2, label=f"{name} - Average food price")
    elif metric == "Food Price Change":
        ax.plot(x, summary["price_change"], linewidth=2, label=f"{name} - Average food price change")
    elif metric == "Protests vs Food Prices":
        ax.plot(x, summary["protests"], linewidth=2, label=f"{name} - Total protests")
        ax.plot(x, summary["price"], linewidth=2, label=f"{name} - Average food price")
    elif metric == "Protests vs Food Price Change":
        ax.plot(x, summary["protests"], linewidth=2, label=f"{name} - Total protests")
        ax.plot(x, summary["price_change"], linewidth=2, label=f"{name} - Average food price change")

    format_time_axis(ax, x, time_unit)

# ----------------------------
# MAP DATA FUNCTION
# ----------------------------
def build_map_data(df, metric):
    grouped = df.groupby("admin1", as_index=False).agg({
        "protests": "sum",
        "civilian_targeting": "sum",
        "civilian_targeting_lag1": "mean",
        "battles": "sum",
        "remote_violence": "sum",
        "riots": "sum",
        "price": "mean",
        "price_change": "mean",
        "price_change_lag1": "mean"
    })

    if metric == "Protests":
        color_col = "protests"
    elif metric == "Civilian Targeting":
        color_col = "civilian_targeting"
    elif metric == "Battles":
        color_col = "battles"
    elif metric == "Remote Violence":
        color_col = "remote_violence"
    elif metric == "Riots":
        color_col = "riots"
    elif metric == "Food Prices":
        color_col = "price"
    elif metric == "Food Price Change":
        color_col = "price_change"
    elif metric == "Protests vs Food Prices":
        color_col = "price"
    elif metric == "Protests vs Food Price Change":
        color_col = "price_change"
    elif metric == "Protests vs Food Price Change (Lagged)":
        color_col = "price_change_lag1"
    elif metric == "Protests vs Civilian Targeting (Lagged)":
        color_col = "civilian_targeting_lag1"
    elif metric == "Total Protests by Repression":
        color_col = "civilian_targeting_lag1"
    elif metric == "Total Protests by Price Change (Binned by Repression)":
        color_col = "price_change_lag1"
    else:
        color_col = "protests"

    return grouped, color_col

# ----------------------------
# MAP CENTER / ZOOM
# ----------------------------
def get_map_view(gdf_map, selected_region, zoom_mode):
    if zoom_mode == "Overall map" or selected_region == "Overall":
        centroid = gdf_map.geometry.union_all().centroid
        return centroid.y, centroid.x, 5.2

    selected = gdf_map[gdf_map["admin1"] == selected_region]
    if selected.empty:
        centroid = gdf_map.geometry.union_all().centroid
        return centroid.y, centroid.x, 5.2

    centroid = selected.geometry.union_all().centroid
    return centroid.y, centroid.x, 5.9

# ----------------------------
# RELATIONSHIP CHARTS
# ----------------------------
def plot_relationship_scatter(ax, data, x_col, y_col, title, x_label, y_label):
    plot_df = data.dropna(subset=[x_col, y_col]).copy()
    ax.scatter(plot_df[x_col], plot_df[y_col], s=55, alpha=0.7)
    ax.set_title(title, pad=15)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.2)

# ----------------------------
# GRAPH VIEW
# ----------------------------
st.subheader(graph_title)
st.caption(graph_description)

fig, ax = plt.subplots(figsize=(10, 5.5))

if metric == "Total Protests by Repression":
    total_by_repression = (
        region_data.groupby("repression_level", observed=False)["protests"]
        .sum()
        .reindex(["Low", "Medium", "High"])
        .fillna(0)
    )
    ax.bar(total_by_repression.index.astype(str), total_by_repression.values)
    ax.set_xlabel("Repression level (lagged civilian targeting)")
    ax.set_ylabel("Total protest event count")

elif metric == "Total Protests by Price Change (Binned by Repression)":
    grouped = (
        region_data.groupby(["price_shock_bin", "repression_level"], observed=False)["protests"]
        .sum()
        .unstack()
        .reindex(["Price Drop", "Stable", "Price Increase"])
        .fillna(0)
    )
    grouped.plot(kind="bar", ax=ax)
    ax.set_xlabel("Lagged food price change bin")
    ax.set_ylabel("Total protest event count")
    ax.legend(title="Repression Level")

elif metric == "Protests vs Civilian Targeting (Lagged)":
    plot_relationship_scatter(
        ax=ax,
        data=region_data,
        x_col="civilian_targeting_lag1",
        y_col="protests",
        title=graph_title,
        x_label="Lagged civilian targeting",
        y_label="Total protest event count"
    )

elif metric == "Protests vs Food Price Change (Lagged)":
    plot_relationship_scatter(
        ax=ax,
        data=region_data,
        x_col="price_change_lag1",
        y_col="protests",
        title=graph_title,
        x_label="Lagged food price change",
        y_label="Total protest event count"
    )

elif metric == "Protests vs Food Prices":
    plot_relationship_scatter(
        ax=ax,
        data=region_data,
        x_col="price",
        y_col="protests",
        title=graph_title,
        x_label="Average food price",
        y_label="Total protest event count"
    )

elif metric == "Protests vs Food Price Change":
    plot_relationship_scatter(
        ax=ax,
        data=region_data,
        x_col="price_change",
        y_col="protests",
        title=graph_title,
        x_label="Average food price change",
        y_label="Total protest event count"
    )

else:
    plot_region(ax, region_data, region)

    if compare and region2:
        region2_data = df_filtered[df_filtered["admin1"] == region2]
        plot_region(ax, region2_data, region2)

    ax.set_title(graph_title, pad=15)
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

plt.tight_layout()
st.pyplot(fig, use_container_width=True)

st.divider()

# ----------------------------
# MAP VIEW
# ----------------------------
st.subheader(map_title)
st.caption(map_description)

map_df, color_col = build_map_data(df_filtered, metric)
choropleth = gdf.merge(map_df, on="admin1", how="left").copy()

hover_dict = {
    "protests": ":,.0f",
    "civilian_targeting": ":,.0f",
    "civilian_targeting_lag1": ":,.2f",
    "battles": ":,.0f",
    "remote_violence": ":,.0f",
    "riots": ":,.0f",
    "price": ":,.2f",
    "price_change": ":,.3f",
    "price_change_lag1": ":,.3f"
}

geojson = json.loads(choropleth[["admin1", "geometry"]].to_json())
lat, lon, zoom = get_map_view(choropleth, region, map_zoom_mode)

fig_map = px.choropleth_mapbox(
    choropleth,
    geojson=geojson,
    locations="admin1",
    featureidkey="properties.admin1",
    color=color_col,
    hover_name="admin1",
    hover_data=hover_dict,
    color_continuous_scale="OrRd",
    mapbox_style="white-bg",
    center={"lat": lat, "lon": lon},
    zoom=zoom,
    opacity=0.82
)

fig_map.update_traces(
    hovertemplate="<b>%{hovertext}</b><br>" +
                  f"{map_label}: %{{z}}<extra></extra>"
)

fig_map.update_layout(
    title={"text": map_title, "x": 0.5, "xanchor": "center"},
    height=700,
    margin={"r": 0, "t": 60, "l": 0, "b": 0},
    coloraxis_colorbar_title=map_label
)

st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# ----------------------------
# REGRESSION SECTION
# ----------------------------
st.subheader("District-Level Regression Results")
st.markdown(
    """
Model run separately for each district:

**protests ~ price_change_lag1 + civilian_targeting_lag1 + price_change_lag1 × civilian_targeting_lag1**
"""
)

@st.cache_data
def run_regressions(df):
    reg_df = df.dropna(subset=[
        "protests",
        "price_change_lag1",
        "civilian_targeting_lag1"
    ]).copy()

    results = []

    for district in sorted(reg_df["admin1"].dropna().unique()):
        sub = reg_df[reg_df["admin1"] == district].copy()

        if len(sub) < 24:
            continue
        if sub["protests"].nunique() < 2:
            continue

        try:
            model = smf.ols(
                "protests ~ price_change_lag1 + civilian_targeting_lag1 + price_change_lag1:civilian_targeting_lag1",
                data=sub
            ).fit(cov_type="HC1")

            results.append({
                "admin1": district,
                "n_obs": int(model.nobs),
                "r_squared": model.rsquared,
                "coef_price_change_lag1": model.params.get("price_change_lag1", float("nan")),
                "p_price_change_lag1": model.pvalues.get("price_change_lag1", float("nan")),
                "coef_civilian_targeting_lag1": model.params.get("civilian_targeting_lag1", float("nan")),
                "p_civilian_targeting_lag1": model.pvalues.get("civilian_targeting_lag1", float("nan")),
                "coef_interaction": model.params.get("price_change_lag1:civilian_targeting_lag1", float("nan")),
                "p_interaction": model.pvalues.get("price_change_lag1:civilian_targeting_lag1", float("nan"))
            })
        except Exception:
            continue

    return pd.DataFrame(results)

regression_results = run_regressions(df_filtered)

if regression_results.empty:
    st.warning("No district-level regressions could be estimated for the selected years/category.")
else:
    reg_left, reg_right = st.columns([1.2, 1])

    with reg_left:
        st.markdown("### All district results")
        st.dataframe(
            regression_results.sort_values("admin1").reset_index(drop=True),
            use_container_width=True
        )

    with reg_right:
        st.markdown("### Selected district summary")
        if region == "Overall":
            st.info("Select a district to view its district-specific regression summary.")
        else:
            selected_result = regression_results[regression_results["admin1"] == region]
            if not selected_result.empty:
                st.dataframe(selected_result.reset_index(drop=True), use_container_width=True)
            else:
                st.info("No regression result available for the selected district under the current filters.")

    st.markdown("### Interaction coefficient by district")

    coef_plot = regression_results.sort_values("coef_interaction").copy()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.barh(coef_plot["admin1"], coef_plot["coef_interaction"])
    ax2.set_title("Interaction Effect: Price Change × Civilian Targeting", pad=12)
    ax2.set_xlabel("Coefficient")
    ax2.set_ylabel("District")
    ax2.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

    st.markdown("### Districts where the interaction is statistically significant (p < 0.05)")
    sig = regression_results[regression_results["p_interaction"] < 0.05].sort_values("p_interaction")

    if sig.empty:
        st.write("None under the current filters.")
    else:
        st.dataframe(sig.reset_index(drop=True), use_container_width=True)