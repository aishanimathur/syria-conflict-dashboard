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

# GLOBAL PROJECT PATH
BASE_DIR = Path(__file__).resolve().parent

# PAGE SETUP
st.set_page_config(
    page_title="Syria Conflict Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Syria Conflict Dashboard")
st.caption(
    "Track protests, violence, food prices, and district-level relationship patterns across Syrian regions."
)

# LOAD DATA
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

# SIDEBAR
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
        "Protests vs Food Price Change (Lagged)",
        "Protests vs Civilian Targeting (Lagged)",
        "Current Protests by Lagged Repression",
        "Current Protests by Lagged Price Change and Lagged Repression",
        "Do Food Prices Matter?",
        "Does Repression Matter?",
        "Does Repression Change the Effect of Food Prices?"
    ]
)

food_metrics = {
    "Food Prices",
    "Food Price Change",
    "Protests vs Food Price Change (Lagged)",
    "Current Protests by Lagged Price Change and Lagged Repression",
    "Do Food Prices Matter?",
    "Does Repression Matter?",
    "Does Repression Change the Effect of Food Prices?"
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

# significance cutoff for regression maps
significance_threshold = 0.05

# PREP DATA
df_panel = merge_food(df_panel_base, food, selected_category)

df_panel["civilian_targeting_lag1"] = (
    df_panel.groupby("admin1")["civilian_targeting"].shift(1)
)

df_panel["price_change_lag1"] = (
    df_panel.groupby("admin1")["price_change"].shift(1)
)

valid_repression = df_panel["civilian_targeting_lag1"].dropna()
if len(valid_repression) > 0:
    df_panel["repression_level"] = pd.qcut(
        df_panel["civilian_targeting_lag1"],
        q=3,
        labels=["Low", "Medium", "High"],
        duplicates="drop"
    )
else:
    df_panel["repression_level"] = pd.Series(dtype="object")

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

region2_data = None
if compare and region2 is not None:
    region2_data = df_filtered[df_filtered["admin1"] == region2].copy()

# KPI ROW
k1, k2, k3, k4 = st.columns(4)
k1.metric("Region", region)
k2.metric("Years", f"{year_range[0]}–{year_range[1]}")
k3.metric("View", time_unit)
k4.metric("Food Category", selected_category if metric in food_metrics else "N/A")

st.divider()

# HELPERS
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
        "Food Price Change": "price_change"
    }

    return grouped, mapping.get(metric, "protests")


def build_correlation_map(df, x_var):
    rows = []

    for district in sorted(df["admin1"].dropna().unique()):
        sub = df[df["admin1"] == district][["protests", x_var]].dropna()

        if len(sub) < 4:
            continue
        if sub["protests"].nunique() < 2 or sub[x_var].nunique() < 2:
            continue

        corr = sub["protests"].corr(sub[x_var])

        rows.append({
            "admin1": district,
            "correlation": corr,
            "n_obs": len(sub)
        })

    return pd.DataFrame(rows)


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
                "coef_price_lag1": round(
                    model.params.get("price_change_lag1", float("nan")),
                    3
                ),
                "p_price_lag1": round(
                    model.pvalues.get("price_change_lag1", float("nan")),
                    3
                ),
                "coef_repression_lag1": round(
                    model.params.get("civilian_targeting_lag1", float("nan")),
                    3
                ),
                "p_repression_lag1": round(
                    model.pvalues.get("civilian_targeting_lag1", float("nan")),
                    3
                ),
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


def get_map_view(gdf_map):
    if map_zoom_mode == "Overall map" or region == "Overall":
        centroid = gdf_map.geometry.union_all().centroid
        return centroid.y, centroid.x, 5.2

    selected = gdf_map[gdf_map["admin1"] == region]
    if selected.empty:
        centroid = gdf_map.geometry.union_all().centroid
        return centroid.y, centroid.x, 5.2

    centroid = selected.geometry.union_all().centroid
    return centroid.y, centroid.x, 5.9


def build_scatter_data(data, xcol, ycol):
    if time_unit == "Yearly":
        return (
            data.groupby("year", as_index=False)
            .agg({
                xcol: "mean",
                ycol: "sum"
            })
            .dropna()
        )
    else:
        return data[[xcol, ycol]].dropna()


def prepare_regression_map_df(regression_results, coef_col, p_col):
    if regression_results is not None and not regression_results.empty:
        map_df = regression_results[
            ["admin1", coef_col, p_col, "r_squared", "n_obs"]
        ].copy()

        map_df["coef_sig"] = map_df[coef_col].where(
            map_df[p_col] < significance_threshold,
            other=float("nan")
        )
        map_df["is_insignificant"] = map_df[p_col] >= significance_threshold
    else:
        map_df = pd.DataFrame(
            columns=[
                "admin1",
                coef_col,
                p_col,
                "r_squared",
                "n_obs",
                "coef_sig",
                "is_insignificant"
            ]
        )
    return map_df


# GRAPH
regression_metrics = {
    "Do Food Prices Matter?",
    "Does Repression Matter?",
    "Does Repression Change the Effect of Food Prices?"
}

st.subheader(metric)

if metric in regression_metrics:
    if metric == "Do Food Prices Matter?":
        st.caption(
            "Map and regression results show whether lagged food price changes predict protests across districts."
        )
    elif metric == "Does Repression Matter?":
        st.caption(
            "Map and regression results show whether lagged civilian targeting predicts protests across districts."
        )
    else:
        st.caption(
            "Map and regression results show whether repression changes the effect of lagged food price changes on protests."
        )

else:
    if metric == "Protests vs Food Price Change (Lagged)":
        st.caption(
            f"Scatterplot reflects the selected chart view ({time_unit.lower()}). "
            "The map below remains monthly-only and shows district-level correlation "
            "between protests and lagged food price change using monthly observations."
        )
    elif metric == "Protests vs Civilian Targeting (Lagged)":
        st.caption(
            f"Scatterplot reflects the selected chart view ({time_unit.lower()}). "
            "The map below remains monthly-only and shows district-level correlation "
            "between protests and lagged civilian targeting using monthly observations."
        )
    elif metric == "Current Protests by Lagged Repression":
        st.caption(
            "Bars show current protests grouped by lagged repression level."
        )
    elif metric == "Current Protests by Lagged Price Change and Lagged Repression":
        st.caption(
            "Bars show current protests grouped by lagged price change bins and lagged repression levels."
        )
    else:
        st.caption("Chart view for selected filters.")

    fig, ax = plt.subplots(figsize=(10, 5.5))

    summary, x = summarize(region_data)

    summary2, x2 = (None, None)
    if region2_data is not None and not region2_data.empty:
        summary2, x2 = summarize(region2_data)

    line_metrics = {
        "Protests": "protests",
        "Civilian Targeting": "civilian_targeting",
        "Battles": "battles",
        "Remote Violence": "remote_violence",
        "Riots": "riots",
        "Food Prices": "price",
        "Food Price Change": "price_change"
    }

    scatter_metrics = {
        "Protests vs Food Price Change (Lagged)": (
            "price_change_lag1",
            "protests",
            "Lagged Food Price Change",
            "Protests"
        ),
        "Protests vs Civilian Targeting (Lagged)": (
            "civilian_targeting_lag1",
            "protests",
            "Lagged Civilian Targeting",
            "Protests"
        )
    }

    if metric in line_metrics:
        col = line_metrics[metric]

        ax.plot(x, summary[col], linewidth=2, label=region)

        if summary2 is not None:
            ax.plot(x2, summary2[col], linewidth=2, linestyle="--", label=region2)

        format_axis(ax, x)
        ax.set_xlabel("Time")
        ax.set_ylabel(metric)
        ax.legend()

    elif metric in scatter_metrics:
        xcol, ycol, xlabel, ylabel = scatter_metrics[metric]

        scatter_df = build_scatter_data(region_data, xcol, ycol)
        if not scatter_df.empty:
            ax.scatter(scatter_df[xcol], scatter_df[ycol], alpha=0.7, label=region)

        if region2_data is not None:
            scatter_df2 = build_scatter_data(region2_data, xcol, ycol)
            if not scatter_df2.empty:
                ax.scatter(scatter_df2[xcol], scatter_df2[ycol], alpha=0.7, label=region2)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

    elif metric == "Current Protests by Lagged Repression":
        tmp = region_data.groupby("repression_level", observed=False)["protests"].sum()
        ax.bar(tmp.index.astype(str), tmp.values)

        ax.set_xlabel("Lagged Repression Level")
        ax.set_ylabel("Current Protests")

    elif metric == "Current Protests by Lagged Price Change and Lagged Repression":
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
        ax.set_xlabel("Lagged Price Change Bin")
        ax.set_ylabel("Current Protests")

    title = f"{metric} in {region}"
    if region2_data is not None:
        title += f" vs {region2}"

    ax.set_title(title)
    ax.grid(alpha=0.2)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

st.divider()

# MAP
st.subheader(f"{metric} Map")

regression_results = None
regression_coef_col = None
regression_p_col = None
color_col = None

if metric == "Protests vs Food Price Change (Lagged)":
    st.caption(
        "Districts are shaded by the monthly district-level correlation between protests "
        "and lagged food price change. This map uses monthly observations only and does "
        "not change when the chart view is switched between Yearly and Monthly."
    )
    map_df = build_correlation_map(df_filtered, "price_change_lag1")
    color_col = "correlation"

elif metric == "Protests vs Civilian Targeting (Lagged)":
    st.caption(
        "Districts are shaded by the monthly district-level correlation between protests "
        "and lagged civilian targeting. This map uses monthly observations only and does "
        "not change when the chart view is switched between Yearly and Monthly."
    )
    map_df = build_correlation_map(df_filtered, "civilian_targeting_lag1")
    color_col = "correlation"

elif metric == "Current Protests by Lagged Repression":
    st.caption(
        "Districts are shaded by the monthly district-level correlation between current protests "
        "and lagged civilian targeting. This map uses monthly observations only."
    )
    map_df = build_correlation_map(df_filtered, "civilian_targeting_lag1")
    color_col = "correlation"

elif metric == "Current Protests by Lagged Price Change and Lagged Repression":
    st.caption(
        f"Districts with insignificant interaction effects are shown in grey. Colored districts have "
        f"statistically significant interaction effects (p < {significance_threshold:.2f}) in the monthly "
        "regression: protests ~ lagged price change + lagged repression + interaction."
    )
    regression_results = run_regressions(df_filtered)
    regression_coef_col = "coef_interaction"
    regression_p_col = "p_interaction"
    map_df = prepare_regression_map_df(regression_results, regression_coef_col, regression_p_col)
    color_col = "coef_sig"

elif metric == "Do Food Prices Matter?":
    st.caption(
        f"Districts with insignificant lagged food price effects are shown in grey. Colored districts have "
        f"statistically significant lagged food price coefficients (p < {significance_threshold:.2f}) "
        "from the monthly regression."
    )
    regression_results = run_regressions(df_filtered)
    regression_coef_col = "coef_price_lag1"
    regression_p_col = "p_price_lag1"
    map_df = prepare_regression_map_df(regression_results, regression_coef_col, regression_p_col)
    color_col = "coef_sig"

elif metric == "Does Repression Matter?":
    st.caption(
        f"Districts with insignificant lagged repression effects are shown in grey. Colored districts have "
        f"statistically significant lagged repression coefficients (p < {significance_threshold:.2f}) "
        "from the monthly regression."
    )
    regression_results = run_regressions(df_filtered)
    regression_coef_col = "coef_repression_lag1"
    regression_p_col = "p_repression_lag1"
    map_df = prepare_regression_map_df(regression_results, regression_coef_col, regression_p_col)
    color_col = "coef_sig"

elif metric == "Does Repression Change the Effect of Food Prices?":
    st.caption(
        f"Districts with insignificant interaction effects are shown in grey. Colored districts have "
        f"statistically significant interaction effects (p < {significance_threshold:.2f}) "
        "from the monthly regression."
    )
    regression_results = run_regressions(df_filtered)
    regression_coef_col = "coef_interaction"
    regression_p_col = "p_interaction"
    map_df = prepare_regression_map_df(regression_results, regression_coef_col, regression_p_col)
    color_col = "coef_sig"

else:
    st.caption("Map view across districts.")
    map_df, color_col = build_map_data(df_filtered)

choropleth = gdf.merge(map_df, on="admin1", how="left").copy()

geojson = json.loads(choropleth[["admin1", "geometry"]].to_json())
lat, lon, zoom = get_map_view(choropleth)

if color_col == "correlation":
    fig_map = px.choropleth_mapbox(
        choropleth,
        geojson=geojson,
        locations="admin1",
        featureidkey="properties.admin1",
        color="correlation",
        hover_name="admin1",
        hover_data={
            "correlation": ":.3f",
            "n_obs": True
        },
        color_continuous_scale="RdBu_r",
        range_color=(-1, 1),
        mapbox_style="white-bg",
        center={"lat": lat, "lon": lon},
        zoom=zoom,
        opacity=0.82
    )
    fig_map.update_coloraxes(colorbar_title="Correlation")

elif color_col == "coef_sig":
    grey_df = choropleth[choropleth["is_insignificant"] == True].copy()

    fig_map = px.choropleth_mapbox(
        grey_df,
        geojson=geojson,
        locations="admin1",
        featureidkey="properties.admin1",
        color_discrete_sequence=["lightgrey"],
        hover_name="admin1",
        hover_data={
            regression_coef_col: ':.3f',
            regression_p_col: ':.3f',
            "r_squared": ':.3f',
            "n_obs": True
        },
        mapbox_style="white-bg",
        center={"lat": lat, "lon": lon},
        zoom=zoom,
        opacity=0.75
    )

    sig_df = choropleth[choropleth["is_insignificant"] == False].copy()

    fig_sig = px.choropleth_mapbox(
        sig_df,
        geojson=geojson,
        locations="admin1",
        featureidkey="properties.admin1",
        color="coef_sig",
        hover_name="admin1",
        hover_data={
            regression_coef_col: ':.3f',
            regression_p_col: ':.3f',
            "r_squared": ':.3f',
            "n_obs": True
        },
        color_continuous_scale="RdBu_r",
        mapbox_style="white-bg",
        center={"lat": lat, "lon": lon},
        zoom=zoom,
        opacity=0.9
    )

    for tr in fig_sig.data:
        fig_map.add_trace(tr)

    if metric == "Do Food Prices Matter?":
        fig_map.update_coloraxes(colorbar_title="Price Effect")
    elif metric == "Does Repression Matter?":
        fig_map.update_coloraxes(colorbar_title="Repression Effect")
    else:
        fig_map.update_coloraxes(colorbar_title="Interaction Effect")

else:
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
    fig_map.update_coloraxes(colorbar_title=color_col)

fig_map.update_layout(
    height=700,
    margin={"r": 0, "t": 0, "l": 0, "b": 0}
)

st.plotly_chart(fig_map, width="stretch")

# REGRESSIONS - ONLY FOR REGRESSION VIEWS
regression_metrics = {
    "Do Food Prices Matter?",
    "Does Repression Matter?",
    "Does Repression Change the Effect of Food Prices?"
}

if metric in regression_metrics:
    st.divider()
    st.subheader("District-Level Regression Results")

    if regression_results is None:
        regression_results = run_regressions(df_filtered)

    if regression_results.empty:
        st.warning("No regressions available.")
    else:
        st.dataframe(regression_results, width="stretch")

        fig2, ax2 = plt.subplots(figsize=(10, 6))

        if metric == "Do Food Prices Matter?":
            plot_col = "coef_price_lag1"
            plot_title = "Lagged Food Price Effect by District"
            xlabel = "Coefficient"
        elif metric == "Does Repression Matter?":
            plot_col = "coef_repression_lag1"
            plot_title = "Lagged Repression Effect by District"
            xlabel = "Coefficient"
        else:
            plot_col = "coef_interaction"
            plot_title = "Interaction Effect by District"
            xlabel = "Interaction Coefficient"

        coef_plot = regression_results.sort_values(plot_col)

        ax2.barh(
            coef_plot["admin1"],
            coef_plot[plot_col]
        )
        ax2.set_title(plot_title)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("District")
        ax2.grid(axis="x", alpha=0.2)

        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)