import pandas as pd


# ----------------------------
# LOAD + CLEAN ACLED
# ----------------------------
def load_clean_acled(path):
    df = pd.read_csv(path)

    df = df[df["country"] == "Syria"].copy()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date", "admin1", "event_type"])

    df["month"] = df["event_date"].dt.to_period("M")
    df["year"] = df["event_date"].dt.year

    # normalize region names
    df["admin1"] = df["admin1"].replace({
        "Al Hasakeh": "Al Hasakah",
        "Deir ez Zor": "Deir Ez-Zor",
        "Ar Raqqa": "Ar-Raqqa",
        "As Sweida": "As-Sweida"
    })

    return df


# ----------------------------
# LOAD + CLEAN FOOD DATA
# ----------------------------
def load_clean_food(path):
    food = pd.read_csv(path)

    food["date"] = pd.to_datetime(food["date"], errors="coerce")
    food = food.dropna(subset=["date", "admin1", "usdprice"])

    food["month"] = food["date"].dt.to_period("M")
    food["year"] = food["date"].dt.year

    # fix region names
    food["admin1"] = food["admin1"].replace({
        "Al-Hasakeh": "Al Hasakah",
        "Hasakeh": "Al Hasakah",
        "Raqqa": "Ar-Raqqa",
        "Sweida": "As-Sweida",
        "Latakia": "Lattakia",
        "Deir-ez-Zor": "Deir Ez-Zor",
        "Dar'a": "Dara"
    })

    return food


# ----------------------------
# BUILD PANEL
# ----------------------------
def build_conflict_panel(acled):
    acled["protest_flag"] = (acled["event_type"] == "Protests").astype(int)
    acled["civilian_targeting_flag"] = (
        acled["event_type"] == "Violence against civilians"
    ).astype(int)
    acled["battle_flag"] = (acled["event_type"] == "Battles").astype(int)
    acled["remote_violence_flag"] = (
        acled["event_type"] == "Explosions/Remote violence"
    ).astype(int)
    acled["riot_flag"] = (acled["event_type"] == "Riots").astype(int)

    monthly = (
        acled.groupby(["admin1", "month"])
        .agg(
            protests=("protest_flag", "sum"),
            civilian_targeting=("civilian_targeting_flag", "sum"),
            battles=("battle_flag", "sum"),
            remote_violence=("remote_violence_flag", "sum"),
            riots=("riot_flag", "sum"),
        )
        .reset_index()
    )

    all_regions = sorted(acled["admin1"].dropna().unique())
    all_months = pd.period_range(
        acled["month"].min(),
        acled["month"].max(),
        freq="M"
    )

    full_index = pd.MultiIndex.from_product(
        [all_regions, all_months],
        names=["admin1", "month"]
    )

    panel = (
        monthly.set_index(["admin1", "month"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    panel["year"] = panel["month"].dt.year
    panel = panel.sort_values(["admin1", "month"])

    return panel


# ----------------------------
# ADD FOOD DATA
# ----------------------------
def merge_food(panel, food, category="All categories"):

    if category != "All categories":
        food = food[food["category"] == category]

    food_region = (
        food.groupby(["admin1", "month"])["usdprice"]
        .mean()
        .reset_index(name="price")
    )

    merged = pd.merge(panel, food_region, on=["admin1", "month"], how="left")

    # correct fill (IMPORTANT)
    merged["price"] = (
        merged.groupby("admin1")["price"]
        .transform(lambda x: x.ffill().bfill())
    )

    merged["price_change"] = merged.groupby("admin1")["price"].pct_change()

    return merged


# ----------------------------
# DEBUG FUNCTION
# ----------------------------
def check_missing_regions(panel, food):
    acled_regions = set(panel["admin1"].unique())
    food_regions = set(food["admin1"].unique())

    return {
        "in_acled_not_food": acled_regions - food_regions,
        "in_food_not_acled": food_regions - acled_regions
    }


# ----------------------------
# SAFE DEBUG BLOCK
# ----------------------------
if __name__ == "__main__":
    acled = load_clean_acled("/Users/aishanimathur/Downloads/ACLED Data_2026-03-18.csv")
    food = load_clean_food("/Users/aishanimathur/Downloads/wfp_food_prices_syr (1).csv")

    panel = build_conflict_panel(acled)
    merged = merge_food(panel, food)

    print("Coverage check:")
    print(
        merged.groupby("admin1")["price"]
        .apply(lambda x: x.notna().sum())
    )