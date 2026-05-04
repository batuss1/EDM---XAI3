from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "reports" / "figures"
RANDOM_STATE = 42


def fit_random_forest(x: pd.DataFrame, y: pd.Series) -> tuple[RandomForestRegressor, dict[str, float]]:
    x = x.astype(float)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return model, {
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
    }


def bike_1d_pdp(day: pd.DataFrame) -> dict[str, float]:
    bike = day.copy()
    bike["days_since_2011"] = (
        pd.to_datetime(bike["dteday"]) - pd.Timestamp("2011-01-01")
    ).dt.days

    features = [
        "days_since_2011",
        "season",
        "yr",
        "mnth",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "hum",
        "windspeed",
    ]
    x = bike[features].astype(float)
    y = bike["cnt"]
    model, metrics = fit_random_forest(x, y)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    PartialDependenceDisplay.from_estimator(
        model,
        x,
        ["days_since_2011", "temp", "hum", "windspeed"],
        ax=axes.ravel(),
        kind="average",
        grid_resolution=50,
    )
    fig.suptitle("Bike rentals: one-dimensional PDPs", fontsize=14)
    fig.savefig(FIGURES / "bike_1d_pdp.png", dpi=180)
    plt.close(fig)
    return metrics


def bike_2d_pdp(day: pd.DataFrame) -> None:
    bike = day.copy()
    bike["days_since_2011"] = (
        pd.to_datetime(bike["dteday"]) - pd.Timestamp("2011-01-01")
    ).dt.days

    features = [
        "days_since_2011",
        "season",
        "yr",
        "mnth",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "hum",
        "windspeed",
    ]
    x = bike[features].astype(float)
    y = bike["cnt"]
    model, _ = fit_random_forest(x, y)

    sample = x.sample(n=min(500, len(x)), random_state=RANDOM_STATE)
    result = partial_dependence(
        model,
        sample,
        features=[(features.index("hum"), features.index("temp"))],
        grid_resolution=45,
    )
    hum_grid, temp_grid = result["grid_values"]
    values = result["average"][0].T

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    tile = ax.pcolormesh(
        hum_grid,
        temp_grid,
        values,
        shading="auto",
        cmap="viridis",
    )
    ax.scatter(sample["hum"], sample["temp"], s=12, c="white", alpha=0.45, edgecolors="none")
    ax.set_title("Bike rentals: 2D PDP for humidity and temperature")
    ax.set_xlabel("Humidity")
    ax.set_ylabel("Temperature")
    cbar = fig.colorbar(tile, ax=ax)
    cbar.set_label("Predicted bike count")
    fig.savefig(FIGURES / "bike_2d_pdp_hum_temp.png", dpi=180)
    plt.close(fig)


def house_pdp(houses: pd.DataFrame) -> dict[str, float]:
    features = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "yr_built"]
    x = houses[features].astype(float)
    y = houses["price"]

    sample_idx = x.sample(n=6000, random_state=RANDOM_STATE).index
    model, metrics = fit_random_forest(x.loc[sample_idx], y.loc[sample_idx])

    pdp_sample = x.sample(n=4000, random_state=RANDOM_STATE)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    PartialDependenceDisplay.from_estimator(
        model,
        pdp_sample,
        ["bedrooms", "bathrooms", "sqft_living", "floors"],
        ax=axes.ravel(),
        kind="average",
        grid_resolution=45,
    )
    fig.suptitle("House prices: one-dimensional PDPs", fontsize=14)
    fig.savefig(FIGURES / "house_1d_pdp.png", dpi=180)
    plt.close(fig)
    return metrics


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    day = pd.read_csv(DATA / "day.csv")
    houses = pd.read_csv(DATA / "kc_house_data.csv")

    bike_metrics = bike_1d_pdp(day)
    bike_2d_pdp(day)
    house_metrics = house_pdp(houses)

    summary = pd.DataFrame(
        [
            {"model": "Bike rentals", **bike_metrics},
            {"model": "House prices", **house_metrics},
        ]
    )
    summary.to_csv(ROOT / "reports" / "model_metrics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
