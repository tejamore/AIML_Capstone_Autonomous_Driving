"""
analysis.py
-----------
Exploratory data analysis for the Autonomous Driving capstone (Part 2):
"Analyze the usage of autopilot and its effect on road safety."

Reads data/Tesla-Deaths.csv and answers the questions posed in the
capstone problem statement, returning matplotlib figures encoded as
base64 PNGs so they can be dropped straight into an HTML page.
"""

import base64
import io
import os

import matplotlib
matplotlib.use("Agg")  # headless rendering, required on Render
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Tesla-Deaths.csv")

plt.rcParams["figure.autolayout"] = True


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load and lightly clean the Tesla deaths dataset."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Basic cleaning: drop exact duplicates, coerce numeric columns
    df = df.drop_duplicates()
    numeric_cols = [
        "Deaths", "Tesla driver", "Tesla occupant", "Other vehicle",
        "Cyclists/Peds", "Verified Tesla Autopilot Deaths",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Weekday"] = df["Date"].dt.day_name()

    return df


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_dashboard(df: pd.DataFrame) -> dict:
    """
    Produce every chart + headline stat needed for the analytics page.
    Returns a dict of {key: base64 png} and a dict of {key: stat}.
    """
    charts = {}

    # 1. Events per year
    fig, ax = plt.subplots(figsize=(6, 3.5))
    df["Year"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#3b82f6")
    ax.set_title("Number of Events per Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Events")
    charts["events_per_year"] = _fig_to_base64(fig)

    # 2. Events by country
    fig, ax = plt.subplots(figsize=(6, 3.5))
    df["Country"].value_counts().plot(kind="bar", ax=ax, color="#f59e0b")
    ax.set_title("Number of Events per Country")
    ax.set_ylabel("Events")
    charts["events_per_country"] = _fig_to_base64(fig)

    # 3. Events by weekday (per day)
    if "Weekday" in df.columns:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        counts = df["Weekday"].value_counts().reindex(order).fillna(0)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        counts.plot(kind="bar", ax=ax, color="#10b981")
        ax.set_title("Number of Events per Day of Week")
        ax.set_ylabel("Events")
        charts["events_per_weekday"] = _fig_to_base64(fig)

    # 4. Distribution of deaths per accident
    fig, ax = plt.subplots(figsize=(6, 3.5))
    df["Deaths"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#ef4444")
    ax.set_title("Number of Victims (Deaths) per Accident")
    ax.set_xlabel("Deaths in a single accident")
    ax.set_ylabel("Number of accidents")
    charts["deaths_per_accident"] = _fig_to_base64(fig)

    # 5. Event distribution across models
    if "Model" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        df["Model"].value_counts().plot(kind="bar", ax=ax, color="#8b5cf6")
        ax.set_title("Event Distribution Across Tesla Models")
        ax.set_ylabel("Events")
        charts["events_per_model"] = _fig_to_base64(fig)

    # 6. Verified Tesla Autopilot deaths distribution
    if "Verified Tesla Autopilot Deaths" in df.columns:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        df["Verified Tesla Autopilot Deaths"].value_counts().sort_index().plot(
            kind="bar", ax=ax, color="#0ea5e9"
        )
        ax.set_title("Verified Tesla Autopilot Deaths (per event)")
        ax.set_ylabel("Number of events")
        charts["verified_autopilot_deaths"] = _fig_to_base64(fig)

    # Headline stats answering the specific capstone questions
    stats = {
        "total_events": int(len(df)),
        "total_deaths": int(df["Deaths"].sum()) if "Deaths" in df.columns else None,
        "tesla_driver_deaths": int(df["Tesla driver"].sum()) if "Tesla driver" in df.columns else None,
        "pct_events_with_occupant_death": round(
            100 * (df["Tesla driver"].add(df.get("Tesla occupant", 0)) > 0).mean(), 1
        ) if "Tesla driver" in df.columns else None,
        "cyclist_ped_events": int(df["Cyclists/Peds"].sum()) if "Cyclists/Peds" in df.columns else None,
        "occupant_plus_cyclist_events": int(
            (
                (df["Tesla driver"].add(df.get("Tesla occupant", 0)) > 0)
                & (df["Cyclists/Peds"] > 0)
            ).sum()
        ) if "Cyclists/Peds" in df.columns and "Tesla driver" in df.columns else None,
        "other_vehicle_collisions": int(df["Other vehicle"].sum()) if "Other vehicle" in df.columns else None,
        "verified_autopilot_deaths": int(df["Verified Tesla Autopilot Deaths"].sum())
        if "Verified Tesla Autopilot Deaths" in df.columns else None,
    }

    return {"charts": charts, "stats": stats}


if __name__ == "__main__":
    dframe = load_data()
    result = build_dashboard(dframe)
    print(result["stats"])
