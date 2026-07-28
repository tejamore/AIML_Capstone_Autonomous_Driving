"""
generate_sample_data.py
------------------------
Creates a SYNTHETIC sample Tesla-Deaths.csv that mirrors the schema of the
real dataset described in the capstone problem statement (Part 2).

IMPORTANT: This is randomly generated demo data only, used so the analytics
dashboard and notebook run out-of-the-box. Replace data/Tesla-Deaths.csv
with the real dataset (e.g. from tesladeaths.com) before submitting your
actual capstone analysis.
"""
import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

N = 220
years = list(range(2013, 2024))
countries = ["USA"] * 14 + ["Canada", "China", "Mexico", "Germany", "Netherlands", "Norway"]
us_states = ["CA", "TX", "FL", "NY", "WA", "AZ", "GA", "IL", "OH", "PA", "NC", "MO", "OR", "NV"]
models = ["Model S", "Model 3", "Model X", "Model Y"]

rows = []
for i in range(1, N + 1):
    year = random.choice(years)
    country = random.choice(countries)
    state = random.choice(us_states) if country == "USA" else "-"
    deaths = np.random.choice([1, 1, 1, 1, 2, 2, 3], p=[0.55, 0.001, 0.001, 0.208, 0.15, 0.05, 0.04][:7] if False else None) if False else random.choices([1, 2, 3], weights=[80, 15, 5])[0]
    tesla_driver = random.choices([1, 0], weights=[55, 45])[0]
    tesla_occupant = random.choices([1, 0], weights=[20, 80])[0] if tesla_driver == 0 else 0
    other_vehicle = random.choices([1, 0], weights=[30, 70])[0]
    cyclist_peds = random.choices([1, 0], weights=[15, 85])[0]
    model = random.choices(models, weights=[25, 45, 20, 10])[0]
    autopilot_claimed = random.choices(["Y", "-"], weights=[18, 82])[0]
    verified_autopilot = random.choices([1, 0], weights=[10, 90])[0] if autopilot_claimed == "Y" else 0
    verified_plus_nhtsa = verified_autopilot + random.choices([0, 1], weights=[85, 15])[0]

    rows.append({
        "Case #": i,
        "Year": year,
        "Date": f"{random.randint(1,12):02d}/{random.randint(1,28):02d}/{year}",
        "Country": country,
        "State": state,
        "Description": "Tesla crash",
        "Deaths": deaths,
        "Tesla driver": tesla_driver,
        "Tesla occupant": tesla_occupant,
        "Other vehicle": other_vehicle,
        "Cyclists/Peds": cyclist_peds,
        "TSLA+cycl/peds": cyclist_peds,
        "Model": model,
        "Autopilot claimed": autopilot_claimed,
        "Verified Tesla Autopilot Deaths": verified_autopilot,
        "Verified Tesla Autopilot Deaths + All Deaths Reported to NHTSA SGO": verified_plus_nhtsa,
        "Source": "https://example.com/source",
        "Note": "Synthetic sample record for demo purposes",
    })

df = pd.DataFrame(rows)
df.to_csv("Tesla-Deaths.csv", index=False)
print(df.shape)
print(df.head())
