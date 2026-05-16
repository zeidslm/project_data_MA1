"""
MECA-H419 Project — Step 1B: Clustering Preparation
===================================================
Create a cross-sectional dataset of countries.
Calculates the average emission profile (sector percentages) 
over a recent time window (2010-2024) and standardizes it 
to prepare for K-Means clustering.

Input : EDGAR_2025_GHG_booklet_2025.xlsx
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- paths
DATA_FILE = "EDGAR_2025_GHG_booklet_2025.xlsx"
OUT_DIR   = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 1. Load Data
print("Loading data...")
df = pd.read_excel(DATA_FILE, sheet_name="GHG_by_sector_and_country")

# We focus on the most recent 15 years to capture modern industrial profiles
recent_years = [y for y in range(2010, 2025) if y in df.columns]

# ---------------------------------------------------------------- 2. Filter Countries
# Remove global aggregates and transport zones that aren't individual countries
aggregates_to_remove = ["EU27", "GLOBAL TOTAL", "Int. Aviation", "Int. Shipping"]
df_countries = df[~df["EDGAR Country Code"].isin(aggregates_to_remove)].copy()

# ---------------------------------------------------------------- 3. Aggregate & Average
print(f"Calculating average emissions over {recent_years[0]}-{recent_years[-1]}...")

# Sum all substances (CO2, CH4, N2O, F-gases) per country and per sector for the recent years
grouped = df_countries.groupby(["EDGAR Country Code", "Country", "Sector"])[recent_years].sum()

# Calculate the mean across the recent years for each row
grouped["Average_Emissions"] = grouped[recent_years].mean(axis=1)
grouped = grouped.reset_index()

# Pivot so that each country is a row, and each sector is a column
profile_df = grouped.pivot(index=["EDGAR Country Code", "Country"], columns="Sector", values="Average_Emissions")

# Fill any missing sectors with 0 (some countries might not have specific industries)
profile_df = profile_df.fillna(0)

# ---------------------------------------------------------------- 4. Convert to Percentages
print("Converting to structural percentages...")
# Calculate total average emissions per country
country_totals = profile_df.sum(axis=1)

# Remove countries with absolute zero emissions to avoid division by zero
valid_countries = country_totals > 0
profile_df = profile_df[valid_countries]
country_totals = country_totals[valid_countries]

# Divide each sector by the country's total to get the percentage [0.0 to 1.0]
profile_pct = profile_df.div(country_totals, axis=0)

# ---------------------------------------------------------------- 5. Standardize (Z-Score)
print("Standardizing data for K-Means...")
# K-Means requires standardized data so variance in one sector doesn't dominate others
scaler = StandardScaler()
profile_std_array = scaler.fit_transform(profile_pct)

# Create a DataFrame for the standardized data
profile_std = pd.DataFrame(profile_std_array, index=profile_pct.index, columns=profile_pct.columns)

# ---------------------------------------------------------------- 6. Save Outputs
# Save the raw percentages (great for interpreting the clusters later!)
out_pct_file = OUT_DIR / "country_profiles_percentages.csv"
profile_pct.to_csv(out_pct_file, sep=';', decimal=',', encoding='utf-8-sig')

# Save the standardized data (this is what you feed into K-Means)
out_std_file = OUT_DIR / "country_profiles_standardized.csv"
profile_std.to_csv(out_std_file, sep=';', decimal=',', encoding='utf-8-sig')

