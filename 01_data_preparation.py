"""
MECA-H419 Project — Step 1: Data Preparation
=============================================
Aggregate EU27 sector-level GHG emissions, apply log-transformation
and standardization. Build the supervised learning setup:
    X_t (8 sector emissions at year t) → y_{t+1} (total EU27 emissions at year t+1)

Input : EDGAR_2025_GHG_booklet_2025.xlsx
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------- paths
DATA_FILE = "EDGAR_2025_GHG_booklet_2025.xlsx"
OUT_DIR   = Path("outputs")
FIG_DIR   = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 1. Load
df = pd.read_excel(DATA_FILE, sheet_name="GHG_by_sector_and_country")
year_cols = [c for c in df.columns if isinstance(c, int)]
years     = np.array(year_cols)

print(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols  |  years {years[0]}-{years[-1]}")

# ---------------------------------------------------------------- 2. Filter EU27
eu27 = df[df["EDGAR Country Code"] == "EU27"].copy()
print(f"EU27 rows: {len(eu27)} "
      f"(4 substances x 8 sectors - 7 absent F-gas combinations = 25)")

# ---------------------------------------------------------------- 3. Aggregate sectors
# Sum the four substances (CO2, CH4, N2O, F-gases - all in Mt CO2-eq) per sector
sector_year = eu27.groupby("Sector")[year_cols].sum()        # (8, 55)
sector_year = sector_year.T                                   # (55, 8)  years x sectors
sector_year.index.name = "Year"

SECTORS = sector_year.columns.tolist()
print(f"\nSector matrix: {sector_year.shape}  (years x sectors)")
print(f"NaN count:     {sector_year.isna().sum().sum()}")

# Target: total emissions = sum across all sectors
total = sector_year.sum(axis=1).rename("Total")
print(f"Total emissions {years[0]}: {total.iloc[0]:8.1f}  Mt CO2-eq")
print(f"Total emissions {years[-1]}: {total.iloc[-1]:8.1f}  Mt CO2-eq")
print(f"Change: {(total.iloc[-1]/total.iloc[0]-1)*100:+.1f} %")

# ---------------------------------------------------------------- 4. Log-transform
# Emission magnitudes span ~3 orders of magnitude across sectors,
# log brings them onto a comparable scale and stabilises variance.
sector_log = np.log(sector_year)         # all values > 0, safe
total_log  = np.log(total)

# ---------------------------------------------------------------- 5. Standardize
# z-score using global mean/std (exploratory view).
# NOTE: For final modelling, the scaler must be fitted on TRAIN data
#       only to avoid data leakage. We re-do that in step 3.
mu_X, sd_X = sector_log.mean(), sector_log.std(ddof=0)
mu_y, sd_y = total_log.mean(),  total_log.std(ddof=0)

sector_std = (sector_log - mu_X) / sd_X
total_std  = (total_log  - mu_y) / sd_y

# ---------------------------------------------------------------- 6. Supervised setup (t -> t+1)
# Features at year t  ->  target at year t+1
X = sector_log.iloc[:-1].values          # rows 1970..2023   (54, 8)
y = total_log.iloc[1:].values            # rows 1971..2024   (54,)
X_years = years[:-1]                     # year of features
y_years = years[1:]                      # year of target

print(f"\nSupervised matrices  X: {X.shape}   y: {y.shape}")
print(f"  X covers years {X_years[0]}-{X_years[-1]}  (predictors)")
print(f"  y covers years {y_years[0]}-{y_years[-1]}  (targets at t+1)")

# ---------------------------------------------------------------- 7. Save
np.savez(OUT_DIR / "data_prepared.npz",
         X_log=X, y_log=y,
         X_years=X_years, y_years=y_years,
         sectors=np.array(SECTORS),
         sector_raw=sector_year.values,
         total_raw=total.values,
         years=years,
         mu_X=mu_X.values, sd_X=sd_X.values,
         mu_y=float(mu_y), sd_y=float(sd_y))

inspection = sector_year.copy()
inspection["Total"] = total
inspection.to_csv(OUT_DIR / "data_prepared.csv")
print(f"\nSaved -> {OUT_DIR/'data_prepared.npz'}")
print(f"Saved -> {OUT_DIR/'data_prepared.csv'}")

# ---------------------------------------------------------------- 8. Diagnostic plots
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

# --- (a) raw vs log vs standardized sector trajectories
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
for ax, data, title, ylab in zip(
        axes,
        [sector_year, sector_log, sector_std],
        ["Raw emissions",    "After log-transform", "After standardization"],
        ["Mt CO$_2$-eq / yr","ln(Mt CO$_2$-eq)",    "z-score"]):
    for s in SECTORS:
        ax.plot(years, data[s], label=s, lw=1.3)
    ax.set_title(title)
    ax.set_xlabel("Year"); ax.set_ylabel(ylab)
    ax.grid(alpha=0.3)
axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
fig.suptitle("EU27 sector-level GHG emissions - three preprocessing stages",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "01_preprocessing_stages.png", bbox_inches="tight")
plt.close(fig)

# --- (b) total EU27 emissions
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(years, total, color="#c0392b", lw=2)
ax.fill_between(years, total, alpha=0.15, color="#c0392b")
ax.set_title("Total EU27 GHG emissions, 1970-2024")
ax.set_xlabel("Year"); ax.set_ylabel("Mt CO$_2$-eq / yr")
ax.grid(alpha=0.3)
ax.annotate(f"{total.iloc[0]:.0f}",  (years[0],  total.iloc[0]),
            textcoords="offset points", xytext=(5, 5))
ax.annotate(f"{total.iloc[-1]:.0f}", (years[-1], total.iloc[-1]),
            textcoords="offset points", xytext=(-30, 5))
fig.tight_layout()
fig.savefig(FIG_DIR / "01_total_emissions.png", bbox_inches="tight")
plt.close(fig)

# --- (c) distribution check: log brings each sector to a roughly comparable spread
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, data, title in zip(axes,
                           [sector_year, sector_log],
                           ["Raw - wide range, skewed",
                            "Log - comparable spreads, more symmetric"]):
    ax.boxplot([data[s].values for s in SECTORS], tick_labels=SECTORS)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("Mt CO$_2$-eq")
axes[1].set_ylabel("ln(Mt CO$_2$-eq)")
fig.tight_layout()
fig.savefig(FIG_DIR / "01_distribution_check.png", bbox_inches="tight")
plt.close(fig)

print(f"Saved -> {FIG_DIR}/01_*.png  (3 figures)")
print("\nStep 1 done.")
