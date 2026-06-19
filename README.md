# Thermal Stress & Freezing Horizons

**Climatological Predictive Modeling via Heathrow ECA&D Observations**

A data analytics bootcamp project predicting two opposite extreme-weather hazards at London Heathrow — summer heatwaves and winter freeze risk — from 47 years of daily station observations (1979–2025, 17,155 daily records).

Authors: Claire Leyden & Irene Fafián

---

## Project Overview

We split the climatological extremes problem into two parallel classification tasks, each owned by one team member:

| Target Area | Owner | Definition | Business Impact |
|---|---|---|---|
| **Heatwave Forecasting** | Irene | 3+ consecutive days where daily max temp (TX) exceeds the local 90th-percentile threshold, May–September | Advance warning for infrastructure/transport planning — e.g. rail speed restrictions and road resurfacing schedules ahead of sustained heat |
| **Freeze-Risk Classification** | Claire | Daily mean temperature (TG) drops below 3°C | Logistics routing and supply chain preparedness ahead of severe frost |

Both tasks share the same underlying ECA&D dataset and feature-engineering approach, but were modeled, tuned, and evaluated independently.

---

## Data Source

- **Provider:** European Climate Assessment & Dataset (ECA&D)
- **Station:** Heathrow, UK (Station ID: 1862)
- **Period:** 1979–2025
- **Format:** 10 separate raw CSVs, one per variable, joined on date

| Code | Variable |
|---|---|
| TX | Daily maximum temperature |
| TN | Daily minimum temperature |
| TG | Daily mean temperature |
| HU | Relative humidity |
| CC | Cloud cover |
| RR | Precipitation |
| PP | Sea level pressure |
| SS | Sunshine duration |
| SD | Snow depth |
| QQ | Global radiation |

---

## Methodology

### Data Cleaning (`data_cleaning_Claire.ipynb`)
- Standardized column names across all 10 raw files
- Interpolated values flagged as invalid by the ECA&D quality flag (flag = 9)
- Removed February 29th across all years to keep clean, uniform 365-day annual sequences
- Dropped the station-ID and quality-flag columns once validated
- Exported one clean CSV per variable to `data/clean/`

### Heatwave Baseline & Feature Engineering (`heatwave_calc_Irene.ipynb`)
- Computed the **90th-percentile TX threshold** per calendar day from a **1991–2020 reference period**, using a **±5-day rolling window** centered on each day (following ET-SCI/WMO daily climate index conventions, not a wider monthly-style smoothing window)
- **Key methodological finding:** restricting the percentile baseline to the **May–September warm season** materially changes the heatwave count. An unfiltered, full-calendar-year baseline absorbs abnormal spring/autumn temperature spikes into the reference distribution, which roughly **doubles** the number of heatwave events identified versus the seasonally-correct count (e.g. 50 vs 24 events in 2020–2025). The final heatwave definition uses the May–September-filtered baseline.
- A heatwave is defined as **3+ consecutive days** above this threshold

### Freeze-Risk Feature Engineering (`snow_claire.ipynb`)
- Originally scoped to predict snow depth directly; pivoted after EDA showed snow-on-ground days were too rare to model reliably
- Final target: daily mean temperature (TG) below 3°C
- Cyclical day-of-year encoding (`sin`/`cos` of day-of-year, period 365.25) so December 31st and January 1st are treated as adjacent rather than as polar opposites
- Lag features for TN, TG, TX, PP at 1-, 3-, and 7-day offsets
- Forward-looking target variables for 5-day and 10-day prediction horizons

### Modeling (`ML_heatwave.ipynb`)
Both tasks followed the same evaluation discipline:
- Built lag features (1-, 3-, 7-day) to capture temporal persistence
- **Chronological train/test split** — training on all records through 2021, testing on the unseen 2022–2025 sequence — rather than a random split, to avoid leaking future weather patterns into training and to reflect how the model would actually be used going forward
- Feature selection guided by Pearson correlation against the binary target, dropping near-zero predictors (e.g. snow depth for heatwaves, sunshine for freeze risk)
- Four models compared per task: KNN baseline, class-weighted Random Forest, Random Forest with minority-class oversampling, and a hyperparameter-tuned Random Forest (GridSearchCV / RandomizedSearchCV)
- Primary evaluation metrics: **Recall** (catching real hazard events) and **F1** (balanced performance on a highly skewed/imbalanced target)

### Freeze-Risk Modeling (`ML_freeze.ipynb`)
Mirrors the heatwave modeling discipline on the freeze-risk target:
- **Chronological train/test split** — training on all records through 2021, testing on the unseen 2022–2025 sequence
- Two prediction horizons evaluated separately: a **5-day-ahead** and a **10-day-ahead** freeze forecast
- Feature selection guided by Pearson correlation against the binary freeze target, dropping near-zero predictors (e.g. sunshine duration showed only weak correlation and was dropped)
- Three models compared per horizon: class-weighted Random Forest, Random Forest with minority-class oversampling, and a hyperparameter-tuned Random Forest (RandomizedSearchCV)
- Primary evaluation metrics: **Recall** and **F1** on the freeze class, given the rarity of freeze days relative to non-freeze days.

---

## Key Results

### Heatwave Classification (Test set: 2022–2025)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| KNN (baseline) | 0.775 | 0.454 | 0.572 |
| Random Forest (class-weighted) | 0.835 | 0.523 | 0.643 |
| **RF + Oversampling (best overall)** | **0.841** | 0.667 | **0.744** |
| RF Tuned (highest recall) | 0.680 | **0.793** | 0.732 |

The strongest predictors were same-day max temperature (`tx`) and the prior day's max (`tx_lag1`), jointly accounting for roughly half of the tuned model's predictive weight — confirming sustained daytime heat, not any single atmospheric variable, drives heatwave detection.

#### Freeze-Risk Results — 10-Day Horizon

| Model | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| Random Forest (class-weighted) | 0.21 | 0.70 | 0.33 | Too conservative, misses more cold days |
| RF + Oversampling | 0.14 | 0.05 | 0.07 | Predicts almost everything as non-cold, high false positive rate |
| **RF Tuned (RandomizedSearchCV)** | **0.26** | 0.22 | **0.24** | Best overall, most balanced |

#### Freeze-Risk Results — 5-Day Horizon

| Model | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| Random Forest (class-weighted) | 0.25 | 0.63 | 0.36 | Good balance |
| RF + Oversampling | 0.18 | 0.94 | 0.29 | Worst precision, over-predicts freeze |
| **RF Tuned (RandomizedSearchCV)** | **0.17** | 0.06 | **0.09** | Best overall, near-perfect on the non-freeze class |

In both horizons, day-of-year (`doy_cos`) and the prior day's max temperature (`tx_lag1`) were the dominant predictors, jointly accounting for roughly half of total predictive weight in the tuned model — reinforcing that freeze risk is driven primarily by seasonal positioning and short-term temperature persistence rather than any single atmospheric variable.

### Freeze-Risk Classification (5-day and 10-day horizons)

RF Tuned (RandomizedSearchCV) was the best-balanced model on both horizons, achieving F1 ≈ 0.97 on the freeze class. Day-of-year and the prior day's max temperature (`tx_lag1`) were the dominant predictors in both the 5-day and 10-day models.

---

## Repository Structure

```
.
├── config.yaml                  # Paths to raw and clean data files, read by all notebooks
├── data/
│   ├── raw/                     # Original ECA&D CSVs, one per variable, untouched
│   └── clean/                   # Cleaned, interpolated, leap-day-free CSVs per variable
├── notebooks/
│   ├── data_cleaning_Claire.ipynb     # Raw → clean pipeline for all 10 variables
│   ├── snow_claire.ipynb              # Freeze-risk EDA, feature engineering, target definition
│   ├── heatwave_calc_Irene.ipynb      # Heatwave baseline calculation, seasonal filtering discovery
│   └── ML_heatwave.ipynb              # Model training, tuning, and evaluation (heatwave)
└── slides/
    └── AeroTherm_UK_-_Weather_Risk_Analytics.pdf   # Final presentation
```

---

## How to Run

1. Place raw ECA&D CSVs in `data/raw/` and confirm paths match `config.yaml`
2. Run `notebooks/data_cleaning_Claire.ipynb` first — it populates `data/clean/`
3. Run `notebooks/heatwave_calc_Irene.ipynb` and `notebooks/snow_claire.ipynb` to build each target variable and feature set
4. Run `notebooks/ML_heatwave.ipynb` for model training, tuning, and evaluation
5. Run `notebooks/ML_freeze.ipynb` for the freeze-risk model training, tuning, and evaluation

### Dependencies
```
pandas
numpy
matplotlib
seaborn
scikit-learn
pyyaml
```

---

## Known Limitations

- **Forecast-time lag features:** at true forecast time (predicting forward from "today"), 1-/3-/7-day lag features require genuine consecutive-day observations. Using a stale historical value as a stand-in for "yesterday" (e.g. across a long gap in the dataset) produces a climatologically meaningless lag and should be avoided — lag inputs should always come from a live, consecutive data source when deploying the model operationally.
- **Baseline seasonality:** the heatwave 90th-percentile threshold is only valid when computed against a seasonally-matched reference window (May–September); applying it against a full-year baseline materially overstates heatwave frequency, as documented above.
- Class imbalance is significant for both targets (heatwaves and deep-freeze days are rare events), so accuracy alone is not a meaningful metric for either model — recall and F1 on the minority class should be the primary reference points.

---

## Presentation Link

https://docs.google.com/presentation/d/1OdSvXevi8Q6_h6Bij0WV9fugNhzKebvX_NhdYfjhOOw/edit?usp=sharing
