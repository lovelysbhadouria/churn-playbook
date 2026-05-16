# =============================================================================
#  CHURN PLAYBOOK — PHASE 2
#  SQL Exploration with DuckDB
# =============================================================================

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os

# ── VISUAL STYLE ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi":        130,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  True,
    "axes.spines.bottom":True,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlepad":     14,
})

CHURN_COLOUR    = "#E24B4A"
STAY_COLOUR     = "#378ADD"
NEUTRAL_COLOUR  = "#888780"

os.makedirs("outputs", exist_ok=True)

# =============================================================================
#  STEP 1 — LOAD THE DATA
# =============================================================================

print("Loading data...")

df_raw = pd.read_excel("Telco_customer_churn.xlsx")

# =============================================================================
# FIX COLUMN NAMES ONLY
# =============================================================================

df_raw.columns = df_raw.columns.str.strip()

# ── FIX THE TOTAL CHARGES COLUMN ─────────────────────────────────────────────

df_raw["Total Charges"] = pd.to_numeric(
    df_raw["Total Charges"], errors="coerce"
).fillna(0)

print(f"  Rows loaded:  {len(df_raw):,}")
print(f"  Columns:      {df_raw.shape[1]}")
print(f"  Churn rate:   {(df_raw['Churn Label'] == 'Yes').mean():.1%}")

# ── CONNECT TO DUCKDB ────────────────────────────────────────────────────────

con = duckdb.connect()
con.register("customers", df_raw)

print("\nDuckDB connected. Running queries...\n")
print("=" * 60)

# =============================================================================
#  QUERY 1 — OVERALL SNAPSHOT
# =============================================================================

q1 = con.execute("""
    SELECT
        COUNT(*) AS total_customers,

        SUM(CASE WHEN "Churn Label" = 'Yes' THEN 1 ELSE 0 END)
            AS churned,

        COUNT(*) - SUM(CASE WHEN "Churn Label" = 'Yes' THEN 1 ELSE 0 END)
            AS stayed,

        ROUND(
            SUM(CASE WHEN "Churn Label" = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
            1
        ) AS churn_rate_pct

    FROM customers
""").df()

print("QUERY 1 — Overall snapshot")
print(q1.to_string(index=False))
print()

q1.to_csv("outputs/q1_overall_snapshot.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(
    ["Stayed", "Churned"],
    [q1["stayed"].iloc[0], q1["churned"].iloc[0]],
    color=[STAY_COLOUR, CHURN_COLOUR],
    width=0.45,
    edgecolor="white"
)

for bar in bars:
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + 30,
        f"{int(h):,}",
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )

ax.set_title(f"Customer retention overview  —  churn rate: {q1['churn_rate_pct'].iloc[0]}%")
ax.set_ylabel("Number of customers")
ax.set_ylim(0, q1["stayed"].iloc[0] * 1.15)

plt.tight_layout()
plt.savefig("outputs/chart1_overall_snapshot.png", bbox_inches="tight")
plt.show()

print("Saved: chart1_overall_snapshot.png\n" + "=" * 60)

# =============================================================================
#  QUERY 2 — CHURN BY CONTRACT TYPE
# =============================================================================

q2 = con.execute("""
    SELECT
        Contract,

        COUNT(*) AS total_customers,

        SUM(CASE WHEN "Churn Label" = 'Yes' THEN 1 ELSE 0 END)
            AS churned,

        ROUND(
            SUM(CASE WHEN "Churn Label" = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
            1
        ) AS churn_rate_pct

    FROM customers
    GROUP BY Contract
    ORDER BY churn_rate_pct DESC
""").df()

print("QUERY 2 — Churn by contract type")
print(q2.to_string(index=False))
print()

q2.to_csv("outputs/q2_churn_by_contract.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4))

colours = [CHURN_COLOUR if r > 20 else NEUTRAL_COLOUR for r in q2["churn_rate_pct"]]

bars = ax.barh(
    q2["Contract"],
    q2["churn_rate_pct"],
    color=colours,
    height=0.45,
    edgecolor="white"
)

for bar, val, count in zip(bars, q2["churn_rate_pct"], q2["total_customers"]):
    ax.text(
        val + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{val}%   ({count:,} customers)",
        va="center", fontsize=10
    )

ax.set_xlim(0, q2["churn_rate_pct"].max() * 1.35)
ax.set_xlabel("Churn rate (%)")
ax.set_title("Churn rate by contract type\nMonth-to-month customers leave at much higher rates")
ax.xaxis.set_major_formatter(mtick.PercentFormatter())

plt.tight_layout()
plt.savefig("outputs/chart2_churn_by_contract.png", bbox_inches="tight")
plt.show()

print("Saved: chart2_churn_by_contract.png\n" + "=" * 60)

# =============================================================================
#  QUERY 3 — CHURN BY TENURE BAND
# =============================================================================

q3 = con.execute("""
    SELECT
        CASE
            WHEN "Tenure Months" <= 12 THEN '1. New (0–12 months)'
            WHEN "Tenure Months" <= 24 THEN '2. Developing (13–24 months)'
            WHEN "Tenure Months" <= 48 THEN '3. Established (25–48 months)'
            ELSE '4. Loyal (49+ months)'
        END AS tenure_band,

        COUNT(*) AS total_customers,

        SUM(CASE WHEN "Churn Label" = 'Yes' THEN 1 ELSE 0 END)
            AS churned,

        ROUND(
            SUM(CASE WHEN "Churn Label" = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
            1
        ) AS churn_rate_pct,

        ROUND(AVG("Monthly Charges"), 2)
            AS avg_monthly_charges

    FROM customers
    GROUP BY tenure_band
    ORDER BY tenure_band
""").df()

print("QUERY 3 — Churn by tenure band")
print(q3.to_string(index=False))
print()

q3.to_csv("outputs/q3_churn_by_tenure.csv", index=False)

 
# Chart: bar with churn rate line (dual-axis) — shows both volume and rate
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()   # twinx creates a second Y-axis on the right side
 
# Strip the numbers from band names so chart labels are clean
labels = [b.split(". ")[1] for b in q3["tenure_band"]]
x = range(len(labels))
 
ax1.bar(x, q3["total_customers"], color="#B5D4F4",
        width=0.55, label="Total customers", edgecolor="white")
ax2.plot(x, q3["churn_rate_pct"], "o-",
         color=CHURN_COLOUR, linewidth=2.2, markersize=8,
         label="Churn rate %")
 
for i, (rate, pos) in enumerate(zip(q3["churn_rate_pct"], x)):
    ax2.annotate(f"{rate}%", (pos, rate), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=10,
                 color=CHURN_COLOUR, fontweight="bold")
 
ax1.set_xticks(list(x))
ax1.set_xticklabels(labels)
ax1.set_ylabel("Number of customers", color="#378ADD")
ax2.set_ylabel("Churn rate (%)", color=CHURN_COLOUR)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.set_ylim(0, q3["churn_rate_pct"].max() * 1.4)
ax1.set_title("Churn rate by customer tenure\nNew customers are most at risk")
plt.tight_layout()
plt.savefig("outputs/chart3_churn_by_tenure.png", bbox_inches="tight")
plt.show()
print("Saved: chart3_churn_by_tenure.png\n" + "=" * 60)
 
# =============================================================================
#  QUERY 4 — CHURN BY MONTHLY CHARGE BAND
# =============================================================================

q4 = con.execute("""
    SELECT
        CASE
            WHEN "Monthly Charges" < 30 THEN '1. Budget (under £30)'
            WHEN "Monthly Charges" < 60 THEN '2. Mid (£30–£59)'
            WHEN "Monthly Charges" < 80 THEN '3. High (£60–£79)'
            ELSE '4. Premium (£80+)'
        END AS charge_band,

        COUNT(*) AS total_customers,

        SUM(CASE WHEN "Churn Label" = 'Yes' THEN 1 ELSE 0 END)
            AS churned,

        ROUND(
            SUM(CASE WHEN "Churn Label" = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
            1
        ) AS churn_rate_pct,

        ROUND(AVG("Tenure Months"), 1)
            AS avg_tenure_months

    FROM customers
    GROUP BY charge_band
    ORDER BY charge_band
""").df()

print("QUERY 4 — Churn by monthly charge band")
print(q4.to_string(index=False))
print()

q4.to_csv("outputs/q4_churn_by_charges.csv", index=False)

 
# Chart: grouped view — churn rate bars coloured by band
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
labels_q4 = [b.split(". ")[1] for b in q4["charge_band"]]
bar_colours = [STAY_COLOUR, NEUTRAL_COLOUR, CHURN_COLOUR, CHURN_COLOUR]
 
bars = axes[0].bar(
    labels_q4, q4["churn_rate_pct"],
    color=bar_colours, width=0.5, edgecolor="white"
)
for bar, val in zip(bars, q4["churn_rate_pct"]):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.5, f"{val}%",
        ha="center", fontsize=10, fontweight="bold"
    )
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[0].set_ylabel("Churn rate (%)")
axes[0].set_title("Churn rate by monthly spend")
 
# Second chart: show average tenure per band — are premium churners new?
axes[1].bar(
    labels_q4, q4["avg_tenure_months"],
    color="#B5D4F4", width=0.5, edgecolor="white"
)
for i, (val, total) in enumerate(zip(q4["avg_tenure_months"], q4["total_customers"])):
    axes[1].text(i, val + 0.4, f"{val} mo\n({total:,} customers)",
                 ha="center", fontsize=9, color="#333333")
axes[1].set_ylabel("Average tenure (months)")
axes[1].set_title("Average tenure per charge band\n(lower = newer customers)")
 
plt.suptitle("Do premium customers churn more — and are they newer?",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("outputs/chart4_churn_by_charges.png", bbox_inches="tight")
plt.show()
print("Saved: chart4_churn_by_charges.png\n" + "=" * 60)

# =============================================================================
#  QUERY 5 — SERVICES COMPARISON
# =============================================================================

q5 = con.execute("""
    SELECT
        "Churn Label",

        ROUND(AVG(CASE WHEN "Online Security" = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 1)
            AS pct_online_security,

        ROUND(AVG(CASE WHEN "Tech Support" = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 1)
            AS pct_tech_support,

        ROUND(AVG(CASE WHEN "Online Backup" = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 1)
            AS pct_online_backup,

        ROUND(AVG(CASE WHEN "Device Protection" = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 1)
            AS pct_device_protection,

        ROUND(AVG(CASE WHEN "Streaming TV" = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 1)
            AS pct_streaming_tv,

        COUNT(*) AS total_in_group

    FROM customers
    GROUP BY "Churn Label"
    ORDER BY "Churn Label"
""").df()

print("QUERY 5 — Services: churners vs stayers")
print(q5.to_string(index=False))
print()

q5.to_csv("outputs/q5_services_comparison.csv", index=False)

 
# Chart: grouped bar chart — each service, two bars side by side (stayed vs churned)
services = ["pct_online_security", "pct_tech_support",
            "pct_online_backup", "pct_device_protection", "pct_streaming_tv"]
service_labels = ["Online\nSecurity", "Tech\nSupport",
                  "Online\nBackup", "Device\nProtection", "Streaming\nTV"]
 
stayed  = q5[q5["Churn Label"] == "No"][services].values.flatten()
churned = q5[q5["Churn Label"] == "Yes"][services].values.flatten()
 
x      = range(len(services))
width  = 0.35

services = [
    "pct_online_security",
    "pct_tech_support",
    "pct_online_backup",
    "pct_device_protection",
    "pct_streaming_tv"
]

service_labels = [
    "Online\nSecurity",
    "Tech\nSupport",
    "Online\nBackup",
    "Device\nProtection",
    "Streaming\nTV"
]

stayed  = q5[q5["Churn Label"] == "No"][services].values.flatten()
churned = q5[q5["Churn Label"] == "Yes"][services].values.flatten()

x = range(len(services))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5))

bars1 = ax.bar(
    [i - width/2 for i in x],
    stayed,
    width,
    label="Stayed",
    color=STAY_COLOUR,
    edgecolor="white"
)

bars2 = ax.bar(
    [i + width/2 for i in x],
    churned,
    width,
    label="Churned",
    color=CHURN_COLOUR,
    edgecolor="white"
)

ax.set_xticks(list(x))
ax.set_xticklabels(service_labels)

ax.set_ylabel("% of customers with this service")
ax.set_ylim(0, max(stayed.max(), churned.max()) * 1.2)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

ax.legend(fontsize=10)

ax.set_title(
    "Service adoption: churned vs stayed customers\n"
    "Churners consistently use fewer add-on services"
)

plt.tight_layout()
plt.savefig("outputs/chart5_services_comparison.png", bbox_inches="tight")
plt.show()

print("Saved: chart5_services_comparison.png\n" + "=" * 60)


# =============================================================================
#  CHURN PLAYBOOK — PHASE 3
#  Data Cleaning → Encoding → XGBoost Model → Evaluation
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import joblib   # used to save the trained model to disk

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

plt.rcParams.update({
    "figure.dpi":         130,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.titlepad":      14,
})

CHURN_COLOUR   = "#E24B4A"
STAY_COLOUR    = "#378ADD"
FEATURE_COLOUR = "#5B8FCC"

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("  CHURN PLAYBOOK — PHASE 3: MODEL TRAINING")
print("=" * 60)

# =============================================================================
#  SECTION 1 — LOAD THE RAW DATA
# =============================================================================

df = pd.read_excel("Telco_customer_churn.xlsx")

print(f"\nLoaded {len(df):,} rows and {df.shape[1]} columns")

# Always good practice: print the column names and data types first
# so you know what you are working with before touching anything
print("\nColumn names and types:")
print(df.dtypes.to_string())

# =============================================================================
#  SECTION 2 — DATA CLEANING
# =============================================================================
#
#  Cleaning means fixing problems in the raw data before modelling.
#  There are 3 problems to fix in this dataset:
#    1. customerID is a unique identifier — useless for prediction
#    2. TotalCharges is a text column but should be a number
#    3. Some TotalCharges values are blank spaces (new customers)

print("\n--- SECTION 2: CLEANING ---")

# ── FIX 1: Drop customerID ────────────────────────────────────────────────────
#
# customerID is like a passport number — it uniquely identifies each person
# but tells us nothing about WHY they might churn.
# If we left it in, the model might accidentally "learn" customer IDs,
# which would be completely useless on new data.

df = df.drop(columns=["CustomerID"])
print("  ✓ Dropped customerID column (unique ID, not useful for prediction)")

# ── FIX 2 & 3: TotalCharges column ───────────────────────────────────────────
#
# The TotalCharges column looks like numbers but Pandas reads it as text.
# Why? Because some rows have " " (a space) instead of a number.
# These are brand-new customers who have been with the company for 0 months
# so their total charges are genuinely zero.
#
# pd.to_numeric() tries to convert each value to a number.
# errors="coerce" means: if a value cannot be converted (like a space),
# turn it into NaN (which means "missing" in Pandas) instead of crashing.
# Then .fillna(0) replaces all those NaN values with 0.

problem_rows_before = df["Total Charges"].astype(str).str.strip().eq("").sum()
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0)
print(f"  ✓ Fixed Total Charges: {problem_rows_before} blank values → set to 0")

# Quick check: any missing values left?
missing = df.isnull().sum().sum()
print(f"  ✓ Missing values remaining: {missing}")

# =============================================================================
#  SECTION 3 — ENCODING: CONVERTING TEXT COLUMNS TO NUMBERS
# =============================================================================
#
#  This is the most important data preparation step.
#
#  Machine learning models are essentially very sophisticated maths.
#  Maths only works on numbers. So we need to convert every text value
#  to a number before the model can use it.
#
#  There are two types of text columns in this dataset, and each needs
#  a different approach:
#
#  TYPE A — Binary columns (only 2 options: Yes/No, Male/Female)
#  ──────────────────────────────────────────────────────────────
#  These are simple: Yes=1, No=0. One column stays, just with numbers.
#
#  TYPE B — Multi-category columns (3+ options like contract type)
#  ───────────────────────────────────────────────────────────────
#  These need "one-hot encoding".
#  Example: Contract has 3 values: Month-to-month, One year, Two year
#  We create 2 new columns (we drop one to avoid redundancy):
#    Contract_One year:    1 if yes, 0 if no
#    Contract_Two year:    1 if yes, 0 if no
#  If both are 0, that means Month-to-month (the "baseline" we dropped).
#
#  Why not just use 1, 2, 3 for the three contract types?
#  Because that would imply Two year (3) is "3 times more than" Month-to-month (1),
#  which is nonsense. One-hot encoding avoids that false ordering.

print("\n--- SECTION 3: ENCODING ---")

# ── STEP 3A: Target column (what we are predicting) ───────────────────────────
#
# The Churn column currently has "Yes" or "No".
# We need 1 for churned and 0 for stayed.

df["Churn Label"] = df["Churn Label"].map({"Yes": 1, "No": 0})
print(f"  ✓ Target encoded — Churn: Yes→1, No→0")
print(f"    Churners: {df['Churn Label'].sum():,} ({df['Churn Label'].mean():.1%})")
print(f"    Stayed:   {(df['Churn Label']==0).sum():,} ({(df['Churn Label']==0).mean():.1%})")

# ── STEP 3B: Simple binary columns (Yes/No → 1/0) ────────────────────────────
#
# These columns only have "Yes" or "No" as values.
# We can directly map Yes→1 and No→0.

binary_yes_no = [
    "Partner",         # does the customer have a partner?
    "Dependents",      # does the customer have dependents?
    "Phone Service",    # do they have phone service?
    "Paperless Billing" # are they on paperless billing?
]

for col in binary_yes_no:
    df[col] = df[col].map({"Yes": 1, "No": 0})
    print(f"  ✓ Encoded {col}: Yes→1, No→0")

# ── STEP 3C: Gender column ────────────────────────────────────────────────────
#
# Gender has "Male" or "Female".
# We encode one of them as 1 — it does not matter which one.

df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
print("  ✓ Encoded Gender: Male→1, Female→0")

# ── STEP 3D: SeniorCitizen column ────────────────────────────────────────────
#
# This one is already 0 or 1 in the original data — nothing to do.

print("  ✓ Senior Citizen already numeric (0/1) — no change needed")

# ── STEP 3E: "Three-value" columns that are really binary ────────────────────
#
# Some columns have three values like:
#   "Yes", "No", "No internet service"
# "No internet service" effectively means the same as "No" for our purposes.
# We simplify by treating anything that is NOT "Yes" as 0.
#
# This is a judgement call in data science — we are deciding that
# "doesn't have internet service" and "has internet but no security"
# both mean the customer does not have that add-on service.
#
# The map with a lambda checks: "is this value 'Yes'? If so, give me 1. Else 0."
# A lambda is a tiny one-line function in Python.

three_value_cols = [
    "Multiple Lines",    # Yes / No / No phone service
    "Online Security",   # Yes / No / No internet service
    "Online Backup",     # Yes / No / No internet service
    "Device Protection", # Yes / No / No internet service
    "Tech Support",      # Yes / No / No internet service
    "Streaming TV",      # Yes / No / No internet service
    "Streaming Movies",  # Yes / No / No internet service
]

for col in three_value_cols:
    df[col] = df[col].apply(lambda x: 1 if x == "Yes" else 0)
    print(f"  ✓ Encoded {col}: 'Yes'→1, everything else→0")

# ── STEP 3F: Multi-category columns — one-hot encoding ───────────────────────
#
# These columns have 3 or more categories that have NO natural order.
# We use pd.get_dummies() which automatically creates new columns for each
# category and drops one (drop_first=True) to avoid redundancy.
#
# Example BEFORE for Contract:
#   Month-to-month | One year | Two year
#
# Example AFTER (drop_first drops Month-to-month as the baseline):
#   Contract_One year | Contract_Two year
#   If both are 0 → we know it's Month-to-month

multi_cat_cols = [
    "Internet Service",  # DSL / Fiber optic / No
    "Contract",         # Month-to-month / One year / Two year
    "Payment Method",    # Electronic check / Mailed check / Bank transfer / Credit card
]

original_col_count = df.shape[1]
df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)
new_cols_created = df.shape[1] - original_col_count
print(f"  ✓ One-hot encoded {multi_cat_cols}")
print(f"    Created {new_cols_created} new columns from 3 multi-category columns")

print(f"\nFinal column count after encoding: {df.shape[1]}")
print("All text columns have been converted to numbers. ✓")

# =============================================================================
#  SECTION 4 — SEPARATE FEATURES AND TARGET
# =============================================================================
#
#  "Features" = the columns used to predict (inputs to the model).
#  "Target"   = the column we are trying to predict (output).
#
#  Think of it like a doctor's appointment:
#    Features = blood pressure, age, symptoms, test results
#    Target   = does this patient have the disease?
#
#  The model learns the relationship between features and target
#  during training, then uses that relationship to predict target
#  values for customers it has never seen before.

print("\n--- SECTION 4: FEATURES AND TARGET ---")

X = df.drop(columns=["Churn Label"])  # everything except Churn
y = df["Churn Label"]                  # only the Churn column

# 🔥 FIX: remove ALL remaining object columns (this is causing XGBoost crash)
object_cols = X.select_dtypes(include=["object"]).columns
if len(object_cols) > 0:
    print("\n  ⚠ Removing leftover object columns:", list(object_cols))
    X = X.drop(columns=object_cols)

# =============================================================================
#  SECTION 5 — TRAIN / TEST SPLIT
# =============================================================================
#
#  We split the data into two groups:
#
#  TRAINING SET (80%) — the model learns from this data.
#  TEST SET (20%)     — we hide this from the model during training,
#                       then use it to check how well the model learned.
#
#  Why hide part of the data?
#  Imagine a student who memorises all the exam questions in advance.
#  They score 100% on that exam but fail any new exam.
#  That is called "overfitting" — the model memorised the training data
#  instead of learning general patterns.
#
#  Testing on unseen data checks for real learning, not memorisation.
#
#  stratify=y ensures both sets have the same proportion of churners.
#  Without this, by bad luck you might get most churners in one set.

print("\n--- SECTION 5: TRAIN / TEST SPLIT ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing, 80% for training
    random_state=42,    # random_state=42 means the split is the same every run
    stratify=y          # keep the same churn ratio in both sets
)

print(f"  Training set: {len(X_train):,} customers "
      f"({y_train.mean():.1%} churn rate)")
print(f"  Test set:     {len(X_test):,} customers  "
      f"({y_test.mean():.1%} churn rate)")

# =============================================================================
#  SECTION 6 — HANDLING CLASS IMBALANCE
# =============================================================================
#
#  We have a problem: only 26% of customers churned.
#
#  Imagine you build a model that just predicts "No churn" for everyone.
#  It would be RIGHT 74% of the time. But it would be completely useless —
#  it would never identify a single customer who is about to leave.
#
#  This is the "class imbalance" problem.
#
#  XGBoost has a fix called scale_pos_weight.
#  It tells the model: "when you get a churn prediction wrong, that mistake
#  costs X times more than getting a non-churn prediction wrong."
#  This forces the model to pay more attention to the rare churners.
#
#  Formula: number of non-churners / number of churners
#  If there are 3 non-churners for every 1 churner, set it to 3.
#  The model then treats each churner as if it appears 3 times in the data.

print("\n--- SECTION 6: CLASS IMBALANCE ---")

n_negative  = (y_train == 0).sum()   # customers who stayed
n_positive  = (y_train == 1).sum()   # customers who churned
scale_weight = round(n_negative / n_positive, 2)

print(f"  Stayed (0):  {n_negative:,} customers")
print(f"  Churned (1): {n_positive:,} customers")
print(f"  Imbalance ratio: {scale_weight:.2f}")
print(f"  → scale_pos_weight = {scale_weight}")
print(f"    (XGBoost will treat each churner as if it appears {scale_weight}× in the data)")

# =============================================================================
#  SECTION 7 — TRAIN THE XGBOOST MODEL
# =============================================================================
#
#  XGBoost stands for "eXtreme Gradient Boosting".
#  Do not worry about the name. Here is the simple version:
#
#  1. Build a small decision tree (like a flowchart of yes/no questions)
#  2. See where it went wrong
#  3. Build another small tree that focuses on fixing those mistakes
#  4. Repeat 500 times
#  5. Combine all 500 trees by weighted voting
#
#  Each individual tree is weak (like asking one person with limited information).
#  Combined, they become very strong (like asking 500 specialists and voting).
#  This is called "ensemble learning."
#
#  PARAMETERS EXPLAINED:
#  n_estimators=500    → build 500 trees (more = more accurate up to a point)
#  learning_rate=0.05  → each new tree corrects mistakes SLOWLY and carefully
#                        (like adjusting gradually vs overcorrecting)
#  max_depth=5         → each tree can ask at most 5 yes/no questions
#                        (keeps trees simple, prevents memorisation)
#  subsample=0.8       → each tree only sees 80% of the rows, chosen randomly
#                        (prevents any one row from dominating)
#  colsample_bytree=0.8 → each tree only sees 80% of features, chosen randomly
#                        (forces trees to find different patterns)
#  eval_metric="auc"   → measure accuracy during training using AUC score
#  random_state=42     → makes results reproducible

print("\n--- SECTION 7: TRAINING XGBOOST ---")
print("  Training model... (this takes about 10-30 seconds)")

model = XGBClassifier(
    n_estimators      = 500,
    learning_rate     = 0.05,
    max_depth         = 5,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = scale_weight,
    eval_metric       = "auc",
    use_label_encoder = False,
    random_state      = 42,
    n_jobs            = -1       # use all CPU cores to train faster
)
print("\n--- FINAL SAFETY CHECK BEFORE TRAINING ---")
print("Object columns in X_train:", X_train.select_dtypes(include=["object"]).columns.tolist())

# eval_set lets us monitor performance on the test set during training
# verbose=100 prints the AUC score every 100 trees so we can watch it improve
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100
)

print("\n  ✓ Model trained successfully")

# Save the trained model to disk.
# joblib.dump() saves a Python object to a file.
# This means we can load the model later without retraining
# (training takes time — we do not want to repeat it every time).
joblib.dump(model, "outputs/churn_model.pkl")
print("  ✓ Model saved to outputs/churn_model.pkl")

# =============================================================================
#  SECTION 8 — GENERATE PREDICTIONS
# =============================================================================
#
#  The model outputs two types of predictions:
#
#  predict_proba()  → a probability between 0 and 1
#                     e.g. 0.82 means "82% chance this customer churns"
#                     This is the USEFUL output — it tells us HOW RISKY each customer is
#
#  predict()        → a hard Yes/No decision based on a 0.5 threshold
#                     i.e. if probability > 0.5 → predict churn (1), else no churn (0)
#                     This is used for the confusion matrix

print("\n--- SECTION 8: PREDICTIONS ---")

# [:, 1] means: take column index 1 (the "churn" probability)
# Column 0 would be the "not churn" probability
# They always add up to 1.0 (e.g. 0.82 + 0.18 = 1.0)
y_proba = model.predict_proba(X_test)[:, 1]
y_pred  = model.predict(X_test)

print(f"  Sample probabilities from first 5 test customers:")
for i in range(5):
    actual = "Churned" if y_test.iloc[i] == 1 else "Stayed"
    predicted = "Churn" if y_pred[i] == 1 else "No churn"
    print(f"    Customer {i+1}: {y_proba[i]:.0%} churn risk | "
          f"Predicted: {predicted} | Actual: {actual}")

# =============================================================================
#  SECTION 9 — EVALUATE THE MODEL
# =============================================================================
#
#  We need to know: how good is the model?
#  We use several metrics because no single number tells the whole story.

print("\n--- SECTION 9: EVALUATION ---")

# ── METRIC 1: ROC-AUC Score ───────────────────────────────────────────────────
#
#  AUC = Area Under the Curve. It measures how well the model SEPARATES
#  churners from non-churners.
#
#  Simple interpretation:
#    0.5  = as good as random guessing (flipping a coin)
#    0.7  = decent
#    0.8  = good
#    0.9  = excellent
#    1.0  = perfect (usually means something is wrong/data leakage)
#
#  For the Telco dataset, a well-tuned XGBoost typically gets 0.84 to 0.89.
#
#  AUC does NOT depend on the threshold (0.5 cutoff).
#  It measures the overall quality of the probability scores.

auc_score = roc_auc_score(y_test, y_proba)
print(f"\n  ROC-AUC Score: {auc_score:.4f}")

if auc_score >= 0.85:
    print("  → Excellent! The model distinguishes churners very well.")
elif auc_score >= 0.75:
    print("  → Good performance. The model has real predictive power.")
else:
    print("  → Decent start. Consider tuning hyperparameters.")

# ── METRIC 2: Confusion Matrix ────────────────────────────────────────────────
#
#  The confusion matrix shows 4 types of predictions:
#
#  True Positive  (TP): Model said "will churn"   — customer DID churn   ✓
#  True Negative  (TN): Model said "will stay"    — customer DID stay    ✓
#  False Positive (FP): Model said "will churn"   — customer STAYED      ✗ (false alarm)
#  False Negative (FN): Model said "will stay"    — customer CHURNED     ✗ (missed churner)
#
#  For churn, False Negatives are EXPENSIVE.
#  A missed churner means we did nothing to retain them and lost the customer.
#  A false alarm just means we offered a discount to someone who wasn't leaving
#  — slightly wasteful but not catastrophic.

cm      = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n  Confusion Matrix:")
print(f"    True Negatives  (correctly predicted STAYED):  {tn:4,}")
print(f"    False Positives (predicted churn, actually stayed): {fp:4,}")
print(f"    False Negatives (predicted stayed, actually churned): {fn:4,}")
print(f"    True Positives  (correctly predicted CHURNED): {tp:4,}")
print(f"\n    Missed churners (FN): {fn} — these are customers we failed to flag")

# ── METRIC 3: Precision, Recall, F1 ──────────────────────────────────────────
#
#  Precision: Of all customers we PREDICTED would churn, how many actually did?
#             "When we raise the alarm, how often are we right?"
#
#  Recall:    Of all customers who ACTUALLY churned, how many did we catch?
#             "What % of real churners did we identify?"
#
#  F1 Score:  A single number that balances precision and recall.
#
#  For churn, RECALL is usually more important than precision.
#  Missing a churner is more costly than a false alarm.

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

# =============================================================================
#  SECTION 10 — CHARTS
# =============================================================================

print("\n--- SECTION 10: CREATING CHARTS ---")

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# ── CHART 1: Confusion Matrix ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stayed", "Churned"]
).plot(ax=ax1, colorbar=False, cmap="Blues")
ax1.set_title(f"Confusion Matrix\nAUC = {auc_score:.3f}")

# Add annotations explaining each quadrant
ax1.text(0, 0, f"Correctly\npredicted\nstayed", ha="center",
         va="center", fontsize=8, color="grey", style="italic",
         transform=ax1.transData)

# ── CHART 2: ROC Curve ────────────────────────────────────────────────────────
#
#  The ROC curve shows model performance at every possible threshold.
#  The dashed line (diagonal) shows what random guessing looks like.
#  The further our curve bends toward the top-left corner, the better.
#  AUC is the area under that curve — higher is better.

ax2 = fig.add_subplot(gs[0, 1])
RocCurveDisplay.from_predictions(
    y_test, y_proba, ax=ax2,
    color=CHURN_COLOUR, name=f"XGBoost (AUC = {auc_score:.3f})"
)
ax2.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random guess (AUC = 0.5)")
ax2.legend(fontsize=9)
ax2.set_title("ROC Curve\n(closer to top-left = better)")

# ── CHART 3: Churn Probability Distribution ───────────────────────────────────
#
#  This shows whether the model clearly separates churners from stayers.
#  A good model should produce:
#    - Non-churners clustered near 0% probability
#    - Churners clustered near 100% probability
#  If both groups overlap heavily, the model is uncertain.

ax3 = fig.add_subplot(gs[1, 0])

# Split probabilities by actual churn status
prob_stayed  = y_proba[y_test == 0]
prob_churned = y_proba[y_test == 1]

ax3.hist(prob_stayed,  bins=40, alpha=0.7, color=STAY_COLOUR,
         label="Stayed", density=True)
ax3.hist(prob_churned, bins=40, alpha=0.7, color=CHURN_COLOUR,
         label="Churned", density=True)
ax3.axvline(0.5, color="black", linestyle="--", linewidth=1.2,
            label="Decision threshold (0.5)")
ax3.set_xlabel("Predicted churn probability")
ax3.set_ylabel("Density")
ax3.legend(fontsize=9)
ax3.set_title("Predicted Probability Distribution\n"
              "(good model = two separate peaks)")

# ── CHART 4: Top 15 Feature Importance ───────────────────────────────────────
#
#  Feature importance shows which columns the model relies on most.
#  XGBoost calculates this by counting how often each feature is used
#  to split data across all 500 trees.
#
#  Note: this is different from SHAP values (Phase 4).
#  Feature importance says "which features are used most."
#  SHAP says "which features push each customer toward or away from churn."
#  SHAP is more informative — but this gives us a quick sanity check now.

ax4 = fig.add_subplot(gs[1, 1])

importance_df = pd.DataFrame({
    "feature":    X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=True).tail(15)

bars = ax4.barh(
    importance_df["feature"],
    importance_df["importance"],
    color=FEATURE_COLOUR,
    edgecolor="white",
    height=0.6
)
ax4.set_xlabel("Importance score")
ax4.set_title("Top 15 Most Important Features\n(quick view — SHAP gives richer detail)")

plt.suptitle(
    f"Model Evaluation Summary  |  XGBoost  |  AUC = {auc_score:.4f}",
    fontsize=14, fontweight="bold", y=1.01
)
plt.savefig("outputs/chart_model_evaluation.png", bbox_inches="tight")
plt.show()
print("  Saved: outputs/chart_model_evaluation.png")

results    = model.evals_result()
train_auc  = results["validation_0"]["auc"]
test_auc   = results["validation_1"]["auc"]

fig2, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_auc, color=STAY_COLOUR,  linewidth=1.8, label="Train AUC", alpha=0.9)
ax.plot(test_auc,  color=CHURN_COLOUR, linewidth=1.8, label="Test AUC",  alpha=0.9)
ax.axhline(max(test_auc), color="gray", linestyle="--", linewidth=0.8,
           label=f"Best test AUC: {max(test_auc):.4f}")
ax.set_xlabel("Number of trees trained")
ax.set_ylabel("AUC score")
ax.legend()
ax.set_title("Learning Curve — How AUC Changed During Training\n"
             "Healthy = both lines improve, then level off together")
plt.tight_layout()
plt.savefig("outputs/chart_learning_curve.png", bbox_inches="tight")
plt.show()
print("  Saved: outputs/chart_learning_curve.png")

# =============================================================================
#  SECTION 11 — EXPORT PREDICTIONS FOR POWER BI
# =============================================================================
#
#  We attach the model's predictions back to the original (uncleaned) data.
#  This gives us a CSV with both the raw customer info AND the model's
#  churn probability for each customer.
#

print("\n--- SECTION 11: EXPORTING PREDICTIONS ---")

# Reload the original data to get readable column values (not encoded ones)
df_original = pd.read_excel("Telco_customer_churn.xlsx")
df_original["Total Charges"] = pd.to_numeric(
    df_original["Total Charges"], errors="coerce").fillna(0)

# The test set has 1,409 rows. We need to line them up with the original data.
# X_test.index contains the original row numbers — use those.
df_predictions = df_original.loc[X_test.index].copy()
df_predictions["churn_probability"] = y_proba
df_predictions["predicted_churn"]   = y_pred
df_predictions["actual_churn"]      = y_test.values

# Create a readable risk label based on probability
def risk_label(prob):
    if prob >= 0.75:   return "High risk"
    elif prob >= 0.50: return "Medium risk"
    elif prob >= 0.25: return "Low risk"
    else:              return "Very low risk"

df_predictions["risk_level"] = df_predictions["churn_probability"].apply(risk_label)

df_predictions = df_predictions.sort_values("churn_probability", ascending=False)
df_predictions.to_csv("outputs/predictions_with_risk.csv", index=False)

# Print a summary of the risk tiers
risk_counts = df_predictions["risk_level"].value_counts()
print("\n  Risk tier breakdown in test set:")
for tier, count in risk_counts.items():
    pct = count / len(df_predictions) * 100
    print(f"    {tier:<15}: {count:4,}  ({pct:.1f}%)")

print(f"\n  Saved: outputs/predictions_with_risk.csv")
print(f"  Total rows: {len(df_predictions):,}")
print(f"  Columns: {list(df_predictions.columns)}")
