# =============================================================================
#  CHURN PLAYBOOK — PHASE 4
#  SHAP: Explaining WHY Each Customer Is Predicted to Churn
# =============================================================================
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

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

CHURN_COLOUR  = "#E24B4A"
STAY_COLOUR   = "#378ADD"

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("  CHURN PLAYBOOK — PHASE 4: SHAP EXPLANATIONS")
print("=" * 60)

# =============================================================================
#  SECTION 1 — RELOAD DATA AND MODEL
# =============================================================================


print("\nLoading data and model...")

# ── Load raw data ─────────────────────────────────────────────────────────────
df = pd.read_excel("Telco_customer_churn.xlsx")
df_original = df.copy()   # keep a clean copy with original text values for display later

# ── Cleaning (same steps as Phase 3) ──────────────────────────────────────────
df = df.drop(columns=["CustomerID"])
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0)

# ── Encoding (same steps as Phase 3) ──────────────────────────────────────────
df["Churn Label"] = df["Churn Label"].map({"Yes": 1, "No": 0})

binary_yes_no = ["Partner", "Dependents", "Phone Service", "Paperless Billing"]
for col in binary_yes_no:
    df[col] = df[col].map({"Yes": 1, "No": 0})

df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

three_value_cols = ["Multiple Lines", "Online Security", "Online Backup",
                    "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"]
for col in three_value_cols:
    df[col] = df[col].apply(lambda x: 1 if x == "Yes" else 0)

df = pd.get_dummies(df, columns=["Internet Service", "Contract", "Payment Method"],
                    drop_first=True)

# ── Recreate the same train/test split ────────────────────────────────────────
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Churn Label"])
y = df["Churn Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Load the trained model saved by Phase 3 ───────────────────────────────────
# joblib.load() reads the model file back from disk.
# This is the same model that took 30 seconds to train — no need to retrain.
model = joblib.load("outputs/churn_model.pkl")

# Also load the predictions file so we know each customer's risk score
df_preds = pd.read_csv("outputs/predictions_with_risk.csv")

print(f"  ✓ Data loaded:  {len(X_test):,} test customers, {X_test.shape[1]} features")
print(f"  ✓ Model loaded: {model.n_estimators} trees")

# =============================================================================
#  SECTION 2 — WHAT IS SHAP? THE SIMPLEST POSSIBLE EXPLANATION
# =============================================================================
#

print("\n--- SECTION 3: COMPUTING SHAP VALUES ---")
print("  This takes 20–60 seconds...")

explainer   = shap.TreeExplainer(model)
# Remove leftover object columns before SHAP
object_cols = X_test.select_dtypes(include=["object"]).columns

if len(object_cols) > 0:
    print("\n  ⚠ Removing object columns before SHAP:", list(object_cols))
    X_test = X_test.drop(columns=object_cols)
shap_values = explainer(X_test)   # shape: (1409 customers, 26 features)

# Let us understand the shape of what we got
print(f"\n  ✓ SHAP values computed")
print(f"    Shape of SHAP matrix: {shap_values.values.shape}")
print(f"    → {shap_values.values.shape[0]:,} customers × {shap_values.values.shape[1]} features")
print(f"\n  Base value (average churn prob across all customers):")
print(f"    {shap_values.base_values[0]:.4f}  ({shap_values.base_values[0]*100:.1f}%)")
print(f"    Every customer starts here before their features adjust the prediction")

# Verify the maths — SHAP values should sum to the prediction
sample_idx        = 0
shap_sum          = shap_values.values[sample_idx].sum()
base              = shap_values.base_values[sample_idx]
manual_prediction = base + shap_sum
model_prediction  = model.predict_proba(X_test.iloc[[sample_idx]])[0, 1]

print(f"\n  Verification (customer #{sample_idx}):")
print(f"    Base value:            {base:.4f}")
print(f"    Sum of SHAP values:   +{shap_sum:.4f}")
print(f"    Manual total:          {manual_prediction:.4f}")
print(f"    Model prediction:      {model_prediction:.4f}")
print(f"    Match: {'✓ Yes' if abs(manual_prediction - model_prediction) < 0.001 else '✗ No'}")
print(f"    (They should match — proves SHAP values are consistent with the model)")

# =============================================================================
#  SECTION 4 — BEESWARM PLOT: THE GLOBAL OVERVIEW
# =============================================================================
#

print("\n--- SECTION 4: BEESWARM PLOT ---")
print("  Creating beeswarm plot...")

plt.figure(figsize=(11, 8))
shap.plots.beeswarm(
    shap_values,
    max_display=18,    # show top 18 features (sorted by importance automatically)
    show=False
)
plt.title(
    "Beeswarm Plot — What Drives Churn Across All Customers\n"
    "Each dot = one customer  |  Right = pushed toward churn  |  Colour = feature value",
    pad=14, fontsize=12
)
plt.tight_layout()
plt.savefig("outputs/chart_shap_beeswarm.png", bbox_inches="tight")
plt.show()
print("  ✓ Saved: outputs/chart_shap_beeswarm.png")
print("    → This is your most important chart. Use it in your LinkedIn post and GitHub README.")

# =============================================================================
#  SECTION 5 — BAR CHART: RANKED FEATURE IMPORTANCE
# =============================================================================
#

print("\n--- SECTION 5: FEATURE IMPORTANCE BAR CHART ---")
print("  Creating bar chart...")

plt.figure(figsize=(9, 7))
shap.plots.bar(
    shap_values,
    max_display=15,
    show=False
)
plt.title(
    "Top 15 Features by Mean |SHAP Value|\n"
    "Bar length = how much this feature moves the prediction on average",
    pad=14, fontsize=12
)
plt.tight_layout()
plt.savefig("outputs/chart_shap_bar.png", bbox_inches="tight")
plt.show()
print("  ✓ Saved: outputs/chart_shap_bar.png")

# Print the numbers as a table so you can reference them
mean_shap = pd.DataFrame({
    "feature":    X_test.columns,
    "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print("\n  Top 10 features by mean |SHAP| value:")
print(f"  {'Feature':<45} {'Mean |SHAP|':>12}")
print("  " + "-" * 58)
for _, row in mean_shap.head(10).iterrows():
    bar = "█" * int(row["mean_abs_shap"] * 200)
    print(f"  {row['feature']:<45} {row['mean_abs_shap']:>10.4f}  {bar}")

# =============================================================================
#  SECTION 6 — WATERFALL PLOT: HIGH-RISK CUSTOMER
# =============================================================================
#
#  A waterfall plot is a "receipt" for one specific customer.
#  It shows exactly how the model arrived at their churn probability.
#
print("\n--- SECTION 6: WATERFALL — HIGH RISK CUSTOMER ---")

# Get churn probabilities for all test customers
y_proba = model.predict_proba(X_test)[:, 1]

# Find the customer with the HIGHEST predicted churn probability
high_risk_pos  = np.argmax(y_proba)    # position in X_test (0 to 1408)
high_risk_prob = y_proba[high_risk_pos]
high_risk_actual = y_test.iloc[high_risk_pos]

# Get this customer's original (readable) information
high_risk_original_idx = X_test.index[high_risk_pos]
high_risk_original     = df_original.loc[high_risk_original_idx]

print(f"\n  Highest-risk customer found (test position #{high_risk_pos}):")
print(f"    Predicted churn probability: {high_risk_prob:.1%}")
print(f"    Actual outcome:  {'CHURNED ✓' if high_risk_actual == 1 else 'STAYED ✗'}")
print(f"\n  Their profile (original values before encoding):")
profile_cols = ["Gender", "Senior Citizen", "Tenure Months", "Contract",
                "Monthly Charges", "Total Charges", "Internet Service",
                "Tech Support", "Online Security"]
for col in profile_cols:
    print(f"    {col:<25}: {high_risk_original[col]}")

# Create the waterfall plot
plt.figure(figsize=(11, 7))
shap.plots.waterfall(
    shap_values[high_risk_pos],
    max_display=14,     # show the 14 biggest contributors
    show=False
)
plt.title(
    f"HIGH-RISK Customer — Predicted Churn: {high_risk_prob:.0%}  "
    f"|  Actual: {'Churned' if high_risk_actual == 1 else 'Stayed'}\n"
    f"Red = pushed toward churn  |  Blue = pushed away  "
    f"|  Numbers = this customer's actual feature values",
    pad=14, fontsize=11
)
plt.tight_layout()
plt.savefig("outputs/chart_waterfall_high_risk.png", bbox_inches="tight")
plt.show()
print("  ✓ Saved: outputs/chart_waterfall_high_risk.png")
print("    → This is the chart you show when explaining: 'here is why THIS customer is at risk'")

# Print a plain-English reading of the top SHAP drivers for this customer
print(f"\n  Plain-English explanation for this customer:")
shap_this_customer = pd.Series(
    shap_values.values[high_risk_pos],
    index=X_test.columns
).sort_values(ascending=False)

print(f"  Factors pushing TOWARD churn (red):")
for feat, val in shap_this_customer[shap_this_customer > 0.01].head(5).items():
    print(f"    +{val:.3f}  {feat} = {X_test.iloc[high_risk_pos][feat]:.0f}")

print(f"  Factors pushing AWAY from churn (blue):")
for feat, val in shap_this_customer[shap_this_customer < -0.01].tail(5).items():
    print(f"    {val:.3f}  {feat} = {X_test.iloc[high_risk_pos][feat]:.0f}")

# =============================================================================
#  SECTION 7 — WATERFALL PLOT: LOW-RISK CUSTOMER
# =============================================================================
#

print("\n--- SECTION 7: WATERFALL — LOW RISK CUSTOMER ---")

low_risk_pos    = np.argmin(y_proba)
low_risk_prob   = y_proba[low_risk_pos]
low_risk_actual = y_test.iloc[low_risk_pos]

low_risk_original_idx = X_test.index[low_risk_pos]
low_risk_original     = df_original.loc[low_risk_original_idx]

print(f"\n  Lowest-risk customer found (test position #{low_risk_pos}):")
print(f"    Predicted churn probability: {low_risk_prob:.1%}")
print(f"    Actual outcome:  {'CHURNED ✗' if low_risk_actual == 1 else 'STAYED ✓'}")
print(f"\n  Their profile:")
for col in profile_cols:
    print(f"    {col:<25}: {low_risk_original[col]}")

plt.figure(figsize=(11, 7))
shap.plots.waterfall(
    shap_values[low_risk_pos],
    max_display=14,
    show=False
)
plt.title(
    f"LOW-RISK Customer — Predicted Churn: {low_risk_prob:.0%}  "
    f"|  Actual: {'Churned' if low_risk_actual == 1 else 'Stayed'}\n"
    f"Blue bars dominate → features are protecting this customer from churning",
    pad=14, fontsize=11
)
plt.tight_layout()
plt.savefig("outputs/chart_waterfall_low_risk.png", bbox_inches="tight")
plt.show()
print("  ✓ Saved: outputs/chart_waterfall_low_risk.png")

# =============================================================================
#  SECTION 8 — SIDE-BY-SIDE COMPARISON CHART
# =============================================================================
#

print("\n--- SECTION 8: SIDE-BY-SIDE COMPARISON ---")

# Get SHAP values for both customers
shap_high = pd.Series(shap_values.values[high_risk_pos], index=X_test.columns)
shap_low  = pd.Series(shap_values.values[low_risk_pos],  index=X_test.columns)

# Find the top drivers across BOTH customers (union of top features)
top_feats_high = set(shap_high.abs().nlargest(8).index)
top_feats_low  = set(shap_low.abs().nlargest(8).index)
top_feats      = sorted(top_feats_high | top_feats_low,
                         key=lambda f: (shap_high[f] + shap_low[f]),
                         reverse=True)[:10]

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

def plot_comparison_bars(ax, shap_series, title, prob, colour_pos, colour_neg):
    values = shap_series[top_feats]
    colors = [colour_pos if v > 0 else colour_neg for v in values]
    y_pos  = range(len(top_feats))

    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.6)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top_feats, fontsize=10)
    ax.set_xlabel("SHAP value\n(+ = pushes toward churn,  – = pushes away)")
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        x_pos = val + 0.003 if val >= 0 else val - 0.003
        align = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=align, fontsize=9)

    ax.set_xlim(
        min(values.min() * 1.4, -0.05),
        max(values.max() * 1.4,  0.05)
    )

plot_comparison_bars(
    axes[0], shap_high,
    f"HIGH-RISK Customer\nPredicted: {high_risk_prob:.0%} churn",
    CHURN_COLOUR, CHURN_COLOUR, STAY_COLOUR
)
plot_comparison_bars(
    axes[1], shap_low,
    f"LOW-RISK Customer\nPredicted: {low_risk_prob:.0%} churn",
    STAY_COLOUR, CHURN_COLOUR, STAY_COLOUR
)

axes[1].tick_params(labelleft=True)

fig.suptitle(
    "SHAP Driver Comparison — High Risk vs Low Risk Customer\n"
    "Same features, opposite effects — this is what the model learned",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig("outputs/chart_shap_comparison.png", bbox_inches="tight")
plt.show()
print("  ✓ Saved: outputs/chart_shap_comparison.png")
print("    → Best image for a LinkedIn post — shows contrast clearly")

# =============================================================================
#  SECTION 9 — DEPENDENCE PLOT: HOW TENURE AFFECTS CHURN RISK
# =============================================================================
#

print("\n--- SECTION 9: DEPENDENCE PLOT ---")

# Find the top feature from the bar chart (usually Contract or tenure)
top_feature = mean_shap.iloc[0]["feature"]

# Also do tenure specifically since it has a nice non-linear pattern
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Top feature from model
ax = axes[0]
tenure_shap = shap_values.values[:, list(X_test.columns).index("Tenure Months")]
tenure_vals  = X_test["Tenure Months"].values

scatter = ax.scatter(
    tenure_vals, tenure_shap,
    c=tenure_vals,
    cmap="RdYlBu_r",
    alpha=0.5, s=12, edgecolors="none"
)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Tenure (months with company)")
ax.set_ylabel("SHAP value for tenure\n(+ = pushing toward churn)")
ax.set_title(
    "How Tenure Affects Churn Risk\n"
    "New customers → high positive SHAP (at risk)\n"
    "Long-tenure customers → negative SHAP (protected)"
)
plt.colorbar(scatter, ax=ax, label="Tenure value")

# Add annotations to explain the pattern
ax.annotate("New customers\nat high risk",
            xy=(5, tenure_shap[tenure_vals < 10].mean()),
            xytext=(15, 0.35),
            arrowprops=dict(arrowstyle="->", color=CHURN_COLOUR),
            fontsize=9, color=CHURN_COLOUR)

ax.annotate("Long-tenure customers\nrarely churn",
            xy=(65, tenure_shap[tenure_vals > 55].mean()),
            xytext=(35, -0.25),
            arrowprops=dict(arrowstyle="->", color=STAY_COLOUR),
            fontsize=9, color=STAY_COLOUR)

# Plot 2: Monthly charges dependence
ax2 = axes[1]
charges_col  = "Monthly Charges"
charges_shap = shap_values.values[:, list(X_test.columns).index(charges_col)]
charges_vals  = X_test[charges_col].values

tenure_colours = X_test["Tenure Months"].values  # colour dots by tenure

scatter2 = ax2.scatter(
    charges_vals, charges_shap,
    c=tenure_colours,
    cmap="RdYlBu_r",
    alpha=0.5, s=12, edgecolors="none"
)
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax2.set_xlabel("Monthly Charges (£)")
ax2.set_ylabel("SHAP value for MonthlyCharges\n(+ = pushing toward churn)")
ax2.set_title(
    "How Monthly Charges Affect Churn Risk\n"
    "(dot colour = tenure — red = new customer)"
)
plt.colorbar(scatter2, ax=ax2, label="Tenure (months)")

plt.suptitle(
    "Dependence Plots — How Feature VALUES Relate to SHAP Values\n"
    "These show the NON-LINEAR patterns the model learned",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig("outputs/chart_shap_dependence.png", bbox_inches="tight")
plt.show()
print("  ✓ Saved: outputs/chart_shap_dependence.png")
print("    → Shows the model has learned non-linear patterns, not just rules")

# =============================================================================
#  SECTION 10 — EXPORT SHAP VALUES FOR PHASE 5 (CLUSTERING)
# =============================================================================


print("\n--- SECTION 10: EXPORTING SHAP VALUES ---")

# ── Export 1: Full SHAP values matrix ────────────────────────────────────────
shap_df = pd.DataFrame(
    shap_values.values,
    columns=[f"shap_{col}" for col in X_test.columns],
    index=X_test.index
)
shap_df["churn_probability"] = y_proba
shap_df["actual_churn"]      = y_test.values
shap_df.to_csv("outputs/shap_values_matrix.csv")
print(f"  ✓ Saved SHAP matrix: outputs/shap_values_matrix.csv")
print(f"    Shape: {shap_df.shape}  ({shap_df.shape[0]:,} customers × {shap_df.shape[1]} columns)")

# ── Export 2: Human-readable top-3 SHAP drivers per customer ─────────────────
#
# For each customer, find which 3 features had the BIGGEST positive SHAP
# (i.e. the top 3 reasons the model thinks they will churn).
# This becomes the "churn reason" we use in the playbook.

def get_top_churn_drivers(row_shap, n=3):
    """Returns the top n positive SHAP drivers for one customer."""
    s = pd.Series(row_shap, index=X_test.columns)
    top = s[s > 0].nlargest(n)
    return [f"{feat}: +{val:.3f}" for feat, val in top.items()]

print("\n  Building human-readable summary per customer...")

summary_rows = []
for i in range(len(X_test)):
    drivers = get_top_churn_drivers(shap_values.values[i])
    summary_rows.append({
        "customer_idx":     X_test.index[i],
        "churn_probability": round(y_proba[i], 4),
        "actual_churn":     int(y_test.iloc[i]),
        "risk_level":       ("High" if y_proba[i] >= 0.75 else
                             "Medium" if y_proba[i] >= 0.50 else
                             "Low" if y_proba[i] >= 0.25 else "Very Low"),
        "top_churn_driver_1": drivers[0] if len(drivers) > 0 else "",
        "top_churn_driver_2": drivers[1] if len(drivers) > 1 else "",
        "top_churn_driver_3": drivers[2] if len(drivers) > 2 else "",
    })

summary_df = pd.DataFrame(summary_rows).sort_values(
    "churn_probability", ascending=False)
summary_df.to_csv("outputs/shap_customer_summary.csv", index=False)
print(f"  ✓ Saved summary: outputs/shap_customer_summary.csv")

# Preview top 5 at-risk customers
print(f"\n  Top 5 highest-risk customers with their churn reasons:")
print(f"  {'Prob':>6}  {'Risk':<9}  {'Top Driver'}")
print("  " + "-" * 65)
for _, row in summary_df.head(5).iterrows():
    print(f"  {row['churn_probability']:>5.0%}  "
          f"{row['risk_level']:<9}  {row['top_churn_driver_1']}")

