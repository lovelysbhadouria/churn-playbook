# =============================================================================
#  CHURN PLAYBOOK — PHASE 5
#  Clustering At-Risk Customers by Their Churn REASON
# =============================================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster        import KMeans
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split

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

CLUSTER_PALETTE = ["#E24B4A", "#378ADD", "#F4A623", "#5CB85C", "#9B59B6", "#1ABC9C"]
os.makedirs("outputs", exist_ok=True)

print("=" * 65)
print("  CHURN PLAYBOOK — PHASE 5: CLUSTERING BY CHURN REASON")
print("=" * 65)

# =============================================================================
#  SECTION 2 — LOAD DATA FROM PREVIOUS PHASES
# =============================================================================

print("\nLoading outputs from Phase 3 and Phase 4...")

# Load the SHAP values matrix saved by Phase 4
# This has one row per test customer and one column per feature's SHAP value
shap_df = pd.read_csv("outputs/shap_values_matrix.csv", index_col=0)

# Separate the SHAP columns from the metadata columns
# SHAP columns start with "shap_" — the rest are churn_probability and actual_churn
shap_cols = [c for c in shap_df.columns if c.startswith("shap_")]
meta_cols  = [c for c in shap_df.columns if not c.startswith("shap_")]

print(f"  ✓ SHAP matrix loaded")
print(f"    Customers: {len(shap_df):,}")
print(f"    SHAP features: {len(shap_cols)}")
print(f"    Churn probability range: "
      f"{shap_df['churn_probability'].min():.1%} – "
      f"{shap_df['churn_probability'].max():.1%}")

# Also reload original data so we can show readable profiles later
df_original = pd.read_excel("Telco_customer_churn.xlsx")
df_original["Total Charges"] = pd.to_numeric(
    df_original["Total Charges"], errors="coerce").fillna(0)
df_original = df_original.set_index(df_original.index)

# Reload the predictions file from Phase 3 for the risk labels
df_preds = pd.read_csv("outputs/predictions_with_risk.csv")

# Recreate X_test so we can access original encoded feature values
# (Same preprocessing + same random_state=42 = same split every time)
df_enc = df_original.copy().drop(columns=["CustomerID"])
df_enc["Total Charges"] = pd.to_numeric(df_enc["Total Charges"], errors="coerce").fillna(0)
df_enc["Churn Label"]         = df_enc["Churn Label"].map({"Yes": 1, "No": 0})
for col in ["Partner", "Dependents", "Phone Service", "Paperless Billing"]:
    df_enc[col] = df_enc[col].map({"Yes": 1, "No": 0})
df_enc["Gender"] = df_enc["Gender"].map({"Male": 1, "Female": 0})
for col in ["Multiple Lines", "Online Security", "Online Backup",
            "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"]:
    df_enc[col] = df_enc[col].apply(lambda x: 1 if x == "Yes" else 0)
df_enc = pd.get_dummies(df_enc, columns=["Internet Service", "Contract", "Payment Method"],
                        drop_first=True)
X = df_enc.drop(columns=["Churn Label"])
y = df_enc["Churn Label"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"  ✓ Original data and X_test reconstructed")

# =============================================================================
#  SECTION 3 — FILTER TO PREDICTED CHURNERS ONLY
# =============================================================================
#
#  We only want to cluster customers the model flagged as at-risk.
#  Clustering all 1,409 customers including the safe ones would:
#    a) Make the clusters less meaningful (mixing at-risk with safe)
#    b) Waste effort — the business only needs to act on at-risk customers
#
#  We use a threshold of 0.5 (50%) — same as the model's default
#  predict() threshold. Customers above 0.5 are "predicted to churn."
#
#  You can lower this to 0.4 if you want to catch borderline cases too.

print("\n--- SECTION 3: FILTERING AT-RISK CUSTOMERS ---")

CHURN_THRESHOLD = 0.50   # change this to 0.40 to be more aggressive

at_risk_mask   = shap_df["churn_probability"] >= CHURN_THRESHOLD
shap_at_risk   = shap_df[at_risk_mask]
shap_values_at_risk = shap_at_risk[shap_cols].values   # just the numbers

print(f"  Threshold: {CHURN_THRESHOLD:.0%}")
print(f"  All test customers:      {len(shap_df):,}")
print(f"  Predicted churners:      {at_risk_mask.sum():,}  "
      f"({at_risk_mask.mean():.1%} of test set)")
print(f"  Customers we will cluster: {len(shap_at_risk):,}")
print(f"\n  We will cluster THESE {len(shap_at_risk):,} customers")
print(f"  by the REASON the model thinks they will churn.")

# =============================================================================
#  SECTION 4 — SCALE THE SHAP VALUES
# =============================================================================
#
#  PROBLEM: SHAP values for different features have different scales.
#    SHAP for MonthlyCharges might range from -0.4 to +0.8
#    SHAP for SeniorCitizen might range from -0.05 to +0.15
#

print("\n--- SECTION 4: SCALING SHAP VALUES ---")

scaler             = StandardScaler()
shap_scaled        = scaler.fit_transform(shap_values_at_risk)

print(f"  BEFORE scaling — first feature stats:")
print(f"    Mean:  {shap_values_at_risk[:, 0].mean():.4f}")
print(f"    Std:   {shap_values_at_risk[:, 0].std():.4f}")
print(f"    Range: {shap_values_at_risk[:, 0].min():.4f} to {shap_values_at_risk[:, 0].max():.4f}")

print(f"\n  AFTER scaling — first feature stats:")
print(f"    Mean:  {shap_scaled[:, 0].mean():.4f}  (always 0 after StandardScaler)")
print(f"    Std:   {shap_scaled[:, 0].std():.4f}   (always 1 after StandardScaler)")
print(f"    Range: {shap_scaled[:, 0].min():.4f} to {shap_scaled[:, 0].max():.4f}")
print(f"\n  ✓ All {len(shap_cols)} features now on equal footing")

# =============================================================================
#  SECTION 5 — ELBOW METHOD: FINDING THE RIGHT NUMBER OF CLUSTERS
# =============================================================================
#

print("\n--- SECTION 5: ELBOW METHOD ---")
print("  Testing k from 2 to 9 (this takes about 30 seconds)...")

inertias          = []
silhouette_scores = []
k_range           = range(2, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=15, max_iter=300)
    km.fit(shap_scaled)
    inertias.append(km.inertia_)

    # Silhouette score: how well-separated are the clusters?
    # Ranges from -1 (bad, overlapping) to +1 (perfect, well-separated)
    # 0.2-0.4 is decent for real-world data
    # We only compute this for k >= 2
    sil = silhouette_score(shap_scaled, km.labels_, sample_size=500, random_state=42)
    silhouette_scores.append(sil)

    print(f"    k={k}: inertia={km.inertia_:,.0f}  |  silhouette={sil:.4f}")

# Chart: Elbow + Silhouette side by side
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Elbow plot
axes[0].plot(list(k_range), inertias, "o-",
             color="#378ADD", linewidth=2.2, markersize=8, zorder=3)
axes[0].fill_between(list(k_range), inertias, alpha=0.08, color="#378ADD")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (total within-cluster distance)")
axes[0].set_xticks(list(k_range))
axes[0].set_title("Elbow Method\n"
                  "Look for where the curve stops dropping sharply")
axes[0].grid(axis="y", alpha=0.3)

# Annotate the likely elbow
inertia_drops = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
elbow_k       = list(k_range)[1 + inertia_drops.index(max(inertia_drops[1:]))]
axes[0].axvline(elbow_k, color="#E24B4A", linestyle="--", linewidth=1.2,
                label=f"Suggested k = {elbow_k}")
axes[0].legend(fontsize=9)

# Silhouette plot
axes[1].plot(list(k_range), silhouette_scores, "o-",
             color="#5CB85C", linewidth=2.2, markersize=8, zorder=3)
axes[1].fill_between(list(k_range), silhouette_scores, alpha=0.08, color="#5CB85C")
best_k_sil = list(k_range)[silhouette_scores.index(max(silhouette_scores))]
axes[1].axvline(best_k_sil, color="#E24B4A", linestyle="--", linewidth=1.2,
                label=f"Best silhouette at k = {best_k_sil}")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette score (higher = better separated)")
axes[1].set_xticks(list(k_range))
axes[1].set_title("Silhouette Score\n"
                  "Higher = clusters are more distinct from each other")
axes[1].legend(fontsize=9)
axes[1].grid(axis="y", alpha=0.3)

plt.suptitle("How Many Clusters Should We Use?\n"
             "Look at BOTH charts — pick k where elbow bends AND silhouette peaks",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/chart_elbow_silhouette.png", bbox_inches="tight")
plt.show()
print("\n  ✓ Saved: outputs/chart_elbow_silhouette.png")
print(f"\n  Elbow method suggests: k = {elbow_k}")
print(f"  Silhouette suggests:   k = {best_k_sil}")

# =============================================================================
#  SECTION 6 — RUN KMEANS WITH CHOSEN K
# =============================================================================
#

K_CLUSTERS = 4   # ← CHANGE THIS based on your elbow/silhouette chart

print(f"\n--- SECTION 6: RUNNING KMEANS WITH k={K_CLUSTERS} ---")
print(f"  (Change K_CLUSTERS at line above if you want a different number)")

final_km      = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=15, max_iter=500)
cluster_labels = final_km.fit_predict(shap_scaled)

# Add cluster labels to the SHAP dataframe
shap_at_risk   = shap_at_risk.copy()
shap_at_risk["cluster"] = cluster_labels

final_sil = silhouette_score(shap_scaled, cluster_labels)
print(f"  ✓ KMeans complete")
print(f"    Final silhouette score: {final_sil:.4f}")
print(f"\n  Cluster sizes:")
for c in range(K_CLUSTERS):
    n     = (cluster_labels == c).sum()
    avg_p = shap_at_risk[shap_at_risk["cluster"] == c]["churn_probability"].mean()
    print(f"    Cluster {c}: {n:4,} customers  |  avg churn prob: {avg_p:.0%}")

# =============================================================================
#  SECTION 7 — NAME EACH CLUSTER BY ITS DOMINANT CHURN REASON
# =============================================================================
#

print(f"\n--- SECTION 7: NAMING CLUSTERS ---")

# Human-readable name for each encoded feature
FEATURE_NAMES = {
    "shap_tenure":                              "Short tenure (new customer)",
    "shap_MonthlyCharges":                      "High monthly charges",
    "shap_TotalCharges":                        "Low total spend",
    "shap_Contract_One year":                   "No annual contract",
    "shap_Contract_Two year":                   "No two-year contract",
    "shap_InternetService_Fiber optic":         "Fiber optic internet",
    "shap_InternetService_No":                  "No internet service",
    "shap_PaymentMethod_Electronic check":      "Electronic check payment",
    "shap_PaymentMethod_Mailed check":          "Mailed check payment",
    "shap_PaymentMethod_Credit card (automatic)":"Credit card payment",
    "shap_TechSupport":                         "No tech support",
    "shap_OnlineSecurity":                      "No online security",
    "shap_OnlineBackup":                        "No online backup",
    "shap_DeviceProtection":                    "No device protection",
    "shap_StreamingTV":                         "No streaming TV",
    "shap_StreamingMovies":                     "No streaming movies",
    "shap_MultipleLines":                       "Single phone line",
    "shap_PhoneService":                        "Has phone service",
    "shap_PaperlessBilling":                    "Paperless billing",
    "shap_SeniorCitizen":                       "Senior citizen",
    "shap_gender":                              "Gender",
    "shap_Partner":                             "No partner",
    "shap_Dependents":                          "No dependents",
}

cluster_info = {}

print(f"\n  {'='*60}")
for c in range(K_CLUSTERS):

    mask_c       = shap_at_risk["cluster"] == c
    cluster_shap = shap_at_risk.loc[mask_c, shap_cols]

    mean_shap_c  = cluster_shap.mean()
    n_customers  = mask_c.sum()
    avg_prob     = shap_at_risk.loc[mask_c, "churn_probability"].mean()

    # Top positive SHAP features = primary reasons for churn in this cluster
    top_drivers  = mean_shap_c[mean_shap_c > 0].nlargest(5)
    top_protectors = mean_shap_c[mean_shap_c < 0].nsmallest(3)

    # Primary name = the single biggest positive SHAP driver
    primary_feat = top_drivers.index[0] if len(top_drivers) > 0 else mean_shap_c.idxmax()
    cluster_name = FEATURE_NAMES.get(primary_feat, primary_feat.replace("shap_", ""))

    cluster_info[c] = {
        "name":            cluster_name,
        "n_customers":     n_customers,
        "avg_prob":        avg_prob,
        "mean_shap":       mean_shap_c,
        "top_drivers":     top_drivers,
        "top_protectors":  top_protectors,
        "primary_feature": primary_feat,
    }

    print(f"\n  CLUSTER {c} — {n_customers} customers | avg churn prob: {avg_prob:.0%}")
    print(f"  Primary churn reason: {cluster_name}")
    print(f"  Top SHAP drivers (pushing TOWARD churn):")
    for feat, val in top_drivers.items():
        readable = FEATURE_NAMES.get(feat, feat.replace("shap_", ""))
        print(f"    {readable:<40} SHAP = {val:+.4f}")
    print(f"  Top protectors (pushing AWAY from churn):")
    for feat, val in top_protectors.items():
        readable = FEATURE_NAMES.get(feat, feat.replace("shap_", ""))
        print(f"    {readable:<40} SHAP = {val:+.4f}")

print(f"  {'='*60}")

# =============================================================================
#  SECTION 8 — HEATMAP: CHURN REASONS PER CLUSTER
# =============================================================================
#

print(f"\n--- SECTION 8: CHURN REASON HEATMAP ---")

# Find the most important features across ALL clusters
# (show the ones that differ most between clusters)
all_mean_shap = pd.DataFrame(
    {f"Cluster {c}": cluster_info[c]["mean_shap"] for c in range(K_CLUSTERS)}
).T

# Only show the top features — those with the highest variance across clusters
# A feature that is the same in all clusters tells us nothing
feature_variance = all_mean_shap.var(axis=0)
top_features     = feature_variance.nlargest(14).index.tolist()

heatmap_data = all_mean_shap[top_features].copy()

# Make row and column labels readable
readable_rows = [
    f"Cluster {c}  ({cluster_info[c]['n_customers']} customers)\n"
    f"↳ {cluster_info[c]['name']}"
    for c in range(K_CLUSTERS)
]
readable_cols = [FEATURE_NAMES.get(f, f.replace("shap_", "")) for f in top_features]

heatmap_data.index   = readable_rows
heatmap_data.columns = readable_cols

fig, ax = plt.subplots(figsize=(15, max(5, K_CLUSTERS * 1.8)))

# Find symmetric colour scale so 0 (no effect) is always white/grey
max_abs_val = max(heatmap_data.abs().max().max(), 0.01)

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".3f",
    cmap="RdBu_r",
    center=0,
    vmin=-max_abs_val,
    vmax=max_abs_val,
    linewidths=0.5,
    linecolor="white",
    ax=ax,
    cbar_kws={"label": "Mean SHAP value\n(red = pushes toward churn, blue = protects)",
              "shrink": 0.8}
)

ax.set_title(
    "Why Each Cluster Churns — Mean SHAP Value per Feature\n"
    "Dark red cell = this feature is the primary churn reason for this cluster\n"
    "Dark blue cell = this feature is protecting this cluster from churning",
    pad=16, fontsize=12
)
ax.set_xlabel("Features (churn drivers)", fontsize=11)
ax.tick_params(axis="x", rotation=40, labelsize=9)
ax.tick_params(axis="y", rotation=0,   labelsize=9)

plt.tight_layout()
plt.savefig("outputs/chart_cluster_heatmap.png", bbox_inches="tight")
plt.show()
print("  ✓ Saved: outputs/chart_cluster_heatmap.png")
print("    → This is your most important chart — read it to name your segments")
print("    → Each row tells you the primary intervention for that group")


# =============================================================================
#  SECTION 9 — CLUSTER PROFILE CHARTS
# =============================================================================
#
#  Now we look at the ORIGINAL feature values (not SHAP) per cluster.
#  This answers: "what kind of PEOPLE are in each cluster?"
#
#  SHAP heatmap answered: WHY do they churn?
#  Profile charts answer:  WHO are they?
#
#  Together, these two views give you everything needed to write
#  the intervention playbook.

print(f"\n--- SECTION 9: CLUSTER PROFILES ---")

# Get original feature values for at-risk customers
X_test_at_risk  = X_test.loc[shap_at_risk.index].copy()
X_test_at_risk["cluster"] = cluster_labels

profile_features = ["Tenure Months", "Monthly Charges", "Total Charges",
                    "Senior Citizen", "Tech Support", "Online Security",
                    "Contract_Two year", "Contract_One year"]
profile_features = [f for f in profile_features if f in X_test_at_risk.columns]

for col in profile_features:
    if X_test_at_risk[col].dtype == 'object':
        X_test_at_risk[col] = X_test_at_risk[col].map({'Yes': 1, 'No': 0, 'Yes ': 1, 'No ': 0}).fillna(X_test_at_risk[col])

cluster_profiles = X_test_at_risk.groupby("cluster")[profile_features].mean()


# Readable names for profile features
PROFILE_NAMES = {
    "Tenure Months":                "Avg tenure\n(months)",
    "Monthly Charges":        "Avg monthly\ncharges (£)",
    "Total Charges":          "Avg total\nspend (£)",
    "Senior Citizen":         "% Senior\ncitizens",
    "Tech Support":           "% With tech\nsupport",
    "Online Security":        "% With online\nsecurity",
    "Contract_Two year":     "% On 2-year\ncontract",
    "Contract_One year":     "% On 1-year\ncontract",
}

n_profiles = len(profile_features)
fig, axes  = plt.subplots(2, 4, figsize=(16, 8))
axes_flat  = axes.flatten()

for i, feat in enumerate(profile_features):
    ax = axes_flat[i]
    values  = [cluster_profiles.loc[c, feat] for c in range(K_CLUSTERS)]
    colours = CLUSTER_PALETTE[:K_CLUSTERS]
    labels  = [f"C{c}" for c in range(K_CLUSTERS)]

    bars = ax.bar(labels, values, color=colours, width=0.55, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + max(values) * 0.02,
                f"{val:.1f}" if feat in ["Tenure Months", "Monthly Charges", "Total Charges"]
                else f"{val:.0%}",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_title(PROFILE_NAMES.get(feat, feat), fontsize=10, pad=8)
    ax.set_ylim(0, max(values) * 1.25)

# Turn off any unused subplots
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].axis("off")

fig.suptitle(
    "Cluster Profiles — WHO Is In Each Group\n"
    "(Read alongside the SHAP heatmap to write the playbook)",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig("outputs/chart_cluster_profiles.png", bbox_inches="tight")
plt.show()
print("  ✓ Saved: outputs/chart_cluster_profiles.png")

# =============================================================================
#  SECTION 10 — CLUSTER SUMMARY BAR CHARTS
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Chart 1: Cluster sizes
sizes = [cluster_info[c]["n_customers"] for c in range(K_CLUSTERS)]
probs = [cluster_info[c]["avg_prob"] for c in range(K_CLUSTERS)]
names = [cluster_info[c]["name"] for c in range(K_CLUSTERS)]
xlabels = [f"C{c}\n({n})" for c, n in enumerate(sizes)]

bars1 = axes[0].bar(xlabels, sizes,
                    color=CLUSTER_PALETTE[:K_CLUSTERS],
                    width=0.5, edgecolor="white")
for bar, n, name in zip(bars1, sizes, names):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 n + max(sizes) * 0.02,
                 f"{n:,}", ha="center", fontsize=10, fontweight="bold")
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 n / 2,
                 "\n".join(name.split(" ", 2)[:2]),
                 ha="center", va="center", fontsize=7.5,
                 color="white", fontweight="bold")
axes[0].set_title("At-Risk Customers per Cluster")
axes[0].set_ylabel("Number of customers")

# Chart 2: Average churn probability
bars2 = axes[1].bar(xlabels, probs,
                    color=CLUSTER_PALETTE[:K_CLUSTERS],
                    width=0.5, edgecolor="white")
for bar, p in zip(bars2, probs):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 p + 0.005, f"{p:.0%}",
                 ha="center", fontsize=10, fontweight="bold")
axes[1].set_title("Average Predicted Churn Probability")
axes[1].set_ylabel("Churn probability")
axes[1].set_ylim(0, max(probs) * 1.2)
axes[1].yaxis.set_major_formatter(
    plt.matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

plt.suptitle("Cluster Overview — Size and Urgency",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/chart_cluster_overview.png", bbox_inches="tight")
plt.show()
print("  ✓ Saved: outputs/chart_cluster_overview.png")

# =============================================================================
#  SECTION 11 — EXPORT: AT-RISK CUSTOMER LIST WITH CLUSTER LABELS
# =============================================================================
#

print(f"\n--- SECTION 11: EXPORTING RESULTS ---")

# Build the final export: original customer data + cluster + probability + action
df_original_indexed = df_original.set_index(
    pd.RangeIndex(len(df_original)))

at_risk_export = df_original_indexed.loc[shap_at_risk.index].copy()
at_risk_export["churn_probability"] = shap_at_risk["churn_probability"].values
at_risk_export["cluster_id"]        = cluster_labels
at_risk_export["cluster_name"]      = [cluster_info[c]["name"] for c in cluster_labels]
at_risk_export["top_churn_reason"]  = [
    FEATURE_NAMES.get(cluster_info[c]["primary_feature"], "")
    for c in cluster_labels
]

# Sort by churn probability — highest risk first
at_risk_export = at_risk_export.sort_values("churn_probability", ascending=False)
at_risk_export.to_csv("outputs/at_risk_customers_clustered.csv", index=True)

print(f"  ✓ Saved: outputs/at_risk_customers_clustered.csv")
print(f"    Rows: {len(at_risk_export):,}")
print(f"    Columns: {list(at_risk_export.columns)}")

# Cluster summary statistics for Power BI
cluster_summary = pd.DataFrame([{
    "cluster_id":        c,
    "cluster_name":      cluster_info[c]["name"],
    "n_customers":       cluster_info[c]["n_customers"],
    "avg_churn_prob":    round(cluster_info[c]["avg_prob"], 4),
    "pct_of_at_risk":    round(cluster_info[c]["n_customers"] / len(shap_at_risk), 4),
    "top_driver_1":      FEATURE_NAMES.get(
        cluster_info[c]["top_drivers"].index[0], "") if len(cluster_info[c]["top_drivers"]) > 0 else "",
    "top_driver_2":      FEATURE_NAMES.get(
        cluster_info[c]["top_drivers"].index[1], "") if len(cluster_info[c]["top_drivers"]) > 1 else "",
    "top_driver_3":      FEATURE_NAMES.get(
        cluster_info[c]["top_drivers"].index[2], "") if len(cluster_info[c]["top_drivers"]) > 2 else "",
} for c in range(K_CLUSTERS)])

cluster_summary.to_csv("outputs/cluster_summary.csv", index=False)
print(f"  ✓ Saved: outputs/cluster_summary.csv  (for Power BI segment table)")

# =============================================================================
#  SECTION 12 — THE PLAYBOOK: WHAT TO DO FOR EACH CLUSTER
# =============================================================================
#

PLAYBOOK = {
    0: {
        "action":   "Offer a discounted annual contract. Frame it as 'lock in your price for 12 months.' "
                    "Target within 30 days of billing cycle. Expected conversion: 15-25%.",
        "urgency":  "HIGH",
        "channel":  "Outbound call + email",
        "timing":   "Within 7 days",
    },
    1: {
        "action":   "Trigger onboarding sequence: day-14 and day-45 check-in emails. "
                    "Offer a 3-month free trial of Tech Support and Online Security. "
                    "New customers who adopt two or more services churn at 3× lower rates.",
        "urgency":  "HIGH",
        "channel":  "Email sequence + in-app notification",
        "timing":   "Day 14 and day 45 of tenure",
    },
    2: {
        "action":   "Send a 'loyalty thank you' with a £10 monthly credit for 6 months. "
                    "Personalise with years-on-platform stat. "
                    "Long-tenure customers respond well to recognition — churn is often triggered "
                    "by a single bad experience, not dissatisfaction.",
        "urgency":  "MEDIUM",
        "channel":  "Personalised email + SMS",
        "timing":   "Before next billing date",
    },
    3: {
        "action":   "Activate Online Security and Device Protection at no charge for 3 months. "
                    "Follow up with a usage email showing their security activity. "
                    "Getting customers to USE services is the strongest retention lever.",
        "urgency":  "MEDIUM",
        "channel":  "Email + account dashboard notification",
        "timing":   "Within 14 days",
    },
}

print(f"\n{'='*65}")
print(f"  CHURN INTERVENTION PLAYBOOK — {len(shap_at_risk):,} customers at risk")
print(f"{'='*65}")

urgency_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}

for c in range(K_CLUSTERS):
    p    = PLAYBOOK.get(c, {"action": "Review cluster heatmap and define action",
                            "urgency": "MEDIUM", "channel": "TBD", "timing": "TBD"})
    info = cluster_info[c]
    icon = urgency_icon.get(p["urgency"], "⚪")

    print(f"\n  {icon}  CLUSTER {c}: {info['name'].upper()}")
    print(f"     Customers:   {info['n_customers']:,}  |  Avg risk: {info['avg_prob']:.0%}  |  Urgency: {p['urgency']}")
    print(f"     Top drivers: {', '.join([FEATURE_NAMES.get(f,'') for f in info['top_drivers'].index[:3]])}")
    print(f"     Recommended action:")
    print(f"       {p['action']}")
    print(f"     Channel: {p['channel']}  |  Timing: {p['timing']}")
    print("  " + "-" * 60)

print(f"""
  SUMMARY:
  Total at-risk customers identified:  {len(shap_at_risk):,}
  Covered by playbook interventions:   {len(shap_at_risk):,}  (100%)

  NEXT STEPS:
  1. Open chart_cluster_heatmap.png
  2. Check the dark red cells — do the cluster names above match your heatmap?
  3. If not, update the PLAYBOOK dictionary above with your actual cluster names
  4. Hand at_risk_customers_clustered.csv to your CRM / marketing team
  5. Phase 6 → Build the Power BI dashboard using this data
""")

print(f"{'='*65}")
print(f"  FILES SAVED IN outputs/ FOLDER")
print(f"{'='*65}")
files = [
    ("chart_elbow_silhouette.png",      "Elbow + silhouette charts for picking k"),
    ("chart_cluster_heatmap.png",       "★ Main chart — WHY each cluster churns"),
    ("chart_cluster_profiles.png",      "WHO is in each cluster (original features)"),
    ("chart_cluster_overview.png",      "Cluster sizes and average churn probabilities"),
    ("at_risk_customers_clustered.csv", "★ Full at-risk list with cluster labels → Power BI"),
    ("cluster_summary.csv",             "★ Segment table for Power BI dashboard"),
]
for fname, desc in files:
    print(f"  {fname:<42}  {desc}")
print(f"\n  NEXT → Phase 6 (06_powerbi_prep.py):")
print(f"  Connect cluster_summary.csv and at_risk_customers_clustered.csv")
print(f"  into Power BI Desktop to build the dashboard.")
print(f"{'='*65}")
