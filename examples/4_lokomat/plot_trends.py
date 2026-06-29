import pandas as pd
import matplotlib.pyplot as plt

# --- Load your data ---
df = pd.read_excel("/Volumes/SP UFD U2/PhD/Stage Nantes/LOKOMAT/all_reports.xlsx")

# --- Config ---
id_col = "ID"  # column for patient ID
session_col = "Session(s)"  # column for session number
features = [
    "Distance_m",
    "Durée_min",
    "Vitesse_kmh_MOY",
    "BWS_%_MOY",
    "Guidage_G_%_MOY",
    "Guidage_D_%_MOY",
]  # replace with your actual column names

# --- Create subplots ---
n_cols = 2  # number of columns in the grid
n_rows = -(-len(features) // n_cols)  # ceiling division to get enough rows

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows), sharex=True)
axes = axes.flatten()  # makes it easy to iterate

# --- Plot each feature ---
for i, feature in enumerate(features):
    ax = axes[i]
    for patient_id, group in df.groupby(id_col):
        group_sorted = group.sort_values(session_col)
        ax.plot(group_sorted[session_col], group_sorted[feature], marker="o", alpha=0.5, label=str(patient_id))
    ax.set_title(feature)
    ax.set_xlabel("Session number")
    ax.set_ylabel(feature)

# --- Hide any unused subplots ---
for j in range(len(features), len(axes)):
    axes[j].set_visible(False)

# --- Shared legend ---
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title=id_col, bbox_to_anchor=(1.02, 0.5), loc="center left")

plt.suptitle("Individual trends per patient", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("trends_subplots.png", dpi=150, bbox_inches="tight")
plt.show()
