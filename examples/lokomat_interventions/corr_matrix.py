import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel("/Volumes/SP UFD U2/PhD/Stage Nantes/LOKOMAT/reports_final_table.xlsx")

correlation_matrix = data.corr(numeric_only=True)
plt.figure(figsize=(12, 10))

sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 10})

plt.title("Correlation Matrix")
plt.savefig("/Volumes/SP UFD U2/PhD/Stage Nantes/LOKOMAT/Reports/correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
