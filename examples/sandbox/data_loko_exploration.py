# %% IMPORTS
import pandas as pd

data = pd.read_excel(
    "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/lokomat_reports/all_reports.xlsx"
)

# %% DATA
vit_means = data.groupby("ID")["Vitesse_kmh_MOY"].mean()

# %% LINEAR REGRESSION
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Data
feature = "Vitesse_kmh_MOY"
X = np.arange(1, 21).reshape(-1, 1)
y = data[feature][0:20]
# 2. Initialisation et entraînement du modèle
modele = LinearRegression()
modele.fit(X, y)

# Récupération des paramètres de la droite
pente = modele.coef_[0]
ordonnee_origine = modele.intercept_
print(f"Équation de la droite : y = {pente:.2f}x + {ordonnee_origine:.2f}")

# Visualisation
plt.scatter(X, y, color="blue", label="Données réelles")
plt.plot(X, modele.predict(X), color="red", label="Droite de régression")
plt.xlim(0, 21)
plt.xticks(np.arange(0, 21))
plt.title("Régression Linéaire")
plt.text(min(X), min(y), f"y = {pente:.2f}x + {ordonnee_origine:.2f}")
plt.xlabel("Nb sessions")
plt.ylabel(feature)
plt.legend()
plt.grid(True, color="0.9")
plt.show()
