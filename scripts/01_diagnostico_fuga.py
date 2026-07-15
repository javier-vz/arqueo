"""01 - Diagnóstico de la fuga de información (paper Cañete).
Demuestra con los datos originales: (1) dist_sitio_mas_cercano codifica la
etiqueta por construcción (buffer 30 m); (2) SMOTE antes del split produjo
un test 50% sintético. Ejecución: ~2-3 min. Salidas en ../salidas/."""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from comun import DATOS, SALIDAS

N_ESTIMATORS = 50  # subir a 100 para cifras finales

df = pd.read_csv(DATOS / "grilla_modelo2_620k.csv.gz", low_memory=False)
df = df.dropna(subset=["altitud", "pendiente", "dist_acequia", "dist_rio"]).reset_index(drop=True)
y = df["label"].values
print(f"Celdas: {len(df):,} | positivas: {int(y.sum())} | prevalencia: {y.mean():.5%}\n")

# --- 1. La variable que es la etiqueta ---
print("dist_sitio_mas_cercano por clase:")
print(df.groupby("label")["dist_sitio_mas_cercano"].agg(["min", "max"]).to_string())
pred_umbral = (df["dist_sitio_mas_cercano"] <= 30).astype(int)
print(f"\n'Clasificador' umbral<=30m (sin modelo): accuracy={(pred_umbral == y).mean():.6f}, "
      f"recall={pred_umbral[y == 1].mean():.6f}, precision={y[pred_umbral == 1].mean():.6f}\n")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(df.loc[y == 0, "dist_sitio_mas_cercano"], bins=np.logspace(0, 4.5, 60),
        alpha=.6, label="label = 0 (sin registro)", color="steelblue")
ax.hist(df.loc[y == 1, "dist_sitio_mas_cercano"], bins=np.logspace(0, 4.5, 60),
        alpha=.85, label="label = 1 (con sitio)", color="firebrick")
ax.axvline(30, color="k", ls="--", lw=1.5, label="30 m (buffer de etiquetado)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("dist_sitio_mas_cercano (m, log)"); ax.set_ylabel("celdas (log)")
ax.legend(); ax.set_title("La variable dist_sitio codifica la etiqueta por construcción")
plt.tight_layout(); plt.savefig(SALIDAS / "fig_fuga_dist_sitio.png", dpi=200)
print(f"figura guardada: {SALIDAS/'fig_fuga_dist_sitio.png'}\n")

# --- 2. Reproducción del pipeline original (SMOTE -> split) ---
FEAT = ["altitud", "pendiente", "dist_acequia", "dist_rio", "dist_sitio_mas_cercano"]
Xr, yr = SMOTE(random_state=42).fit_resample(df[FEAT].values, y)
Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.3, random_state=42)
m = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1).fit(Xtr, ytr)
pred = m.predict(Xte)
print(f"PIPELINE ORIGINAL — test: {len(yte):,} celdas | balance: {np.bincount(yte)}  <- ~50/50 = sintético")
print(f"accuracy={accuracy_score(yte, pred):.5f}  precision={precision_score(yte, pred):.5f}  "
      f"recall={recall_score(yte, pred):.5f}  f1={f1_score(yte, pred):.5f}")
print("\nConclusión: el 0.9995 de la tesis = variable-etiqueta + test sintético. Ver 02 para la evaluación honesta.")
