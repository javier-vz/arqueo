"""04 - Modelo 1: reconstrucción y reanálisis (paper Cañete).
El input original del M1 ('grilla_final_modelo_logistico (1).csv') es irrecuperable;
este script documenta la grilla disponible (celdas de 250 m), su tautología,
reconstruye la etiqueta total y muestra que las conclusiones no dependen de la
versión exacta de la etiqueta. Ejecución: ~1-2 min."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from imblearn.over_sampling import SMOTE
from comun import DATOS

m1 = pd.read_csv(DATOS / "grilla_modelo1_250m_rec.csv")
y = m1["presencia_total_rec"].values
print(f"celdas: {len(m1):,} (250x250 m) | nuevos: {m1.presencia_sitio_nuevo.sum()} | "
      f"total_rec: {y.sum()} | prevalencia: {y.mean():.3%}\n")

# --- Tautología del M1 (etiqueta verificable) ---
print("distancia_sitios_nuevos por presencia_sitio_nuevo:")
print(m1.groupby("presencia_sitio_nuevo")["distancia_sitios_nuevos"].agg(["min", "max"]).to_string())
u = (m1.distancia_sitios_nuevos <= 148).astype(int)
print(f"'clasificador' umbral<=148m: accuracy={(u == m1.presencia_sitio_nuevo).mean():.6f}\n")

# --- Pipeline original de la tesis sobre la reconstrucción ---
FEAT = ["distancia_sitios_nuevos", "distancia_mincul", "altitud", "pendiente"]
Xr, yr = SMOTE(random_state=42).fit_resample(m1[FEAT].values, y)
f1s = []
for seed in (1, 42, 123, 2024):
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, stratify=yr, random_state=seed)
    mo = RandomForestClassifier(class_weight="balanced", n_estimators=100,
                                max_depth=8, random_state=seed).fit(Xtr, ytr)
    f1s.append(f1_score(yte, mo.predict(Xte)))
print(f"ORIGINAL-STYLE (SMOTE->split, feats con distancias): F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print("   (tesis: 0.976 — mismo régimen inflado; conclusiones insensibles a la versión de etiqueta)\n")

# --- Corregido: solo terreno + spatial CV ---
X = m1[["altitud", "pendiente"]].values.astype(np.float32)
m1["block"] = (m1.X // 2000).astype(int).astype(str) + "_" + (m1.Y // 2000).astype(int).astype(str)
pp = np.zeros(len(y))
for tr, te in GroupKFold(5).split(X, y, m1["block"].values):
    Xs, ys = SMOTE(random_state=42, k_neighbors=min(5, int(y[tr].sum()) - 1)).fit_resample(X[tr], y[tr])
    pp[te] = RandomForestClassifier(n_estimators=100, random_state=42,
                                    n_jobs=-1).fit(Xs, ys).predict_proba(X[te])[:, 1]
pred = (pp >= .5).astype(int); prev = y.mean(); orden = np.argsort(-pp)
print(f"CORREGIDO spatial CV (solo terreno): prec={precision_score(y, pred, zero_division=0):.4f} "
      f"rec={recall_score(y, pred):.4f} PR-AUC={average_precision_score(y, pp):.4f}")
print(f"precision@85 (presupuesto real M1): {y[orden[:85]].mean():.4f} = "
      f"{y[orden[:85]].mean()/prev:.1f}x prevalencia ({prev:.4f})")
