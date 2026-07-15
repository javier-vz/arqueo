"""02 - Reanálisis corregido de los modelos (paper Cañete).
Evaluaciones honestas sobre la grilla del Modelo 2, solo variables de paisaje:
  B) partición aleatoria estratificada -> SMOTE solo en train
  C) validación cruzada espacial por bloques (GroupKFold 5, checkpoints resumibles)
  + precisión@K, baselines no-ML.
Parámetros ligeros por defecto; para cifras finales: N_ESTIMATORS=100, MAX_DEPTH=None
(requiere ~8+ GB RAM). Ejecución ligera: ~5 min; es re-ejecutable (retoma folds hechos)."""
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score)
from imblearn.over_sampling import SMOTE
from comun import DATOS, SALIDAS, PAISAJE

N_ESTIMATORS = 30
MAX_DEPTH = 12          # None para el paper
MAX_SAMPLES = 0.5
BLOCK_M = 2000          # sensibilidad: 1000 y 4000

def rf():
    return RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                                  max_samples=MAX_SAMPLES, random_state=42, n_jobs=-1)

def reporte(nombre, y_true, proba, umbral=0.5):
    pred = (proba >= umbral).astype(int)
    r = dict(pipeline=nombre,
             accuracy=round(accuracy_score(y_true, pred), 4),
             precision=round(precision_score(y_true, pred, zero_division=0), 4),
             recall=round(recall_score(y_true, pred, zero_division=0), 4),
             f1=round(f1_score(y_true, pred, zero_division=0), 4),
             pr_auc=round(average_precision_score(y_true, proba), 5))
    print(r); return r

df = pd.read_csv(DATOS / "grilla_modelo2_620k.csv.gz", low_memory=False)
df = df.dropna(subset=PAISAJE).reset_index(drop=True)
y = df["label"].values
X = df[PAISAJE].values.astype(np.float32)
print(f"Celdas: {len(df):,} | positivas: {int(y.sum())} | prevalencia: {y.mean():.5%}\n")
resultados = []

# --- B: partición aleatoria corregida ---
print("B — split aleatorio estratificado -> SMOTE solo en train")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
Xr, yr = SMOTE(random_state=42).fit_resample(Xtr, ytr)
mB = rf().fit(Xr, yr)
resultados.append(reporte("B_aleatoria_SMOTE_en_train", yte, mB.predict_proba(Xte)[:, 1]))

# --- C: spatial block CV con checkpoints ---
print(f"\nC — spatial block CV (bloques {BLOCK_M} m, 5 folds; checkpoints en salidas/)")
df["block"] = (df["x"] // BLOCK_M).astype(int).astype(str) + "_" + (df["y"] // BLOCK_M).astype(int).astype(str)
pp = np.zeros(len(y))
for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, df["block"].values)):
    ckpt = SALIDAS / f"_fold{k}_b{BLOCK_M}.npz"
    if ckpt.exists():
        d = np.load(ckpt); pp[d["te"]] = d["proba"]; print(f"  fold {k}: checkpoint"); continue
    npos = int(y[tr].sum())
    Xs, ys = SMOTE(random_state=42, k_neighbors=min(5, npos - 1)).fit_resample(X[tr], y[tr])
    m = rf().fit(Xs, ys)
    pp[te] = m.predict_proba(X[te])[:, 1]
    np.savez(ckpt, te=te, proba=pp[te]); print(f"  fold {k}: ok (train_pos={npos})")
resultados.append(reporte("C_spatial_block_CV", y, pp))

# --- precisión@K y baselines ---
orden = np.argsort(-pp); prev = y.mean()
print(f"\nprevalencia base: {prev:.5%}")
for K in (50, 100, 146, 500):
    pk = y[orden[:K]].mean()
    print(f"precision@{K:>3}: {pk:.4f} | enriquecimiento: {pk/prev:6.1f}x")
print(f"\nBaseline prevalencia — accuracy: {(y == 0).mean():.5f}, recall: 0, PR-AUC≈{prev:.5f}")
for Xm in (50, 100, 200):
    ph = (df["dist_acequia"] < Xm).astype(int).values
    pr = precision_score(y, ph, zero_division=0); rc = recall_score(y, ph)
    print(f"heurística acequia<{Xm:>3}m — precisión: {pr:.4f} ({pr/prev:.0f}x), recall: {rc:.3f}, alertas: {ph.sum():,}")

df[["x", "y", "label"]].assign(proba_spatial_cv=pp).to_csv(SALIDAS / "predicciones_spatial_cv.csv", index=False)
json.dump(resultados, open(SALIDAS / "resultados_reanalisis.json", "w"), indent=1)
print(f"\nGuardado en {SALIDAS}: predicciones_spatial_cv.csv, resultados_reanalisis.json")
