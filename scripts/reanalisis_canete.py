"""
Reanálisis del modelo predictivo del valle bajo de Cañete
=========================================================
Script limpio y reproducible para el paper (Royal Society Open Science).
Reproduce (A) el pipeline original de la tesis y lo compara con
(B) evaluación con SMOTE solo en entrenamiento y (C) validación
cruzada por bloques espaciales, usando solo variables de paisaje.

Requiere: grilla_codigo2.csv (columnas: x, y, altitud, pendiente,
label, dist_acequia, dist_rio, dist_sitio_mas_cercano)

Uso: python reanalisis_canete.py ruta/a/grilla_codigo2.csv
"""
import sys, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score)
from imblearn.over_sampling import SMOTE

RUTA = sys.argv[1] if len(sys.argv) > 1 else "datos/grilla_modelo2_620k.csv.gz"
PAISAJE = ['altitud', 'pendiente', 'dist_acequia', 'dist_rio']
CON_LEAK = PAISAJE + ['dist_sitio_mas_cercano']
RF = dict(n_estimators=100, random_state=42, n_jobs=-1)
resultados = []

def evaluar(nombre, y_true, proba, umbral=0.5, extra=None):
    pred = (proba >= umbral).astype(int)
    r = dict(pipeline=nombre,
             accuracy=round(accuracy_score(y_true, pred), 4),
             precision=round(precision_score(y_true, pred, zero_division=0), 4),
             recall=round(recall_score(y_true, pred, zero_division=0), 4),
             f1=round(f1_score(y_true, pred, zero_division=0), 4),
             pr_auc=round(average_precision_score(y_true, proba), 4))
    if extra: r.update(extra)
    resultados.append(r)
    print(json.dumps(r, indent=1))

df = pd.read_csv(RUTA, low_memory=False).dropna(subset=PAISAJE).reset_index(drop=True)
y = df['label'].values
print(f"Celdas: {len(df):,} | positivos: {int(y.sum())} | prevalencia: {y.mean():.5%}")

# --- Diagnóstico de fuga: dist_sitio_mas_cercano codifica la etiqueta ---
sep = df.groupby('label')['dist_sitio_mas_cercano'].agg(['min', 'max'])
print("\ndist_sitio_mas_cercano por clase (nótese la separación perfecta ~30 m):")
print(sep.to_string())

# === A. Pipeline original de la tesis: SMOTE -> split (contaminado) ===
X = df[CON_LEAK].values
Xr, yr = SMOTE(random_state=42).fit_resample(X, y)
Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.3, random_state=42)
m = RandomForestClassifier(**RF).fit(Xtr, ytr)
evaluar('A_original_SMOTE_antes_de_split', yte, m.predict_proba(Xte)[:, 1],
        extra=dict(test_n=len(yte), test_sinteticos=True))

# === B. Corregido: split -> SMOTE solo en train, solo paisaje ===
X = df[PAISAJE].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
Xr, yr = SMOTE(random_state=42).fit_resample(Xtr, ytr)
m = RandomForestClassifier(**RF).fit(Xr, yr)
evaluar('B_SMOTE_solo_train_particion_aleatoria', yte, m.predict_proba(Xte)[:, 1],
        extra=dict(test_n=len(yte), test_pos=int(yte.sum())))

# === C. Corregido: validación cruzada por bloques espaciales de 2 km ===
df['block'] = (df['x'] // 2000).astype(int).astype(str) + '_' + \
              (df['y'] // 2000).astype(int).astype(str)
X, g = df[PAISAJE].values, df['block'].values
pp = np.zeros(len(y))
for tr, te in GroupKFold(n_splits=5).split(X, y, g):
    npos = int(y[tr].sum())
    Xr, yr = SMOTE(random_state=42, k_neighbors=min(5, npos - 1)).fit_resample(X[tr], y[tr])
    m = RandomForestClassifier(**RF).fit(Xr, yr)
    pp[te] = m.predict_proba(X[te])[:, 1]
extra = dict(n_bloques=int(df['block'].nunique()), prevalencia=round(float(y.mean()), 6))
orden = np.argsort(-pp)
for K in (50, 146, 500):
    extra[f'precision_top{K}'] = round(float(y[orden[:K]].mean()), 4)
evaluar('C_spatial_block_CV_5fold', y, pp, extra=extra)

df[['x', 'y', 'label']].assign(proba_spatial_cv=pp)\
  .to_csv('predicciones_spatial_cv.csv', index=False)
json.dump(resultados, open('resultados_reanalisis.json', 'w'), indent=1)
print("\nGuardado: predicciones_spatial_cv.csv, resultados_reanalisis.json")
