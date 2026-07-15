"""03 - Validación de campo: trazabilidad y estadística (paper Cañete).
Reconstruye la contabilidad de las 2 salidas (alerta->GPS->foto->resultado),
IC de Wilson, Fisher, y el AUC de campo. Ejecución: <30 s.
ATENCIÓN: los CSV de campo contienen coordenadas exactas de evidencias; no publicar crudos."""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from comun import DATOS, SALIDAS, dist_m, wilson

m1 = pd.read_csv(DATOS / "alertas_modelo1_85.csv")
m2 = pd.read_csv(DATOS / "alertas_modelo2_146.csv")
c1 = pd.read_csv(DATOS / "campo_salida1.csv")
c2 = pd.read_csv(DATOS / "campo_salida2.csv")
print(f"alertas M1: {len(m1)} | alertas M2: {len(m2)} | campo: {len(c1)}+{len(c2)} puntos\n")

def clasificar(pred, es_sitio, d):
    conf = str(es_sitio).strip().lower() not in ("no", "nan")
    if str(pred).strip() == "si":
        return "alerta_confirmada" if conf else "alerta_descartada_FP"
    return "hallazgo_adyacente" if d < 110 else "sitio_no_alertado_independiente"

rows = []
for campo, alertas, salida, pcol in [(c1, m1, 1, "prioridad"), (c2, m2, 2, "probabilidad")]:
    for _, r in campo.iterrows():
        d = dist_m(r["latitud"], r["longitud"], alertas.lat, alertas.lon)
        i = int(d.idxmin())
        rows.append(dict(salida=salida, foto=r["foto"], predicho=r["predicho"],
                         alerta=int(alertas.loc[i, "alerta_num"]), info_alerta=alertas.loc[i, pcol],
                         dist_alerta_m=round(float(d.min()), 1), resultado=r["es_sitio"],
                         categoria=clasificar(r["predicho"], r["es_sitio"], d.min())))
traz = pd.DataFrame(rows)
print(traz.to_string(index=False))
print("\n" + traz.categoria.value_counts().to_string() + "\n")
traz.to_csv(SALIDAS / "tabla_trazabilidad_31pts.csv", index=False)

# --- Tasas con incertidumbre ---
for s, nombre in [(1, "Modelo 1"), (2, "Modelo 2")]:
    t = traz[(traz.salida == s) & (traz.predicho == "si")]
    k = (t.categoria == "alerta_confirmada").sum(); n = len(t)
    lo, hi = wilson(k, n)
    print(f"{nombre}: {k}/{n} = {k/n:.1%}  IC95% Wilson [{lo:.1%}, {hi:.1%}]")
odds, p = fisher_exact([[4, 5], [12, 5]])
print(f"Fisher exacto M1 vs M2: p = {p:.3f}  -> diferencia exploratoria, no concluyente\n")

# --- AUC de campo (Modelo 2) ---
t2 = traz[(traz.salida == 2) & (traz.predicho == "si")].copy()
t2["conf"] = (t2.categoria == "alerta_confirmada").astype(int)
p_conf = t2.loc[t2.conf == 1, "info_alerta"].astype(float)
p_fp = t2.loc[t2.conf == 0, "info_alerta"].astype(float)
u, pv = mannwhitneyu(p_conf, p_fp, alternative="greater")
auc = u / (len(p_conf) * len(p_fp))
print(f"prob media confirmadas (n={len(p_conf)}): {p_conf.mean():.3f} | "
      f"descartadas (n={len(p_fp)}): {p_fp.mean():.3f}")
print(f"Mann-Whitney U={u:.0f}, p={pv:.3f}  ->  AUC de campo = {auc:.2f}\n")

fig, ax = plt.subplots(figsize=(6.5, 4))
rng = np.random.default_rng(0)
for vals, x0, col, lab in [(p_fp, 0, "steelblue", "descartada (FP)"),
                           (p_conf, 1, "firebrick", "confirmada")]:
    ax.scatter(x0 + rng.uniform(-.08, .08, len(vals)), vals, s=70, alpha=.8, color=col, label=lab, zorder=3)
    ax.hlines(vals.mean(), x0 - .18, x0 + .18, color=col, lw=2)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Alerta descartada", "Alerta confirmada"])
ax.set_ylabel("Probabilidad del modelo")
ax.set_title(f"El ranking llevó señal al campo (AUC={auc:.2f}, p={pv:.3f}, n=17)")
ax.legend(); plt.tight_layout(); plt.savefig(SALIDAS / "fig_auc_campo.png", dpi=200)

na = traz[traz.predicho != "si"][["salida", "foto", "dist_alerta_m", "categoria"]]
print("Sitios no alertados:"); print(na.to_string(index=False))
print(f"\nSalidas guardadas en {SALIDAS}")
