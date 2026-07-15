"""00 - Verificación de procedencia (paper Cañete).
Audita que las variables de grilla_codigo2 provengan de los insumos primarios
que la tesis declara: DEM del IGN, shapefiles de acequias (Fernandini 2022),
ríos (ANA) y sitios. Recalcula cada variable desde el crudo y la compara.
Ejecución: ~2-3 min (muestra de 30k celdas + todos los positivos)."""
import os; os.environ["SHAPE_RESTORE_SHX"] = "YES"
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import shapely
from scipy.spatial import cKDTree
from comun import DATOS

GEO = DATOS / "geodatos"
rng = np.random.default_rng(42)

df = pd.read_csv(DATOS / "grilla_modelo2_620k.csv.gz", low_memory=False)
df = df.dropna(subset=["altitud", "pendiente", "dist_acequia", "dist_rio"]).reset_index(drop=True)
y = df["label"].values
idx = np.unique(np.concatenate([np.where(y == 1)[0], rng.choice(len(df), 30000, replace=False)]))
s = df.iloc[idx].reset_index(drop=True)
print(f"Grilla: {len(df):,} celdas | muestra de auditoría: {len(s):,} (incluye {int(y.sum())} positivos)\n")
pts = shapely.points(s["x"].values, s["y"].values)

def comparar(nombre, recalc, original):
    d = np.abs(recalc - original)
    corr = np.corrcoef(recalc, original)[0, 1]
    print(f"{nombre:14s} corr={corr:.4f}  MAE={d.mean():8.2f}  mediana={np.median(d):7.2f}  max={d.max():9.2f}")
    return corr

print("Variable       | acuerdo recalculado-desde-crudo vs. columna de la grilla")
print("-" * 78)

# --- altitud desde el DEM del IGN ---
with rasterio.open(GEO / "DEM_vallebajo.tif") as dem:
    alt = np.array([v[0] for v in dem.sample(zip(s["x"], s["y"]))], dtype=float)
ok = alt > -1000
comparar("altitud (DEM)", alt[ok], s["altitud"].values[ok])

# --- dist_acequia desde el shapefile de Fernandini ---
aceq = gpd.read_file(GEO / "Acequias_antiguas.shp")
u_aceq = shapely.union_all(aceq.geometry.values)
comparar("dist_acequia", shapely.distance(pts, u_aceq), s["dist_acequia"].values)

# --- dist_rio desde la cartografía ANA ---
rios = gpd.read_file(GEO / "Rios_vallebajoCanete.shp")
u_rios = shapely.union_all(rios.geometry.values)
comparar("dist_rio", shapely.distance(pts, u_rios), s["dist_rio"].values)

# --- dist_sitio y etiqueta desde los 188 sitios unificados (grilla completa) ---
sitios = pd.read_csv(DATOS / "sitios_unificados_188.csv")
tree = cKDTree(sitios[["x", "y"]].values)
dist_rec, _ = tree.query(df[["x", "y"]].values, k=1)
comparar("dist_sitio", dist_rec, df["dist_sitio_mas_cercano"].values)
lab_rec = (dist_rec <= 30).astype(int)
print(f"{'label (buf30m)':14s} coincidencia con label de la grilla: {(lab_rec == y).mean():.6f} "
      f"({int((lab_rec != y).sum())} discrepancias de {len(y):,})")

print("""
Lectura: corr≈1 y MAE pequeño = la variable proviene del insumo declarado.
Con esto, la cadena DEM/shapefiles -> variables -> etiquetas queda auditada y
el paper puede afirmar reproducibilidad desde datos primarios.""")
