# Paquete de reanálisis — Tesis de Cañete → paper (Royal Society Open Science)
### Notebooks corregidos + datos canónicos + trazabilidad completa
Preparado para Dina Cornejo · julio 2026

## Estructura

```
paper_canete/
├── README.md                      <- este archivo
├── datos/
│   ├── grilla_modelo2_620k.csv.gz        Grilla canónica del Modelo 2 (620,296 celdas, 196 positivas).
│   │                                      Columnas: x, y, altitud, pendiente, label, dist_acequia,
│   │                                      dist_rio, dist_sitio_mas_cercano (¡esta última NO usar como
│   │                                      predictor: es la etiqueta por construcción, ver notebook 01!)
│   ├── alertas_modelo1_85.csv            Las 85 alertas del M1 (del KML numerado): número, prioridad, lat/lon.
│   ├── alertas_modelo2_146.csv           Las 146 alertas del M2 con probabilidad y coordenadas.
│   │                                      alerta_num es 1-based y coincide con la columna "Cercano"
│   │                                      de los registros de campo.
│   ├── campo_salida1.csv                 Salida 1 (9 puntos, Modelo 1) — limpiado del Excel original.
│   ├── campo_salida2.csv                 Salida 2 (22 puntos, Modelo 2) — limpiado del Excel original.
│   ├── tabla_trazabilidad_31pts.csv      LA tabla: cada punto de campo → alerta → distancia → categoría.
│   │                                      Categorías: alerta_confirmada (16) / alerta_descartada_FP (10) /
│   │                                      hallazgo_adyacente (2, a <110 m de una alerta) /
│   │                                      sitio_no_alertado_independiente (3).
│   ├── sitios_unificados_188.csv         Sitios usados para etiquetar (MINCUL + Williams-Merino + campo).
│   └── predicciones_spatial_cv.csv.gz    Probabilidades fuera-de-bloque del CV espacial (corrida previa).
├── scripts/                       EJECUTAR EN ORDEN (ver instrucciones abajo):
│   ├── comun.py                          Utilidades compartidas (rutas, distancias, Wilson).
│   ├── 01_diagnostico_fuga.py            Las dos fugas demostradas con los datos (~2-3 min). VALIDADO ✓
│   ├── 02_reanalisis_modelos.py          Evaluación honesta: SMOTE-en-train, CV espacial, precisión@K,
│   │                                      baselines (~5 min; RESUMIBLE por checkpoints). VALIDADO ✓
│   ├── 03_validacion_campo.py            Trazabilidad, Wilson, Fisher, AUC campo=0.80 (<30 s). VALIDADO ✓
│   └── 04_modelo1_reconstruido.py        Modelo 1 reconstruido y corregido (~1-2 min). VALIDADO ✓
├── environment.yml                Entorno Anaconda (conda env create -f environment.yml).
├── salidas/                       Aquí escriben los scripts: figuras, tablas, predicciones.
└── figuras_preview/
    ├── fig_fuga_dist_sitio.png           El histograma de la tautología de los 30 m.
    └── fig_auc_campo.png                 Probabilidad vs resultado de campo (AUC=0.80).

```

## Resultados clave ya verificados con estos datos

| Qué | Valor |
|---|---|
| Pipeline original (SMOTE→split, con dist_sitio) | accuracy ~1.0 sobre test 50% sintético — reproduce la Tabla 1 de la tesis |
| Umbral dist_sitio ≤ 30 m, sin modelo | accuracy = recall = precisión = 1.000 (la variable ES la etiqueta) |
| Evaluación honesta (paisaje, CV espacial) | PR-AUC ~0.003–0.012 según regularización |
| Enriquecimiento del ranking (top-146) | ~20× sobre prevalencia; heurística acequias: solo 5–6× |
| Campo M1 | 4/9 = 44.4% [IC95: 18.9–73.3] |
| Campo M2 | 12/17 = 70.6% [IC95: 46.9–86.7]; Fisher M1 vs M2 p=0.234 (exploratorio) |
| **AUC de campo (M2)** | **0.80** (p=0.032): prob media 0.725 confirmadas vs 0.624 descartadas |

## Para las cifras FINALES del paper
1. En notebook 02: `N_ESTIMATORS = 100`, `MAX_DEPTH = None` (necesita ~8+ GB RAM), y sensibilidad
   de bloques con `BLOCK_M` en {1000, 2000, 4000}.
2. Falta un archivo que no estaba en la carpeta: `grilla_final_modelo_logistico (1).csv`
   (input real del Modelo 1, con `presencia_sitio_total`). Sin él, el M1 se reporta desde
   el notebook original pero no es re-ejecutable. También faltan las fuentes de sitios:
   `W y M.xlsx`, `mincul.xlsx`, `puntos.csv`.
3. ⚠️ COORDENADAS SENSIBLES: campo_salida*.csv, tabla_trazabilidad y sitios contienen ubicaciones
   exactas de evidencias arqueológicas. Este paquete es de trabajo interno. Para el repositorio
   público del paper: generalizar a la celda de 50 m o aplicar desplazamiento controlado, y
   documentar el mecanismo de acceso (vía Ministerio de Cultura / proyecto).

## Correcciones respecto al código original (resumen)
- SMOTE ahora SIEMPRE después del split / dentro de cada fold.
- Variables: solo paisaje (altitud, pendiente, dist_acequia, dist_rio). Excluidas: x, y,
  dist_sitio_mas_cercano, distancia_mincul, distancia_sitios_nuevos (fuga/circularidad).
- Validación espacial por bloques (GroupKFold) en lugar de partición aleatoria.
- Métricas centradas en PR-AUC y precisión@K (accuracy es engañosa con prevalencia 0.03%).
- Baselines no-ML incluidos (prevalencia y heurística de acequias).
- Contabilidad de campo en 4 categorías excluyentes con trazabilidad alerta-por-alerta.

## Actualización v2 — Modelo 1 resuelto
- Se recibió `grilla_final_modelo_logistico.csv`: es idéntico al del zip (etiqueta parcial
  `presencia_sitio_nuevo`, 84 positivos). El input exacto del M1 ("(1)", con
  `presencia_sitio_total`) sigue irrecuperable.
- **`datos/grilla_modelo1_250m_rec.csv`**: grilla del M1 (celdas de **250 m** — dato nuevo
  para Métodos) con etiqueta total **reconstruida** (188 positivos; la original tenía ~169
  según la matriz de la tesis). Regla documentada en el script 04.
- **`scripts/04_modelo1_reconstruido.py`** (VALIDADO ✓): tautología del M1
  (separación perfecta en ~148 m para la etiqueta verificable), pipeline original sobre la
  reconstrucción (F1 0.998 — mismo régimen que la tesis, conclusiones insensibles a la versión
  de la etiqueta) y evaluación corregida (solo terreno, spatial CV: PR-AUC 0.026,
  precision@85 = 8.2% = 5.6x prevalencia).
- Los `W y M.xlsx`, `mincul.xlsx`, `puntos.csv` originales siguen pendientes; si aparecen,
  la reconstrucción se reemplaza por la etiqueta exacta en una celda.

## Actualización v3 — Scripts .py + entorno Anaconda (recomendado si Jupyter da problemas)

La carpeta `scripts/` ahora contiene versiones .py de los cuatro análisis, equivalentes a los
notebooks y VALIDADAS de punta a punta. No requieren Jupyter: solo Anaconda.

### Instalación (una vez, en Anaconda Prompt)
```
cd ruta\a\paper_canete
conda env create -f environment.yml
conda activate canete
```

### Ejecución (en orden, desde la carpeta scripts/)
```
cd scripts
python 01_diagnostico_fuga.py        (~2-3 min)
python 02_reanalisis_modelos.py      (~5 min; RESUMIBLE: si se corta, volver a ejecutar
                                      y retoma desde el último fold guardado)
python 03_validacion_campo.py        (<30 s)
python 04_modelo1_reconstruido.py    (~1-2 min)
```
Todos los resultados (figuras PNG, tabla de trazabilidad, predicciones, JSON de métricas)
se escriben en `salidas/`. `comun.py` contiene las utilidades compartidas; los scripts se
ejecutan desde `scripts/` (las rutas a `../datos/` se resuelven solas).

Para las cifras FINALES del paper, editar al inicio de `02_reanalisis_modelos.py`:
`N_ESTIMATORS = 100`, `MAX_DEPTH = None` (necesita ~8+ GB de RAM), y repetir con
`BLOCK_M` en {1000, 2000, 4000} para la sensibilidad (los checkpoints se separan por bloque).


## Actualización v4 — Procedencia auditada desde datos primarios

Rastreando la metodología de la tesis (§5.2–5.3) contra los archivos de la carpeta, se
recuperó la **cadena de construcción completa** del Modelo 2 y se añadió su auditoría:

- **`datos/geodatos/`** — los insumos primarios que la tesis declara: `DEM_vallebajo.tif`
  (IGN, ~31 m), `Acequias_antiguas.shp` (19 trazas, prospección Fernandini 2022),
  `Rios_vallebajoCanete.shp` (860 líneas, cartografía ANA), `Valle_bajo_Canete.shp` y
  `Sitios_Williams_Merino_shape.shp` (120 puntos). Todo en EPSG:32718 (UTM 18S).
- **`scripts/00_verificacion_procedencia.py`** (VALIDADO ✓, ~2-3 min): recalcula cada
  variable desde el crudo y la compara con la grilla. Resultado: **acuerdo perfecto**
  (corr=1.0000, error 0) en altitud, dist_acequia, dist_rio y dist_sitio, y coincidencia
  100% de las etiquetas reconstruidas (buffer 30 m sobre los 188 sitios) en las 620,296
  celdas. El paper puede afirmar reproducibilidad desde datos primarios.

### Trazabilidad de nombres (tesis → archivo → estado)
| La tesis dice | Archivo real | Estado |
|---|---|---|
| grilla_codigo2.csv (§5.3, Modelo 2) | datos/grilla_modelo2_620k.csv.gz | ✓ verificado desde crudo |
| **grilla_codigo1.csv (§5.3, Modelo 1)** | — | **FALTA: buscar en Drive/Colab con ese nombre exacto** |
| DEM del IGN "50 m" | geodatos/DEM_vallebajo.tif | ✓ (resolución real ~31 m — corregir en el paper) |
| acequias Fernandini 2022 | geodatos/Acequias_antiguas.shp | ✓ |
| ríos (ANA) | geodatos/Rios_vallebajoCanete.shp | ✓ |
| etiqueta "por intersección de celda" | código: buffer de 30 m | discrepancia texto-código, documentada |
| "filtro de exclusión de 15 m" | — | no existe en ningún código; eliminar o implementar |
| fuentes de sitios (W y M.xlsx, mincul.xlsx, puntos.csv) | — | faltan los crudos; sitios_unificados_188.csv es el consolidado |

Cadena intermedia disponible en la carpeta original (no incluida aquí por peso, 1,175,331
celdas cada una): grilla_utm_50m → grilla_con_altura_pendiente → grilla_etiquetada_con_sitios
→ grilla_con_distancias_agua_y_sitios → grilla_codigo2 (620,296 tras eliminar celdas sin DEM).

Nuevo orden de ejecución: 00 (opcional, auditoría) → 01 → 02 → 03 → 04.
