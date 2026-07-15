"""Utilidades compartidas del reanálisis de Cañete."""
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATOS = BASE / "datos"
SALIDAS = BASE / "salidas"
SALIDAS.mkdir(exist_ok=True)

PAISAJE = ["altitud", "pendiente", "dist_acequia", "dist_rio"]

def dist_m(lat1, lon1, lat2, lon2, lat_ref=-13.1):
    """Distancia aproximada en metros entre coordenadas geográficas."""
    return np.sqrt(((lat2 - lat1) * 111320) ** 2 +
                   ((lon2 - lon1) * 111320 * np.cos(np.radians(lat_ref))) ** 2)

def wilson(k, n, z=1.96):
    """Intervalo de confianza de Wilson para una proporción."""
    import math
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
