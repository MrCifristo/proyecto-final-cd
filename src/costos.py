"""Supuestos de costo y regla de decisión de tres acciones (fase 1, framing).

Todos los costos son SUPUESTOS del equipo, en unidades relativas: 1 unidad es lo
que pierde el banco cuando le reduce el límite a un cliente que sí iba a pagar
(margen perdido y riesgo de que se vaya). No son cifras del banco.

Matriz de costo (fila = acción, columna = lo que pasa en realidad):

                    Paga (0)              Impaga (1)
    Mantener        0                     C_IMPAGO
    Revisar         C_REVISION            C_REVISION + (1 - DETECCION_REVISION) * C_IMPAGO
    Reducir         C_REDUCIR_BUENO       FRACCION_EXPOSICION_REDUCIDA * C_IMPAGO
"""

import numpy as np

# Supuestos (cambiar aquí y todo el resto se recalcula).
C_IMPAGO = 5.0  # mantener la línea a quien impaga: 5 veces el costo de un falso positivo
C_REDUCIR_BUENO = 1.0  # reducir la línea a quien sí paga (unidad de referencia)
C_REVISION = 0.3  # tiempo de analista por caso revisado
DETECCION_REVISION = 0.5  # el analista detecta la mitad de los impagos que revisa
FRACCION_EXPOSICION_REDUCIDA = 0.3  # reducir la línea deja el 30 % de la pérdida

ACCIONES = ("Mantener", "Revisar", "Reducir")


def matriz_costos(
    c_impago: float = C_IMPAGO,
    c_reducir_bueno: float = C_REDUCIR_BUENO,
    c_revision: float = C_REVISION,
    deteccion: float = DETECCION_REVISION,
    fraccion_reducida: float = FRACCION_EXPOSICION_REDUCIDA,
) -> np.ndarray:
    """Devuelve una matriz 3x2: filas = ACCIONES, columnas = (paga, impaga)."""
    return np.array(
        [
            [0.0, c_impago],
            [c_revision, c_revision + (1 - deteccion) * c_impago],
            [c_reducir_bueno, fraccion_reducida * c_impago],
        ]
    )


def costo_esperado_por_accion(p, matriz: np.ndarray | None = None) -> np.ndarray:
    """Costo esperado de cada acción para una probabilidad de impago p.

    Devuelve un arreglo de forma (len(p), 3).
    """
    m = matriz_costos() if matriz is None else matriz
    p = np.atleast_1d(np.asarray(p, dtype=float))[:, None]
    return (1 - p) * m[:, 0] + p * m[:, 1]


def umbrales_teoricos(matriz: np.ndarray | None = None) -> tuple[float, float]:
    """Umbrales (t1, t2) donde cambia la acción de menor costo esperado.

    p < t1 -> Mantener; t1 <= p < t2 -> Revisar; p >= t2 -> Reducir.
    Solo valen si las probabilidades del modelo están calibradas.
    """
    m = matriz_costos() if matriz is None else matriz

    def cruce(a: int, b: int) -> float:
        # (1-p)*m[a,0] + p*m[a,1] = (1-p)*m[b,0] + p*m[b,1]
        d0 = m[b, 0] - m[a, 0]
        d1 = m[a, 1] - m[b, 1]
        return d0 / (d0 + d1)

    t1, t2 = cruce(0, 1), cruce(1, 2)
    if not 0 < t1 < t2 < 1:
        raise ValueError(
            f"Con estos costos la acción Revisar nunca es la mejor (t1={t1:.3f}, t2={t2:.3f})."
        )
    return t1, t2


def asignar_acciones(p, t1: float, t2: float) -> np.ndarray:
    """0 = Mantener, 1 = Revisar, 2 = Reducir."""
    p = np.asarray(p, dtype=float)
    return np.where(p < t1, 0, np.where(p < t2, 1, 2))


def costo_politica(y, p, t1: float, t2: float, matriz: np.ndarray | None = None) -> float:
    """Costo promedio por cliente de aplicar los umbrales (t1, t2) a las probabilidades p."""
    m = matriz_costos() if matriz is None else matriz
    y = np.asarray(y, dtype=int)
    acciones = asignar_acciones(p, t1, t2)
    return float(m[acciones, y].mean())


def tabla_decisiones(y, p, t1: float, t2: float):
    """Matriz de confusión de la decisión: filas = acción tomada, columnas = lo que pasó.

    Es la versión de 3 acciones de la matriz de confusión binaria. Multiplicada celda
    por celda por matriz_costos() da el costo total de la política.
    """
    import pandas as pd

    acciones = pd.Categorical.from_codes(asignar_acciones(p, t1, t2), ACCIONES)
    real = pd.Categorical.from_codes(np.asarray(y, dtype=int), ["Paga (0)", "Impaga (1)"])
    return pd.crosstab(acciones, real, rownames=["Acción"], colnames=["Realidad"], dropna=False)


def costo_politica_trivial(y, accion: int, matriz: np.ndarray | None = None) -> float:
    """Costo promedio de aplicar la misma acción a todos (línea base sin modelo)."""
    m = matriz_costos() if matriz is None else matriz
    y = np.asarray(y, dtype=int)
    return float(m[accion, y].mean())
