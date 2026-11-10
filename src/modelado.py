"""Piezas compartidas de modelado y evaluación (fases 2 y 3).

Los notebooks 02 y 03 arman los modelos con estas funciones para que el modelo que
se evalúa en la prueba sea exactamente el que se eligió en validación cruzada.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import costos, datos

SEMILLA_CV = 0
MONTOS = ["LIMIT_BAL"] + datos.BILL + datos.PAY_AMT
CATEGORICAS = ["SEX", "EDUCATION", "MARRIAGE"]


def particiones() -> StratifiedKFold:
    """Los mismos 5 simulacros para todos los modelos: barajados y estratificados."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA_CV)


def log_con_signo(a):
    """sign(x) * log1p(|x|): comprime las colas largas y acepta facturas negativas."""
    return np.sign(a) * np.log1p(np.abs(a))


def logistica(columnas, pay_categorica: bool = True, C: float = 1.0, class_weight=None):
    """Regresión logística regularizada (L2) dentro de un Pipeline.

    Montos:      log con signo y luego escala estándar.
    Categóricas: una columna por categoría (one-hot). Si pay_categorica, las PAY_* también.
    Resto:       escala estándar.

    Todo lo que aprende de los datos (promedios, dispersiones, categorías) vive dentro
    del Pipeline, así que en validación cruzada se aprende solo con los datos de estudio.
    """
    columnas = list(columnas)
    montos = [c for c in MONTOS if c in columnas]
    cat = [c for c in CATEGORICAS if c in columnas]
    if pay_categorica:
        cat += [c for c in datos.PAY if c in columnas]
    resto = [c for c in columnas if c not in montos + cat]
    pre = ColumnTransformer(
        [
            ("montos", make_pipeline(FunctionTransformer(log_con_signo, feature_names_out="one-to-one"), StandardScaler()), montos),
            # min_frequency agrupa los códigos con menos de 30 clientes (PAY de 5 a 8 meses)
            ("categoricas", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30), cat),
            ("resto", StandardScaler(), resto),
        ]
    )
    return make_pipeline(pre, LogisticRegression(C=C, max_iter=2000, class_weight=class_weight))


def arbol(**perillas):
    return DecisionTreeClassifier(random_state=0, **perillas)


def bosque(**perillas):
    return RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1, **perillas)


def boosting(**perillas):
    return HistGradientBoostingClassifier(random_state=0, **perillas)


def _menos_costo(y, p):
    t1, t2 = costos.umbrales_teoricos()
    return -costos.costo_politica(y, p, t1, t2)


# Para GridSearchCV: sklearn maximiza, así que el puntaje es el costo con signo negativo.
PUNTAJE_COSTO = make_scorer(_menos_costo, response_method="predict_proba")


def costo_por_ronda(y, p, cv, X=None) -> np.ndarray:
    """Costo promedio por cliente en cada uno de los 5 simulacros."""
    t1, t2 = costos.umbrales_teoricos()
    y = np.asarray(y)
    X = np.zeros(len(y)) if X is None else X
    return np.array([costos.costo_politica(y[te], p[te], t1, t2) for _, te in cv.split(X, y)])


# --- Modelo elegido en notebooks/02_modelado.ipynb (sección 9) ---------------------------
# C se eligió con validación cruzada sobre el costo. El notebook 02 comprueba con un assert
# que este valor sigue siendo el que gana.
C_FINAL = 0.03


def columnas_finales(X) -> list[str]:
    """Las columnas del modelo elegido: originales + candidatas, sin SEX ni MARRIAGE."""
    from src import variables

    return [c for c in X.columns if c not in variables.PROTEGIDAS]


def modelo_final(X):
    """Logística regularizada, PAY_* como categoría, sin pesos por clase. Sin entrenar."""
    return logistica(columnas_finales(X), pay_categorica=True, C=C_FINAL)
