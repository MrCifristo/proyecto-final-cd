"""Carga, limpieza y división del dataset de UCI (fase 1).

Toda la limpieza vive aquí para que el notebook de EDA y el de modelado usen
exactamente las mismas reglas. Las razones de cada regla están en
notebooks/01_framing_datos_eda.ipynb.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[1]
RUTA_CRUDO = RAIZ / "data" / "raw" / "default of credit card clients.xls"
DIR_PROCESADO = RAIZ / "data" / "processed"

OBJETIVO = "DEFAULT"
SEMILLA = 42
PROPORCION_TEST = 0.20

# Historial de 6 meses. Índice 1 = septiembre 2005 (el más reciente), 6 = abril.
MESES = range(1, 7)
PAY = [f"PAY_{i}" for i in MESES]
BILL = [f"BILL_AMT{i}" for i in MESES]
PAY_AMT = [f"PAY_AMT{i}" for i in MESES]
DEMOGRAFICAS = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]

ETIQUETAS = {
    "SEX": {1: "Hombre", 2: "Mujer"},
    "EDUCATION": {1: "Posgrado", 2: "Universidad", 3: "Secundaria", 4: "Otros"},
    "MARRIAGE": {1: "Casado", 2: "Soltero", 3: "Otros"},
}


def cargar_crudo(ruta: Path = RUTA_CRUDO) -> pd.DataFrame:
    """Lee el .xls tal como viene de UCI (los nombres reales están en la fila 2)."""
    return pd.read_excel(ruta, header=1)


def limpiar(crudo: pd.DataFrame) -> pd.DataFrame:
    """Aplica las reglas de limpieza documentadas en el notebook 01.

    1. Renombra el objetivo a DEFAULT y PAY_0 a PAY_1 (así coincide con BILL_AMT1/PAY_AMT1).
    2. EDUCATION 0, 5 y 6 (no documentados) -> 4 "Otros".
    3. MARRIAGE 0 (no documentado) -> 3 "Otros".
    4. ID pasa a ser el índice.

    No elimina filas: los 35 duplicados son en su mayoría cuentas sin movimiento,
    que es lo esperable de clientes distintos con el mismo perfil.
    PAY_* conserva sus códigos -2, -1 y 0 (ver notebook).
    """
    df = crudo.rename(columns={"default payment next month": OBJETIVO, "PAY_0": "PAY_1"})
    df = df.set_index("ID")
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    assert df.index.is_unique
    assert df.notna().all().all()
    assert set(df["EDUCATION"].unique()) <= {1, 2, 3, 4}
    assert set(df["MARRIAGE"].unique()) <= {1, 2, 3}
    assert set(df[OBJETIVO].unique()) == {0, 1}
    return df


def dividir(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split estratificado 80/20 con semilla fija. El test se usa una sola vez."""
    train, test = train_test_split(
        df, test_size=PROPORCION_TEST, stratify=df[OBJETIVO], random_state=SEMILLA
    )
    return train.sort_index(), test.sort_index()


def cargar_train() -> pd.DataFrame:
    """Para la fase de modelado: lee el entrenamiento ya limpio."""
    return pd.read_csv(DIR_PROCESADO / "train.csv", index_col="ID")


def cargar_test() -> pd.DataFrame:
    """Leer SOLO para la evaluación final. Una vez."""
    return pd.read_csv(DIR_PROCESADO / "test.csv", index_col="ID")
