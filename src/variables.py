"""Variables candidatas que salieron del EDA (fase 2, modelado).

Cada una viene de un hallazgo del notebook 01. Ninguna usa el objetivo ni aprende
nada de los datos (son cuentas fila por fila), así que se pueden calcular antes
de la validación cruzada sin hacer trampa.
"""

import pandas as pd

from src import datos

CANDIDATAS = ["MESES_ATRASO", "ATRASO_ULTIMO", "USO_LINEA", "INACTIVO"]
PROTEGIDAS = ["SEX", "MARRIAGE"]


def agregar_candidatas(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia de df con las cuatro variables candidatas.

    MESES_ATRASO   cuántos de los 6 meses tienen atraso (PAY > 0). Cuenta meses, no suma
                   códigos: el código 1 no significa lo mismo en septiembre que en agosto.
    ATRASO_ULTIMO  1 si hubo atraso en septiembre, el mes más reciente.
    USO_LINEA      factura de septiembre entre el límite de crédito.
    INACTIVO       1 si no facturó ni pagó nada en los 6 meses.
    """
    out = df.copy()
    out["MESES_ATRASO"] = (df[datos.PAY] > 0).sum(axis=1)
    out["ATRASO_ULTIMO"] = (df["PAY_1"] > 0).astype(int)
    out["USO_LINEA"] = df["BILL_AMT1"] / df["LIMIT_BAL"]
    out["INACTIVO"] = (
        df[datos.BILL].eq(0).all(axis=1) & df[datos.PAY_AMT].eq(0).all(axis=1)
    ).astype(int)
    return out
