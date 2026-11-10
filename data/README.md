# Datos: procedencia

| Campo | Valor |
|---|---|
| Dataset | Default of Credit Card Clients |
| Fuente | UCI Machine Learning Repository, id 350 |
| Página | https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients |
| Descarga | https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip |
| Autor | I-Cheng Yeh (donado en 2016) |
| Licencia | CC BY 4.0: se puede redistribuir citando la fuente |
| Descargado | 2026-11-09 |
| Archivo | `raw/default of credit card clients.xls` (el .zip trae solo este archivo) |
| SHA-256 | `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933` |

`raw/` no se edita a mano. Toda la limpieza se hace en el notebook a partir de este archivo.

## Cómo leerlo

```python
import pandas as pd
df = pd.read_excel("data/raw/default of credit card clients.xls", header=1)
```

La primera fila del Excel trae etiquetas `X1…X23, Y`. Los nombres reales de las columnas
están en la segunda fila, por eso va `header=1`.

## Primera revisión (2026-11-09)

- 30,000 filas y 25 columnas: `ID`, 23 explicativas y el objetivo `default payment next month`.
- Ningún valor faltante. Ningún `ID` repetido.
- Impago: 22.12 % (6,636 clientes), lo que confirma el ~22 % de la propuesta.
- Códigos fuera de la documentación de UCI:
  - `EDUCATION` 0, 5 y 6 (UCI documenta 1–4).
  - `MARRIAGE` 0 (UCI documenta 1–3).
  - `PAY_*` -2 y 0 (UCI documenta -1 = pagó a tiempo y 1–9 = meses de atraso).
- 35 filas idénticas si se quita el `ID`.
- El historial de pagos se llama `PAY_0, PAY_2, …, PAY_6`: no hay `PAY_1`.

Las decisiones de limpieza que salieron de esta revisión, y sus razones, están en
`notebooks/01_framing_datos_eda.ipynb` (sección 3). El código está en `src/datos.py`.

## Datos procesados

`processed/train.csv` y `processed/test.csv` salen de correr el notebook 01: datos limpios,
split estratificado 80/20 con semilla 42, índice `ID`. `PAY_0` se renombra a `PAY_1` y el
objetivo a `DEFAULT`. El test se abre una sola vez, en la evaluación final.
