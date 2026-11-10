# Riesgo de impago de tarjetas de crédito

**¿Va a dejar de pagar este cliente el mes que viene, y qué hace el banco con esa respuesta?**

Proyecto Final de Ciencia de Datos (Sección A, segundo semestre 2026), Universidad Francisco Marroquín.
George Albadr y Milton Beltrán.

---

## El problema

Un banco que emite tarjetas de crédito tiene que decidir qué hacer con la línea de cada cliente.
Si el banco no detecta a alguien que va a caer en mora, pierde dinero directamente. Si castiga
a alguien que sí iba a pagar, le quita crédito a un buen cliente y pierde su negocio. Los
afectados son el analista de crédito, el área de riesgos y el propio cliente.

## La decisión

El modelo estima la probabilidad de impago del mes siguiente. Con dos umbrales, esa probabilidad
se convierte en una de tres acciones:

| Probabilidad de impago | Acción |
|---|---|
| Baja | Mantener el límite |
| Media | Revisión manual por un analista |
| Alta | Reducir el límite |

Los dos tipos de error no cuestan lo mismo, así que **la exactitud no es la métrica**. Los
modelos se comparan con precisión y recall sobre la clase de impago. Los umbrales se fijan
minimizando el costo esperado bajo supuestos de costo que declaramos explícitamente.

## Los datos

[Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients),
del repositorio UCI Machine Learning (I-Cheng Yeh, licencia CC BY 4.0).

- 30,000 clientes de un banco de Taiwán, de abril a septiembre de 2005.
- 23 variables: límite de crédito, datos demográficos, seis meses de historial de pagos, montos
  facturados y montos pagados.
- Objetivo: impago en el mes siguiente. Ocurre en el 22.1 % de los casos.

La fuente, la licencia, el hash del archivo y la primera revisión de calidad están en
[`data/README.md`](data/README.md).

## Enfoque

1. **Línea base interpretable:** regresión logística regularizada.
2. **Comparación:** un modelo de árboles (bosque aleatorio o gradient boosting).
3. **Validación:** hiperparámetros elegidos con validación cruzada sobre entrenamiento. El
   conjunto de prueba se usa **una sola vez**.
4. **Evaluación:** curva precisión-recall, costo esperado y análisis de errores por edad, sexo
   y nivel educativo.

## Lo que este proyecto no puede concluir

Los datos son de **un solo banco, de un solo país y de 2005**. El modelo no se presenta como
válido para otro banco, otro mercado ni otra época. Los costos de cada error son supuestos
nuestros y no cifras del banco.

## Estructura

```
.
├── data/
│   ├── raw/          # dataset original de UCI, sin modificar
│   ├── processed/    # train.csv (24,000) y test.csv (6,000), limpios; los genera el notebook 01
│   └── README.md     # procedencia y primera revisión
├── docs/
│   ├── Enunciado.pdf
│   ├── Propuesta.pdf # propuesta aprobada
│   └── bitacora.md   # colaboración con AI: qué se preguntó, usó, verificó y descartó
├── figures/          # gráficas para las diapositivas
├── notebooks/
│   ├── 01_framing_datos_eda.ipynb   # fase 1: framing, costos, limpieza, split y EDA
│   ├── 02_modelado.ipynb            # fase 2: comparación de modelos, perillas y modelo elegido
│   └── 03_evaluacion_despliegue.ipynb  # fase 3: prueba (una vez), dónde falla, despliegue y límites
├── src/
│   ├── datos.py      # carga, limpieza y split (las mismas reglas para todas las fases)
│   ├── costos.py     # supuestos de costo, umbrales y costo de una política
│   ├── variables.py  # variables candidatas que salieron del EDA
│   └── modelado.py   # logística, árbol, bosque, boosting y el modelo elegido
├── requirements.txt
└── README.md
```

## Cómo correrlo

Requiere Python 3.14 (probado con 3.14.6).

```bash
git clone https://github.com/MrCifristo/proyecto-final-cd.git
cd proyecto-final-cd
python3 -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Los datos vienen incluidos en el repositorio, así que no hay que descargar nada aparte.

Para regenerar los datos procesados y las figuras de la fase 1, abrir
`notebooks/01_framing_datos_eda.ipynb` con el kernel de `.venv` y correr todo.

## Estado y siguiente paso

| Parte | Estado |
|---|---|
| 1. Framing y costos de los errores | Hecho en el notebook 01 (sección 1). Los costos son supuestos y se cambian en `src/costos.py` |
| 2. Datos, limpieza y EDA | Hecho en el notebook 01 (secciones 2 a 6) |
| 3. Modelado | Hecho en el notebook 02. Modelo elegido: logística regularizada (`modelado.modelo_final`) |
| 4. Evaluación | Hecho en el notebook 03. La prueba se abrió una sola vez: costo 0.740 por cliente, IC 95 % de 0.712 a 0.767, contra 0.853 de la vara. Ahorro 13.3 % |
| 5. Despliegue y límites | Hechos en el notebook 03 (secciones 6 a 8) |

Para empezar el modelado, leer las secciones 6 a 8 del notebook 01: el resumen de hallazgos con la
decisión que cambia cada uno, qué archivos y funciones quedan listos (`src/datos.py`, `src/costos.py`)
y qué quedó abierto a propósito para modelado (cómo entran las `PAY_*`, variables candidatas, con o
sin `SEX` y `MARRIAGE`, calibración antes de aplicar los umbrales).
