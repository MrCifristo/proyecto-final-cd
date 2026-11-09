# Riesgo de impago de tarjetas de crédito

Proyecto Final · Ciencia de Datos, Sección A · Universidad Francisco Marroquín · Segundo semestre 2026

**Integrantes:** George Albadr y Milton Beltrán

## La pregunta

¿Va a caer este cliente en impago el mes que viene? La predicción alimenta una decisión con
tres acciones: **mantener** el límite, mandarlo a **revisión manual** o **reducir** el límite.
Los dos errores no cuestan lo mismo, así que los umbrales se fijan según el costo de cada error
y no según la exactitud.

Detalle en [`docs/Propuesta.pdf`](docs/Propuesta.pdf) (aprobada). Enunciado en
[`docs/Enunciado.pdf`](docs/Enunciado.pdf).

## Datos

Default of Credit Card Clients (UCI, CC BY 4.0): 30,000 clientes de un banco de Taiwán,
de abril a septiembre de 2005. Procedencia, hash y primera revisión en [`data/README.md`](data/README.md).

## División del trabajo

| Sección del enunciado | Responsable |
|---|---|
| 1. Framing y costos de los errores | Milton |
| 2. Datos y EDA | Milton |
| 3. Modelado | George |
| 4. Evaluación | George |
| 5. Despliegue y límites | George |
| 6. Bitácora de colaboración con AI | Los dos |

## Estructura

```
.
├── data/
│   ├── raw/          # dataset original de UCI, sin tocar
│   └── README.md     # procedencia
├── docs/             # enunciado y propuesta
├── requirements.txt
└── README.md
```

## Entorno

Probado con Python 3.14.6.

```bash
python3 -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Entrega

Lunes 9 de noviembre de 2026, 23:59: notebook ejecutable, datos y diapositivas.
