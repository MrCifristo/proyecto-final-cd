# Bitácora de colaboración con AI

> Proyecto final · Ciencia de Datos, Sección A · Actualizado: 09/11/2026
> Fase 1 anotada (notebook `01_framing_datos_eda.ipynb`). Modelado, evaluación y diapositivas se agregan abajo, con el mismo formato.

La AI propuso opciones. La decisión y la verificación son del equipo. Cada entrada dice qué se preguntó, qué se usó, qué se verificó y qué se descartó.

## 1. Filas idénticas salvo por el ID

- **Qué se preguntó:** ¿Borramos las 35 filas que son idénticas a otra si se quita el `ID`?
- **Qué se usó:** No se eliminan. El archivo limpio sigue teniendo 30,000 clientes. `src/datos.limpiar` no borra filas.
- **Qué se verificó:** En el crudo hay 35 filas sobrantes (70 involucradas). 60 de esas 70 son cuentas sin factura ni pago en seis meses; las otras 10 son cargos fijos que se pagan completos. Los `ID` no se repiten. Sección 3 del notebook, corrida el 09/11/2026.
- **Qué se descartó:** Tratarlas como duplicados y borrarlas. Son clientes distintos con un perfil común (sobre todo cuentas inactivas). Borrarlas sesgaría la muestra contra esas cuentas. Son 35 de 30,000, así que el riesgo de que un perfil quede en entrenamiento y en prueba es pequeño.

## 2. Códigos no documentados de educación y estado civil

- **Qué se preguntó:** ¿Qué hacemos con `EDUCATION` 0, 5 y 6, y con `MARRIAGE` 0? UCI no los documenta.
- **Qué se usó:** `EDUCATION` 0, 5 y 6 pasan a 4 ("Otros"): 345 clientes, 1.2 %. `MARRIAGE` 0 pasa a 3 ("Otros"): 54 clientes. Después de limpiar, educación queda en {1: 10,585, 2: 14,030, 3: 4,917, 4: 468} y estado civil en {1: 13,659, 2: 15,964, 3: 377}.
- **Qué se verificó:** Conteos y tasa de impago por código en el archivo crudo, contra la documentación de UCI (educación 1–4, estado civil 1–3). El `assert` de `limpiar` comprueba que no quede ningún código fuera de esos rangos.
- **Qué se descartó:** Dejar 0, 5 y 6 como categorías propias, o imputarlas. Son muy pocos clientes y no hay forma de saber qué significan.

## 3. Montos extremos

- **Qué se preguntó:** ¿Recortamos los pagos y facturas extremos (un pago llega a 1.7 millones)?
- **Qué se usó:** No se recorta nadie. Para la regresión logística, el plan que queda escrito es transformar con `log1p` (en facturas, `sign(x)·log1p(|x|)`, porque pueden ser negativas) y estandarizar. Un modelo de árboles no necesita esa transformación. Sección 5.6.
- **Qué se verificó:** El límite máximo es 1,000,000 y el pago máximo (`PAY_AMT2`) es 1,684,259: la cola es larga, no un valor imposible. La edad va de 21 a 79. Las facturas negativas (1,930 clientes) son saldo a favor, no errores, y también se conservan.
- **Qué se descartó:** Winsorizar o borrar atípicos. Recortar cambiaría clientes reales. El tratamiento se deja para el modelo, no para la limpieza.

## 4. Conjunto de prueba cerrado

- **Qué se preguntó:** ¿Sobre qué datos hacemos el EDA y cuándo se abre la prueba?
- **Qué se usó:** Split 80/20 estratificado, semilla 42, antes del EDA. Entrenamiento: 24,000. Prueba: 6,000, en `data/processed/test.csv`. El EDA, los costos y los umbrales se calculan solo sobre entrenamiento. La prueba se abre una vez, al reportar el número final.
- **Qué se verificó:** Los índices de train y test no se cruzan, 24,000 + 6,000 = 30,000, y la tasa de impago queda en 0.221 en los dos. Sección 4, corrida el 09/11/2026.
- **Qué se descartó:** Explorar los 30,000 juntos, o mirar la prueba para elegir umbrales o el modelo.

## Cómo seguir

Cada decisión nueva va en una sección como las de arriba, con las mismas cuatro líneas. `data/processed/test.csv` no se abre hasta el número final. Lo que la fase 1 dejó abierto para modelado:

- Si `PAY_*` entra como número o como categoría.
- Si meses con atraso, atraso en el último mes, uso de la línea e `INACTIVO` bajan el costo en validación cruzada.
- Modelo con y sin `SEX` y `MARRIAGE`.
- Calibrar las probabilidades antes de usar los umbrales 0.12 y 0.35.
