# Bitácora de colaboración con AI

> Proyecto final · Ciencia de Datos, Sección A · Actualizado: 09/11/2026
> Fases 1, 2 y 3 anotadas (notebooks `01`, `02` y `03`). Modelado, evaluación final y revisión de los costos registrados.

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

## 5. Variables de atraso y uso de la línea

- **Qué se preguntó:** ¿Cómo construyo variables para contar los meses con atraso, el atraso reciente y el uso de la línea sin fugar información del conjunto de prueba?
- **Qué se usó:** Tres variables en `src/variables.py` calculadas fila por fila: `MESES_ATRASO` (meses con `PAY` mayor a 0), `ATRASO_ULTIMO` (septiembre) y `USO_LINEA` (factura de septiembre entre límite).
- **Qué se verificó:** Que al combinarlas con `PAY_*` el costo baja a 0.725 en los simulacros (con `PAY_*` como número bajaban de 0.760 a 0.741). Notebook 02, sección 2.
- **Qué se descartó:** Calcular agregaciones globales o promedios del grupo. Se calculan cliente por cliente para evitar fuga de datos.

## 6. Pesos por clase y calibración

- **Qué se preguntó:** ¿Conviene ponerle `class_weight="balanced"` a la logística por el desbalance de clases, o eso arruina las probabilidades para los umbrales de costo?
- **Qué se usó:** Regresión logística sin pesos por clase (`class_weight=None`). La asimetría de costos (5 a 1) ya está contemplada en los umbrales de decisión (0.12 y 0.35).
- **Qué se verificó:** Que con pesos la probabilidad media predicha se infla de 0.22 a 0.43 (como si la mitad impagara) y el costo con los umbrales sube a 0.868, peor que la vara de 0.853. Con calibración isotónica vuelve a 0.723, igual que sin pesos (0.724). Curva f10, Notebook 02, sección 4.
- **Qué se descartó:** Usar pesos balanceados más calibración. Añade pasos extra para terminar en el mismo resultado que ya da la logística natural.

## 7. Comparación de modelos: logística frente a ensambles

- **Qué se preguntó:** ¿Cómo armo una comparación justa bajo las mismas 5 particiones entre la logística y modelos más complejos como Random Forest y HistGradientBoosting usando la métrica de costo?
- **Qué se usó:** Pipeline con 5 simulacros idénticos para todos. Costos promedio: bosque 0.720, boosting 0.721, logística regularizada (`C = 0.03`) 0.724, árbol 0.731 (vara: 0.853).
- **Qué se verificó:** Que el boosting gana por solo 0.003 por cliente (0.4 % del costo, diez veces menor al ruido de la prueba). Además, la logística demostró ser muy estable ante `C` entre 0.01 y 10. Notebook 02, secciones 3 a 7.
- **Qué se descartó:** Boosting y Random Forest. El equipo eligió la regresión logística porque el banco necesita justificar con pesos directos por qué le baja el límite a un cliente, y no vale la pena perder esa explicabilidad por un 0.4 % marginal.

## 8. Evaluación en la prueba y cálculo con bootstrap

- **Qué se preguntó:** ¿Cómo armo la función de evaluación en `test.csv` para abrirlo una sola vez con un interruptor, y cómo calculo el intervalo de confianza del costo con bootstrap?
- **Qué se usó:** `ABRIR_PRUEBA = True` en el notebook 03, corrido una sola vez sobre los 6,000 clientes de prueba (1,327 impagos) con 2,000 remuestreos bootstrap al 95 % de confianza.
- **Qué se verificó:** Costo en prueba de 0.740 (IC 95 %: [0.712, 0.767]), superando la vara de 0.853 con un ahorro del 13.3 % (IC 95 %: [11.5 %, 15.1 %]). El 0.725 de los simulacros quedó dentro del intervalo. Notebook 03, secciones 2 a 4.
- **Qué se descartó:** Ajustar hiperparámetros o umbrales después de ver el examen. El número final que se reporta es 0.740.

## 9. Costos de los errores: ¿nos quedamos con los de la fase 1?

- **Qué se preguntó:** Milton propuso los costos iniciales (impago no detectado 5, reducir a quien paga 1, revisar 0.3, el analista detecta el 50 %, reducir deja el 30 % de la pérdida). Después de abrir la prueba le preguntamos a la AI cómo evaluar si esos supuestos se sostenían o si convenía cambiarlos.
- **Qué se usó:** Un análisis de sensibilidad con las probabilidades de los simulacros del modelo final (solo entrenamiento): 18 escenarios moviendo un supuesto a la vez y una malla de 270 combinaciones. Para cada una, los umbrales que les corresponden, el ahorro, el % a revisión y cuánto se pierde por seguir con 0.12 y 0.35. Notebook 02, sección 12.
- **Qué se verificó:** Que con los umbrales recalculados el modelo nunca queda claramente debajo de la vara (ahorro de −0.1 % a 32.7 %). Que no recalcularlos cuesta una mediana de 6.9 % más. Que el supuesto que más mueve la política es la revisión (cuánto cuesta y cuánto detecta el analista), no la razón 5 : 1. Y que esa razón coincide con la matriz de costos de German Credit (Statlog, UCI).
- **Qué se descartó:** Cambiar los costos. La prueba ya se había abierto, y cambiarlos habría obligado a reevaluar en ella. Sin cifras del banco, unos costos nuevos serían igual de supuestos, y elegirlos por el ahorro que dan sería circular. Nos quedamos con los de la fase 1 (decisión de Milton y George) y dejamos escrito que, al desplegar, los umbrales se recalculan con los costos reales del banco.

## Cómo seguir

Cada decisión nueva va en una sección como las de arriba, con las mismas cuatro líneas. `data/processed/test.csv` ya se abrió en la entrada 8. No se vuelve a correr para ajustar nada. El modelado y la evaluación cerraron los puntos que la fase 1 había dejado abiertos:

- `PAY_*` entra como categoría (captura la no linealidad principal).
- Las variables de atraso y uso de línea se calculan fila por fila y ayudan al modelo.
- `SEX` y `MARRIAGE` se retiraron por criterio ético y legal, con un impacto de costo de solo 0.2 %.
- Las probabilidades se usan sin pesos por clase (bien calibradas al prior real) con los umbrales teóricos 0.12 y 0.35.
- La evaluación final confirmó un ahorro del 13.3 % (IC 95 %: 11.5 %–15.1 %) frente a la política base.
