# Trabajo Final — Simulación de Sistemas

Análisis y propuesta de mejora del servicio de un call center de 4 agentes, mediante simulación de eventos discretos, a partir de datos reales de 2021.

**Universidad Nacional Federico Villarreal** — Facultad de Ingeniería Industrial y de Sistemas
Curso: Simulación de Sistemas (Mg. Maurice Frayssinet Delgado)

## Resumen

El call center registró 51,708 llamadas durante 2021. Mientras el volumen mensual se duplicó (3,000 en enero → 6,249 en diciembre), el cumplimiento del estándar de servicio (90% de llamadas atendidas en ≤60 segundos) cayó de 97.47% a 83.45%, con una capacidad fija de 4 agentes durante todo el año.

Se construyó y validó un modelo de simulación de eventos discretos (distribuciones Exponenciales, motor de asignación de agentes tipo FIFO) que reproduce el comportamiento real con menos de 1 punto porcentual de diferencia en los 12 meses del año. Con el modelo validado, se determinó que **incorporar un quinto agente** durante los meses de alta demanda recupera el cumplimiento del SLA (82.39% → 94.43%) y reduce la espera promedio en 75.6%.

Las columnas de  datos de Dataset fueron traducidas al español para su mejor entendimiento.

## Estructura del repositorio

```
trabajo-final-simulacion/
├── datos/                      # Dataset  (Bangs, 2021, Kaggle)
├── modelo/
│   ├── simulacion_as_is.py     # Parámetros, motor, validación, escenarios
│   ├── simulacion_to_be.py     # Barrido de agentes, propuesta de mejora
│   └── notebook_resultados.ipynb  # Presentación de resultados y gráficos
├── diagramas/                  # BPMN AS-IS y TO-BE 
├── resultados/                 # CSV de resultados 
├── informe/                    # Informe técnico completo (PDF)
└── presentacion/                # Presentación final (PPTX)
```

## Cómo correr el modelo

```bash
pip install numpy pandas scipy matplotlib
cd modelo
python simulacion_as_is.py
python simulacion_to_be.py
```
-
## Fuente de datos

Bangs, D. (2021). *Call Centre Queue Simulation* [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/donovanbangs/call-centre-queue-simulation
