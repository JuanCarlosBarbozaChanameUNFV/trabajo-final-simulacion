# ======================================================
# TRABAJO FINAL - SIMULACION DE SISTEMAS
# SIMULACION TO-BE: propuesta de aumento de agentes
# ======================================================


# Reutiliza el mismo motor de simulacion de simulacion_as_is.py
# ,variando unicamente el numero de
# agentes bajo la demanda de diciembre (el escenario mas
# exigente / peor caso del añoo).
# ======================================================

import os
import numpy as np
import pandas as pd

# Rutas basadas en la ubicacion del propio archivo .py (no en el
# directorio desde donde se ejecute), igual que en simulacion_as_is.py.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(SCRIPT_DIR, "..", "datos", "SimulacionCallCenter.csv")
CARPETA_RESULTADOS = os.path.join(SCRIPT_DIR, "..", "resultados")

SLA_SEGUNDOS = 60
N_REPLICAS = 30
OBJETIVO_SLA = 90.0


# ------------------------------------------------------
# 1. Cargar datos y calcular parametros de diciembre (peor caso)
# ------------------------------------------------------
df = pd.read_csv(CSV)
df["hora_inicio_llamada_dt"] = pd.to_datetime(
    df["fecha"] + " " + df["hora_inicio_llamada"], format="%Y-%m-%d %I:%M:%S %p"
)
df = df.sort_values(["fecha", "hora_inicio_llamada_dt"]).reset_index(drop=True)
df["tiempo_entre_llegadas_seg"] = (
    df.groupby("fecha")["hora_inicio_llamada_dt"].diff().dt.total_seconds()
)

MU = 1 / df["tiempo_servicio_seg"].mean()
diciembre = df[df["fecha"].str.startswith("2021-12")]
LAMBDA_DICIEMBRE = 1 / diciembre["tiempo_entre_llegadas_seg"].mean()
N_LLAMADAS_DICIEMBRE = len(diciembre)


# ------------------------------------------------------
# 2. Motor de simulacion 
# ------------------------------------------------------
def simular_call_center(lam, mu, n_agentes, n_llamadas, semilla):
    np.random.seed(semilla)
    tiempos_entre_llegadas = np.random.exponential(scale=1 / lam, size=n_llamadas)
    tiempos_servicio = np.random.exponential(scale=1 / mu, size=n_llamadas)
    tiempos_llegada = np.cumsum(tiempos_entre_llegadas)

    tiempo_libre_agente = np.zeros(n_agentes)
    inicio_atencion = np.zeros(n_llamadas)
    fin_atencion = np.zeros(n_llamadas)
    espera = np.zeros(n_llamadas)

    for i in range(n_llamadas):
        agente = np.argmin(tiempo_libre_agente)
        inicio_atencion[i] = max(tiempos_llegada[i], tiempo_libre_agente[agente])
        fin_atencion[i] = inicio_atencion[i] + tiempos_servicio[i]
        espera[i] = inicio_atencion[i] - tiempos_llegada[i]
        tiempo_libre_agente[agente] = fin_atencion[i]

    tiempo_total = fin_atencion[-1] - tiempos_llegada[0]
    return {
        "espera_promedio": espera.mean(),
        "pct_cumple_sla": (espera <= SLA_SEGUNDOS).mean() * 100,
        "utilizacion_teorica": lam / (n_agentes * mu) * 100,
        "utilizacion_simulada": np.sum(tiempos_servicio) / (n_agentes * tiempo_total) * 100,
    }


# ------------------------------------------------------
# 3. Barrido de 4 (AS-IS) a 8 agentes, 30 replicas cada uno
# ------------------------------------------------------
filas = []
for n_agentes in range(4, 9):
    resultados = [
        simular_call_center(LAMBDA_DICIEMBRE, MU, n_agentes, N_LLAMADAS_DICIEMBRE, semilla=300 + r)
        for r in range(N_REPLICAS)
    ]
    dfr = pd.DataFrame(resultados)
    filas.append({
        "n_agentes": n_agentes,
        "espera_promedio_media": dfr["espera_promedio"].mean(),
        "pct_cumple_sla_media": dfr["pct_cumple_sla"].mean(),
        "utilizacion_simulada_media": dfr["utilizacion_simulada"].mean(),
        "cumple_objetivo_90": dfr["pct_cumple_sla"].mean() >= OBJETIVO_SLA,
    })

df_to_be = pd.DataFrame(filas)
df_to_be.to_csv(os.path.join(CARPETA_RESULTADOS, "resultados_to_be.csv"), index=False)

print("=== Barrido de agentes (diciembre, 30 replicas cada uno) ===")
print(df_to_be.round(2).to_string(index=False))

cumplen = df_to_be[df_to_be["cumple_objetivo_90"]]
minimo_agentes = int(cumplen.iloc[0]["n_agentes"]) if len(cumplen) > 0 else None
print(f"\nNumero minimo de agentes para cumplir el 90% de SLA: {minimo_agentes}")
print(f"Guardado: {os.path.normpath(os.path.join(CARPETA_RESULTADOS, 'resultados_to_be.csv'))}")


# ------------------------------------------------------
# 4. Comparacion AS-IS (4 agentes) vs TO-BE (agentes minimos)
# ------------------------------------------------------
as_is = df_to_be[df_to_be["n_agentes"] == 4].iloc[0]
to_be = df_to_be[df_to_be["n_agentes"] == minimo_agentes].iloc[0]

valores_as_is = [as_is["espera_promedio_media"], as_is["pct_cumple_sla_media"], as_is["utilizacion_simulada_media"]]
valores_to_be = [to_be["espera_promedio_media"], to_be["pct_cumple_sla_media"], to_be["utilizacion_simulada_media"]]

comparacion = pd.DataFrame({
    "Indicador": ["Espera promedio (s)", "% cumple SLA", "Utilizacion simulada (%)"],
    "AS-IS (4 agentes)": valores_as_is,
    f"TO-BE ({minimo_agentes} agentes)": valores_to_be,
})
# Mejora: para espera y utilizacion, % de cambio relativo (negativo = baja);
# para % cumple SLA, diferencia en puntos porcentuales (no % relativo, porque
# ya es un porcentaje).
comparacion["Mejora"] = [
    f"{(valores_to_be[0] - valores_as_is[0]) / valores_as_is[0] * 100:+.1f}%",
    f"{valores_to_be[1] - valores_as_is[1]:+.2f} puntos",
    f"{(valores_to_be[2] - valores_as_is[2]) / valores_as_is[2] * 100:+.1f}%",
]

comparacion.to_csv(os.path.join(CARPETA_RESULTADOS, "comparacion_as_is_to_be.csv"), index=False)
print(f"\n=== Comparacion final: AS-IS (4 agentes) vs. TO-BE ({minimo_agentes} agentes) ===")
print(comparacion.round(2).to_string(index=False))
print(f"Guardado: {os.path.normpath(os.path.join(CARPETA_RESULTADOS, 'comparacion_as_is_to_be.csv'))}")
