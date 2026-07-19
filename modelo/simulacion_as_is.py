# ======================================================
# TRABAJO FINAL - SIMULACION DE SISTEMAS
# SIMULACION AS-IS: parametros, motor, validacion y escenarios
# ======================================================
# Motor de simulacion de eventos discretos (patron de asignacion
# de servidor con np.argmin + max, generalizado a multiples
# agentes), calibrado con datos reales de un call center de
# 4 agentes (dataset: Bangs, 2021, Kaggle).
# ======================================================

import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Rutas basadas en la ubicacion del propio archivo .py, no en el
# directorio desde donde se ejecute (asi funciona igual si se corre
# con "python simulacion_as_is.py" desde modelo/, o con el boton
# "Run" de VS Code desde la raiz del repo, o de cualquier otro lado).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(SCRIPT_DIR, "..", "datos", "SimulacionCallCenter.csv")
CARPETA_RESULTADOS = os.path.join(SCRIPT_DIR, "..", "resultados")

N_AGENTES_AS_IS = 4
SLA_SEGUNDOS = 60
N_REPLICAS = 30


# ------------------------------------------------------
# 1. Cargar datos y derivar tiempo entre llegadas
# ------------------------------------------------------
df = pd.read_csv(CSV)
df["hora_inicio_llamada_dt"] = pd.to_datetime(
    df["fecha"] + " " + df["hora_inicio_llamada"], format="%Y-%m-%d %I:%M:%S %p"
)
df = df.sort_values(["fecha", "hora_inicio_llamada_dt"]).reset_index(drop=True)
df["tiempo_entre_llegadas_seg"] = (
    df.groupby("fecha")["hora_inicio_llamada_dt"].diff().dt.total_seconds()
)
df["mes"] = df["fecha"].str[:7]

# mu (tasa de servicio) se calcula con TODO el anio: se asume
# constante en el tiempo (supuesto documentado en el informe).
MU = 1 / df["tiempo_servicio_seg"].mean()

diciembre = df[df["fecha"].str.startswith("2021-12")]
LAMBDA_DICIEMBRE = 1 / diciembre["tiempo_entre_llegadas_seg"].mean()
N_LLAMADAS_DICIEMBRE = len(diciembre)


# ------------------------------------------------------
# 2. Motor de simulacion (reutilizado por AS-IS y escenarios)
# ------------------------------------------------------
def simular_call_center(lam, mu, n_agentes, n_llamadas, semilla):
    """Simula n_llamadas en un call center con n_agentes en paralelo (FIFO)."""
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
    }, espera


# ------------------------------------------------------
# 3. Corrida de demostracion + validacion contra diciembre real
# ------------------------------------------------------
resultado_demo, espera_demo = simular_call_center(
    LAMBDA_DICIEMBRE, MU, N_AGENTES_AS_IS, N_LLAMADAS_DICIEMBRE, semilla=42
)
espera_real_dic = diciembre["tiempo_espera_seg"]
pct_cumple_real_dic = (diciembre["cumple_estandar"] == "Si").mean() * 100

print("=== Corrida de demostracion AS-IS (diciembre, 4 agentes, semilla 42) ===")
print(f"Espera promedio: {resultado_demo['espera_promedio']:.2f}s simulado vs "
      f"{espera_real_dic.mean():.2f}s real")
print(f"% cumple SLA: {resultado_demo['pct_cumple_sla']:.2f}% simulado vs "
      f"{pct_cumple_real_dic:.2f}% real")

estadistico_ks, p_valor_ks = ks_2samp(espera_demo, espera_real_dic)
print(f"Prueba K-S (2 muestras) simulado vs real: D={estadistico_ks:.4f}, "
      f"p-valor={p_valor_ks:.4f}")


# ------------------------------------------------------
# 4. 30 replicas del escenario AS-IS
# ------------------------------------------------------
filas_replicas = []
for r in range(N_REPLICAS):
    res, _ = simular_call_center(
        LAMBDA_DICIEMBRE, MU, N_AGENTES_AS_IS, N_LLAMADAS_DICIEMBRE, semilla=100 + r
    )
    res["replica"] = r + 1
    filas_replicas.append(res)

df_replicas = pd.DataFrame(filas_replicas)
df_replicas.to_csv(os.path.join(CARPETA_RESULTADOS, "resultados_as_is.csv"), index=False)
print("\n=== Resumen de las 30 replicas AS-IS ===")
print(df_replicas[["espera_promedio", "pct_cumple_sla", "utilizacion_simulada"]].describe().round(2))
print(f"Guardado: {os.path.normpath(os.path.join(CARPETA_RESULTADOS, 'resultados_as_is.csv'))}")


# ------------------------------------------------------
# 5. Validacion contra los 12 meses reales (no solo diciembre)
# ------------------------------------------------------
filas_validacion = []
for mes_actual in sorted(df["mes"].unique()):
    datos_mes = df[df["mes"] == mes_actual]
    lam_mes = 1 / datos_mes["tiempo_entre_llegadas_seg"].mean()
    pct_real_mes = (datos_mes["cumple_estandar"] == "Si").mean() * 100
    resultados_mes = [
        simular_call_center(lam_mes, MU, N_AGENTES_AS_IS, len(datos_mes), semilla=200 + r)[0]["pct_cumple_sla"]
        for r in range(10)
    ]
    filas_validacion.append({
        "mes": mes_actual,
        "pct_cumple_real": round(pct_real_mes, 2),
        "pct_cumple_simulado": round(np.mean(resultados_mes), 2),
    })

df_validacion = pd.DataFrame(filas_validacion)
df_validacion["diferencia"] = (df_validacion["pct_cumple_simulado"] - df_validacion["pct_cumple_real"]).round(2)
df_validacion.to_csv(os.path.join(CARPETA_RESULTADOS, "validacion_12_meses.csv"), index=False)
print("\n=== Validacion del modelo contra los 12 meses reales ===")
print(df_validacion.to_string(index=False))
print(f"Guardado: {os.path.normpath(os.path.join(CARPETA_RESULTADOS, 'validacion_12_meses.csv'))}")


# ------------------------------------------------------
# 6. Escenarios What-If (Optimista / Normal / Pesimista)
# ------------------------------------------------------
enero = df[df["fecha"].str.startswith("2021-01")]
escenarios = {
    "Optimista (enero)": {"lam": 1 / enero["tiempo_entre_llegadas_seg"].mean(), "n": len(enero)},
    "Normal (promedio anual)": {"lam": 1 / df["tiempo_entre_llegadas_seg"].mean(), "n": round(len(df) / 12)},
    "Pesimista (diciembre)": {"lam": LAMBDA_DICIEMBRE, "n": N_LLAMADAS_DICIEMBRE},
}

filas_escenarios = []
for nombre, params in escenarios.items():
    resultados = [
        simular_call_center(params["lam"], MU, N_AGENTES_AS_IS, params["n"], semilla=300 + r)[0]
        for r in range(N_REPLICAS)
    ]
    dfr = pd.DataFrame(resultados)
    filas_escenarios.append({
        "escenario": nombre,
        "espera_promedio_media": dfr["espera_promedio"].mean(),
        "pct_cumple_sla_media": dfr["pct_cumple_sla"].mean(),
        "utilizacion_simulada_media": dfr["utilizacion_simulada"].mean(),
    })

df_escenarios = pd.DataFrame(filas_escenarios)
df_escenarios.to_csv(os.path.join(CARPETA_RESULTADOS, "resultados_escenarios.csv"), index=False)
print("\n=== Escenarios What-If (Optimista / Normal / Pesimista) ===")
print(df_escenarios.round(2).to_string(index=False))
print(f"Guardado: {os.path.normpath(os.path.join(CARPETA_RESULTADOS, 'resultados_escenarios.csv'))}")
