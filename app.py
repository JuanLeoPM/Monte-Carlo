# Monte-Carlo
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

#  Función para generar una distribución triangular
def triangular(inferior, moda, superior):
    return stats.triang(c=(moda - inferior) / (superior - inferior), loc=inferior, scale=superior - inferior)

# Función de simulación de Monte Carlo
def simulacion_montecarlo_valuacion(
    tasa_libre_riesgo, prima_riesgo_mercado, valor_capital, valor_deuda, beta_no_apalancada,
    beta_no_apalancada_terminal, costo_deuda_pretax_actual, costo_deuda_pretax_terminal,
    tasa_impuesto_efectiva_actual, tasa_impuesto_marginal,
    ingresos_base, cantidad_muestras
):
    np.random.seed(42)  
    datos = {
        "tasa_libre_riesgo": tasa_libre_riesgo.rvs(cantidad_muestras),
        "prima_riesgo_mercado": prima_riesgo_mercado.rvs(cantidad_muestras),
        "valor_capital": valor_capital.rvs(cantidad_muestras),
        "valor_deuda": valor_deuda.rvs(cantidad_muestras),
        "beta_no_apalancada": beta_no_apalancada.rvs(cantidad_muestras),
        "beta_no_apalancada_terminal": beta_no_apalancada_terminal.rvs(cantidad_muestras),
        "costo_deuda_pretax_actual": costo_deuda_pretax_actual.rvs(cantidad_muestras),
        "costo_deuda_pretax_terminal": costo_deuda_pretax_terminal.rvs(cantidad_muestras),
        "tasa_impuesto_efectiva_actual": tasa_impuesto_efectiva_actual.rvs(cantidad_muestras),
        "tasa_impuesto_marginal": tasa_impuesto_marginal.rvs(cantidad_muestras),
        "ingresos_base": ingresos_base.rvs(cantidad_muestras)
    }
    return pd.DataFrame(datos)

# Ejecutar la simulación con distribuciones
df_valuacion = simulacion_montecarlo_valuacion(
    tasa_libre_riesgo=stats.norm(0.04, 0.002),
    prima_riesgo_mercado=stats.norm(0.048, 0.001),
    valor_capital=triangular(45, 51.016, 57),
    valor_deuda=triangular(3.7, 3.887, 4),
    beta_no_apalancada=triangular(0.8, 0.9, 1),
    beta_no_apalancada_terminal=triangular(0.8, 0.9, 1),
    costo_deuda_pretax_actual=triangular(0.057, 0.06, 0.063),
    costo_deuda_pretax_terminal=triangular(0.052, 0.055, 0.058),
    tasa_impuesto_efectiva_actual=triangular(0.23, 0.24, 0.25),
    tasa_impuesto_marginal=triangular(0.23, 0.25, 0.27),
    ingresos_base=triangular(8.8, 9.2, 9.6),
    cantidad_muestras=20000
)

# Mostrar resumen estadístico
print(df_valuacion.describe())

# Función para describir y calcular valores por acción
def valuation_describer(df, sharesOutstanding):
    description = df.describe()
    description.loc['per_share'] = description.loc['mean'] / sharesOutstanding
    print(description)

valuation_describer(df_valuacion, sharesOutstanding=0.027589800)

# Generar gráficos 📊
plt.figure(figsize=(12, 8))

# Histograma de Valor de Capital
plt.subplot(2, 2, 1)
sns.histplot(df_valuacion["valor_capital"], bins=50, kde=True, color="blue")
plt.title("Distribución del Valor de Capital")
plt.xlabel("Valor de Capital")
plt.ylabel("Frecuencia")

# Histograma de Valor de Deuda
plt.subplot(2, 2, 2)
sns.histplot(df_valuacion["valor_deuda"], bins=50, kde=True, color="green")
plt.title("Distribución del Valor de Deuda")
plt.xlabel("Valor de Deuda")
plt.ylabel("Frecuencia")

# Boxplot de tasas
plt.subplot(2, 2, 3)
sns.boxplot(data=df_valuacion[["tasa_libre_riesgo", "prima_riesgo_mercado"]])
plt.title("Distribución de Tasas")
plt.xlabel("Variable")
plt.ylabel("Valor")

#  Scatter Plot Ingresos Base vs Beta No Apalancada
plt.subplot(2, 2, 4)
sns.scatterplot(x=df_valuacion["ingresos_base"], y=df_valuacion["beta_no_apalancada"], alpha=0.3)
plt.title("Ingresos Base vs Beta No Apalancada")
plt.xlabel("Ingresos Base")
plt.ylabel("Beta No Apalancada")

#  Ajustar el diseño y mostrar gráficos
plt.tight_layout()
plt.show()

# Guardar los gráficos
plt.savefig("resultados_montecarlo.png", dpi=300)


