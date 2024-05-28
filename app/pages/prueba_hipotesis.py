import scipy.stats as stats
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from utils import is_exam_period
from connection import load_data


df = load_data('./database.db')

# Split the data into exam periods and non-exam periods
exam_data = df[df['Is Exam Period'] == True]
non_exam_data = df[df['Is Exam Period'] == False]

print(exam_data.head())
print(non_exam_data.head())
# Perform t-test
t_stat, p_value = stats.ttest_ind(
    exam_data['Loan Time Seconds'], non_exam_data['Loan Time Seconds'], equal_var=False)

# Display t-test results
st.subheader("Prueba de Hipótesis")
st.write(f"Estadístico t: {t_stat:.4f}")
st.write(f"p-valor: {p_value:.4f}")

if p_value < 0.05:
    st.write("Rechazamos la hipótesis nula: Hay una diferencia significativa en la demanda de libros durante los periodos de exámenes.")
else:
    st.write("No podemos rechazar la hipótesis nula: No hay una diferencia significativa en la demanda de libros durante los periodos de exámenes.")

# Plot the box plot
# Plot the box plot
st.subheader("Distribución de los Tiempos de Préstamo")
fig, ax = plt.subplots(figsize=(10, 6))
data_to_plot = [exam_data['Loan Time Seconds'],
                non_exam_data['Loan Time Seconds']]
ax.boxplot(data_to_plot, labels=['Periodo de Exámenes', 'No Exámenes'])
ax.set_xlabel('Periodo')
ax.set_ylabel('Tiempo de Préstamo en Segundos')
ax.set_title(
    'Distribución de los Tiempos de Préstamo en Periodos de Exámenes y No Exámenes')
st.pyplot(fig)


confidence_level = 0.95
# confidence interval for the mean loan time during exam periods
exam_time_mean = df['Loan Time Seconds'].mean()
exam_time_sem = stats.sem(df['Loan Time Seconds'])
exam_time_ci = stats.t.interval(confidence_level, len(
    df['Loan Time Seconds'])-1, loc=exam_time_mean, scale=exam_time_sem)

st.write(f"Intervalo de Confianza para la Media del Tiempo de Préstamo en Periodos de Exámenes: {
         exam_time_ci}")

# confidence interval for the mean loan time outside exam periods
non_df = df[df['Is Exam Period'] == False]
non_exam_time_mean = non_df['Loan Time Seconds'].mean()
non_exam_time_sem = stats.sem(non_df['Loan Time Seconds'])
non_exam_time_ci = stats.t.interval(confidence_level, len(
    non_df['Loan Time Seconds'])-1, loc=non_exam_time_mean, scale=non_exam_time_sem)

st.write(f"Intervalo de Confianza para la Media del Tiempo de Préstamo fuera de Periodos de Exámenes: {
         non_exam_time_ci}")
# Cerrar la conexión

st.markdown("""
## Explicación del Análisis

### Prueba de Hipótesis
La prueba de hipótesis se utiliza para determinar si hay una diferencia significativa en la demanda de libros durante los periodos de exámenes en comparación con los periodos fuera de exámenes. Utilizamos una prueba t para comparar los tiempos de préstamo promedio en segundos entre los dos periodos.

- **Estadístico t**: Mide la diferencia entre las medias de los dos grupos en unidades de desviación estándar.
- **p-valor**: Indica la probabilidad de observar una diferencia tan grande como la observada, bajo la hipótesis nula de que no hay diferencia.

Si el p-valor es menor que 0.05, rechazamos la hipótesis nula, indicando que hay una diferencia significativa.

### Distribución de los Tiempos de Préstamo
El box plot muestra la distribución de los tiempos de préstamo en segundos para los periodos de exámenes y no exámenes. Esta visualización ayuda a comparar la dispersión y las características de los tiempos de préstamo en los dos periodos.

- **Eje X (Periodo)**: Representa los periodos de exámenes y no exámenes.
- **Eje Y (Tiempo de Préstamo en Segundos)**: Representa los tiempos de préstamo en segundos.

### Intervalos de Confianza
Los intervalos de confianza proporcionan un rango estimado en el que probablemente se encuentra la media del tiempo de préstamo para cada periodo.

- **Intervalo de Confianza para la Media del Tiempo de Préstamo en Periodos de Exámenes**: Proporciona un rango en el que se espera que se encuentre la media del tiempo de préstamo durante los periodos de exámenes, con un 95% de confianza.
- **Intervalo de Confianza para la Media del Tiempo de Préstamo fuera de Periodos de Exámenes**: Proporciona un rango en el que se espera que se encuentre la media del tiempo de préstamo fuera de los periodos de exámenes, con un 95% de confianza.
""")
